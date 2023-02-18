#include <cstddef>
#include <array>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

namespace {
template <typename T, std::size_t DIM>
struct ArrayWrapper {
  ArrayWrapper(const std::array<T, DIM>& a) {
    for (std::size_t i = 0; i < DIM; ++i) data[i] = a[i];
  }

  T data[DIM];
};
}  // namespace

template <typename T, std::size_t DIM>
static __global__ void assign_group_kernel(std::size_t n,
                                           ArrayWrapper<std::size_t, DIM> gridDimensions,
                                           const T* __restrict__ minCoordsMaxGlobal,
                                           ArrayWrapper<const T*, DIM> coord,
                                           std::size_t* __restrict__ out) {
  T minCoords[DIM];
  T maxCoords[DIM];
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    minCoords[dimIdx] = minCoordsMaxGlobal[dimIdx];
  }
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    maxCoords[dimIdx] = minCoordsMaxGlobal[DIM + dimIdx];
  }

  T gridSpacingInv[DIM];
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    gridSpacingInv[dimIdx] = gridDimensions.data[dimIdx] / (maxCoords[dimIdx] - minCoords[dimIdx]);
  }

  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    std::size_t groupIndex = 0;
    for (std::size_t dimIdx = DIM - 1;; --dimIdx) {
      const T* __restrict__ coordDimPtr = coord.data[dimIdx];
      groupIndex = groupIndex * gridDimensions.data[dimIdx] +
                   max(min(static_cast<std::size_t>(gridSpacingInv[dimIdx] *
                                                    (coordDimPtr[i] - minCoords[dimIdx])),
                           gridDimensions.data[dimIdx] - 1),
                       std::size_t(0));

      if (!dimIdx) break;
    }

    out[i] = groupIndex;
  }
}


template <std::size_t BLOCK_THREADS>
static __global__ void group_count_kernel(std::size_t nGroups, std::size_t n,
                                          const std::size_t* __restrict__ in,
                                          std::size_t* groupCount) {
  using BlockExchangeType = api::cub::BlockExchange<std::size_t, BLOCK_THREADS, BLOCK_THREADS>;
  __shared__ std::size_t inCache[BLOCK_THREADS];

  for (std::size_t groupStart = blockIdx.y * BLOCK_THREADS; groupStart < nGroups;
       groupStart += gridDim.y * BLOCK_THREADS) {
    std::size_t myCount = 0;
    std::size_t myGroup = groupStart + threadIdx.x;
    for (std::size_t inStart = 0; inStart < n; inStart += BLOCK_THREADS) {
      if (inStart + threadIdx.x < n) inCache[threadIdx.x] = in[inStart + threadIdx.x];
      __syncthreads();

      for (std::size_t i = 0; i < min(BLOCK_THREADS, n - inStart); ++i) {
        myCount += (inCache[i] == myGroup);
      }
    }

    if(myGroup < nGroups) groupCount[myGroup] = myCount;
  }
}

template <typename T>
static __global__ void apply_permut_kernel(std::size_t n, const std::size_t* __restrict__ permut,
                                           const T* __restrict__ in, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[permut[i]] = in[i];
  }
}

template <typename T>
static __global__ void reverse_permut_kernel(std::size_t n, const std::size_t* __restrict__ permut,
                                             const T* __restrict__ in, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = in[permut[i]];
  }
}

template <typename T, typename>
auto DomainPartition::grid(const std::shared_ptr<ContextInternal>& ctx,
                           std::array<std::size_t, 3> gridDimensions, std::size_t n,
                           std::array<const T*, 3> coord) -> DomainPartition {
  constexpr std::size_t DIM = 3;

  const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                        std::size_t(1), std::multiplies<std::size_t>());
  if (gridSize <= 1) return DomainPartition::single(ctx, n);

  auto& q = ctx->gpu_queue();

  auto minMaxBuffer = q.create_device_buffer<T>(2 * DIM);

  // Compute the minimum and maximum in each dimension stored in minMax as
  // (min_x, min_y, ..., max_x, max_y, ...)
  {
    std::size_t worksize = 0;
    api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(nullptr, worksize, nullptr, nullptr,
                                                                n, q.stream()));

    auto workBuffer = q.create_device_buffer<char>(worksize);

    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(
          workBuffer.get(), worksize, coord[dimIdx], minMaxBuffer.get() + dimIdx, n, q.stream()));
    }

    api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(nullptr, worksize, nullptr, nullptr,
                                                                n, q.stream()));
    if (worksize > workBuffer.size()) workBuffer = q.create_device_buffer<char>(worksize);

    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(
          workBuffer.get(), worksize, coord[dimIdx], minMaxBuffer.get() + DIM + dimIdx, n,
          q.stream()));
    }
    q.sync(); //TODO: remove
  }

  // Assign the group idx to each input element and store temporarily in the permutation array
  auto permutBuffer = q.create_device_buffer<std::size_t>(n);
  {
    constexpr int blockSize = 256;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
    api::launch_kernel(assign_group_kernel<T, DIM>, grid, block, 0, q.stream(), n, gridDimensions,
                       minMaxBuffer.get(), coord, permutBuffer.get());
    q.sync(); //TODO: remove
  }

  // Compute the number of elements in each group
  auto groupSizesBuffer = q.create_device_buffer<std::size_t>(gridSize);
  auto groupBeginBuffer = q.create_device_buffer<std::size_t>(gridSize);
  {
    // constexpr int blockSize =
    //     128;  // should be small, since each thread will create local array of size blockSize
    // const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    // const auto grid = kernel_launch_grid(q.device_prop(), {n, gridSize / blockSize + 1, 1}, block);
    // api::launch_kernel(
    //     group_count_kernel<blockSize, api::cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>,
    //     grid, block, 0, q.stream(), gridSize, n, permutBuffer.get(), groupSizesBuffer.get());

    constexpr int blockSize = 512;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {n, gridSize / blockSize + 1, 1}, block);
    api::launch_kernel(group_count_kernel<blockSize>, grid, block, 0, q.stream(), gridSize, n,
                       permutBuffer.get(), groupSizesBuffer.get());

    q.sync(); //TODO: remove
  }

  // Compute the rolling sum through exclusive scan to get the start index for each group
  {
    std::size_t worksize = 0;
    api::check_status(api::cub::DeviceScan::ExclusiveSum<const std::size_t*, std::size_t*>(
        nullptr, worksize, nullptr, nullptr, gridSize, q.stream()));
    auto workBuffer = q.create_device_buffer<char>(worksize);

    api::check_status(api::cub::DeviceScan::ExclusiveSum<const std::size_t*, std::size_t*>(
        workBuffer.get(), worksize, groupSizesBuffer.get(), groupBeginBuffer.get(), gridSize,
        q.stream()));
    q.sync(); //TODO: remove
  }

  auto groupBufferHost = q.create_pinned_buffer<Group>(gridSize);
  auto permutBufferHost = q.create_host_buffer<std::size_t>(permutBuffer.size());
  // Finally compute permutation array
  {
    static_assert(sizeof(Group) == 2 * sizeof(std::size_t),
                  "Assuming interleaved begin, size for copy");

    gpu::api::memcpy_2d_async(groupBufferHost.get(), sizeof(Group), groupBeginBuffer.get(),
                              sizeof(std::size_t), sizeof(std::size_t), gridSize,
                              gpu::api::flag::MemcpyDeviceToHost, q.stream());
    q.sync(); //TODO: remove

    gpu::api::memcpy_2d_async(reinterpret_cast<std::size_t*>(groupBufferHost.get()) + 1,
                              sizeof(Group), groupSizesBuffer.get(), sizeof(std::size_t),
                              sizeof(std::size_t), gridSize, gpu::api::flag::MemcpyDeviceToHost,
                              q.stream());
    q.sync(); //TODO: remove
    gpu::api::memcpy_async(permutBufferHost.get(), permutBuffer.get(), permutBuffer.size_in_bytes(),
                           gpu::api::flag::MemcpyDeviceToHost, q.stream());
    q.sync(); //TODO: remove

    // make sure copy operations are done
    q.sync();

    // compute permutation index for each data point and restore group sizes.
    auto* __restrict__ groupsPtr = groupBufferHost.get();
    auto* __restrict__ permutPtr = permutBufferHost.get();
    for (std::size_t i = 0; i < n; ++i) {
      permutPtr[i] = groupsPtr[permutPtr[i]].begin++;
    }

    // Restore begin index
    for (std::size_t i = 0; i < groupBufferHost.size(); ++i) {
      groupsPtr[i].begin -= groupsPtr[i].size;
    }

    // copy permutation back to device
    gpu::api::memcpy_async(permutBuffer.get(), permutBufferHost.get(), permutBuffer.size_in_bytes(),
                           gpu::api::flag::MemcpyHostToDevice, q.stream());
  }

  return DomainPartition(ctx, std::move(permutBuffer), std::move(groupBufferHost));
}

template <typename F, typename>
auto DomainPartition::apply(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), inDevice, outDevice);

        } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
          gpu::api::memcpy_async(outDevice, inDevice, sizeof(F) * arg,
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::apply(F* inOutDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          auto tmpBuffer = q.create_device_buffer<F>(arg.size());
          gpu::api::memcpy_async(tmpBuffer.get(), inOutDevice, sizeof(F) * arg.size(),
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), tmpBuffer.get(), inOutDevice);
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::reverse(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), inDevice, outDevice);

        } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
          gpu::api::memcpy_async(outDevice, inDevice, sizeof(F) * arg,
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::reverse(F* inOutDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          auto tmpBuffer = q.create_device_buffer<F>(arg.size());
          gpu::api::memcpy_async(tmpBuffer.get(), inOutDevice, sizeof(F) * arg.size(),
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), tmpBuffer.get(), inOutDevice);
        }
      },
      permut_);
}

template auto DomainPartition::grid<float>(const std::shared_ptr<ContextInternal>& ctx,
                                           std::array<std::size_t, 3> gridDimensions, std::size_t n,
                                           std::array<const float*, 3> coord) -> DomainPartition;

template auto DomainPartition::grid<double>(const std::shared_ptr<ContextInternal>& ctx,
                                            std::array<std::size_t, 3> gridDimensions,
                                            std::size_t n, std::array<const double*, 3> coord)
    -> DomainPartition;

template auto DomainPartition::apply<float>(const float* __restrict__ inDevice,
                                            float* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<double>(const double* __restrict__ inDevice,
                                             double* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<float>(float* inOutDevice) -> void;

template auto DomainPartition::apply<double>(double* inOutDevice) -> void;

template auto DomainPartition::apply<api::ComplexFloatType>(api::ComplexFloatType* inOutDevice)
    -> void;

template auto DomainPartition::apply<api::ComplexDoubleType>(api::ComplexDoubleType* inOutDevice)
    -> void;

template auto DomainPartition::apply<api::ComplexFloatType>(
    const api::ComplexFloatType* __restrict__ inDevice,
    api::ComplexFloatType* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<api::ComplexDoubleType>(
    const api::ComplexDoubleType* __restrict__ inDevice,
    api::ComplexDoubleType* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<float>(const float* __restrict__ inDevice,
                                              float* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<double>(const double* __restrict__ inDevice,
                                               double* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<float>(float* inOutDevice) -> void;

template auto DomainPartition::reverse<double>(double* inOutDevice) -> void;

template auto DomainPartition::reverse<api::ComplexFloatType>(api::ComplexFloatType* inOutDevice)
    -> void;

template auto DomainPartition::reverse<api::ComplexDoubleType>(api::ComplexDoubleType* inOutDevice)
    -> void;

template auto DomainPartition::reverse<api::ComplexFloatType>(
    const api::ComplexFloatType* __restrict__ inDevice,
    api::ComplexFloatType* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<api::ComplexDoubleType>(
    const api::ComplexDoubleType* __restrict__ inDevice,
    api::ComplexDoubleType* __restrict__ outDevice) -> void;

}  // namespace gpu
}  // namespace bipp
