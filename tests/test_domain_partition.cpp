#include <array>
#include <memory>
#include <tuple>
#include <numeric>
#include <vector>
#include <variant>
#include <random>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gtest/gtest.h"
#include "host/domain_partition.hpp"
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/domain_partition.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

template <typename T>
class DomainPartitionTest : public ::testing::TestWithParam<std::tuple<BippProcessingUnit>> {
public:
  using ValueType = T;

  DomainPartitionTest() : ctx_(new bipp::ContextInternal(std::get<0>(GetParam()))) {}


  auto test_grid(std::array<std::size_t, 3> gridDimensions, std::array<std::vector<T>, 3> domain) {
    ASSERT_EQ(domain[0].size(), domain[1].size());
    ASSERT_EQ(domain[0].size(), domain[2].size());

    const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                          std::size_t(1), std::multiplies<std::size_t>());

    std::array<T, 3> minCoord, maxCoord, gridSpacing;

    for (std::size_t dimIdx = 0; dimIdx < minCoord.size(); ++dimIdx) {
      minCoord[dimIdx] = *std::min_element(domain[dimIdx].begin(), domain[dimIdx].end());
      maxCoord[dimIdx] = *std::max_element(domain[dimIdx].begin(), domain[dimIdx].end());
      gridSpacing[dimIdx] = (maxCoord[dimIdx] - minCoord[dimIdx]) / gridDimensions[dimIdx];
    }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    std::variant<bipp::host::DomainPartition, bipp::gpu::DomainPartition> partition =
        bipp::host::DomainPartition::single(ctx_, domain[0].size());
#else
    std::variant<bipp::host::DomainPartition> partition =
        bipp::host::DomainPartition::single(ctx_, domain[0].size());
#endif

    if(ctx_->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto bufferX = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());
      auto bufferY = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());
      auto bufferZ = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());

      bipp::gpu::api::memcpy_async(bufferX.get(), domain[0].data(), bufferX.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
      bipp::gpu::api::memcpy_async(bufferY.get(), domain[1].data(), bufferY.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
      bipp::gpu::api::memcpy_async(bufferZ.get(), domain[2].data(), bufferZ.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

      partition = bipp::gpu::DomainPartition::grid<T>(
          ctx_, gridDimensions, domain[0].size(), {bufferX.get(), bufferY.get(), bufferZ.get()});

#else
      ASSERT_TRUE(false);
#endif
    } else {
      partition = bipp::host::DomainPartition::grid<T, 3>(
          ctx_, gridDimensions, domain[0].size(),
          {domain[0].data(), domain[1].data(), domain[2].data()});
    }

    std::visit(
        [&](auto&& arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          // Make sure groups cover all indices
          std::vector<bool> inputCover(domain[0].size());
          for (const auto& [begin, size] : arg.groups()) {
            ASSERT_LE(begin + size, domain[0].size());
            for (std::size_t i = begin; i < begin + size; ++i) {
              inputCover[i] = true;
            }
          }
          for (std::size_t i = 0; i < domain[0].size(); ++i) {
            ASSERT_TRUE(inputCover[i]);
          }

          for (std::size_t dimIdx = 0; dimIdx < minCoord.size(); ++dimIdx) {
            auto dataInPlace = domain[dimIdx];
            auto dataOutOfPlace = std::vector<T>(dataInPlace.size());

            // apply in place and out of place
            if constexpr (std::is_same_v<variantType, bipp::host::DomainPartition>) {
              arg.apply(dataInPlace.data(), dataOutOfPlace.data());
              arg.apply(dataInPlace.data());
            } else {
              auto dataInPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataInPlace.size());
              auto dataOutOfPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataOutOfPlace.size());

              bipp::gpu::api::memcpy_async(
                  dataInPlaceDevice.get(), dataInPlace.data(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

              arg.apply(dataInPlaceDevice.get(), dataOutOfPlaceDevice.get());
              arg.apply(dataInPlaceDevice.get());

              bipp::gpu::api::memcpy_async(
                  dataInPlace.data(), dataInPlaceDevice.get(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
              bipp::gpu::api::memcpy_async(dataOutOfPlace.data(), dataOutOfPlaceDevice.get(),
                                           dataOutOfPlaceDevice.size_in_bytes(),
                                           bipp::gpu::api::flag::MemcpyDefault,
                                           ctx_->gpu_queue().stream());

              ctx_->gpu_queue().sync();
            }

            // check data
            for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
              ASSERT_EQ(dataInPlace[i], dataOutOfPlace[i]);
            }

            for (const auto& [begin, size] : arg.groups()) {
              if (size) {
                auto minGroup = *std::min_element(dataInPlace.begin() + begin,
                                                  dataInPlace.begin() + begin + size);
                auto maxGroup = *std::max_element(dataInPlace.begin() + begin,
                                                  dataInPlace.begin() + begin + size);

                ASSERT_LE(maxGroup - minGroup, gridSpacing[dimIdx]);
              }
            }

            // reverse in place and out of place
            if constexpr (std::is_same_v<variantType, bipp::host::DomainPartition>) {
              arg.reverse(dataInPlace.data(), dataOutOfPlace.data());
              arg.reverse(dataInPlace.data());
            } else {
              auto dataInPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataInPlace.size());
              auto dataOutOfPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataOutOfPlace.size());

              bipp::gpu::api::memcpy_async(
                  dataInPlaceDevice.get(), dataInPlace.data(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

              arg.reverse(dataInPlaceDevice.get(), dataOutOfPlaceDevice.get());
              arg.reverse(dataInPlaceDevice.get());

              bipp::gpu::api::memcpy_async(
                  dataInPlace.data(), dataInPlaceDevice.get(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
              bipp::gpu::api::memcpy_async(dataOutOfPlace.data(), dataOutOfPlaceDevice.get(),
                                           dataOutOfPlaceDevice.size_in_bytes(),
                                           bipp::gpu::api::flag::MemcpyDefault,
                                           ctx_->gpu_queue().stream());

              ctx_->gpu_queue().sync();
            }


            // check reversed data
            for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
              ASSERT_EQ(dataInPlace[i], dataOutOfPlace[i]);
              ASSERT_EQ(dataInPlace[i], domain[dimIdx][i]);
            }

          }


        },
        partition);



  }

  std::shared_ptr<bipp::ContextInternal> ctx_;
};

using DomainPartitionSingle = DomainPartitionTest<float>;
using DomainPartitioDouble = DomainPartitionTest<double>;

template <typename T>
static auto test_grid_simple(DomainPartitionTest<T>& t) ->  void {
  std::minstd_rand randGen(42);
  std::uniform_real_distribution<T> distri(-5.0, 10.0);

  std::vector<T> x(100);
  std::vector<T> y(100);
  std::vector<T> z(100);

  for (auto& val : x) val = distri(randGen);
  for (auto& val : y) val = distri(randGen);
  for (auto& val : z) val = distri(randGen);

  t.test_grid({2, 2, 2}, {x, y, z});
}

TEST_P(DomainPartitionSingle, gridSimple) { test_grid_simple(*this);}


static auto param_type_names(const ::testing::TestParamInfo<std::tuple<BippProcessingUnit>>& info)
    -> std::string {
  std::stringstream stream;

  if (std::get<0>(info.param) == BIPP_PU_CPU)
    stream << "CPU";
  else
    stream << "GPU";

  return stream.str();
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#define TEST_PROCESSING_UNITS BIPP_PU_CPU, BIPP_PU_GPU
#else
#define TEST_PROCESSING_UNITS BIPP_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(DomainPartition, DomainPartitionSingle,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);

// INSTANTIATE_TEST_SUITE_P(Lofar, StandardSynthesisLofarDouble,
//                          ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
//                          param_type_names);
