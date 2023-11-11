#include "gpu/nufft_synthesis.hpp"

#include <cassert>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/min_max_element.hpp"
#include "gpu/kernels/nuft_sum.hpp"
#include "gpu/kernels/scale_vector.hpp"
#include "gpu/nufft_3d3.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace gpu {

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                  std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                                  HostArray<BippFilter, 1> filter, DeviceArray<T, 2> pixel)

    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nIntervals_(nIntervals),
      nFilter_(filter.shape()),
      nPixel_(pixel.shape(0)),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      filter_(std::move(filter)),
      pixel_(std::move(pixel)),
      img_(ctx_->gpu_queue().create_device_array<T, 3>({nPixel_, nIntervals_, nFilter_})),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      collectCount_(0),
      totalCollectCount_(0) {
  auto& queue = ctx_->gpu_queue();
  api::memset_async(img_.data(), 0, img_.size() * sizeof(T), queue.stream());

  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          imgPartition_ = DomainPartition::grid<T>(
              ctx_, arg.dimensions,
              {pixel.slice_view(0), pixel.slice_view(1), pixel.slice_view(2)});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(pixel_.slice_view(0));
  imgPartition_.apply(pixel_.slice_view(1));
  imgPartition_.apply(pixel_.slice_view(2));

  if (opt_.collectGroupSize && opt_.collectGroupSize.value() > 0) {
    nMaxInputCount_ = opt_.collectGroupSize.value();
  } else {
    // use at most 20% of memory more accumulation, but not more than 200
    // iterations.
    std::size_t freeMem, totalMem;
    api::mem_get_info(&freeMem, &totalMem);
    nMaxInputCount_ = (totalMem / 5) / (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ *
                                        sizeof(api::ComplexType<T>));
    nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);
  }

  virtualVis_ = queue.create_device_array<api::ComplexType<T>, 3>(
      {nAntenna_ * nAntenna_ * nMaxInputCount_, nIntervals_, nFilter_});
  uvw_ = queue.create_device_array<T, 2>({nAntenna_ * nAntenna_ * nMaxInputCount_, 3});
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
                                ConstHostView<api::ComplexType<T>, 2> sHost,
                                ConstDeviceView<api::ComplexType<T>, 2> s,
                                ConstDeviceView<api::ComplexType<T>, 2> w,
                                ConstDeviceView<T, 2> xyz, ConstDeviceView<T, 2> uvw) -> void {
  assert(xyz.shape(0) == nAntenna_);
  assert(xyz.shape(1) == 3);
  assert(intervals.shape(1) == nIntervals_);
  assert(intervals.shape(0) == 2);
  assert(w.shape(0) == nAntenna_);
  assert(w.shape(1) == nBeam_);
  assert(!s.size() || s.shape(0) == nBeam_);
  assert(!s.size() || s.shape(1) == nBeam_);
  assert(uvw.shape(0) == nAntenna_ * nAntenna_);
  assert(uvw.shape(1) == 3);

  auto& queue = ctx_->gpu_queue();

  // store coordinates
  copy(queue, uvw,
       uvw_.sub_view({collectCount_ * nAntenna_ * nAntenna_, 0}, {nAntenna_ * nAntenna_, 3}));

  auto v = queue.create_device_array<api::ComplexType<T>, 2>({nBeam_, nEig});
  auto d = queue.create_device_array<T, 1>(nEig);

  {
    auto g = queue.create_device_array<api::ComplexType<T>, 2>({nBeam_, nBeam_});

    gram_matrix<T>(*ctx_, w, xyz, wl, g);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s.size())
      eigh<T>(*ctx_, nEig, sHost, s, g, d, v);
    else {
      auto gHost = queue.create_pinned_array<api::ComplexType<T>, 2>(g.shape());
      copy(queue, g, gHost);
      queue.sync();  // finish copy
      eigh<T>(*ctx_, nEig, gHost, g, s, d, v);
    }
  }

  // Reverse beamforming
  auto vUnbeam = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna_, v.shape(1)});
  api::blas::gemm<api::ComplexType<T>>(queue.blas_handle(), api::blas::operation::None,
                                       api::blas::operation::None, {1, 0}, w, v, {0, 0}, vUnbeam);

  // slice virtual visibility for current step
  auto virtVisCurrent =
      virtualVis_.sub_view({collectCount_ * nAntenna_ * nAntenna_, 0, 0},
                           {nAntenna_ * nAntenna_, virtualVis_.shape(1), virtualVis_.shape(2)});

  // compute virtual visibilities
  virtual_vis(*ctx_, filter_, intervals, d, vUnbeam, virtVisCurrent);

  ++collectCount_;
  ++totalCollectCount_;
  ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "collect count: {} / {}", collectCount_, nMaxInputCount_);
  if (collectCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  auto& queue = ctx_->gpu_queue();

  if (collectCount_) {
    auto output = queue.create_device_array<api::ComplexType<T>, 1>(nPixel_);

    const auto nInputPoints = nAntenna_ * nAntenna_ * collectCount_;

    auto uvwX = uvw_.slice_view(0).sub_view({0}, {nInputPoints});
    auto uvwY = uvw_.slice_view(1).sub_view({0}, {nInputPoints});
    ;
    auto uvwZ = uvw_.slice_view(2).sub_view({0}, {nInputPoints});
    ;

    auto currentVirtualVis = virtualVis_.sub_view(
        {0, 0, 0}, {nInputPoints, virtualVis_.shape(1), virtualVis_.shape(2)});

    auto pixelX = pixel_.slice_view(0);
    auto pixelY = pixel_.slice_view(1);
    auto pixelZ = pixel_.slice_view(2);

    auto inputPartition = std::visit(
        [&](auto&& arg) -> DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                               arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
            return DomainPartition::grid<T>(ctx_, arg.dimensions, {uvwX, uvwY, uvwZ});

          } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
            return DomainPartition::none(ctx_, nInputPoints);

          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
            auto minMaxDevice = queue.create_device_array<T, 1>(12);
            min_element(queue, nInputPoints, uvwX.data(), minMaxDevice.data());
            max_element(queue, nInputPoints, uvwX.data(), minMaxDevice.data() + 1);

            min_element(queue, nInputPoints, uvwY.data(), minMaxDevice.data() + 2);
            max_element(queue, nInputPoints, uvwY.data(), minMaxDevice.data() + 3);

            min_element(queue, nInputPoints, uvwZ.data(), minMaxDevice.data() + 4);
            max_element(queue, nInputPoints, uvwZ.data(), minMaxDevice.data() + 5);

            min_element(queue, nPixel_, pixelX.data(), minMaxDevice.data() + 6);
            max_element(queue, nPixel_, pixelX.data(), minMaxDevice.data() + 7);

            min_element(queue, nPixel_, pixelY.data(), minMaxDevice.data() + 8);
            max_element(queue, nPixel_, pixelY.data(), minMaxDevice.data() + 9);

            min_element(queue, nPixel_, pixelZ.data(), minMaxDevice.data() + 10);
            max_element(queue, nPixel_, pixelZ.data(), minMaxDevice.data() + 11);

            auto minMaxHost = queue.create_host_array<T, 1>(minMaxDevice.shape());
            copy(queue, minMaxDevice, minMaxHost);

            queue.sync();

            const T* minMaxPtr = minMaxHost.data();

            std::array<double, 3> uvwExtent = {minMaxPtr[1] - minMaxPtr[0],
                                               minMaxPtr[3] - minMaxPtr[2],
                                               minMaxPtr[5] - minMaxPtr[4]};
            std::array<double, 3> imgExtent = {minMaxPtr[7] - minMaxPtr[6],
                                               minMaxPtr[9] - minMaxPtr[8],
                                               minMaxPtr[11] - minMaxPtr[10]};

            // Use at most 12.5% of total memory for fft grid
            const auto gridSize = optimal_nufft_input_partition(
                uvwExtent, imgExtent,
                queue.device_prop().totalGlobalMem / (8 * sizeof(api::ComplexType<T>)));

            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})", gridSize[0],
                               gridSize[1], gridSize[2]);

            // set partition method to grid and create grid partition
            opt_.localUVWPartition.method = Partition::Grid{gridSize};
            return DomainPartition::grid<T>(ctx_, gridSize, {uvwX, uvwY, uvwZ});
          }
        },
        opt_.localUVWPartition.method);

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        inputPartition.apply(currentVirtualVis.slice_view(i).slice_view(j));
      }
    }

    inputPartition.apply(uvwX);
    inputPartition.apply(uvwY);
    inputPartition.apply(uvwZ);

    queue.signal_stream(nullptr);  // cufinufft uses default stream

    for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
      if (!inputSize) continue;
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        if (inputSize <= 1024) {
          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
          // Direct evaluation of sum for small input sizes
          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.slice_view(i).slice_view(j).data() + imgBegin;
              nuft_sum<T>(queue.device_prop(), nullptr, 1.0, inputSize,
                          currentVirtualVis.slice_view(i).slice_view(j).data() + inputBegin,
                          uvwX.data() + inputBegin, uvwY.data() + inputBegin,
                          uvwZ.data() + inputBegin, imgSize, pixelX.data() + imgBegin,
                          pixelY.data() + imgBegin, pixelZ.data() + imgBegin, imgPtr);
            }
          }
        } else {
          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, calling cuFINUFFT", inputSize);
          // Approximate sum through nufft
          Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwX.data() + inputBegin,
                                uvwY.data() + inputBegin, uvwZ.data() + inputBegin, imgSize,
                                pixelX.data() + imgBegin, pixelY.data() + imgBegin,
                                pixelZ.data() + imgBegin);

          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.slice_view(i).slice_view(j).data() + imgBegin;
              transform.execute(currentVirtualVis.slice_view(i).slice_view(j).data() + inputBegin,
                                output.data());
              ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output", imgSize, 1,
                                        output.data(), imgSize);

              add_vector_real_of_complex<T>(queue.device_prop(), nullptr, imgSize, output.data(),
                                            imgPtr);
            }
          }
        }
      }
    }
  }

  queue.sync_with_stream(nullptr);
  collectCount_ = 0;
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, DeviceView<T, 2> out) -> void {
  computeNufft();  // make sure all input has been processed

  auto& queue = ctx_->gpu_queue();

  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filter_.data();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  const T scale =
      totalCollectCount_ ? static_cast<T>(1.0 / static_cast<double>(totalCollectCount_)) : 0;

  auto filterImg = img_.slice_view(index);

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    auto intervalImage = filterImg.slice_view(i);
    auto intervalOut = out.slice_view(i);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", intervalImage);
    imgPartition_.reverse(intervalImage, intervalOut);

    scale_vector<T>(queue.device_prop(), queue.stream(), nPixel_, scale, intervalOut.data());
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", intervalOut);
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
