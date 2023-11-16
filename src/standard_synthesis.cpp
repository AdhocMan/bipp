#include "bipp/standard_synthesis.hpp"

#include <chrono>
#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/view.hpp"
#include "imager.hpp"
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T>
struct StandardSynthesisInternal {
  StandardSynthesisInternal(const std::shared_ptr<ContextInternal>& ctx, std::size_t nLevel,
                            std::size_t nFilter, const BippFilter* filter, std::size_t nPixel,
                            const T* pixelX, const T* pixelY, const T* pixelZ)
      : ctx_(ctx), nLevel_(nLevel), nPixel_(nPixel) {
    ctx_->logger().log(
        BIPP_LOG_LEVEL_DEBUG, "{} StandardSynthesis.create({}, opt, {} ,{} {}, {}, {}, {}, {})",
        (const void*)this, (const void*)ctx_.get(), nLevel, nFilter, (const void*)filter,
        nPixel, (const void*)pixelX, (const void*)pixelY, (const void*)pixelZ);

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnX", nPixel_, 1, pixelX, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnY", nPixel_, 1, pixelY, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnZ", nPixel_, 1, pixelZ, nPixel_);

    if (ctx_->processing_unit() == BIPP_PU_CPU) {
      planHost_.emplace(
          ctx_, nLevel, ConstHostView<BippFilter, 1>(filter, nFilter, 1),
          ConstHostView<T, 1>(pixelX, nPixel, 1), ConstHostView<T, 1>(pixelY, nPixel, 1),
          ConstHostView<T, 1>(pixelZ, nPixel, 1));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      auto filterArray = queue.create_host_array<BippFilter, 1>(nFilter);
      copy(queue, ConstView<BippFilter, 1>(filter, nFilter, 1), filterArray);
      queue.sync();  // make sure filters are available

      auto pixelArray = queue.create_device_array<T, 2>({nPixel_, 3});
      copy(queue, ConstView<T, 1>(pixelX, nPixel_, 1), pixelArray.slice_view(0));
      copy(queue, ConstView<T, 1>(pixelY, nPixel_, 1), pixelArray.slice_view(1));
      copy(queue, ConstView<T, 1>(pixelZ, nPixel_, 1), pixelArray.slice_view(2));

      planGPU_.emplace(ctx_, nLevel, std::move(filterArray), std::move(pixelArray));
#else
      throw GPUSupportError();
#endif
    }
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis created with Context {}",
                       static_cast<const void*>(this), (const void*)ctx_.get());
  }

  ~StandardSynthesisInternal() {
    try {
      ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis destroyed",
                         static_cast<const void*>(this));
    } catch (...) {
    }
  }

  void collect(std::size_t nAntenna, std::size_t nBeam, T wl,
               const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz) {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "------------");
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                       "{} StandardSynthesis.collect({}, {}, {}, {}, {} ,{} ,{} {}, {})",
                       (const void*)this, nAntenna, nBeam, wl, (const void*)s, lds, (const void*)w,
                       ldw, (const void*)xyz, ldxyz);
    if (s) ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "S", nBeam, nBeam, s, lds);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "W", nAntenna, nBeam, w, ldw);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "XYZ", nAntenna, 3, xyz, ldxyz);

    const auto start = std::chrono::high_resolution_clock::now();

    if (planHost_) {
      auto& p = planHost_.value();
      p.collect(wl, eigMaskFunc, ConstHostView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
                ConstHostView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                ConstHostView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      typename ConstView<T, 2>::IndexType sShape = {0, 0};
      if (s) sShape = {nBeam, nBeam};

      ConstHostAccessor<gpu::api::ComplexType<T>, 2> sHost(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s), sShape, {1, lds});
      queue.sync();  // make sure it's accessible on host

      ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> sDevice(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s), sShape, {1, lds});
      ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> wDevice(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(w), {nAntenna, nBeam},
          {1, ldw});
      ConstDeviceAccessor<T, 2> xyzDevice(queue, xyz, {nAntenna, 3}, {1, ldxyz});

      planGPU_->collect(wl, eigMaskFunc, sHost.view(), sDevice.view(), wDevice.view(),
                        xyzDevice.view());
#else
      throw GPUSupportError();
#endif
    }

    const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis.collect() time: {}ms",
                       (const void*)this, time.count());

    if (ctx_->processing_unit() == BIPP_PU_CPU)
      ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} Context memory usage: host {}MB",
                         (const void*)ctx_.get(), ctx_->host_alloc()->size() / 1000000);
    else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      const auto queueMem = ctx_->gpu_queue().allocated_memory();
      ctx_->logger().log(
          BIPP_LOG_LEVEL_INFO, "{} Context memory usage: host {}MB, pinned {}MB, device {}MB",
          (const void*)ctx_.get(), (ctx_->host_alloc()->size() + std::get<0>(queueMem)) / 1000000,
          std::get<1>(queueMem) / 1000000, std::get<2>(queueMem) / 1000000);
#endif
    }
  }

  auto get(BippFilter f, T* img, std::size_t ld) -> void {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "{} StandardSynthesis.get({}, {}, {})",
                       (const void*)this, (int)f, (const void*)img, ld);
    if (planHost_) {
      auto& p = planHost_.value();
      p.get(f, HostView<T, 2>(img, {p.num_pixel(), p.num_level()}, {1, ld}));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      DeviceAccessor<T, 2> imgDevice(queue, img, {nPixel_, nLevel_}, {1, ld});
      planGPU_->get(f, imgDevice.view());
      imgDevice.copy_back(queue);
      ctx_->gpu_queue().sync();
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  std::size_t nLevel_, nPixel_;
  std::optional<host::StandardSynthesis<T>> planHost_;
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::StandardSynthesis<T>> planGPU_;
#endif
};

template <typename T>
StandardSynthesis<T>::StandardSynthesis(Context& ctx, std::size_t nLevel, std::size_t nFilter,
                                        const BippFilter* filter, std::size_t nPixel,
                                        const T* pixelX, const T* pixelY, const T* pixelZ) {
  // try {
  //   plan_ = decltype(plan_)(
  //       new StandardSynthesisInternal<T>(InternalContextAccessor::get(ctx), nLevel, nFilter,
  //                                        filter, nPixel, pixelX, pixelY, pixelZ),
  //       [](auto&& ptr) { delete reinterpret_cast<StandardSynthesisInternal<T>*>(ptr); });
  // } catch (const std::exception& e) {
  //   try {
  //     InternalContextAccessor::get(ctx)->logger().log(
  //         BIPP_LOG_LEVEL_ERROR, "StandardSynthesis creation error: {}", e.what());
  //   } catch (...) {
  //   }
  //   throw;
  // }

  try {
    plan_ = decltype(plan_)(
        new Imager<T>(Imager<T>::standard_synthesis(
            InternalContextAccessor::get(ctx), nLevel, ConstView<BippFilter, 1>(filter, nFilter, 1),
            ConstView<T, 1>(pixelX, nPixel, 1), ConstView<T, 1>(pixelY, nPixel, 1),
            ConstView<T, 1>(pixelZ, nPixel, 1))),
        [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "StandardSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::collect(
    std::size_t nAntenna, std::size_t nBeam, T wl,
    const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc, const std::complex<T>* s,
    std::size_t lds, const std::complex<T>* w, std::size_t ldw, const T* xyz, std::size_t ldxyz)
    -> void {
  // try {
  //   reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
  //       ->collect(nAntenna, nBeam, wl, eigMaskFunc, s, lds, w, ldw, xyz, ldxyz);
  // } catch (const std::exception& e) {
  //   try {
  //     reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
  //         ->ctx_->logger()
  //         .log(BIPP_LOG_LEVEL_ERROR, "{} StandardSynthesis.get() error: {}",
  //              (const void*)plan_.get(), e.what());
  //   } catch (...) {
  //   }
  //   throw;
  // }

  try {
    reinterpret_cast<Imager<T>*>(plan_.get())
        ->collect(wl, eigMaskFunc, ConstView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
                  ConstView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                  ConstView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}), ConstView<T, 2>());
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
          ->ctx_->logger()
          .log(BIPP_LOG_LEVEL_ERROR, "{} StandardSynthesis.get() error: {}",
               (const void*)plan_.get(), e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  // reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())->get(f, out, ld);
  reinterpret_cast<Imager<T>*>(plan_.get())->get(f, out, ld);
}

template class BIPP_EXPORT StandardSynthesis<double>;

template class BIPP_EXPORT StandardSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_standard_synthesis_create_f(BippContext ctx, size_t nLevel,
                                                       size_t nFilter, const BippFilter* filter,
                                                       size_t nPixel, const float* lmnX,
                                                       const float* lmnY, const float* lmnZ,
                                                       BippStandardSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<float>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nLevel, nFilter, filter,
        nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy_f(BippStandardSynthesisF* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesisInternal<float>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect_f(BippStandardSynthesisF plan,
                                                        size_t nAntenna, size_t nBeam, float wl,
                                                        void (*mask)(size_t, size_t, float*),
                                                        const void* s, size_t lds, const void* w,
                                                        size_t ldw, const float* xyz,
                                                        size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<float>*>(s), lds,
        reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get_f(BippStandardSynthesisF plan, BippFilter f,
                                                    float* img, size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_create(BippContext ctx, size_t nLevel,
                                                     size_t nFilter, const BippFilter* filter,
                                                     size_t nPixel, const double* lmnX,
                                                     const double* lmnY, const double* lmnZ,
                                                     BippStandardSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<double>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nLevel, nFilter, filter,
        nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy(BippStandardSynthesis* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesisInternal<double>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect(BippStandardSynthesis plan, size_t nAntenna,
                                                      size_t nBeam, double wl,
                                                      void (*mask)(size_t, size_t, double*),
                                                      const void* s, size_t lds, const void* w,
                                                      size_t ldw, const double* xyz, size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<double>*>(s), lds,
        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get(BippStandardSynthesis plan, BippFilter f,
                                                  double* img, size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
