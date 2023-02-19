#include "bipp/standard_synthesis.hpp"

#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/buffer.hpp"
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T>
struct StandardSynthesisInternal {
  StandardSynthesisInternal(const std::shared_ptr<ContextInternal>& ctx, std::size_t nAntenna,
                            std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                            const BippFilter* filter, std::size_t nPixel, const T* pixelX,
                            const T* pixelY, const T* pixelZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals), nPixel_(nPixel) {
    if (ctx_->processing_unit() == BIPP_PU_CPU) {
      planHost_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, pixelX, pixelY,
                        pixelZ);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      Buffer<T> pixelXBuffer, pixelYBuffer, pixelZBuffer;
      auto pixelXDevice = pixelX;
      auto pixelYDevice = pixelY;
      auto pixelZDevice = pixelZ;

      if (!gpu::is_device_ptr(pixelX)) {
        pixelXBuffer = queue.create_device_buffer<T>(nPixel);
        pixelXDevice = pixelXBuffer.get();
        gpu::api::memcpy_async(pixelXBuffer.get(), pixelX, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(pixelY)) {
        pixelYBuffer = queue.create_device_buffer<T>(nPixel);
        pixelYDevice = pixelYBuffer.get();
        gpu::api::memcpy_async(pixelYBuffer.get(), pixelY, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(pixelZ)) {
        pixelZBuffer = queue.create_device_buffer<T>(nPixel);
        pixelZDevice = pixelZBuffer.get();
        gpu::api::memcpy_async(pixelZBuffer.get(), pixelZ, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }

      planGPU_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, pixelXDevice,
                       pixelYDevice, pixelZDevice);
      ctx_->gpu_queue().sync();
#else
      throw GPUSupportError();
#endif
    }
  }

  void collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz) {
    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      queue.sync_with_stream(nullptr);
      queue.sync();  // make sure unused allocated memory is available

      Buffer<gpu::api::ComplexType<T>> wBuffer, sBuffer;
      Buffer<T> xyzBuffer;

      auto sDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(s);
      auto ldsDevice = lds;
      auto wDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(w);
      auto ldwDevice = ldw;

      if (s && !gpu::is_device_ptr(s)) {
        sBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(nBeam_ * nBeam_);
        ldsDevice = nBeam_;
        sDevice = sBuffer.get();
        gpu::api::memcpy_2d_async(sBuffer.get(), nBeam_ * sizeof(gpu::api::ComplexType<T>), s,
                                  lds * sizeof(gpu::api::ComplexType<T>),
                                  nBeam_ * sizeof(gpu::api::ComplexType<T>), nBeam_,
                                  gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(w)) {
        wBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(nAntenna_ * nBeam_);
        ldwDevice = nAntenna_;
        wDevice = wBuffer.get();
        gpu::api::memcpy_2d_async(wBuffer.get(), nAntenna_ * sizeof(gpu::api::ComplexType<T>), w,
                                  ldw * sizeof(gpu::api::ComplexType<T>),
                                  nAntenna_ * sizeof(gpu::api::ComplexType<T>), nBeam_,
                                  gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }

      // Always copy xyz, even when on device, since it will be overwritten
      xyzBuffer = queue.create_device_buffer<T>(3 * nAntenna_);
      auto ldxyzDevice = nAntenna_;
      auto xyzDevice = xyzBuffer.get();
      gpu::api::memcpy_2d_async(xyzBuffer.get(), nAntenna_ * sizeof(T), xyz, ldxyz * sizeof(T),
                                nAntenna_ * sizeof(T), 3, gpu::api::flag::MemcpyDefault,
                                queue.stream());

      // sync before call, such that host memory can be safely discarded by
      // caller, while computation is continued asynchronously
      queue.sync();

      planGPU_->collect(nEig, wl, intervals, ldIntervals, sDevice, ldsDevice, wDevice, ldwDevice,
                        xyzDevice, ldxyzDevice);
#else
      throw GPUSupportError();
#endif
    }
  }

  auto get(BippFilter f, T* img, std::size_t ld) -> void {
    if (planHost_) {
      planHost_.value().get(f, img, ld);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      planGPU_->get(f, img, ld);
      ctx_->gpu_queue().sync();
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  std::size_t nAntenna_, nBeam_, nIntervals_, nPixel_;
  std::optional<host::StandardSynthesis<T>> planHost_;
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::StandardSynthesis<T>> planGPU_;
#endif
};

template <typename T>
StandardSynthesis<T>::StandardSynthesis(Context& ctx, std::size_t nAntenna, std::size_t nBeam,
                                        std::size_t nIntervals, std::size_t nFilter,
                                        const BippFilter* filter, std::size_t nPixel,
                                        const T* pixelX, const T* pixelY, const T* pixelZ)
    : plan_(new StandardSynthesisInternal<T>(InternalContextAccessor::get(ctx), nAntenna, nBeam,
                                             nIntervals, nFilter, filter, nPixel, pixelX, pixelY,
                                             pixelZ),
            [](auto&& ptr) { delete reinterpret_cast<StandardSynthesisInternal<T>*>(ptr); }) {}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals,
                                   std::size_t ldIntervals, const std::complex<T>* s,
                                   std::size_t lds, const std::complex<T>* w, std::size_t ldw,
                                   const T* xyz, std::size_t ldxyz) -> void {
  reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz);
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())->get(f, out, ld);
}

template class BIPP_EXPORT StandardSynthesis<double>;

template class BIPP_EXPORT StandardSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_standard_synthesis_create_f(BippContext ctx, size_t nAntenna,
                                                       size_t nBeam, size_t nIntervals,
                                                       size_t nFilter, const BippFilter* filter,
                                                       size_t nPixel, const float* lmnX,
                                                       const float* lmnY, const float* lmnZ,
                                                       BippStandardSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<float>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nAntenna, nBeam, nIntervals,
        nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
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

BIPP_EXPORT BippError bipp_standard_synthesis_collect_f(BippStandardSynthesisF plan, size_t nEig,
                                                        float wl, const float* intervals,
                                                        size_t ldIntervals, const void* s,
                                                        size_t lds, const void* w, size_t ldw,
                                                        const float* xyz, size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<float>*>(s), lds,
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

BIPP_EXPORT BippError bipp_standard_synthesis_create(BippContext ctx, size_t nAntenna, size_t nBeam,
                                                     size_t nIntervals, size_t nFilter,
                                                     const BippFilter* filter, size_t nPixel,
                                                     const double* lmnX, const double* lmnY,
                                                     const double* lmnZ,
                                                     BippStandardSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<double>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nAntenna, nBeam, nIntervals,
        nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
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

BIPP_EXPORT BippError bipp_standard_synthesis_collect(BippStandardSynthesis plan, size_t nEig,
                                                      double wl, const double* intervals,
                                                      size_t ldIntervals, const void* s, size_t lds,
                                                      const void* w, size_t ldw, const double* xyz,
                                                      size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<double>*>(s), lds,
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