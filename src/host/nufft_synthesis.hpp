#pragma once

#include <complex>
#include <memory>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "host/grid_decomposition.hpp"

namespace bipp {
namespace host {

template <typename T>
class NufftSynthesis {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, T tol, std::size_t nAntenna,
                 std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                 const BippFilter* filter, std::size_t nPixel, const T* lmnX, const T* lmnY,
                 const T* lmnZ);

  auto collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw) -> void;

  auto get(BippFilter f, T* out, std::size_t ld) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const T tol_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  Buffer<BippFilter> filter_;
  Buffer<T> lmnX_, lmnY_, lmnZ_;
  std::optional<GridDecomposition<T, 2>> imgDecomposition_;

  std::size_t nMaxInputCount_, collectCount_;
  Buffer<std::complex<T>> virtualVis_;
  Buffer<T> uvwX_, uvwY_, uvwZ_;
  Buffer<T> img_;
};

}  // namespace host
}  // namespace bipp
