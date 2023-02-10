#pragma once

#include <cstring>
#include <cstddef>
#include <numeric>
#include <functional>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T, std::size_t DIM>
class GridDecomposition {
public:
  static_assert(DIM > 1, "Dimension must be >= 1");

  struct Tile {
    std::size_t begin = 0;
    std::size_t size = 0;
  };

  GridDecomposition() = default;

  GridDecomposition(ContextInternal& ctx, std::array<T, DIM> min, std::array<T, DIM> max,
                    std::array<std::size_t, DIM> gridSize, std::size_t n,
                    std::array<const T*, DIM> coord)
      : permut_(ctx.host_alloc(), n),
        tiles_(ctx.host_alloc(), std::accumulate(gridSize.begin(), gridSize.end(), std::size_t(1),
                                                 std::multiplies<std::size_t>())) {
    auto* __restrict__ tiles = tiles_.get();
    std::size_t* __restrict__ permut = permut_.get();

    std::array<T, DIM> gridSpacingInv;
    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      gridSpacingInv[dimIdx] = gridSize[dimIdx] / (max[dimIdx] - min[dimIdx]);
    }

    std::memset(tiles, 0, tiles_.size_in_bytes());

    // Compute the assigned tile index in grid for each data point and store temporarily in permut
    // array. Increment tiles array of following tile each time, such that the size of the
    // previous tile is computed.
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t tileIndex = 0;
      for (std::size_t dimIdx = DIM - 1;; --dimIdx) {
        tileIndex = tileIndex * gridSize[dimIdx] +
                    std::max<std::size_t>(
                        std::min(static_cast<std::size_t>(gridSpacingInv[dimIdx] *
                                                          (coord[dimIdx][i] - min[dimIdx])),
                                 gridSize[dimIdx] - 1),
                        0);

        if (!dimIdx) break;
      }

      permut[i] = tileIndex;
      ++tiles[tileIndex].size;
    }

    // Compute the rolling sum, such that each tile has its begin index
    for (std::size_t i = 1; i < tiles_.size(); ++i) {
      tiles[i].begin += tiles[i - 1].begin + tiles[i - 1].size;
    }

    // Finally compute permutation index for each data point and restore tile sizes.
    for (std::size_t i = 0; i < n; ++i) {
      permut[i] = tiles[permut[i]].begin++;
    }

    // Restore begin index
    for (std::size_t i = 0; i < tiles_.size(); ++i) {
      tiles[i].begin -= tiles[i].size;
    }
  }

  inline auto permut_array() const -> const std::size_t* { return permut_.get(); }

  inline auto tile_begin_array() const -> const std::size_t* { return tiles_.get(); }

  inline auto begin() const -> const Tile* { return tiles_.get(); }

  inline auto end() const -> const Tile* { return tiles_.get() + tiles_.size(); }

  template <typename F>
  inline auto apply(const F* __restrict__ in, F* __restrict__ out) -> void {
    const std::size_t* __restrict__ permut = permut_.get();
    for (std::size_t i = 0; i < permut_.size(); ++i) {
      out[i] = in[permut[i]];
    }
  }

  template <typename F>
  inline auto reverse(const F* __restrict__ in, F* __restrict__ out) -> void {
    const std::size_t* __restrict__ permut = permut_.get();
    for (std::size_t i = 0; i < permut_.size(); ++i) {
      out[permut[i]] = in[i];
    }
  }

  inline auto size() const -> std::size_t { return tiles_.size(); }

private:
  Buffer<std::size_t> permut_;
  Buffer<Tile> tiles_;
};

}  // namespace host
}  // namespace bipp
