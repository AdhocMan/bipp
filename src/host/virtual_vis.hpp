#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                 ConstHostView<T, 2> dMasked, ConstHostView<std::complex<T>, 2> v,
                 HostView<std::complex<T>, 3> virtVis) -> void;

}  // namespace host
}  // namespace bipp
