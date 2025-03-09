#include "arch/kokkos_aliases.h"

#include "global.h"

#include <Kokkos_Core.hpp>

template <>
auto CreateRangePolicy<Dim::_1D>(
  const tuple_t<std::size_t, Dim::_1D>& i1,
  const tuple_t<std::size_t, Dim::_1D>& i2) -> range_t<Dim::_1D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  return Kokkos::RangePolicy<AccelExeSpace>(i1min, i1max);
}

template <>
auto CreateRangePolicy<Dim::_2D>(
  const tuple_t<std::size_t, Dim::_2D>& i1,
  const tuple_t<std::size_t, Dim::_2D>& i2) -> range_t<Dim::_2D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  index_t i2min = i1[1];
  index_t i2max = i2[1];
  return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>({ i1min, i2min },
                                                               { i1max, i2max });
}

template <>
auto CreateRangePolicy<Dim::_3D>(
  const tuple_t<std::size_t, Dim::_3D>& i1,
  const tuple_t<std::size_t, Dim::_3D>& i2) -> range_t<Dim::_3D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  index_t i2min = i1[1];
  index_t i2max = i2[1];
  index_t i3min = i1[2];
  index_t i3max = i2[2];
  return Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
    { i1min, i2min, i3min },
    { i1max, i2max, i3max });
}

template <>
auto CreateRangePolicyOnHost<Dim::_1D>(
  const tuple_t<std::size_t, Dim::_1D>& i1,
  const tuple_t<std::size_t, Dim::_1D>& i2) -> range_h_t<Dim::_1D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  return Kokkos::RangePolicy<HostExeSpace>(i1min, i1max);
}

template <>
auto CreateRangePolicyOnHost<Dim::_2D>(
  const tuple_t<std::size_t, Dim::_2D>& i1,
  const tuple_t<std::size_t, Dim::_2D>& i2) -> range_h_t<Dim::_2D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  index_t i2min = i1[1];
  index_t i2max = i2[1];
  return Kokkos::MDRangePolicy<Kokkos::Rank<2>, HostExeSpace>({ i1min, i2min },
                                                              { i1max, i2max });
}

template <>
auto CreateRangePolicyOnHost<Dim::_3D>(
  const tuple_t<std::size_t, Dim::_3D>& i1,
  const tuple_t<std::size_t, Dim::_3D>& i2) -> range_h_t<Dim::_3D> {
  index_t i1min = i1[0];
  index_t i1max = i2[0];
  index_t i2min = i1[1];
  index_t i2max = i2[1];
  index_t i3min = i1[2];
  index_t i3max = i2[2];
  return Kokkos::MDRangePolicy<Kokkos::Rank<3>, HostExeSpace>(
    { i1min, i2min, i3min },
    { i1max, i2max, i3max });
}

auto WaitAndSynchronize(bool debug_only) -> void {
  if (debug_only) {
#ifndef DEBUG
    return;
#endif
  }
  Kokkos::fence();
}
