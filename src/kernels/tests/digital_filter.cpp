#include "kernels/digital_filter.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/formatting.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <typename M>
void testFilter(const std::vector<std::size_t>&      res,
                const boundaries_t<real_t>&          ext,
                const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 2);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");
  using namespace ntt;

  auto boundaries = boundaries_t<FldsBC> {};
  if constexpr (M::CoordType != Coord::Cart) {
    boundaries = {
      { FldsBC::CUSTOM, FldsBC::CUSTOM },
      {   FldsBC::AXIS,   FldsBC::AXIS }
    };
  } else {
    boundaries = {
      { FldsBC::PERIODIC, FldsBC::PERIODIC },
      { FldsBC::PERIODIC, FldsBC::PERIODIC }
    };
  }

  M metric { res, ext, params };

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<Dim::_2D, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
  ndfield_t<Dim::_2D, 3> Jbuff { "Jbuff", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };

  tuple_t<std::size_t, Dim::_2D> size;
  size[0]                                   = nx1;
  size[1]                                   = nx2;
  auto J_h                                  = Kokkos::create_mirror_view(J);
  J_h(5 + N_GHOSTS, 5 + N_GHOSTS, cur::jx1) = 1.0;
  J_h(4 + N_GHOSTS, 5 + N_GHOSTS, cur::jx2) = 1.0;
  J_h(5 + N_GHOSTS, 4 + N_GHOSTS, cur::jx3) = 1.0;
  Kokkos::deep_copy(J, J_h);
  const auto range = CreateRangePolicy<Dim::_2D>(
    { N_GHOSTS, N_GHOSTS },
    { nx1 + N_GHOSTS, nx2 + N_GHOSTS + 1 });
  Kokkos::deep_copy(Jbuff, J);
  Kokkos::parallel_for(
    "CurrentsFilter",
    range,
    kernel::DigitalFilter_kernel<Dim::_2D, M::CoordType>(J, Jbuff, size, boundaries));
  real_t SumJx1 { 0.0 }, SumJx2 { 0.0 }, SumJx3 { 0.0 };
  Kokkos::parallel_reduce(
    "SumJx1",
    range,
    Lambda(index_t i, index_t j, real_t & sum) { sum += J(i, j, cur::jx1); },
    SumJx1);
  Kokkos::parallel_reduce(
    "SumJx2",
    range,
    Lambda(index_t i, index_t j, real_t & sum) { sum += J(i, j, cur::jx2); },
    SumJx2);
  Kokkos::parallel_reduce(
    "SumJx3",
    range,
    Lambda(index_t i, index_t j, real_t & sum) { sum += J(i, j, cur::jx3); },
    SumJx3);

  Kokkos::deep_copy(J_h, J);

  errorIf(not cmp::AlmostEqual(ONE, SumJx1), "DigitalFilter_kernel::SumJx1 != 1");
  errorIf(not cmp::AlmostEqual(ONE, SumJx2), "DigitalFilter_kernel::SumJx2 != 1");
  errorIf(not cmp::AlmostEqual(ONE, SumJx3), "DigitalFilter_kernel::SumJx3 != 1");

  for (auto i1 = 4; i1 < 7; ++i1) {
    for (auto i2 = 4; i2 < 7; ++i2) {
      const real_t expect = math::pow(0.5, math::abs(i1 - 5) + 1) *
                            math::pow(0.5, math::abs(i2 - 5) + 1);
      errorIf(not cmp::AlmostEqual(J_h(i1 + N_GHOSTS, i2 + N_GHOSTS, cur::jx1),
                                   expect),
              fmt::format("J_h(%d, %d, cur::jx1) == %f != %f",
                          i1 + N_GHOSTS,
                          i2 + N_GHOSTS,
                          J_h(i1 + N_GHOSTS, i2 + N_GHOSTS, cur::jx1),
                          expect));
      errorIf(not cmp::AlmostEqual(J_h(i1 + N_GHOSTS - 1, i2 + N_GHOSTS, cur::jx2),
                                   expect),
              fmt::format("J_h(%d, %d, cur::jx1) == %f != %f",
                          i1 + N_GHOSTS - 1,
                          i2 + N_GHOSTS,
                          J_h(i1 + N_GHOSTS - 1, i2, cur::jx2),
                          expect));
      errorIf(not cmp::AlmostEqual(J_h(i1 + N_GHOSTS, i2 + N_GHOSTS - 1, cur::jx3),
                                   expect),
              fmt::format("J_h(%d, %d, cur::jx1) == %f != %f",
                          i1 + N_GHOSTS,
                          i2 + N_GHOSTS - 1,
                          J_h(i1 + N_GHOSTS, i2 + N_GHOSTS - 1, cur::jx3),
                          expect));
    }
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;

    testFilter<Minkowski<Dim::_2D>>(
      {
        10,
        10
    },
      { { 0.0, 55.0 }, { 0.0, 55.0 } },
      {});

    testFilter<Spherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      {});

    testFilter<QSpherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } });

    testFilter<KerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } });

    testFilter<QKerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 }, { "a", 0.9 } });

    testFilter<KerrSchild0<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
