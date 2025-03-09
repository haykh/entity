#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "kernels/currents_deposit.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

Inline auto equal(real_t a, real_t b, const char* msg = "", real_t acc = ONE) -> bool {
  const auto eps = epsilon * acc;
  if (not cmp::AlmostEqual(a, b, eps)) {
    printf("%.12e != %.12e %s\n", a, b, msg);
    return false;
  }
  return true;
}

template <typename T>
void put_value(array_t<T*> arr, T value, int i) {
  auto arr_h = Kokkos::create_mirror_view(arr);
  arr_h(i)   = value;
  Kokkos::deep_copy(arr, arr_h);
}

template <typename M, ntt::SimEngine::type S>
void testDeposit(const std::vector<std::size_t>&      res,
                 const boundaries_t<real_t>&          ext,
                 const std::map<std::string, real_t>& params = {},
                 const real_t                         acc    = ONE) {
  static_assert(M::Dim == 2);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");
  using namespace ntt;

  M metric { res, ext, params };

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<M::Dim, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
  array_t<int*>        i1 { "i1", 10 };
  array_t<int*>        i2 { "i2", 10 };
  array_t<int*>        i3 { "i3", 10 };
  array_t<int*>        i1_prev { "i1_prev", 10 };
  array_t<int*>        i2_prev { "i2_prev", 10 };
  array_t<int*>        i3_prev { "i3_prev", 10 };
  array_t<prtldx_t*>   dx1 { "dx1", 10 };
  array_t<prtldx_t*>   dx2 { "dx2", 10 };
  array_t<prtldx_t*>   dx3 { "dx3", 10 };
  array_t<prtldx_t*>   dx1_prev { "dx1_prev", 10 };
  array_t<prtldx_t*>   dx2_prev { "dx2_prev", 10 };
  array_t<prtldx_t*>   dx3_prev { "dx3_prev", 10 };
  array_t<real_t*>     ux1 { "ux1", 10 };
  array_t<real_t*>     ux2 { "ux2", 10 };
  array_t<real_t*>     ux3 { "ux3", 10 };
  array_t<real_t*>     phi { "phi", 10 };
  array_t<real_t*>     weight { "weight", 10 };
  array_t<short*>      tag { "tag", 10 };
  const real_t         charge { 1.0 }, inv_dt { 1.0 };

  const int i0 = 4, j0 = 4;

  const prtldx_t dxi = 0.53, dxf = 0.47;
  const prtldx_t dyi = 0.34, dyf = 0.52;
  const real_t   xi = (real_t)i0 + (real_t)dxi, xf = (real_t)i0 + (real_t)dxf;
  const real_t   yi = (real_t)j0 + (real_t)dyi, yf = (real_t)j0 + (real_t)dyf;

  const real_t xr = 0.5 * (xi + xf);
  const real_t yr = 0.5 * (yi + yf);

  const real_t Wx1 = 0.5 * (xi + xr) - (real_t)i0;
  const real_t Wx2 = 0.5 * (xf + xr) - (real_t)i0;

  const real_t Wy1 = 0.5 * (yi + yr) - (real_t)j0;
  const real_t Wy2 = 0.5 * (yf + yr) - (real_t)j0;

  const real_t Fx1 = (xr - xi);
  const real_t Fx2 = (xf - xr);

  const real_t Fy1 = (yr - yi);
  const real_t Fy2 = (yf - yr);

  const real_t Jx1 = Fx1 * (1 - Wy1) + Fx2 * (1 - Wy2);
  const real_t Jx2 = Fx1 * Wy1 + Fx2 * Wy2;

  const real_t Jy1 = Fy1 * (1 - Wx1) + Fy2 * (1 - Wx2);
  const real_t Jy2 = Fy1 * Wx1 + Fy2 * Wx2;

  put_value<int>(i1, i0, 0);
  put_value<int>(i2, j0, 0);
  put_value<int>(i1_prev, i0, 0);
  put_value<int>(i2_prev, j0, 0);
  put_value<prtldx_t>(dx1, dxf, 0);
  put_value<prtldx_t>(dx2, dyf, 0);
  put_value<prtldx_t>(dx1_prev, dxi, 0);
  put_value<prtldx_t>(dx2_prev, dyi, 0);
  put_value<real_t>(weight, 1.0, 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  auto J_scat = Kokkos::Experimental::create_scatter_view(J);

  // clang-format off
  Kokkos::parallel_for("CurrentsDeposit", 10,
                       kernel::DepositCurrents_kernel<S, M>(J_scat,
                                                            i1, i2, i3,
                                                            i1_prev, i2_prev, i3_prev,
                                                            dx1, dx2, dx3,
                                                            dx1_prev, dx2_prev, dx3_prev,
                                                            ux1, ux2, ux3,
                                                            phi, weight, tag,
                                                            metric, charge, inv_dt));
  // clang-format on

  Kokkos::Experimental::contribute(J, J_scat);

  real_t SumDivJ { 0.0 };
  Kokkos::parallel_reduce(
    "SumDivJ",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ N_GHOSTS, N_GHOSTS },
                                           { nx1 + N_GHOSTS, nx2 + N_GHOSTS }),
    Lambda(const int i, const int j, real_t& sum) {
      sum += J(i, j, cur::jx1) - J(i - 1, j, cur::jx1) + J(i, j, cur::jx2) -
             J(i, j - 1, cur::jx2);
    },
    SumDivJ);

  auto J_h = Kokkos::create_mirror_view(J);
  Kokkos::deep_copy(J_h, J);

  if (not cmp::AlmostZero(SumDivJ)) {
    throw std::logic_error("DepositCurrents_kernel::SumDivJ != 0");
  }
  errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx1), Jx1, "", acc),
          "DepositCurrents_kernel::Jx1 is incorrect");
  errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + 1 + N_GHOSTS, cur::jx1), Jx2, "", acc),
          "DepositCurrents_kernel::Jx2 is incorrect");
  errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), Jy1, "", acc),
          "DepositCurrents_kernel::Jy1 is incorrect");
  errorIf(not equal(J_h(i0 + 1 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), Jy2, "", acc),
          "DepositCurrents_kernel::Jy2 is incorrect");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;

    testDeposit<Minkowski<Dim::_2D>, SimEngine::SRPIC>(
      {
        10,
        10
    },
      { { 0.0, 55.0 }, { 0.0, 55.0 } },
      {},
      30);

    testDeposit<Spherical<Dim::_2D>, SimEngine::SRPIC>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      {},
      30);

    testDeposit<QSpherical<Dim::_2D>, SimEngine::SRPIC>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } },
      30);

    testDeposit<KerrSchild<Dim::_2D>, SimEngine::GRPIC>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } },
      30);

    testDeposit<QKerrSchild<Dim::_2D>, SimEngine::GRPIC>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 }, { "a", 0.9 } },
      30);

    testDeposit<KerrSchild0<Dim::_2D>, SimEngine::GRPIC>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } },
      30);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
