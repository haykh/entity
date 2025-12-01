#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "kernels/particle_pusher_sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

void errorIf(bool condition, const std::string& message = "") {
  if (condition) {
    throw std::runtime_error(message);
  }
}

Inline auto equal(real_t a, real_t b, const std::string& msg) -> bool {
  if (not(math::abs(a - b) < 1e-4)) {
    Kokkos::printf("%.12e != %.12e %s\n", a, b, msg.c_str());
    return false;
  }
  return true;
}

template <typename T>
void put_value(array_t<T*>& arr, T v, index_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

template <SimEngine::type S, typename M>
void testPeriodicBC(const std::vector<std::size_t>&      res,
                    const boundaries_t<real_t>&          ext,
                    const std::map<std::string, real_t>& params = {}) {
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");
  errorIf(M::CoordType != Coord::Cart, "M::CoordType != Coord::Cart");
  // aliases
  const auto NoGCA      = false;
  const auto NoExtForce = false;

  real_t sx = ZERO, sy = ZERO, sz = ZERO;
  if (ext.size() > 0) {
    sx = static_cast<real_t>(ext.at(0).second - ext.at(0).first);
  }
  if (ext.size() > 1) {
    sy = static_cast<real_t>(ext.at(1).second - ext.at(1).first);
  }
  if (ext.size() > 2) {
    sz = static_cast<real_t>(ext.at(2).second - ext.at(2).first);
  }

  M metric { res, ext, params };

  int nx1 = 0, nx2 = 0, nx3 = 0;
  if (res.size() > 0) {
    nx1 = static_cast<int>(res.at(0));
  }
  if (res.size() > 1) {
    nx2 = static_cast<int>(res.at(1));
  }
  if (res.size() > 2) {
    nx3 = static_cast<int>(res.at(2));
  }

  const real_t dt    = 0.1 * (ext.at(0).second - ext.at(0).first) / sx;
  const real_t coeff = HALF * dt;

  ndfield_t<M::Dim, 6> emfield;
  if constexpr (M::Dim == Dim::_1D) {
    emfield = ndfield_t<M::Dim, 6> { "emfield", res.at(0) + 2 * N_GHOSTS };
  } else if constexpr (M::Dim == Dim::_2D) {
    emfield = ndfield_t<M::Dim, 6> { "emfield",
                                     res.at(0) + 2 * N_GHOSTS,
                                     res.at(1) + 2 * N_GHOSTS };
  } else {
    emfield = ndfield_t<M::Dim, 6> { "emfield",
                                     res.at(0) + 2 * N_GHOSTS,
                                     res.at(1) + 2 * N_GHOSTS,
                                     res.at(2) + 2 * N_GHOSTS };
  }

  const short        sp_idx = 1;
  // allocate two particles
  array_t<int*>      i1 { "i1", 2 };
  array_t<int*>      i2 { "i2", 2 };
  array_t<int*>      i3 { "i3", 2 };
  array_t<int*>      i1_prev { "i1_prev", 2 };
  array_t<int*>      i2_prev { "i2_prev", 2 };
  array_t<int*>      i3_prev { "i3_prev", 2 };
  array_t<prtldx_t*> dx1 { "dx1", 2 };
  array_t<prtldx_t*> dx2 { "dx2", 2 };
  array_t<prtldx_t*> dx3 { "dx3", 2 };
  array_t<prtldx_t*> dx1_prev { "dx1_prev", 2 };
  array_t<prtldx_t*> dx2_prev { "dx2_prev", 2 };
  array_t<prtldx_t*> dx3_prev { "dx3_prev", 2 };
  array_t<real_t*>   ux1 { "ux1", 2 };
  array_t<real_t*>   ux2 { "ux2", 2 };
  array_t<real_t*>   ux3 { "ux3", 2 };
  array_t<short*>    tag { "tag", 2 };

  array_t<real_t*> phi;

  // init parameters of prtl #1
  real_t xi_1 = ZERO, yi_1 = ZERO, zi_1 = ZERO;
  real_t xi_2 = ZERO, yi_2 = ZERO, zi_2 = ZERO;
  if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
    xi_1 = 0.515460 * sx + ext.at(0).first;
    xi_2 = 0.149088 * sx + ext.at(0).first;
  }
  if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
    yi_1 = 0.340680 * sy + ext.at(1).first;
    yi_2 = 0.997063 * sy + ext.at(1).first;
  }
  if constexpr (M::Dim == Dim::_3D) {
    zi_1 = 0.940722 * sz + ext.at(2).first;
    zi_2 = 0.607354 * sz + ext.at(2).first;
  }
  real_t ux_1    = 0.569197;
  real_t uy_1    = 0.716085;
  real_t uz_1    = 0.760101;
  real_t gamma_1 = math::sqrt(1.0 + SQR(ux_1) + SQR(uy_1) + SQR(uz_1));

  // init parameters of prtl #2
  real_t ux_2    = -0.872069;
  real_t uy_2    = 0.0484461;
  real_t uz_2    = -0.613575;
  real_t gamma_2 = math::sqrt(1.0 + SQR(ux_2) + SQR(uy_2) + SQR(uz_2));

  {
    coord_t<M::PrtlDim> xCd { ZERO }, xi { ZERO };
    std::size_t         prtl_idx;

    // set up particle #1
    prtl_idx = 0;
    if constexpr (M::Dim == Dim::_1D) {
      xi[0] = xi_1;
    }
    if constexpr (M::Dim == Dim::_2D) {
      xi[0] = xi_1;
      xi[1] = yi_1;
    }
    if constexpr (M::Dim == Dim::_3D) {
      xi[0] = xi_1;
      xi[1] = yi_1;
      xi[2] = zi_1;
    }
    metric.template convert_xyz<Crd::XYZ, Crd::Cd>(xi, xCd);
    if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                  M::PrtlDim == Dim::_3D) {
      put_value<int>(i1, static_cast<int>(xCd[0]), prtl_idx);
      put_value<prtldx_t>(dx1,
                          static_cast<prtldx_t>(xCd[0]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[0])),
                          prtl_idx);
    }
    if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
      put_value<int>(i2, static_cast<int>(xCd[1]), prtl_idx);
      put_value<prtldx_t>(dx2,
                          static_cast<prtldx_t>(xCd[1]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[1])),
                          prtl_idx);
    }
    if constexpr (M::PrtlDim == Dim::_3D) {
      put_value<int>(i3, static_cast<int>(xCd[2]), prtl_idx);
      put_value<prtldx_t>(dx3,
                          static_cast<prtldx_t>(xCd[2]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[2])),
                          prtl_idx);
    }
    put_value<real_t>(ux1, ux_1, prtl_idx);
    put_value<real_t>(ux2, uy_1, prtl_idx);
    put_value<real_t>(ux3, uz_1, prtl_idx);
    put_value<short>(tag, ParticleTag::alive, prtl_idx);

    // set up particle #2
    prtl_idx = 1;
    if constexpr (M::Dim == Dim::_1D) {
      xi[0] = xi_2;
    }
    if constexpr (M::Dim == Dim::_2D) {
      xi[0] = xi_2;
      xi[1] = yi_2;
    }
    if constexpr (M::Dim == Dim::_3D) {
      xi[0] = xi_2;
      xi[1] = yi_2;
      xi[2] = zi_2;
    }
    metric.template convert_xyz<Crd::XYZ, Crd::Cd>(xi, xCd);
    if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                  M::PrtlDim == Dim::_3D) {
      put_value<int>(i1, static_cast<int>(xCd[0]), prtl_idx);
      put_value<prtldx_t>(dx1,
                          static_cast<prtldx_t>(xCd[0]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[0])),
                          prtl_idx);
    }
    if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
      put_value<int>(i2, static_cast<int>(xCd[1]), prtl_idx);
      put_value<prtldx_t>(dx2,
                          static_cast<prtldx_t>(xCd[1]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[1])),
                          prtl_idx);
    }
    if constexpr (M::PrtlDim == Dim::_3D) {
      put_value<int>(i3, static_cast<int>(xCd[2]), prtl_idx);
      put_value<prtldx_t>(dx3,
                          static_cast<prtldx_t>(xCd[2]) -
                            static_cast<prtldx_t>(static_cast<int>(xCd[2])),
                          prtl_idx);
    }
    put_value<real_t>(ux1, ux_2, prtl_idx);
    put_value<real_t>(ux2, uy_2, prtl_idx);
    put_value<real_t>(ux3, uz_2, prtl_idx);
    put_value<short>(tag, ParticleTag::alive, prtl_idx);
  }

  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
  };

  real_t     time   = ZERO;
  const auto n_iter = 100;

  for (auto n { 0 }; n < n_iter; ++n) {
    // clang-format off
    Kokkos::parallel_for(
      "pusher", CreateRangePolicy<Dim::_1D>({ 0 }, { 2 }),
      kernel::sr::Pusher_kernel<M>(PrtlPusher::BORIS,
                                   NoGCA, NoExtForce, kernel::sr::Cooling::None,
                                   emfield,
                                   sp_idx,
                                   i1, i2, i3,
                                   i1_prev, i2_prev, i3_prev,
                                   dx1, dx2, dx3,
                                   dx1_prev, dx2_prev, dx3_prev,
                                   ux1, ux2, ux3,
                                   phi, tag,
                                   metric,
                                   time, coeff, dt,
                                   nx1, nx2, nx3,
                                   boundaries,
                                   ZERO, ZERO, ZERO, ZERO));
    // clang-format on
    auto i1_  = Kokkos::create_mirror_view(i1);
    auto i2_  = Kokkos::create_mirror_view(i2);
    auto i3_  = Kokkos::create_mirror_view(i3);
    auto dx1_ = Kokkos::create_mirror_view(dx1);
    auto dx2_ = Kokkos::create_mirror_view(dx2);
    auto dx3_ = Kokkos::create_mirror_view(dx3);
    auto ux1_ = Kokkos::create_mirror_view(ux1);
    auto ux2_ = Kokkos::create_mirror_view(ux2);
    auto ux3_ = Kokkos::create_mirror_view(ux3);
    Kokkos::deep_copy(i1_, i1);
    Kokkos::deep_copy(i2_, i2);
    Kokkos::deep_copy(i3_, i3);
    Kokkos::deep_copy(dx1_, dx1);
    Kokkos::deep_copy(dx2_, dx2);
    Kokkos::deep_copy(dx3_, dx3);
    Kokkos::deep_copy(ux1_, ux1);
    Kokkos::deep_copy(ux2_, ux2);
    Kokkos::deep_copy(ux3_, ux3);

    coord_t<M::PrtlDim> xCd_1 { ZERO }, xCd_2 { ZERO };
    coord_t<M::PrtlDim> xPh_1 { ZERO }, xPh_2 { ZERO };
    if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
      xCd_1[0] = static_cast<real_t>(i1_(0)) + static_cast<real_t>(dx1_(0));
      xCd_2[0] = static_cast<real_t>(i1_(1)) + static_cast<real_t>(dx1_(1));
    }
    if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
      xCd_1[1] = static_cast<real_t>(i2_(0)) + static_cast<real_t>(dx2_(0));
      xCd_2[1] = static_cast<real_t>(i2_(1)) + static_cast<real_t>(dx2_(1));
    }
    if constexpr (M::Dim == Dim::_3D) {
      xCd_1[2] = static_cast<real_t>(i3_(0)) + static_cast<real_t>(dx3_(0));
      xCd_2[2] = static_cast<real_t>(i3_(1)) + static_cast<real_t>(dx3_(1));
    }
    metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xCd_1, xPh_1);
    metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xCd_2, xPh_2);

    if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
      xi_1 += dt * ux_1 / gamma_1;
      xi_2 += dt * ux_2 / gamma_2;
      if (xi_1 >= ext.at(0).second) {
        xi_1 -= sx;
      }
      if (xi_1 < ext.at(0).first) {
        xi_1 += sx;
      }
      if (xi_2 >= ext.at(0).second) {
        xi_2 -= sx;
      }
      if (xi_2 < ext.at(0).first) {
        xi_2 += sx;
      }
      errorIf(not equal(xPh_1[0] / sx,
                        xi_1 / sx,
                        fmt::format("xPh_1[0] != xi_1 @ t = %f", time)));
      errorIf(not equal(xPh_2[0] / sx,
                        xi_2 / sx,
                        fmt::format("xPh_2[0] != xi_2 @ t = %f", time)));
    }
    if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
      yi_1 += dt * uy_1 / gamma_1;
      yi_2 += dt * uy_2 / gamma_2;
      if (yi_1 >= ext.at(1).second) {
        yi_1 -= sy;
      }
      if (yi_1 < ext.at(1).first) {
        yi_1 += sy;
      }
      if (yi_2 >= ext.at(1).second) {
        yi_2 -= sy;
      }
      if (yi_2 < ext.at(1).first) {
        yi_2 += sy;
      }
      errorIf(not equal(xPh_1[1] / sy,
                        yi_1 / sy,
                        fmt::format("xPh_1[1] != yi_1 @ t = %f", time)));
      errorIf(not equal(xPh_2[1] / sy,
                        yi_2 / sy,
                        fmt::format("xPh_2[1] != yi_2 @ t = %f", time)));
    }
    if constexpr (M::Dim == Dim::_3D) {
      zi_1 += dt * uz_1 / gamma_1;
      zi_2 += dt * uz_2 / gamma_2;
      if (zi_1 >= ext.at(2).second) {
        zi_1 -= sz;
      }
      if (zi_1 < ext.at(2).first) {
        zi_1 += sz;
      }
      if (zi_2 >= ext.at(2).second) {
        zi_2 -= sz;
      }
      if (zi_2 < ext.at(2).first) {
        zi_2 += sz;
      }
      errorIf(not equal(xPh_1[2] / sz,
                        zi_1 / sz,
                        fmt::format("xPh_1[2] != zi_1 @ t = %f", time)));
      errorIf(not equal(xPh_2[2] / sz,
                        zi_2 / sz,
                        fmt::format("xPh_2[2] != zi_2 @ t = %f", time)));
    }
    time += dt;
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    const std::vector<std::size_t> res1d { 50 };
    const boundaries_t<real_t>     ext1d {
          { 0.0, 1000.0 },
    };
    const std::vector<std::size_t> res2d { 30, 20 };
    const boundaries_t<real_t>     ext2d {
          { -15.0, 15.0 },
          { -10.0, 10.0 },
    };
    const std::vector<std::size_t> res3d { 10, 10, 10 };
    const boundaries_t<real_t>     ext3d {
          { 0.0, 1.0 },
          { 0.0, 1.0 },
          { 0.0, 1.0 }
    };
    testPeriodicBC<SimEngine::SRPIC, Minkowski<Dim::_1D>>(res1d, ext1d, {});
    testPeriodicBC<SimEngine::SRPIC, Minkowski<Dim::_2D>>(res2d, ext2d, {});
    testPeriodicBC<SimEngine::SRPIC, Minkowski<Dim::_3D>>(res3d, ext3d, {});

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
