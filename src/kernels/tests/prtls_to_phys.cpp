#include "kernels/prtls_to_phys.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

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

using namespace ntt;

template <class M>
struct Checker {
  Checker(const M&                  metric,
          std::size_t               stride,
          const array_t<int*>&      i1,
          const array_t<prtldx_t*>& dx1,
          const array_t<int*>&      i2,
          const array_t<prtldx_t*>& dx2,
          const array_t<real_t*>&   phi,
          const array_t<real_t*>&   ux1,
          const array_t<real_t*>&   ux2,
          const array_t<real_t*>&   ux3,
          const array_t<real_t*>&   weight,
          const array_t<real_t*>&   buff_x1,
          const array_t<real_t*>&   buff_x2,
          const array_t<real_t*>&   buff_x3,
          const array_t<real_t*>&   buff_ux1,
          const array_t<real_t*>&   buff_ux2,
          const array_t<real_t*>&   buff_ux3,
          const array_t<real_t*>&   buff_wei)
    : metric { metric }
    , stride { stride }
    , i1 { i1 }
    , dx1 { dx1 }
    , i2 { i2 }
    , dx2 { dx2 }
    , phi { phi }
    , ux1 { ux1 }
    , ux2 { ux2 }
    , ux3 { ux3 }
    , weight { weight }
    , buff_x1 { buff_x1 }
    , buff_x2 { buff_x2 }
    , buff_x3 { buff_x3 }
    , buff_ux1 { buff_ux1 }
    , buff_ux2 { buff_ux2 }
    , buff_ux3 { buff_ux3 }
    , buff_wei { buff_wei } {}

  Inline void operator()(index_t p) const {
    std::size_t pold = p * stride;
    real_t x1 = static_cast<real_t>(i1(pold)) + static_cast<real_t>(dx1(pold));
    real_t x2 = static_cast<real_t>(i2(pold)) + static_cast<real_t>(dx2(pold));
    real_t x1_phys = metric.template convert<1, Crd::Cd, Crd::Ph>(x1);
    real_t x2_phys = metric.template convert<2, Crd::Cd, Crd::Ph>(x2);
    if (not cmp::AlmostEqual(x1_phys, buff_x1(p))) {
      raise::KernelError(HERE, "x1_phys != buff_x1");
    }
    if (not cmp::AlmostEqual(x2_phys, buff_x2(p))) {
      raise::KernelError(HERE, "x2_phys != buff_x2");
    }
    if (not cmp::AlmostEqual(phi(pold), buff_x3(p))) {
      raise::KernelError(HERE, "phi != buff_x3");
    }
    vec_t<Dim::_3D> u_Phys { ZERO };
    metric.template transform_xyz<Idx::XYZ, Idx::T>(
      { x1, x2, phi(pold) },
      { ux1(pold), ux2(pold), ux3(pold) },
      u_Phys);
    if (not cmp::AlmostEqual(u_Phys[0], buff_ux1(p))) {
      raise::KernelError(HERE, "u_Phys[0] != buff_ux1");
    }
    if (not cmp::AlmostEqual(u_Phys[1], buff_ux2(p))) {
      raise::KernelError(HERE, "u_Phys[1] != buff_ux2");
    }
    if (not cmp::AlmostEqual(u_Phys[2], buff_ux3(p))) {
      raise::KernelError(HERE, "u_Phys[2] != buff_ux3");
    }
    if (not cmp::AlmostEqual(weight(pold), buff_wei(p))) {
      raise::KernelError(HERE, "weight != buff_wei");
    }
  }

private:
  const M                  metric;
  std::size_t              stride;
  const array_t<int*>      i1;
  const array_t<prtldx_t*> dx1;
  const array_t<int*>      i2;
  const array_t<prtldx_t*> dx2;
  const array_t<real_t*>   phi;
  const array_t<real_t*>   ux1;
  const array_t<real_t*>   ux2;
  const array_t<real_t*>   ux3;
  const array_t<real_t*>   weight;
  array_t<real_t*>         buff_x1;
  array_t<real_t*>         buff_x2;
  array_t<real_t*>         buff_x3;
  array_t<real_t*>         buff_ux1;
  array_t<real_t*>         buff_ux2;
  array_t<real_t*>         buff_ux3;
  array_t<real_t*>         buff_wei;
};

template <typename M>
void testPrtl2PhysSR(const std::vector<std::size_t>&      res,
                     const boundaries_t<real_t>&          ext,
                     const std::map<std::string, real_t>& params = {}) {
  static constexpr Dimension D = M::Dim;
  static_assert(D == 2, "Invalid dimension");
  raise::ErrorIf(res.size() != D, "res.size() != D", HERE);

  boundaries_t<real_t> extent;

  extent = {
    ext[0],
    {ZERO, constant::PI}
  };

  const M metric { res, extent, params };

  const std::size_t nprtl = 100;

  array_t<int*>      i1 { "i1", nprtl };
  array_t<prtldx_t*> dx1 { "dx1", nprtl };
  array_t<int*>      i2 { "i2", nprtl };
  array_t<prtldx_t*> dx2 { "dx2", nprtl };
  array_t<real_t*>   phi { "phi", nprtl };
  array_t<real_t*>   ux1 { "ux1", nprtl };
  array_t<real_t*>   ux2 { "ux2", nprtl };
  array_t<real_t*>   ux3 { "ux3", nprtl };
  array_t<real_t*>   weight { "weight", nprtl };

  array_t<int*>      i3;
  array_t<prtldx_t*> dx3;

  Kokkos::parallel_for(
    "Init",
    nprtl,
    Lambda(index_t p) {
      // init "random" values
      i1(p)     = p % 10;
      i2(p)     = (p + nprtl) % 10;
      dx1(p)    = (prtldx_t)(p) / (prtldx_t)(2 * nprtl);
      dx2(p)    = (prtldx_t)(1.0) - (prtldx_t)(p) / (prtldx_t)(2 * nprtl);
      phi(p)    = constant::TWO_PI * (real_t)(p) / (real_t)(nprtl);
      ux1(p)    = ((real_t)(p) - (real_t)(nprtl) / 2) / (real_t)(2 * nprtl);
      ux2(p)    = ((real_t)(p) - (real_t)(nprtl) / 4) / (real_t)(9 * nprtl);
      ux3(p)    = ((real_t)(p) - (real_t)(nprtl) / 2) / (real_t)(5 * nprtl);
      weight(p) = (real_t)(25) + (real_t)(p) / (real_t)(nprtl);
    });

  const std::size_t stride = 2;
  array_t<real_t*>  buff_x1 { "buff_x1", nprtl / stride };
  array_t<real_t*>  buff_x2 { "buff_x2", nprtl / stride };
  array_t<real_t*>  buff_x3 { "buff_x3", nprtl / stride };
  array_t<real_t*>  buff_ux1 { "buff_ux1", nprtl / stride };
  array_t<real_t*>  buff_ux2 { "buff_ux2", nprtl / stride };
  array_t<real_t*>  buff_ux3 { "buff_ux3", nprtl / stride };
  array_t<real_t*>  buff_wei { "buff_wei", nprtl / stride };

  Kokkos::parallel_for(
    "Init",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nprtl / stride),
    kernel::PrtlToPhys_kernel<SimEngine::SRPIC, M>(stride,
                                                   buff_x1,
                                                   buff_x2,
                                                   buff_x3,
                                                   buff_ux1,
                                                   buff_ux2,
                                                   buff_ux3,
                                                   buff_wei,
                                                   i1,
                                                   i2,
                                                   i3,
                                                   dx1,
                                                   dx2,
                                                   dx3,
                                                   ux1,
                                                   ux2,
                                                   ux3,
                                                   phi,
                                                   weight,
                                                   metric));
  Kokkos::parallel_for("Check",
                       nprtl / stride,
                       Checker<M>(metric,
                                  stride,
                                  i1,
                                  dx1,
                                  i2,
                                  dx2,
                                  phi,
                                  ux1,
                                  ux2,
                                  ux3,
                                  weight,
                                  buff_x1,
                                  buff_x2,
                                  buff_x3,
                                  buff_ux1,
                                  buff_ux2,
                                  buff_ux3,
                                  buff_wei));
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace metric;

    testPrtl2PhysSR<Spherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 2.0 } },
      {});

    testPrtl2PhysSR<QSpherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 10.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
