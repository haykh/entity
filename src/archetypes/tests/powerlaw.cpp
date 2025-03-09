#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "archetypes/energy_dist.h"

#include <Kokkos_Core.hpp>

#include <iostream>

using namespace ntt;
using namespace metric;
using namespace arch;

template <class EnrgDist, SimEngine::type S, class M>
struct Caller {
  static constexpr auto D = M::Dim;

  Caller(const M& metric, const EnrgDist& dist)
    : metric { metric }
    , dist { dist } {}

  Inline void operator()(index_t) const {
    vec_t<Dim::_3D> vp { ZERO };
    coord_t<D>      xp { ZERO };
    for (unsigned short d = 0; d < D; ++d) {
      xp[d] = 2.0;
    }
    dist(xp, vp);
    if (not Kokkos::isfinite(vp[0]) or not Kokkos::isfinite(vp[1]) or
        not Kokkos::isfinite(vp[2])) {
      raise::KernelError(HERE, "Non-finite velocity generated");
    }
    if constexpr (S == SimEngine::SRPIC) {
      const auto gamma = math::sqrt(ONE + SQR(vp[0]) + SQR(vp[1]) + SQR(vp[2]));
      if (gamma < 10 or gamma > 1000) {
        raise::KernelError(HERE, "Gamma out of bounds");
      }
    } else {
      vec_t<Dim::_3D> vup { ZERO };
      metric.template transform<Idx::D, Idx::U>(xp, vp, vup);
      const auto gamma = math::sqrt(
        ONE + vup[0] * vp[0] + vup[1] * vp[1] + vup[2] * vp[2]);
      if (gamma < 10 or gamma > 1000) {
        raise::KernelError(HERE, "Gamma out of bounds");
      }
    }
  }

private:
  M        metric;
  EnrgDist dist;
};

template <SimEngine::type S, typename M>
void testEnergyDist(const std::vector<std::size_t>&      res,
                    const boundaries_t<real_t>&          ext,
                    const std::map<std::string, real_t>& params = {}) {
  raise::ErrorIf(res.size() != M::Dim, "res.size() != M::Dim", HERE);

  boundaries_t<real_t> extent;
  if constexpr (M::CoordType == Coord::Cart) {
    extent = ext;
  } else {
    if constexpr (M::Dim == Dim::_2D) {
      extent = {
        ext[0],
        {ZERO, constant::PI}
      };
    } else if constexpr (M::Dim == Dim::_3D) {
      extent = {
        ext[0],
        {ZERO,     constant::PI},
        {ZERO, constant::TWO_PI}
      };
    }
  }
  raise::ErrorIf(extent.size() != M::Dim, "extent.size() != M::Dim", HERE);

  M metric { res, extent, params };

  random_number_pool_t pool { constant::RandomSeed };
  Powerlaw<S, M>       plaw { metric,
                        pool,
                        static_cast<real_t>(10),
                        static_cast<real_t>(1000),
                        static_cast<real_t>(-2.5) };
  Kokkos::parallel_for("Powerlaw", 100, Caller<Powerlaw<S, M>, S, M>(metric, plaw));
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    testEnergyDist<SimEngine::SRPIC, Minkowski<Dim::_1D>>(
      {
        10
    },
      { { 0.0, 55.0 } });

    testEnergyDist<SimEngine::SRPIC, Minkowski<Dim::_2D>>(
      {
        10,
        10
    },
      { { 0.0, 55.0 }, { 0.0, 55.0 } });

    testEnergyDist<SimEngine::SRPIC, Minkowski<Dim::_3D>>(
      {
        10,
        10,
        10
    },
      { { 0.0, 55.0 }, { 0.0, 55.0 }, { 0.0, 55.0 } });

    testEnergyDist<SimEngine::SRPIC, Spherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } });

    testEnergyDist<SimEngine::SRPIC, QSpherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } });

    testEnergyDist<SimEngine::GRPIC, KerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } });

    testEnergyDist<SimEngine::GRPIC, QKerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 }, { "a", 0.9 } });

    testEnergyDist<SimEngine::GRPIC, KerrSchild0<Dim::_2D>>(
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
