#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

#include <string>

namespace user {
  using namespace ntt;

  enum class FieldGeometry {
    dipole,
    monopole
  };

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar, const std::string& field_geometry)
      : Bsurf { bsurf }
      , Rstar { rstar }
      , field_geom { field_geometry == "monopole" ? FieldGeometry::monopole
                                                  : FieldGeometry::dipole } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return Bsurf / SQR(x_Ph[0] / Rstar);
      } else {
        return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return ZERO;
      } else {
        return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

  private:
    const real_t        Bsurf, Rstar;
    const FieldGeometry field_geom;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(simtime_t          time,
                real_t             bsurf,
                real_t             rstar,
                real_t             omega,
                const std::string& field_geometry)
      : InitFields<D> { bsurf, rstar, field_geometry }
      , time { (real_t)time }
      , Omega { omega } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_2D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t      Bsurf, Rstar, Omega;
    const std::string field_geom;
    InitFields<D>     init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , field_geom { p.template get<std::string>("setup.field_geometry", "dipole") }
      , init_flds { Bsurf, Rstar, field_geom } {}

    inline PGen() {}

    auto AtmFields(simtime_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, Bsurf, Rstar, Omega, field_geom };
    }

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return InitFields<D> { Bsurf, Rstar, field_geom };
    }
  };

} // namespace user

#endif
