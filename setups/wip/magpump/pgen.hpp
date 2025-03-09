#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <plog/Log.h>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

  private:
    const real_t Bsurf, Rstar;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar)
      : InitFields<D> { bsurf, rstar }
      , time { time } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time;
  };

  template <SimEngine::type S, class M>
  struct Inflow : public arch::EnergyDistribution<S, M> {
    Inflow(const M& metric, real_t vin)
      : arch::EnergyDistribution<S, M> { metric }
      , vin { vin } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v_Ph,
                           unsigned short) const override {
      v_Ph[0] = -vin;
    }

  private:
    const real_t vin;
  };

  template <SimEngine::type S, class M>
  struct Sphere : public arch::SpatialDistribution<S, M> {
    Sphere(const M& metric, real_t r0, real_t dr)
      : arch::SpatialDistribution<S, M> { metric }
      , r0 { r0 }
      , dr { dr } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      return math::exp(-SQR((x_Ph[0] - r0) / dr)) *
             (x_Ph[1] > 0.25 && x_Ph[1] < constant::PI - 0.25);
    }

  private:
    const real_t r0, dr;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t  Bsurf, pump_period, pump_ampl, pump_radius, Rstar;
    const real_t  vin, drinj;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , pump_period { p.template get<real_t>("setup.pump_period") }
      , pump_ampl { p.template get<real_t>("setup.pump_ampl") }
      , pump_radius { p.template get<real_t>("setup.pump_radius") }
      , Rstar { m.mesh().extent(in::x1).first }
      , vin { p.template get<real_t>("setup.vin") }
      , drinj { p.template get<real_t>("setup.drinj") }
      , init_flds { Bsurf, Rstar } {}

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, Bsurf, Rstar };
    }

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {
      const real_t radius = pump_radius +
                            pump_ampl *
                              math::sin(time * constant::TWO_PI / pump_period);
      const real_t dr     = 1.0;
      const auto&  metric = domain.mesh.metric;
      auto         EM     = domain.fields.em;
      Kokkos::parallel_for(
        "outerBC",
        domain.mesh.rangeActiveCells(),
        Lambda(index_t i1, index_t i2) {
          const auto i1_ = COORD(i1), i2_ = COORD(i2);
          const auto r = metric.template convert<1, Crd::Cd, Crd::Ph>(i1_);
          if (r > radius - 5 * dr) {
            const auto smooth   = HALF * (ONE - math::tanh((r - radius) / dr));
            EM(i1, i2, em::ex1) = smooth * EM(i1, i2, em::ex1);
            EM(i1, i2, em::ex2) = smooth * EM(i1, i2, em::ex2);
            EM(i1, i2, em::ex3) = smooth * EM(i1, i2, em::ex3);
            EM(i1, i2, em::bx1) = smooth * EM(i1, i2, em::bx1);
            EM(i1, i2, em::bx2) = smooth * EM(i1, i2, em::bx2);
            EM(i1, i2, em::bx3) = smooth * EM(i1, i2, em::bx3);
          }
        });

      if (time < pump_period * 0.25) {
        const auto energy_dist = Inflow<S, M>(domain.mesh.metric, vin);
        const auto spatial_dist = Sphere<S, M>(domain.mesh.metric, radius, drinj);
        const auto injector = arch::NonUniformInjector<S, M, Inflow, Sphere>(
          energy_dist,
          spatial_dist,
          { 1, 2 });

        arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, Inflow, Sphere>>(
          params,
          domain,
          injector,
          ONE,
          true);
      }
    }
  };

} // namespace user

#endif
