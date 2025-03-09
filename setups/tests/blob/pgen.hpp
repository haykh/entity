#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct CounterstreamEnergyDist : public arch::EnergyDistribution<S, M> {
    CounterstreamEnergyDist(const M& metric, real_t v_max)
      : arch::EnergyDistribution<S, M> { metric }
      , v_max { v_max } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      v[0] = v_max;
    }

  private:
    const real_t v_max;
  };

  template <SimEngine::type S, class M>
  struct GaussianDist : public arch::SpatialDistribution<S, M> {
    GaussianDist(const M& metric, real_t x1c, real_t x2c, real_t dr)
      : arch::SpatialDistribution<S, M> { metric }
      , x1c { x1c }
      , x2c { x2c }
      , dr { dr } {}

    // to properly scale the number density, the probability should be normalized to 1
    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      if (math::abs(x_Ph[0] - x1c) < dr && math::abs(x_Ph[1] - x2c) < dr) {
        return 1.0;
      } else {
        return 0.0;
      }
    }

  private:
    const real_t x1c, x2c, dr;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temp_1, x1c, x2c, dr, v_max;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temp_1 { p.template get<real_t>("setup.temp_1") }
      , x1c { p.template get<real_t>("setup.x1c") }
      , x2c { p.template get<real_t>("setup.x2c") }
      , v_max { p.template get<real_t>("setup.v_max") }
      , dr { p.template get<real_t>("setup.dr") } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = CounterstreamEnergyDist<S, M>(local_domain.mesh.metric,
                                                             v_max);
      const auto spatial_dist = GaussianDist<S, M>(local_domain.mesh.metric,
                                                   x1c,
                                                   x2c,
                                                   dr);
      const auto injector =
        arch::NonUniformInjector<S, M, CounterstreamEnergyDist, GaussianDist>(
          energy_dist,
          spatial_dist,
          { 1, 2 });

      arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, CounterstreamEnergyDist, GaussianDist>>(
        params,
        local_domain,
        injector,
        1.0);
    }
  };

} // namespace user

#endif
