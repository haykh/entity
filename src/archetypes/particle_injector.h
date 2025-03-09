/**
 * @file archetypes/particle_injector.h
 * @brief Particle injector routines and classes
 * @implements
 *   - arch::UniformInjector<>
 *   - arch::NonUniformInjector<>
 *   - arch::AtmosphereInjector<>
 *   - arch::InjectUniform<> -> void
 *   - arch::InjectGlobally<> -> void
 *   - arch::InjectNonUniform<> -> void
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_PARTICLE_INJECTOR_H
#define ARCHETYPES_PARTICLE_INJECTOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#include <map>
#include <utility>
#include <vector>

namespace arch {
  using namespace ntt;
  using spidx_t = unsigned short;

  template <SimEngine::type S, class M, template <SimEngine::type, class> class ED>
  struct UniformInjector {
    using energy_dist_t = ED<S, M>;
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(energy_dist_t::is_energy_dist,
                  "E must be an energy distribution class");
    static constexpr bool      is_uniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const std::pair<spidx_t, spidx_t> species;

    UniformInjector(const energy_dist_t&               energy_dist,
                    const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , species { species } {}

    ~UniformInjector() = default;
  };

  template <SimEngine::type S,
            class M,
            template <SimEngine::type, class>
            class ED,
            template <SimEngine::type, class>
            class SD>
  struct NonUniformInjector {
    using energy_dist_t  = ED<S, M>;
    using spatial_dist_t = SD<S, M>;
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(energy_dist_t::is_energy_dist,
                  "E must be an energy distribution class");
    static_assert(spatial_dist_t::is_spatial_dist,
                  "SD must be a spatial distribution class");
    static constexpr bool      is_nonuniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const spatial_dist_t              spatial_dist;
    const std::pair<spidx_t, spidx_t> species;

    NonUniformInjector(const energy_dist_t&               energy_dist,
                       const spatial_dist_t&              spatial_dist,
                       const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , spatial_dist { spatial_dist }
      , species { species } {}

    ~NonUniformInjector() = default;
  };

  template <SimEngine::type S, class M, bool P, in O>
  struct AtmosphereInjector {
    struct TargetDensityProfile {
      const real_t nmax, height, xsurf, ds;

      TargetDensityProfile(real_t nmax, real_t height, real_t xsurf, real_t ds)
        : nmax { nmax }
        , height { height }
        , xsurf { xsurf }
        , ds { ds } {}

      Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
        if constexpr ((O == in::x1) or
                      (O == in::x2 and (M::Dim == Dim::_2D or M::Dim == Dim::_3D)) or
                      (O == in::x3 and M::Dim == Dim::_3D)) {
          const auto xi = x_Ph[static_cast<unsigned short>(O)];
          if constexpr (P) {
            // + direction
            if (xi < xsurf - ds or xi >= xsurf) {
              return ZERO;
            } else {
              if constexpr (M::CoordType == Coord::Cart) {
                return nmax * math::exp(-(xsurf - xi) / height);
              } else {
                raise::KernelError(
                  HERE,
                  "Atmosphere in +x cannot be applied for non-cartesian");
                return ZERO;
              }
            }
          } else {
            // - direction
            if (xi < xsurf or xi >= xsurf + ds) {
              return ZERO;
            } else {
              if constexpr (M::CoordType == Coord::Cart) {
                return nmax * math::exp(-(xi - xsurf) / height);
              } else {
                return nmax * math::exp(-(xsurf / height) * (ONE - (xsurf / xi)));
              }
            }
          }
        } else {
          raise::KernelError(HERE, "Wrong direction");
          return ZERO;
        }
      }
    };

    using energy_dist_t  = Maxwellian<S, M>;
    using spatial_dist_t = Replenish<S, M, TargetDensityProfile>;
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr bool      is_nonuniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const TargetDensityProfile        target_density;
    const spatial_dist_t              spatial_dist;
    const std::pair<spidx_t, spidx_t> species;

    AtmosphereInjector(const M&                           metric,
                       const ndfield_t<M::Dim, 6>&        density,
                       real_t                             nmax,
                       real_t                             height,
                       real_t                             xsurf,
                       real_t                             ds,
                       real_t                             T,
                       random_number_pool_t&              pool,
                       const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { metric, pool, T }
      , target_density { nmax, height, xsurf, ds }
      , spatial_dist { metric, density, 0, target_density, nmax }
      , species { species } {}

    ~AtmosphereInjector() = default;
  };

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * @param domain Domain object
   * @param injector Uniform injector object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam I Injector type
   */
  template <SimEngine::type S, class M, class I>
  inline void InjectUniform(const SimulationParams& params,
                            Domain<S, M>&           domain,
                            const I&                injector,
                            real_t                  number_density,
                            bool                    use_weights = false) {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I::is_uniform_injector, "I must be a uniform injector class");
    raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                   "Weights must be used for non-Cartesian coordinates",
                   HERE);
    raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                   "Weights should not be used for Cartesian coordinates",
                   HERE);
    raise::ErrorIf(params.template get<bool>("particles.use_weights") != use_weights,
                   "Weights must be enabled from the input file to use them in "
                   "the injector",
                   HERE);
    if (domain.species[injector.species.first - 1].charge() +
          domain.species[injector.species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }

    {
      auto             ppc0 = params.template get<real_t>("particles.ppc0");
      array_t<real_t*> ni { "ni", M::Dim };
      auto             ni_h   = Kokkos::create_mirror_view(ni);
      std::size_t      ncells = 1;
      for (auto d = 0; d < M::Dim; ++d) {
        ni_h(d)  = domain.mesh.n_active()[d];
        ncells  *= domain.mesh.n_active()[d];
      }
      Kokkos::deep_copy(ni, ni_h);
      const auto nparticles = static_cast<std::size_t>(
        (long double)(ppc0 * number_density * 0.5) * (long double)(ncells));

      Kokkos::parallel_for(
        "InjectUniform",
        nparticles,
        kernel::UniformInjector_kernel<S, M, typename I::energy_dist_t>(
          injector.species.first,
          injector.species.second,
          domain.species[injector.species.first - 1],
          domain.species[injector.species.second - 1],
          domain.species[injector.species.first - 1].npart(),
          domain.species[injector.species.second - 1].npart(),
          domain.mesh.metric,
          ni,
          injector.energy_dist,
          ONE / params.template get<real_t>("scales.V0"),
          domain.random_pool));
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + nparticles);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + nparticles);
    }
  }

  /**
   * @brief Injects particles from a globally-defined map
   * @note very inefficient, should only be used for debug purposes
   * @note (or when injecting very small # of particles)
   * @param global_domain Global metadomain object
   * @param local_domain Local domain object
   * @param spidx Species index
   * @param data Map containing all the coordinates/velocities of particles to inject
   * @param use_weights Boolean toggle to use weights or not
   */
  template <SimEngine::type S, class M>
  inline void InjectGlobally(const Metadomain<S, M>& global_domain,
                             Domain<S, M>&           local_domain,
                             spidx_t                 spidx,
                             const std::map<std::string, std::vector<real_t>>& data,
                             bool use_weights = false) {
    static_assert(M::is_metric, "M must be a metric class");
    const auto n_inject        = data.at("ux1").size();
    auto       injector_kernel = kernel::GlobalInjector_kernel<S, M>(
      local_domain.species[spidx - 1],
      global_domain.mesh().metric,
      local_domain,
      data,
      use_weights);
    Kokkos::parallel_for("InjectGlobally", n_inject, injector_kernel);
    const auto n_inj = injector_kernel.number_injected();
    local_domain.species[spidx - 1].set_npart(
      local_domain.species[spidx - 1].npart() + n_inj);
  }

  /**
   * @brief Injects particles based on spatial distribution function
   * @param params Simulation parameters
   * @param domain Local domain object
   * @param injector Non-uniform injector object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @param box Region to inject the particles in
   */
  template <SimEngine::type S, class M, class I>
  inline void InjectNonUniform(const SimulationParams& params,
                               Domain<S, M>&           domain,
                               const I&                injector,
                               real_t                  number_density,
                               bool                    use_weights = false,
                               boundaries_t<real_t>    box         = {}) {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I::is_nonuniform_injector,
                  "I must be a nonuniform injector class");
    raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                   "Weights must be used for non-Cartesian coordinates",
                   HERE);
    raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                   "Weights should not be used for Cartesian coordinates",
                   HERE);
    raise::ErrorIf(
      params.template get<bool>("particles.use_weights") and not use_weights,
      "Weights are enabled in the input but not enabled in the injector",
      HERE);
    raise::ErrorIf(
      not params.template get<bool>("particles.use_weights") and use_weights,
      "Weights are not enabled in the input but enabled in the injector",
      HERE);
    if (domain.species[injector.species.first - 1].charge() +
          domain.species[injector.species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }
    {
      range_t<M::Dim> cell_range;
      if (box.size() == 0) {
        cell_range = domain.mesh.rangeActiveCells();
      } else {
        raise::ErrorIf(box.size() != M::Dim,
                       "Box must have the same dimension as the mesh",
                       HERE);
        boundaries_t<bool> incl_ghosts;
        for (auto d = 0; d < M::Dim; ++d) {
          incl_ghosts.push_back({ false, false });
        }
        const auto extent = domain.mesh.ExtentToRange(box, incl_ghosts);
        tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
        for (auto d = 0; d < M::Dim; ++d) {
          x_min[d] = extent[d].first;
          x_max[d] = extent[d].second;
        }
        cell_range = CreateRangePolicy<M::Dim>(x_min, x_max);
      }
      const auto ppc = number_density *
                       params.template get<real_t>("particles.ppc0") * HALF;
      auto injector_kernel =
        kernel::NonUniformInjector_kernel<S, M, typename I::energy_dist_t, typename I::spatial_dist_t>(
          ppc,
          injector.species.first,
          injector.species.second,
          domain.species[injector.species.first - 1],
          domain.species[injector.species.second - 1],
          domain.species[injector.species.first - 1].npart(),
          domain.species[injector.species.second - 1].npart(),
          domain.mesh.metric,
          injector.energy_dist,
          injector.spatial_dist,
          ONE / params.template get<real_t>("scales.V0"),
          domain.random_pool);
      Kokkos::parallel_for("InjectNonUniformNumberDensity",
                           cell_range,
                           injector_kernel);
      const auto n_inj = injector_kernel.number_injected();
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + n_inj);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + n_inj);
    }
  }

} // namespace arch

#endif // ARCHETYPES_PARTICLE_INJECTOR_H
