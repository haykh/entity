/**
 * @file archetypes/particle_injector.h
 * @brief Particle injector routines and classes
 * @implements
 *   - arch::DeduceRegion<> -> tuple<bool, array_t<real_t*>, array_t<real_t*>>
 *   - arch::ComputeNumInject<> -> tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>>
 *   - arch::AtmosphereDensityProfile<>
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

#include "metrics/traits.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace arch {
  using namespace ntt;

  /**
   * @brief Deduces the region of injection in computational coordinates
   * @param domain Domain object
   * @param box Region to inject the particles in global coords
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @return Tuple containing:
   *   - bool: whether the region intersects with the local domain
   *   - array_t<real_t*>: minimum coordinates of the region in computational coords
   *   - array_t<real_t*>: maximum coordinates of the region in computational coords
   */
  template <SimEngine::type S, class M>
    requires metric::traits::HasD<M> && metric::traits::HasConvert<M>
  auto DeduceRegion(const Domain<S, M>& domain, const boundaries_t<real_t>& box)
    -> std::tuple<bool, array_t<real_t*>, array_t<real_t*>> {
    if (not domain.mesh.Intersects(box)) {
      return { false, array_t<real_t*> {}, array_t<real_t*> {} };
    }
    coord_t<M::Dim> xCorner_min_Ph { ZERO };
    coord_t<M::Dim> xCorner_max_Ph { ZERO };
    coord_t<M::Dim> xCorner_min_Cd { ZERO };
    coord_t<M::Dim> xCorner_max_Cd { ZERO };

    for (auto d { 0u }; d < M::Dim; ++d) {
      const auto local_xi_min = domain.mesh.extent(static_cast<in>(d)).first;
      const auto local_xi_max = domain.mesh.extent(static_cast<in>(d)).second;
      const auto extent_min   = std::min(std::max(local_xi_min, box[d].first),
                                       local_xi_max);
      const auto extent_max   = std::max(std::min(local_xi_max, box[d].second),
                                       local_xi_min);
      xCorner_min_Ph[d]       = extent_min;
      xCorner_max_Ph[d]       = extent_max;
    }
    domain.mesh.metric.template convert<Crd::Ph, Crd::Cd>(xCorner_min_Ph,
                                                          xCorner_min_Cd);
    domain.mesh.metric.template convert<Crd::Ph, Crd::Cd>(xCorner_max_Ph,
                                                          xCorner_max_Cd);

    array_t<real_t*> xi_min { "xi_min", M::Dim }, xi_max { "xi_max", M::Dim };

    auto xi_min_h = Kokkos::create_mirror_view(xi_min);
    auto xi_max_h = Kokkos::create_mirror_view(xi_max);
    for (auto d { 0u }; d < M::Dim; ++d) {
      xi_min_h(d) = xCorner_min_Cd[d];
      xi_max_h(d) = xCorner_max_Cd[d];
    }
    Kokkos::deep_copy(xi_min, xi_min_h);
    Kokkos::deep_copy(xi_max, xi_max_h);

    return { true, xi_min, xi_max };
  }

  /**
   * @brief Computes the number of particles to inject in a given region
   * @param params Simulation parameters
   * @param domain Domain object
   * @param number_density Number density (in units of n0)
   * @param box Region to inject the particles in global coords
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @return Tuple containing:
   *   - bool: whether the region intersects with the local domain
   *   - npart_t: number of particles to inject
   *   - array_t<real_t*>: minimum coordinates of the region in computational coords
   *   - array_t<real_t*>: maximum coordinates of the region in computational coords
   */
  template <SimEngine::type S, class M>
    requires metric::traits::HasD<M>
  auto ComputeNumInject(const SimulationParams&     params,
                        const Domain<S, M>&         domain,
                        real_t                      number_density,
                        const boundaries_t<real_t>& box)
    -> std::tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>> {
    const auto result = DeduceRegion(domain, box);
    if (not std::get<0>(result)) {
      return { false, (npart_t)0, array_t<real_t*> {}, array_t<real_t*> {} };
    }
    const auto xi_min   = std::get<1>(result);
    const auto xi_max   = std::get<2>(result);
    auto       xi_min_h = Kokkos::create_mirror_view(xi_min);
    auto       xi_max_h = Kokkos::create_mirror_view(xi_max);
    Kokkos::deep_copy(xi_min_h, xi_min);
    Kokkos::deep_copy(xi_max_h, xi_max);

    long double num_cells { 1.0 };
    for (auto d { 0u }; d < M::Dim; ++d) {
      num_cells *= static_cast<long double>(xi_max_h(d)) -
                   static_cast<long double>(xi_min_h(d));
    }

    const auto ppc0       = params.template get<real_t>("particles.ppc0");
    const auto nparticles = static_cast<npart_t>(
      (long double)(ppc0 * number_density * 0.5) * num_cells);

    return { true, nparticles, xi_min, xi_max };
  }

  template <Dimension D, Coord::type C, bool P, in O>
  struct AtmosphereDensityProfile {
    const real_t nmax, height, xsurf, ds;

    AtmosphereDensityProfile(real_t nmax, real_t height, real_t xsurf, real_t ds)
      : nmax { nmax }
      , height { height }
      , xsurf { xsurf }
      , ds { ds } {}

    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr ((O == in::x1) or
                    (O == in::x2 and (D == Dim::_2D or D == Dim::_3D)) or
                    (O == in::x3 and D == Dim::_3D)) {
        const auto xi = x_Ph[static_cast<dim_t>(O)];
        if constexpr (P) {
          // + direction
          if (xi < xsurf - ds or xi >= xsurf) {
            return ZERO;
          } else {
            if constexpr (C == Coord::Cart) {
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
            if constexpr (C == Coord::Cart) {
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

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * @param domain Domain object
   * @param species Pair of species indices
   * @param energy_dists Pair of energy distribution objects
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @param box Region to inject the particles in global coords
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam ED1 Energy distribution type for species 1
   * @tparam ED2 Energy distribution type for species 2
   */
  template <SimEngine::type S, class M, class ED1, class ED2>
    requires metric::traits::HasD<M> && traits::energydist::IsValid<ED1> &&
             traits::energydist::IsValid<ED2>
  inline void InjectUniform(const SimulationParams&            params,
                            Domain<S, M>&                      domain,
                            const std::pair<spidx_t, spidx_t>& species,
                            const std::pair<ED1, ED2>&         energy_dists,
                            real_t                             number_density,
                            bool                        use_weights = false,
                            const boundaries_t<real_t>& box         = {}) {
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
    if (domain.species[species.first - 1].charge() +
          domain.species[species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }

    {
      boundaries_t<real_t> nonempty_box;
      for (auto d { 0u }; d < M::Dim; ++d) {
        if (d < box.size()) {
          nonempty_box.push_back({ box[d].first, box[d].second });
        } else {
          nonempty_box.push_back(Range::All);
        }
      }
      const auto result = ComputeNumInject(params, domain, number_density, nonempty_box);
      if (not std::get<0>(result)) {
        return;
      }
      const auto nparticles = std::get<1>(result);
      const auto xi_min     = std::get<2>(result);
      const auto xi_max     = std::get<3>(result);

      Kokkos::parallel_for("InjectUniform",
                           nparticles,
                           kernel::UniformInjector_kernel<S, M, ED1, ED2>(
                             domain.species[species.first - 1],
                             domain.species[species.second - 1],
                             nparticles,
                             domain.index(),
                             domain.mesh.metric,
                             xi_min,
                             xi_max,
                             energy_dists.first,
                             energy_dists.second,
                             ONE / params.template get<real_t>("scales.V0"),
                             domain.random_pool()));
      domain.species[species.first - 1].set_npart(
        domain.species[species.first - 1].npart() + nparticles);
      domain.species[species.second - 1].set_npart(
        domain.species[species.second - 1].npart() + nparticles);
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
   * @param species Pair of species indices
   * @param energy_dists Pair of energy distribution objects
   * @param spatial_dist Spatial distribution object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @param box Region to inject the particles in
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam ED1 Energy distribution type for species 1
   * @tparam ED2 Energy distribution type for species 2
   * @tparam SD Spatial distribution type
   */
  template <SimEngine::type S, class M, class ED1, class ED2, class SD>
    requires metric::traits::HasD<M> && traits::energydist::IsValid<ED1> &&
             traits::energydist::IsValid<ED2> && traits::spatialdist::IsValid<SD>
  inline void InjectNonUniform(const SimulationParams&            params,
                               Domain<S, M>&                      domain,
                               const std::pair<spidx_t, spidx_t>& species,
                               const std::pair<ED1, ED2>&         energy_dists,
                               const SD&                          spatial_dist,
                               real_t                      number_density,
                               bool                        use_weights = false,
                               const boundaries_t<real_t>& box         = {}) {
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
    if (domain.species[species.first - 1].charge() +
          domain.species[species.second - 1].charge() !=
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
        tuple_t<ncells_t, M::Dim> x_min { 0 }, x_max { 0 };
        for (auto d = 0; d < M::Dim; ++d) {
          x_min[d] = extent[d].first;
          x_max[d] = extent[d].second;
        }
        cell_range = CreateRangePolicy<M::Dim>(x_min, x_max);
      }
      const auto ppc = number_density *
                       params.template get<real_t>("particles.ppc0") * HALF;
      auto injector_kernel = kernel::NonUniformInjector_kernel<S, M, ED1, ED2, SD>(
        ppc,
        domain.species[species.first - 1],
        domain.species[species.second - 1],
        domain.index(),
        domain.mesh.metric,
        energy_dists.first,
        energy_dists.second,
        spatial_dist,
        ONE / params.template get<real_t>("scales.V0"),
        domain.random_pool());
      Kokkos::parallel_for("InjectNonUniformNumberDensity",
                           cell_range,
                           injector_kernel);
      const auto n_inj = injector_kernel.number_injected();
      for (auto sp : { species.first, species.second }) {
        domain.species[sp - 1].set_npart(domain.species[sp - 1].npart() + n_inj);
        domain.species[sp - 1].set_counter(domain.species[sp - 1].counter() + n_inj);
      }
    }
  }

} // namespace arch

#endif // ARCHETYPES_PARTICLE_INJECTOR_H
