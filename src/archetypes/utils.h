/**
 * @file archetypes/utils.h
 * @brief Utility functions that use prefabricated archetypes for the most common tasks
 * @implements
 *   - arch::InjectUniformMaxwellians<> -> void
 *   - arch::InjectUniformMaxwellian<> -> void
 *   - arch::ComputeMomentWithSpecies<> -> void
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_UTILS_H
#define ARCHETYPES_UTILS_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/field_setter.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/particle_moments.hpp"

#include <utility>

namespace arch {

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * following two Maxwellian distributions
   *
   * @param domain Domain object
   * @param tot_number_density Total number density (in units of n0)
   * @param temperatures Temperatures of the two species (in units of m0 c^2)
   * @param species Pair of species indices
   * @param drift_four_vels Pair of drift four-velocities for the two species
   * @param use_weights Use weights
   * @param box Region to inject the particles in global coords (or empty for the whole domain)
   *
   * @tparam S Simulation engine type
   * @tparam M Metric type
   */
  template <SimEngine::type S, MetricClass M>
  inline void InjectUniformMaxwellians(
    const SimulationParams&            params,
    Domain<S, M>&                      domain,
    real_t    tot_number_density,
    const std::pair<real_t, real_t>&   temperatures,
    const std::pair<spidx_t, spidx_t>& species,
    const std::pair<std::vector<real_t>, std::vector<real_t>>& drift_four_vels = {{ ZERO, ZERO, ZERO }, { ZERO, ZERO, ZERO }},
    bool                               use_weights = false,
    const boundaries_t<real_t>&        box         = {}) {

    const auto mass_1        = domain.species[species.first - 1].mass();
    const auto mass_2        = domain.species[species.second - 1].mass();
    const auto temperature_1 = temperatures.first / mass_1;
    const auto temperature_2 = temperatures.second / mass_2;

    const auto maxwellian_1 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
      domain.random_pool(),
      temperature_1,
      drift_four_vels.first);
    const auto maxwellian_2 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
      domain.random_pool(),
      temperature_2,
      drift_four_vels.second);

    arch::InjectUniform<S, M, decltype(maxwellian_1), decltype(maxwellian_2)>(
      params,
      domain,
      species,
      { maxwellian_1, maxwellian_2 },
      tot_number_density,
      use_weights,
      box);
  }

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * with the same temperature for both species
   *
   * @param domain Domain object
   * @param tot_number_density Total number density (in units of n0)
   * @param temperature Temperature (in units of m0 c^2)
   * @param species Pair of species indices
   * @param drift_four_vels Pair of drift four-velocities for the two species
   * @param use_weights Use weights
   * @param box Region to inject the particles in global coords (or empty for the whole domain)
   *
   * @tparam S Simulation engine type
   * @tparam M Metric type
   */
  template <SimEngine::type S, MetricClass M>
  inline void InjectUniformMaxwellian(
    const SimulationParams&            params,
    Domain<S, M>&                      domain,
    real_t   tot_number_density,
    real_t   temperature,
    const std::pair<spidx_t, spidx_t>& species,
    const std::pair<std::vector<real_t>, std::vector<real_t>>& drift_four_vels = {{ ZERO, ZERO, ZERO }, { ZERO, ZERO, ZERO }},
    bool                               use_weights = false,
    const boundaries_t<real_t>&        box         = {}) {

    InjectUniformMaxwellians<S, M>(params,
                                   domain,
                                   tot_number_density,
                                   { temperature, temperature },
                                   species,
                                   drift_four_vels,
                                   use_weights,
                                   box);
  }

  /**
   * @brief Computes the moment of the distribution function for a given set of species and saves it to the provided buffer
   *
   * @param params Simulation parameters
   * @param domain Domain object
   * @param species Vector of species indices to include in the calculation
   * @param buffer Buffer to save the computed moment (must be preallocated with the correct size)
   * @param components Vector of field components to compute (e.g. {} for N, {0}
   * {1} {2} for V, {0, 1} for T, etc., default: empty, i.e. scalar)
   * @param buffer_idx Index of the field component in the buffer to save the result to
   * @param smoothing_order Smoothing order
   * @param smoothing_method Smoothing algorithm (e.g. SPLINE, CONST, etc.)
   *
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam F Field ID for the moment to compute (e.g. FldsID::N, FldsID::T, etc.)
   * @tparam N Last dimension of the buffer (e.g. 3 or 6)
   */
  template <SimEngine::type S, MetricClass M, FldsID::type F, int N>
  inline void ComputeMomentWithSpecies(
    const SimulationParams&     params,
    Domain<S, M>&               domain,
    const std::vector<spidx_t>& species,
    ndfield_t<M::Dim, N>&       buffer,
    const std::vector<uint8_t>& components      = {},
    idx_t                       buffer_idx      = 0u,
    uint8_t                     smoothing_order = 0u,
    OutputSmoothingTypeFlag smoothing_method    = OutputSmoothingType::SPLINE) {
    const auto ni2         = domain.mesh.n_active(in::x2);
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto use_weights = params.template get<bool>("particles.use_weights");

    if constexpr (M::Dim == Dim::_1D) {
      Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL(), buffer_idx), ZERO);
    } else if constexpr (M::Dim == Dim::_2D) {
      Kokkos::deep_copy(
        Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, buffer_idx),
        ZERO);
    } else if constexpr (M::Dim == Dim::_3D) {
      Kokkos::deep_copy(
        Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, buffer_idx),
        ZERO);
    }
    auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
    for (const auto sp : species) {
      const auto& prtl_spec = domain.species[sp - 1];
      Kokkos::parallel_for(
        "ComputeMoment",
        prtl_spec.rangeActiveParticles(),
        kernel::ParticleMoments_kernel<S, M, F, N>(components,
                                                   scatter_buff,
                                                   buffer_idx,
                                                   prtl_spec,
                                                   use_weights,
                                                   domain.mesh.metric,
                                                   domain.mesh.flds_bc(),
                                                   ni2,
                                                   inv_n0,
                                                   smoothing_order,
                                                   smoothing_method));
    }
    Kokkos::Experimental::contribute(buffer, scatter_buff);
  }

  template <SimEngine::type S, MetricClass M, kernel::ParticleMomentsFunctor MF, uint8_t N>
  inline void ComputeMomentWithSpeciesNew(
    const SimulationParams&            params,
    const Domain<S, M>&                domain,
    const std::vector<spidx_t>&        species,
    ndfield_t<M::Dim, N>&              buffer,
    const Kokkos::Array<idx_t, MF::N>& buff_indices,
    const MF&                          func,
    uint8_t                            smoothing_order = 0u,
    OutputSmoothingTypeFlag smoothing_method = OutputSmoothingType::SPLINE) {
    const auto ni2         = domain.mesh.n_active(in::x2);
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto use_weights = params.template get<bool>("particles.use_weights");

    for (const auto idx : buff_indices) {
      if constexpr (M::Dim == Dim::_1D) {
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL(), idx), ZERO);
      } else if constexpr (M::Dim == Dim::_2D) {
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, idx),
                          ZERO);
      } else if constexpr (M::Dim == Dim::_3D) {
        Kokkos::deep_copy(
          Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, idx),
          ZERO);
      }
    }
    auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
    for (const auto sp : species) {
      const auto& prtl_spec = domain.species[sp - 1];
      Kokkos::parallel_for(
        "ComputeMoment",
        prtl_spec.rangeActiveParticles(),
        kernel::ParticleMomentsNew_kernel<M, N, MF>(func,
                                                    scatter_buff,
                                                    buff_indices,
                                                    prtl_spec,
                                                    use_weights,
                                                    domain.mesh.metric,
                                                    domain.mesh.flds_bc(),
                                                    ni2,
                                                    inv_n0,
                                                    smoothing_order,
                                                    smoothing_method));
    }
    Kokkos::Experimental::contribute(buffer, scatter_buff);
  }

  template <SimEngine::type S, MetricClass M, class F>
  inline void UpdateEMFields(Domain<S, M>& domain, const F& fieldsetter) {
    if constexpr (S == SimEngine::SRPIC) {
      Kokkos::deep_copy(domain.fields.bckp, domain.fields.em);
      Kokkos::parallel_for(
        "UpdateEMFields",
        domain.mesh.rangeActiveCells(),
        arch::CustomSetEMFields_kernel<S, M, F>(domain.mesh.metric,
                                                domain.fields.em,
                                                domain.fields.bckp,
                                                fieldsetter));
      // comm here
    } else {
      raise::Error("Custom fieldsetter is only implemented for SRPIC", HERE);
    }
  }

} // namespace arch

#endif // ARCHETYPES_UTILS_H
