/**
 * @file archetypes/utils.h
 * @brief Utility functions that use prefabricated archetypes for the most common tasks
 * @implements
 *   - arch::InjectUniformMaxwellians<> -> void
 *   - arch::InjectUniformMaxwellian<> -> void
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_UTILS_H
#define ARCHETYPES_UTILS_H

#include "enums.h"
#include "global.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

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
   */
  template <SimEngine::type S, class M>
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

    const auto maxwellian_1 = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                     domain.random_pool(),
                                                     temperature_1,
                                                     drift_four_vels.first);
    const auto maxwellian_2 = arch::Maxwellian<S, M>(domain.mesh.metric,
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
   */
  template <SimEngine::type S, class M>
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

} // namespace arch

#endif // ARCHETYPES_UTILS_H
