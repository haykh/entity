/**
 * @file framework/parameters/algorithms.h
 * @brief Auxiliary functions for reading in extra physics parameters
 * @implements
 *   - ntt::params::Extra
 * @cpp:
 *   - extra.cpp
 * @namespaces:
 *   - ntt::params::
 */

#ifndef FRAMEWORK_PARAMETERS_EXTRA_H
#define FRAMEWORK_PARAMETERS_EXTRA_H

#include "global.h"

#include "utils/toml.h"

#include "framework/parameters/parameters.h"

#include <map>
#include <string>

namespace ntt {
  namespace params {

    struct Extra {
      // radiative drag parameters
      real_t synchrotron_gamma_rad;
      real_t compton_gamma_rad;

      // emission parameters
      real_t  synchrotron_energy_min;
      real_t  synchrotron_gamma_qed;
      real_t  synchrotron_photon_weight;
      real_t  synchrotron_photon_sampling_rate;
      spidx_t synchrotron_photon_species;

      real_t  compton_energy_min;
      real_t  compton_gamma_qed;
      real_t  compton_photon_weight;
      real_t  compton_photon_sampling_rate;
      spidx_t compton_photon_species;

      void read(const std::map<std::string, bool>&, const toml::value&);
      void setParams(const std::map<std::string, bool>&, SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_EXTRA_H
