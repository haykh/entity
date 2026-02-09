/**
 * @file framework/parameters/algorithms.h
 * @brief Auxiliary functions for reading in algorithms parameters
 * @implements
 *   - ntt::params::Algorithms
 * @cpp:
 *   - algorithms.cpp
 * @namespaces:
 *   - ntt::params::
 */

#ifndef FRAMEWORK_PARAMETERS_ALGORITHMS_H
#define FRAMEWORK_PARAMETERS_ALGORITHMS_H

#include "global.h"

#include "utils/toml.h"

#include "framework/parameters/parameters.h"

#include <map>
#include <string>

namespace ntt {
  namespace params {

    struct Algorithms {
      real_t CFL;
      real_t dt;
      real_t dt_correction_factor;

      unsigned short number_of_current_filters;

      bool           deposit_enable;
      unsigned short deposit_order;

      bool                          fieldsolver_enable;
      std::map<std::string, real_t> fieldsolver_stencil_coeffs;

      real_t         gr_pusher_eps;
      unsigned short gr_pusher_niter;

      real_t gca_e_ovr_b_max;
      real_t gca_larmor_max;

      real_t synchrotron_gamma_rad;
      real_t compton_gamma_rad;

      void read(real_t, const std::map<std::string, bool>&, const toml::value&);
      void setParams(const std::map<std::string, bool>&, SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_ALGORITHMS_H
