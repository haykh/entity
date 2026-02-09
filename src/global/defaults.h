/**
 * @file defaults.h
 * @brief Default values for the simulation parameters
 * @namespaces:
 *   - ntt::defaults
 * @note These values are used when the user does not provide them in the input file
 */

#ifndef GLOBAL_DEFAULTS_H
#define GLOBAL_DEFAULTS_H

#include "global.h"

#include <string>
#include <string_view>

namespace ntt::defaults {
  constexpr std::string_view input_filename = "input";

  const real_t correction = 1.0;
  const real_t cfl        = 0.95;

  const unsigned short current_filters = 0;

  const std::string em_pusher      = "Boris";
  const std::string ph_pusher      = "Photon";
  const timestep_t  clear_interval = 100;

  namespace fieldsolver {
    const real_t delta_x = 0.0;

    const real_t delta_y = 0.0;
    const real_t beta_xy = 0.0;
    const real_t beta_yx = 0.0;

    const real_t delta_z = 0.0;
    const real_t beta_xz = 0.0;
    const real_t beta_zx = 0.0;
    const real_t beta_yz = 0.0;
    const real_t beta_zy = 0.0;
  } // namespace fieldsolver

  namespace qsph {
    const real_t r0 = 0.0;
    const real_t h  = 0.0;
  } // namespace qsph

  namespace ks {
    const real_t a = 0.0;
  } // namespace ks

  namespace gr {
    const real_t         pusher_eps   = 1e-6;
    const unsigned short pusher_niter = 10;
  } // namespace gr

  namespace bc {
    namespace match {
      const real_t ds_frac = 0.01;
    } // namespace match

    namespace absorb {
      const real_t ds_frac = 0.01;
    } // namespace absorb
  } // namespace bc

  namespace output {
    const std::string              format           = "BPFile";
    const timestep_t               interval         = 100;
    const unsigned short           mom_smooth       = 0;
    const npart_t                  prtl_stride      = 100;
    const real_t                   spec_emin        = 1e-3;
    const real_t                   spec_emax        = 1e3;
    const bool                     spec_log         = true;
    const std::size_t              spec_nbins       = 200;
    const std::vector<std::string> stats_quantities = { "B^2",
                                                        "E^2",
                                                        "ExB",
                                                        "Rho",
                                                        "T00" };
  } // namespace output

  namespace checkpoint {
    const timestep_t  interval   = 1000;
    const int         keep       = 2;
    const std::string walltime   = "00:00:00";
    const std::string write_path = "%s.ckpt";
  } // namespace checkpoint

  namespace diag {
    const timestep_t  interval  = 1;
    const std::string log_level = "VERBOSE";
  } // namespace diag

  namespace gca {
    const real_t EovrB_max = 0.9;
  } // namespace gca

  namespace synchrotron {
    const real_t energy_min = 1e-3;
    const real_t gamma_rad  = 1.0;
    const real_t gamma_qed  = 10.0;
  } // namespace synchrotron

  namespace compton {
    const real_t energy_min = 1e-3;
    const real_t gamma_rad  = 1.0;
    const real_t gamma_qed  = 10.0;
  } // namespace compton
} // namespace ntt::defaults

#endif // GLOBAL_DEFAULTS_H
