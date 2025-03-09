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
  const std::size_t clear_interval = 100;

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
      const real_t coeff   = 1.0;
    } // namespace match

    namespace absorb {
      const real_t ds_frac = 0.01;
    } // namespace absorb
  } // namespace bc

  namespace output {
    const std::string    format      = "hdf5";
    const std::size_t    interval    = 100;
    const unsigned short mom_smooth  = 0;
    const std::size_t    prtl_stride = 100;
    const real_t         spec_emin   = 1e-3;
    const real_t         spec_emax   = 1e3;
    const bool           spec_log    = true;
    const std::size_t    spec_nbins  = 200;
  } // namespace output

  namespace checkpoint {
    const std::size_t interval = 1000;
    const int         keep     = 2;
  } // namespace checkpoint

  namespace diag {
    const std::size_t interval = 1;
  } // namespace diag

  namespace gca {
    const real_t EovrB_max = 0.9;
  } // namespace gca

  namespace synchrotron {
    const real_t gamma_rad = 1.0;
  } // namespace synchrotron

} // namespace ntt::defaults

#endif // GLOBAL_DEFAULTS_H
