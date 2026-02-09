#include "framework/parameters/algorithms.h"

#include "defaults.h"
#include "global.h"

#include "utils/numeric.h"
#include "utils/toml.h"

#include "framework/parameters/parameters.h"

namespace ntt {
  namespace params {

    void Algorithms::read(real_t                             dx0,
                          const std::map<std::string, bool>& extra,
                          const toml::value&                 toml_data) {
      CFL = toml::find_or(toml_data, "algorithms", "timestep", "CFL", defaults::cfl);
      dt                   = CFL * dx0;
      dt_correction_factor = toml::find_or(toml_data,
                                           "algorithms",
                                           "timestep",
                                           "correction",
                                           defaults::correction);

      number_of_current_filters = toml::find_or(toml_data,
                                                "algorithms",
                                                "current_filters",
                                                defaults::current_filters);

      deposit_enable = toml::find_or(toml_data, "algorithms", "deposit", "enable", true);
      deposit_order = static_cast<unsigned short>(SHAPE_ORDER);

      fieldsolver_enable = toml::find_or(toml_data,
                                         "algorithms",
                                         "fieldsolver",
                                         "enable",
                                         true);

      fieldsolver_stencil_coeffs["delta_x"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "delta_x",
        defaults::fieldsolver::delta_x);

      fieldsolver_stencil_coeffs["delta_y"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "delta_y",
        defaults::fieldsolver::delta_y);

      fieldsolver_stencil_coeffs["delta_z"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "delta_z",
        defaults::fieldsolver::delta_z);

      fieldsolver_stencil_coeffs["beta_xy"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_xy",
        defaults::fieldsolver::beta_xy);

      fieldsolver_stencil_coeffs["beta_yx"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_yx",
        defaults::fieldsolver::beta_yx);

      fieldsolver_stencil_coeffs["beta_xz"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_xz",
        defaults::fieldsolver::beta_xz);

      fieldsolver_stencil_coeffs["beta_zx"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_zx",
        defaults::fieldsolver::beta_zx);

      fieldsolver_stencil_coeffs["beta_yz"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_yz",
        defaults::fieldsolver::beta_yz);

      fieldsolver_stencil_coeffs["beta_zy"] = toml::find_or(
        toml_data,
        "algorithms",
        "fieldsolver",
        "beta_zy",
        defaults::fieldsolver::beta_zy);

      if (extra.at("gr")) {
        gr_pusher_eps   = toml::find_or(toml_data,
                                      "algorithms",
                                      "gr",
                                      "pusher_eps",
                                      defaults::gr::pusher_eps);
        gr_pusher_niter = toml::find_or(toml_data,
                                        "algorithms",
                                        "gr",
                                        "pusher_niter",
                                        defaults::gr::pusher_niter);
      }

      if (extra.at("gca")) {
        gca_e_ovr_b_max = toml::find_or(toml_data,
                                        "algorithms",
                                        "gca",
                                        "e_ovr_b_max",
                                        defaults::gca::EovrB_max);
        gca_larmor_max  = toml::find_or(toml_data,
                                       "algorithms",
                                       "gca",
                                       "larmor_max",
                                       ZERO);
      }
    }

    void Algorithms::setParams(const std::map<std::string, bool>& extra,
                               SimulationParams* params) const {
      params->set("algorithms.timestep.CFL", CFL);
      params->set("algorithms.timestep.dt", dt);
      params->set("algorithms.timestep.correction", dt_correction_factor);

      params->set("algorithms.current_filters", number_of_current_filters);

      params->set("algorithms.deposit.enable", deposit_enable);
      params->set("algorithms.deposit.order", deposit_order);

      params->set("algorithms.fieldsolver.enable", fieldsolver_enable);
      for (const auto& [key, value] : fieldsolver_stencil_coeffs) {
        params->set("algorithms.fieldsolver." + key, value);
      }

      if (extra.at("gr")) {
        params->set("algorithms.gr.pusher_eps", gr_pusher_eps);
        params->set("algorithms.gr.pusher_niter", gr_pusher_niter);
      }

      if (extra.at("gca")) {
        params->set("algorithms.gca.e_ovr_b_max", gca_e_ovr_b_max);
        params->set("algorithms.gca.larmor_max", gca_larmor_max);
      }
    }

  } // namespace params
} // namespace ntt
