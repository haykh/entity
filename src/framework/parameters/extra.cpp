#include "framework/parameters/extra.h"

#include "defaults.h"

#include "utils/numeric.h"

namespace ntt {
  namespace params {

    void Extra::read(const std::map<std::string, bool>& extra,
                     const toml::value&                 toml_data) {
      if (extra.at("synchrotron_drag")) {
        synchrotron_gamma_rad = toml::find_or(toml_data,
                                              "radiation",
                                              "drag",
                                              "synchrotron",
                                              "gamma_rad",
                                              defaults::synchrotron::gamma_rad);
      }

      if (extra.at("compton_drag")) {
        compton_gamma_rad = toml::find_or(toml_data,
                                          "radiation",
                                          "drag",
                                          "compton",
                                          "gamma_rad",
                                          defaults::compton::gamma_rad);
      }

      if (extra.at("synchrotron_emission")) {
        synchrotron_gamma_qed      = toml::find_or(toml_data,
                                              "radiation",
                                              "emission",
                                              "synchrotron",
                                              "gamma_qed",
                                              defaults::synchrotron::gamma_qed);
        synchrotron_energy_min     = toml::find_or(toml_data,
                                               "radiation",
                                               "emission",
                                               "synchrotron",
                                               "photon_energy_min",
                                               defaults::synchrotron::energy_min);
        synchrotron_photon_weight  = toml::find_or(toml_data,
                                                  "radiation",
                                                  "emission",
                                                  "synchrotron",
                                                  "photon_weight",
                                                  ONE);
        synchrotron_photon_species = toml::find<spidx_t>(toml_data,
                                                         "radiation",
                                                         "emission",
                                                         "synchrotron",
                                                         "photon_species");
      }

      if (extra.at("compton_emission")) {
        compton_gamma_qed      = toml::find_or(toml_data,
                                          "radiation",
                                          "emission",
                                          "compton",
                                          "gamma_qed",
                                          defaults::compton::gamma_qed);
        compton_energy_min     = toml::find_or(toml_data,
                                           "radiation",
                                           "emission",
                                           "compton",
                                           "photon_energy_min",
                                           defaults::compton::energy_min);
        compton_photon_weight  = toml::find_or(toml_data,
                                              "radiation",
                                              "emission",
                                              "compton",
                                              "photon_weight",
                                              ONE);
        compton_photon_species = toml::find<spidx_t>(toml_data,
                                                     "radiation",
                                                     "emission",
                                                     "compton",
                                                     "photon_species");
      }
    }

    void Extra::setParams(const std::map<std::string, bool>& extra,
                          SimulationParams*                  params) const {
      if (extra.at("synchrotron_drag")) {
        params->set("radiation.drag.synchrotron.gamma_rad", synchrotron_gamma_rad);
      }

      if (extra.at("compton_drag")) {
        params->set("radiation.drag.compton.gamma_rad", compton_gamma_rad);
      }

      if (extra.at("synchrotron_emission")) {
        params->set("radiation.emission.synchrotron.gamma_qed",
                    synchrotron_gamma_qed);
        params->set("radiation.emission.synchrotron.photon_energy_min",
                    synchrotron_energy_min);
        params->set("radiation.emission.synchrotron.photon_weight",
                    synchrotron_photon_weight);
        params->set("radiation.emission.synchrotron.photon_species",
                    synchrotron_photon_species);
      }

      if (extra.at("compton_emission")) {
        params->set("radiation.emission.compton.gamma_qed", compton_gamma_qed);
        params->set("radiation.emission.compton.photon_energy_min",
                    compton_energy_min);
        params->set("radiation.emission.compton.photon_weight",
                    compton_photon_weight);
        params->set("radiation.emission.compton.photon_species",
                    compton_photon_species);
      }
    }
  } // namespace params
} // namespace ntt
