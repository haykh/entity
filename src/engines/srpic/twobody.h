#ifndef ENGINES_SRPIC_TWOBODY_H
#define ENGINES_SRPIC_TWOBODY_H

#include "enums.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "archetypes/qed/compton.h"
#include "framework/domain/domain.h"
#include "framework/parameters/extra.h"
#include "framework/parameters/parameters.h"
#include "kernels/twobody_interactions.hpp"

namespace ntt {
  namespace srpic {

    template <CartesianMetricClass M>
    void TwoBodyInteractions(Domain<SimEngine::SRPIC, M>& domain,
                             const prm::Parameters&       engine_params,
                             const SimulationParams&      params) {
      logger::Checkpoint("Launching TwoBodyInteractions routines", HERE);
      const auto dt   = engine_params.get<real_t>("dt");
      const auto step = engine_params.get<timestep_t>("step");
      for (const auto& interaction :
           params.template get<std::vector<::ntt::params::TwoBodyInteractionParams>>(
             "two_body.interaction")) {
        if (step % interaction.interval == 0u) {
          const auto thomson_optical_depth = params.template get<real_t>(
            "two_body.thomson_optical_depth");
          const auto nominal_thomson_probability_density = thomson_optical_depth *
                                                           dt *
                                                           static_cast<real_t>(
                                                             interaction.interval);
          if (interaction.type == TwoBodyInteraction::COMPTON) {
            prm::Parameters compton_params;
            compton_params.set("compton_scattering.nominal_probability_density",
                               nominal_thomson_probability_density);
            auto launch = [&]<bool R1, bool R2>() {
              auto policy = arch::qed::ComptonScattering<M::Dim, R1, R2>(
                compton_params,
                domain.random_pool());

              std::vector<Particles<M::Dim, M::CoordType>*> group1_species;
              std::vector<Particles<M::Dim, M::CoordType>*> group2_species;

              for (const auto& sp_lepton : interaction.group1) {
                raise::ErrorIf(
                  domain.species[sp_lepton - 1].mass() == ZERO,
                  fmt::format(
                    "Species %u is massless but is in the lepton group "
                    "of a Compton interaction",
                    sp_lepton),
                  HERE);
                group1_species.push_back(&domain.species[sp_lepton - 1]);
              }
              for (const auto& sp_photon : interaction.group2) {
                raise::ErrorIf(
                  domain.species[sp_photon - 1].mass() != ZERO,
                  fmt::format(
                    "Species %u is massive but is in the photon group "
                    "of a Compton interaction",
                    sp_photon),
                  HERE);
                group2_species.push_back(&domain.species[sp_photon - 1]);
              }

              kernel::mink::TwoBodyInteraction<M::Dim>(
                group1_species,
                group2_species,
                domain.mesh.n_active(),
                domain.mesh.extent(),
                interaction.tile_size,
                params.template get<real_t>("particles.ppc0"),
                domain.random_pool(),
                policy);
            };
            if (interaction.recoil1 and interaction.recoil2) {
              launch.template operator()<true, true>();
            } else if (interaction.recoil1 and not interaction.recoil2) {
              launch.template operator()<true, false>();
            } else if (not interaction.recoil1 and interaction.recoil2) {
              launch.template operator()<false, true>();
            } else {
              launch.template operator()<false, false>();
            }
          } else if (interaction.type == TwoBodyInteraction::CUSTOM) {
            raise::Error("Custom two-body interactions not implemented yet", HERE);
          }
        }
      }
    }
  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_TWOBODY_H
