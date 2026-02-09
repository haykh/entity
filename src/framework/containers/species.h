/**
 * @file framework/containers/species.h
 * @brief Container for particle species information
 * @implements
 *   - ntt::ParticleSpecies
 * @namespaces:
 *   - ntt::
 * @note Particles class inherits from this one
 */

#ifndef FRAMEWORK_CONTAINERS_SPECIES_H
#define FRAMEWORK_CONTAINERS_SPECIES_H

#include "enums.h"

#include "utils/formatting.h"
#include "utils/reporter.h"

#include <string>

namespace ntt {

  class ParticleSpecies {
  protected:
    // Species index
    const spidx_t     m_index;
    // Species label
    const std::string m_label;
    // Species mass in units of m0
    const float       m_mass;
    // Species charge in units of q0
    const float       m_charge;
    // Max number of allocated particles for the species
    npart_t           m_maxnpart;

    // Pusher assigned for the species
    const ParticlePusherFlags m_particle_pusher_flags;

    // Use particle tracking for the species
    const bool m_use_tracking;

    // Radiative drag mechanism(s) assigned for the species
    const RadiativeDragFlags m_radiative_drag_flags;

    // Emission policy assigned for the species
    const EmissionTypeFlag m_emission_policy_flag;

    // Number of payloads for the species
    const unsigned short m_npld_r;
    const unsigned short m_npld_i;

  public:
    ParticleSpecies()
      : m_index { 0u }
      , m_label { "" }
      , m_mass { 0.0 }
      , m_charge { 0.0 }
      , m_maxnpart { 0 }
      , m_particle_pusher_flags { ParticlePusher::NONE }
      , m_use_tracking { false }
      , m_radiative_drag_flags { RadiativeDrag::NONE }
      , m_emission_policy_flag { EmissionType::NONE }
      , m_npld_r { 0 }
      , m_npld_i { 0 } {}

    /**
     * @brief Constructor for the particle species container.
     *
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     * @param particle_pusher_flags The pusher(s) assigned for the species.
     * @param use_tracking Use particle tracking for the species.
     * @param radiative_drag_flags The radiative drag mechanism(s) assigned for the species.
     * @param emission_policy_flag The emission policy assigned for the species.
     * @param npld_r The number of real-valued payloads for the species
     * @param npld_i The number of integer-valued payloads for the species
     */
    ParticleSpecies(spidx_t             index,
                    const std::string&  label,
                    float               m,
                    float               ch,
                    npart_t             maxnpart,
                    ParticlePusherFlags particle_pusher_flags,
                    bool                use_tracking,
                    RadiativeDragFlags  radiative_drag_flags,
                    EmissionTypeFlag    emission_policy_flag,
                    unsigned short      npld_r,
                    unsigned short      npld_i)
      : m_index { index }
      , m_label { std::move(label) }
      , m_mass { m }
      , m_charge { ch }
      , m_maxnpart { maxnpart }
      , m_particle_pusher_flags { particle_pusher_flags }
      , m_use_tracking { use_tracking }
      , m_radiative_drag_flags { radiative_drag_flags }
      , m_emission_policy_flag { emission_policy_flag }
      , m_npld_r { npld_r }
      , m_npld_i { npld_i } {
      if (use_tracking) {
#if !defined(MPI_ENABLED)
        raise::ErrorIf(m_npld_i < 1,
                       "npld_i must be at least 1 when tracking is enabled",
                       HERE);
#else
        raise::ErrorIf(
          m_npld_i < 2,
          "npld_i must be at least 2 when tracking is enabled with MPI",
          HERE);
#endif
      }
    }

    ParticleSpecies(const ParticleSpecies&) = default;

    /**
     * @brief Destructor for the particle species container.
     */
    ~ParticleSpecies() = default;

    [[nodiscard]]
    auto index() const -> spidx_t {
      return m_index;
    }

    [[nodiscard]]
    auto label() const -> std::string {
      return m_label;
    }

    [[nodiscard]]
    auto mass() const -> float {
      return m_mass;
    }

    [[nodiscard]]
    auto charge() const -> float {
      return m_charge;
    }

    [[nodiscard]]
    auto maxnpart() const -> npart_t {
      return m_maxnpart;
    }

    [[nodiscard]]
    auto pusher() const -> ParticlePusherFlags {
      return m_particle_pusher_flags;
    }

    [[nodiscard]]
    auto use_tracking() const -> bool {
      return m_use_tracking;
    }

    [[nodiscard]]
    auto radiative_drag_flags() const -> RadiativeDragFlags {
      return m_radiative_drag_flags;
    }

    [[nodiscard]]
    auto emission_policy_flag() const -> EmissionTypeFlag {
      return m_emission_policy_flag;
    }

    [[nodiscard]]
    auto npld_r() const -> unsigned short {
      return m_npld_r;
    }

    [[nodiscard]]
    auto npld_i() const -> unsigned short {
      return m_npld_i;
    }

    /* reporter -------------------------------------------------------------- */
    auto Report() const -> std::string {
      std::string report = "";
      reporter::AddSubcategory(report,
                               4,
                               fmt::format("Species #%d", index()).c_str());
      reporter::AddParam(report, 6, "Label", "%s", label().c_str());
      reporter::AddParam(report, 6, "Mass", "%.1f", mass());
      reporter::AddParam(report, 6, "Charge", "%.1f", charge());
      reporter::AddParam(report, 6, "Max #", "%d [per domain]", maxnpart());
      reporter::AddParam(report,
                         6,
                         "Pusher",
                         "%s",
                         ParticlePusher::to_string(pusher()).c_str());
      reporter::AddParam(report,
                         6,
                         "Radiative drag",
                         "%s",
                         RadiativeDrag::to_string(radiative_drag_flags()).c_str());
      reporter::AddParam(report, 6, "# of real-value payloads", "%d", npld_r());
      reporter::AddParam(report, 6, "# of integer-value payloads", "%d", npld_i());
      return report;
    }
  };
} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_SPECIES_H
