/**
 * @file framework/containers/particles.h
 * @brief Definition of the particle container class
 * @implements
 *   - ntt::Particles<> : ntt::ParticleSpecies
 * @cpp:
 *   - particles.cpp
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef FRAMEWORK_CONTAINERS_PARTICLES_H
#define FRAMEWORK_CONTAINERS_PARTICLES_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "framework/containers/species.h"

#include <Kokkos_Core.hpp>

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
#endif

#include <string>
#include <vector>

namespace ntt {

  /**
   * @brief Container class to carry particle information for a specific species
   * @tparam D The dimension of the simulation
   * @tparam S The simulation engine being used
   */
  template <Dimension D, Coord::type C>
  struct Particles : public ParticleSpecies {
  private:
    // Number of currently active (used) particles
    npart_t m_npart { 0 };
    npart_t m_counter { 0 };
    bool    m_is_sorted { false };

#if !defined(MPI_ENABLED)
    const std::size_t m_ntags { 2 };
#else // MPI_ENABLED
    const std::size_t m_ntags { (std::size_t)(2 + math::pow(3, (int)D) - 1) };
#endif

  public:
    // Cell indices of the current particle
    array_t<int*>      i1, i2, i3;
    // Displacement of a particle within the cell
    array_t<prtldx_t*> dx1, dx2, dx3;
    // Three spatial components of the covariant 4-velocity (physical units)
    array_t<real_t*>   ux1, ux2, ux3;
    // Particle weights.
    array_t<real_t*>   weight;
    // Previous timestep coordinates
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    // Array to tag the particles
    array_t<short*>    tag;
    // Array to store real-valued payloads
    array_t<real_t**>  pld_r;
    // Array to store integer-valued payloads
    array_t<npart_t**> pld_i;
    // phi coordinate (for axisymmetry)
    array_t<real_t*>   phi;

    // for empty allocation
    Particles() {}

    /**
     * @brief Constructor for the particle container
     * @param index The index of the species (starts from 1)
     * @param label The label for the species
     * @param m The mass of the species
     * @param ch The charge of the species
     * @param maxnpart The maximum number of allocated particles for the species
     * @param particle_pusher_flags The pusher(s) assigned for the species
     * @param use_tracking Use particle tracking for the species
     * @param radiative_drag_flags The radiative drag mechanism(s) assigned for the species
     * @param emission_policy_flag The emission policy assigned for the species
     * @param npld_r The number of real-valued payloads for the species
     * @param npld_i The number of integer-valued payloads for the species
     */
    Particles(spidx_t             index,
              const std::string&  label,
              float               m,
              float               ch,
              npart_t             maxnpart,
              ParticlePusherFlags particle_pusher_flags,
              bool                use_tracking,
              RadiativeDragFlags  radiative_drag_flags,
              EmissionTypeFlag    emission_policy_flag,
              unsigned short      npld_r,
              unsigned short      npld_i);

    /**
     * @brief Constructor for the particle container
     * @overload
     * @param spec The particle species container
     */
    Particles(const ParticleSpecies& spec)
      : Particles(spec.index(),
                  spec.label(),
                  spec.mass(),
                  spec.charge(),
                  spec.maxnpart(),
                  spec.pusher(),
                  spec.use_tracking(),
                  spec.radiative_drag_flags(),
                  spec.emission_policy_flag(),
                  spec.npld_r(),
                  spec.npld_i()) {}

    Particles(const Particles&)            = delete;
    Particles& operator=(const Particles&) = delete;

    ~Particles() = default;

    /**
     * @brief Loop over all active particles
     * @returns A 1D Kokkos range policy of size of `npart`
     */
    inline auto rangeActiveParticles() const -> range_t<Dim::_1D> {
      return CreateParticleRangePolicy(0u, npart());
    }

    /**
     * @brief Loop over all particles
     * @returns A 1D Kokkos range policy of size of `npart`
     */
    inline auto rangeAllParticles() const -> range_t<Dim::_1D> {
      return CreateParticleRangePolicy(0u, maxnpart());
    }

    /* getters -------------------------------------------------------------- */
    /**
     * @brief Get the number of active particles
     */
    [[nodiscard]]
    auto npart() const -> npart_t {
      return m_npart;
    }

    /**
     * @brief Get the particle counter
     */
    [[nodiscard]]
    auto counter() const -> npart_t {
      return m_counter;
    }

    /**
     * @brief Check if particles are sorted by tag
     */
    [[nodiscard]]
    auto is_sorted() const -> bool {
      return m_is_sorted;
    }

    /**
     * @brief Get the number of distinct tags possible
     */
    [[nodiscard]]
    auto ntags() const -> std::size_t {
      return m_ntags;
    }

    [[nodiscard]]
    auto memory_footprint() const -> std::size_t {
      std::size_t footprint  = 0;
      footprint             += sizeof(int) * i1.extent(0);
      footprint             += sizeof(int) * i2.extent(0);
      footprint             += sizeof(int) * i3.extent(0);
      footprint             += sizeof(prtldx_t) * dx1.extent(0);
      footprint             += sizeof(prtldx_t) * dx2.extent(0);
      footprint             += sizeof(prtldx_t) * dx3.extent(0);
      footprint             += sizeof(real_t) * ux1.extent(0);
      footprint             += sizeof(real_t) * ux2.extent(0);
      footprint             += sizeof(real_t) * ux3.extent(0);
      footprint             += sizeof(real_t) * weight.extent(0);
      footprint             += sizeof(int) * i1_prev.extent(0);
      footprint             += sizeof(int) * i2_prev.extent(0);
      footprint             += sizeof(int) * i3_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx1_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx2_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx3_prev.extent(0);
      footprint             += sizeof(short) * tag.extent(0);
      footprint += sizeof(real_t) * pld_r.extent(0) * pld_r.extent(1);
      footprint += sizeof(npart_t) * pld_i.extent(0) * pld_i.extent(1);
      footprint += sizeof(real_t) * phi.extent(0);
      return footprint;
    }

    /**
     * @brief Count the number of particles with a specific tag.
     * @return The vector of counts for each tag + offsets
     * @note For instance, given the counts: 0 -> n0, 1 -> n1, 2 -> n2, 3 -> n3,
     * ... it returns:
     * ... [n0, n1, n2, n3, ...] of size ntags
     * ... [n2, n2 + n3, n2 + n3 + n4, ...]  of size ntags - 3
     * ... so in buffer array:
     * ... tag=2 particles are offset by 0
     * ... tag=3 particles are offset by n2
     * ... tag=4 particles are offset by n2 + n3
     * ... etc.
     */
    auto NpartsPerTagAndOffsets() const
      -> std::pair<std::vector<npart_t>, array_t<npart_t*>>;

    /* setters -------------------------------------------------------------- */
    /**
     * @brief Set the number of particles
     * @param npart The number of particles as a npart_t
     */
    void set_npart(npart_t n) {
      raise::ErrorIf(
        n > maxnpart(),
        fmt::format(
          "Trying to set npart to %d which is larger than maxnpart %d",
          n,
          maxnpart()),
        HERE);
      m_npart = n;
    }

    /**
     * @brief Set the particle counter
     * @param n The counter value as a npart_t
     */
    void set_counter(npart_t n) {
      m_counter = n;
    }

    void set_unsorted() {
      m_is_sorted = false;
    }

    /**
     * @brief Move dead particles to the end of arrays
     */
    void RemoveDead();

    /**
     * @brief Copy particle data from device to host.
     */
    void SyncHostDevice();

#if defined(MPI_ENABLED)
    /**
     * @brief Communicate particles across neighboring meshblocks
     * @param dirs_to_comm The directions requiring communication
     * @param shifts_in_x1 The coordinate shifts in x1 direction per each communicated particle
     * @param shifts_in_x2 The coordinate shifts in x2 direction per each communicated particle
     * @param shifts_in_x3 The coordinate shifts in x3 direction per each communicated particle
     * @param send_ranks The map of ranks per each send direction
     * @param recv_ranks The map of ranks per each recv direction
     */
    void Communicate(const dir::dirs_t<D>&,
                     const array_t<int*>&,
                     const array_t<int*>&,
                     const array_t<int*>&,
                     const dir::map_t<D, int>&,
                     const dir::map_t<D, int>&);
#endif

#if defined(OUTPUT_ENABLED)
    void OutputDeclare(adios2::IO&) const;

    template <SimEngine::type S, class M>
    void OutputWrite(adios2::IO&,
                     adios2::Engine&,
                     npart_t,
                     std::size_t,
                     std::size_t,
                     const M&);

    void CheckpointDeclare(adios2::IO&) const;
    void CheckpointRead(adios2::IO&, adios2::Engine&, std::size_t, std::size_t);
    void CheckpointWrite(adios2::IO&, adios2::Engine&, std::size_t, std::size_t) const;
#endif
  };

} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_PARTICLES_H
