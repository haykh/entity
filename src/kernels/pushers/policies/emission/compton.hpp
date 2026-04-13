#ifndef KERNELS_PUSHERS_EMISSION_COMPTON_HPP
#define KERNELS_PUSHERS_EMISSION_COMPTON_HPP

#include "enums.h"
#include "global.h"

#include "metrics/traits.h"

#include "framework/parameters/parameters.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Pair.hpp>

#include <vector>

namespace pusher::emission {
  using namespace ntt;

  template <class M>
    requires metric::traits::HasD<M> && metric::traits::HasCoordType<M> &&
             metric::traits::HasPrtlDim<M>
  struct Compton {
    struct Payload {
      real_t photon_energy;
    };

    const spidx_t emitted_index;
    const real_t  species_mass;
    const real_t  emitted_photon_weight;
    const real_t  emitted_photon_min_energy;

    const real_t nominal_probability;
    const real_t nominal_photon_energy;

    const bool should_drag;

    array_t<int*>      photon_i1, photon_i2, photon_i3;
    array_t<prtldx_t*> photon_dx1, photon_dx2, photon_dx3;
    array_t<real_t*>   photon_ux1, photon_ux2, photon_ux3;
    array_t<real_t*>   photon_phi;
    array_t<real_t*>   photon_weight;
    array_t<short*>    photon_tag;
    array_t<npart_t**> photon_pld_i;

    array_t<npart_t> photon_idx { "idx" };
    const npart_t    photon_offset, photon_cntr, domain_idx;
    const bool       photon_use_tracking;

    random_number_pool_t random_pool;

    Compton(Particles<M::Dim, M::CoordType>& photon_species,
            spidx_t                          emitted_index,
            float                            sp_mass,
            float                            sp_charge,
            RadiativeDragFlags               radiative_drag_flags,
            npart_t                          domain_idx,
            const prm::Parameters&           params,
            random_number_pool_t&            random_pool)
      : emitted_index { emitted_index }
      , species_mass { static_cast<real_t>(sp_mass) }
      , emitted_photon_weight { params.template get<real_t>(
          "radiation.emission.compton.photon_weight") }
      , emitted_photon_min_energy { params.template get<real_t>(
          "radiation.emission.compton.photon_energy_min") }
      , nominal_probability { math::abs(sp_charge / species_mass) *
                              params.template get<real_t>(
                                "radiation.emission.compton.nominal_"
                                "probability") }
      , nominal_photon_energy { species_mass * params.template get<real_t>(
                                                 "radiation.emission.compton."
                                                 "nominal_photon_energy") }
      , should_drag { static_cast<bool>(radiative_drag_flags &
                                        RadiativeDrag::COMPTON) }
      , photon_i1 { photon_species.i1 }
      , photon_i2 { photon_species.i2 }
      , photon_i3 { photon_species.i3 }
      , photon_dx1 { photon_species.dx1 }
      , photon_dx2 { photon_species.dx2 }
      , photon_dx3 { photon_species.dx3 }
      , photon_ux1 { photon_species.ux1 }
      , photon_ux2 { photon_species.ux2 }
      , photon_ux3 { photon_species.ux3 }
      , photon_phi { photon_species.phi }
      , photon_weight { photon_species.weight }
      , photon_tag { photon_species.tag }
      , photon_pld_i { photon_species.pld_i }
      , photon_offset { photon_species.npart() }
      , photon_cntr { photon_species.counter() }
      , domain_idx { domain_idx }
      , photon_use_tracking { photon_species.use_tracking() }
      , random_pool { random_pool } {}

    auto emitted_species_indices() const -> std::vector<spidx_t> {
      return { emitted_index };
    }

    auto numbers_injected() const -> std::vector<npart_t> {
      auto photon_idx_h = Kokkos::create_mirror_view(photon_idx);
      Kokkos::deep_copy(photon_idx_h, photon_idx);
      return { photon_idx_h() };
    }

    /**
     *
     * @brief Determine whether a photon is emitted, the drag is applied,
     *      and compute its energy and the recoil on the emitting particle
     *
     * @param x_Cd Position of the particle (code)
     * @param x_Ph Position of the particle (physical)
     * @param u_Ph Velocity of the particle (physical)
     * @param ep Interpolated electric field at the particle position (physical)
     * @param bp Interpolated magnetic field at the particle position (physical)
     * @param delta_u_Ph Output parameter for the recoil on the emitting particle (physical units)
     *
     * @note
     *
     *   probability at each timestep is:
     *      nominal_probability = omegaB0 * etarec * dt * (gamma_QED / gamma_rad)^2 / photon_weight
     *      p_gamma = (q / q0) / (m / m0) * beta * nominal_probability
     *
     *   mean energy of the emitted photon [units of m0 c^2]:
     *      nominal_photon_energy = (1 / gamma_QED)^2
     *      e_gamma = (gamma)^2 * (m / m0) * nominal_photon_energy
     *
     *   drag acceleration * dt:
     *      du = - photon_weight * p_gamma * e_gamma * u_hat / (m / m0)
     *
     *  @returns Pair of booleans to indicate whether a particle should be emitted
     *      and whether the emitting particle should experience a recoil (i.e. radiative drag)
     *
     */
    Inline auto shouldEmit(const coord_t<M::PrtlDim>&,
                           const coord_t<M::PrtlDim>&,
                           const vec_t<Dim::_3D>& u_Ph,
                           const vec_t<Dim::_3D>&,
                           const vec_t<Dim::_3D>&,
                           vec_t<Dim::_3D>& delta_u_Ph,
                           Payload& payload) const -> Kokkos::pair<bool, bool> {
      const auto u_sqr     = NORM_SQR(u_Ph[0], u_Ph[1], u_Ph[2]);
      const auto gamma_sqr = ONE + u_sqr;

      payload.photon_energy = gamma_sqr * nominal_photon_energy;

      const auto delta_u = -emitted_photon_weight * payload.photon_energy /
                           (math::sqrt(u_sqr) * species_mass);

      delta_u_Ph[0] = delta_u * u_Ph[0];
      delta_u_Ph[1] = delta_u * u_Ph[1];
      delta_u_Ph[2] = delta_u * u_Ph[2];

      auto       rand_gen    = random_pool.get_state();
      // should not emit if photon energy is above 20% of (gamma - 1) m c^2
      const auto should_emit = (Random<real_t>(rand_gen) <
                                (nominal_probability *
                                 math::sqrt(u_sqr / gamma_sqr))) and
                               (payload.photon_energy <
                                species_mass * (math::sqrt(gamma_sqr) - ONE) *
                                  static_cast<real_t>(0.2));
      random_pool.free_state(rand_gen);

      return Kokkos::make_pair(
        should_emit and (payload.photon_energy >= emitted_photon_min_energy),
        should_drag and should_emit);
    }

    Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                     const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                     const vec_t<Dim::_3D>&           direction,
                     real_t                           weight,
                     real_t                           phi,
                     const Payload&                   payload) const {
      const auto index = Kokkos::atomic_fetch_add(&photon_idx(), 1);
      if (not photon_use_tracking) {
        kernel::InjectParticle<M::Dim, M::CoordType, false>(
          photon_offset + index,
          photon_i1,
          photon_i2,
          photon_i3,
          photon_dx1,
          photon_dx2,
          photon_dx3,
          photon_ux1,
          photon_ux2,
          photon_ux3,
          photon_phi,
          photon_weight,
          photon_tag,
          photon_pld_i,
          xi_Cd,
          dxi_Cd,
          { direction[0] * payload.photon_energy,
            direction[1] * payload.photon_energy,
            direction[2] * payload.photon_energy },
          emitted_photon_weight * weight,
          phi);
      } else {
        kernel::InjectParticle<M::Dim, M::CoordType, true>(
          photon_offset + index,
          photon_i1,
          photon_i2,
          photon_i3,
          photon_dx1,
          photon_dx2,
          photon_dx3,
          photon_ux1,
          photon_ux2,
          photon_ux3,
          photon_phi,
          photon_weight,
          photon_tag,
          photon_pld_i,
          xi_Cd,
          dxi_Cd,
          { direction[0] * payload.photon_energy,
            direction[1] * payload.photon_energy,
            direction[2] * payload.photon_energy },
          emitted_photon_weight * weight,
          phi,
          domain_idx,
          photon_cntr + index);
      }
    }
  };

} // namespace pusher::emission

#endif // KERNELS_PUSHERS_EMISSION_COMPTON_HPP
