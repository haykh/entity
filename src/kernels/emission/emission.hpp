#ifndef KERNELS_EMISSION_EMISSION_HPP
#define KERNELS_EMISSION_EMISSION_HPP

#include "enums.h"
#include "global.h"

#include "metrics/traits.h"

#include "framework/parameters/parameters.h"

#include "kernels/injectors.hpp"

namespace kernel {
  using namespace ntt;

  template <class M, EmissionTypeFlag E>
  struct EmissionPolicy;

  template <class M>
  struct EmissionPolicy<M, EmissionType::NONE> {};

  template <class M>
  struct EmissionPolicy<M, EmissionType::STRONGFIELDPP> {};

  template <class M>
  struct EmissionPolicy<M, EmissionType::SYNCHROTRON> {};

  template <class M>
    requires metric::traits::HasD<M> && metric::traits::HasCoordType<M> &&
             metric::traits::HasPrtlDim<M>
  struct EmissionPolicy<M, EmissionType::COMPTON> {
    const real_t nominal_probability;
    const real_t inv_gamma_qed_sqr;

    array_t<int*>      photon_i1, photon_i2, photon_i3;
    array_t<prtldx_t*> photon_dx1, photon_dx2, photon_dx3;
    array_t<real_t*>   photon_ux1, photon_ux2, photon_ux3;
    array_t<real_t*>   photon_phi;
    array_t<real_t*>   photon_weight;
    array_t<short*>    photon_tag;
    array_t<npart_t**> photon_pld_i;

    real_t           emitted_photon_weight;
    array_t<npart_t> photon_idx { "idx" };
    const npart_t    photon_offset, photon_cntr, domain_idx;
    const bool       photon_use_tracking;

    random_number_pool_t random_pool;

    EmissionPolicy(Particles<M::Dim, M::CoordType>& photon_species,
                   float                            species_mass,
                   real_t                           dt,
                   npart_t                          domain_idx,
                   const SimulationParams&          params,
                   random_number_pool_t&            random_pool)
      : nominal_probability { params.get<real_t>("scales.omegaB0") *
                              static_cast<real_t>(0.1) *
                              static_cast<real_t>(species_mass) * dt *
                              SQR(params.get<real_t>(
                                    "radiation.emission.compton.gamma_qed") /
                                  params.get<real_t>(
                                    "radiation.drag.compton.gamma_rad")) /
                              params.get<real_t>(
                                "radiation.emission.compton.photon_weight") }
      , inv_gamma_qed_sqr { ONE / SQR(params.get<real_t>(
                                    "radiation.emission.compton.gamma_qed")) }
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
      , emitted_photon_weight { params.get<real_t>(
          "radiation.emission.compton.photon_weight") }
      , photon_offset { photon_species.npart() }
      , photon_cntr { photon_species.counter() }
      , domain_idx { domain_idx }
      , photon_use_tracking { photon_species.use_tracking() }
      , random_pool { random_pool } {}

    auto number_injected() const -> npart_t {
      auto photon_idx_h = Kokkos::create_mirror_view(photon_idx);
      Kokkos::deep_copy(photon_idx_h, photon_idx);
      return photon_idx_h();
    }

    /*
     * @brief Determine whether a photon is emitted and compute its energy and
     * the recoil on the emitting particle
     *
     * @note
     *
     *   probability at each timestep is:
     *      p_gamma = omegaB * etarec * (m / m_0) * dt * (gamma_QED /
     * gamma_rad)^2 * beta^2 / photon_weight
     *
     *   mean energy of the emitted photon [units of m0 c^2]:
     *      e_gamma = (gamma / gamma_QED)^2
     *
     *   drag force [in units of m c]:
     *      du / dt = - photon_weight * p_gamma * e_gamma * beta_hat
     *
     *  @returns Boolean to indicate whether a particle should be emitted
     */
    Inline auto shouldEmit(const vec_t<Dim::_3D>& u_Ph,
                           real_t&                photon_energy,
                           vec_t<Dim::_3D>&       delta_u_Ph) const -> bool {
      const auto gamma_sqr   = U2GAMMA_SQR(u_Ph[0], u_Ph[1], u_Ph[2]);
      const auto inv_gamma   = ONE / math::sqrt(gamma_sqr);
      const auto beta_sqr    = NORM_SQR(u_Ph[0], u_Ph[1], u_Ph[2]) / gamma_sqr;
      const auto probability = nominal_probability * beta_sqr;
      photon_energy          = gamma_sqr * inv_gamma_qed_sqr;
      delta_u_Ph[0] = -probability * photon_energy * (u_Ph[0] * inv_gamma);
      delta_u_Ph[1] = -probability * photon_energy * (u_Ph[1] * inv_gamma);
      delta_u_Ph[2] = -probability * photon_energy * (u_Ph[2] * inv_gamma);
      auto       rand_gen    = random_pool.get_state();
      const auto should_emit = Random<real_t>(rand_gen) < probability;
      random_pool.free_state(rand_gen);
      return should_emit;
    }

    Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                     const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                     const vec_t<Dim::_3D>&           direction,
                     real_t                           photon_energy,
                     real_t                           weight,
                     real_t                           phi = ZERO) const {
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
          { direction[0] * photon_energy,
            direction[1] * photon_energy,
            direction[2] * photon_energy },
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
          { direction[0] * photon_energy,
            direction[1] * photon_energy,
            direction[2] * photon_energy },
          emitted_photon_weight * weight,
          phi,
          domain_idx,
          photon_cntr + index);
      }
    }
  };

} // namespace kernel

#endif
