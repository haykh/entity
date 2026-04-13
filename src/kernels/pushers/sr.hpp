/**
 * @file kernels/pushers/sr.h
 * @brief Particle pusher for the SR
 * @implements
 *   - kernel::sr::Pusher_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * @note
 * At the end of the boundary condition call, if MPI is enabled particles
 * are additionally tagged depending on which direction they are leaving
 * @note
 * BasePusher_kernel contains all the common functions, while this class
 * implements the main operator() plus extra routines (e.g., drag, emission)
 */

#ifndef KERNELS_PARTICLE_PUSHER_SR_HPP
#define KERNELS_PARTICLE_PUSHER_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include "kernels/pushers/base_sr.hpp"
#include "kernels/pushers/policies_sr.h"
#include "kernels/pushers/traits.h"

namespace pusher::sr {
  using namespace ntt;

  /**
   * @tparam M Metric
   * @tparam P Policy struct containing the emission, custom particle update and external fields policies
   */
  template <class M, kernel::pusher::ValidPolicy<M> P>
    requires metric::traits::HasD<M> && metric::traits::HasTransformXYZ<M> &&
             metric::traits::HasConvertXYZ<M> &&
             metric::traits::HasTransform_i<M> && metric::traits::HasConvert_i<M>
  struct Kernel : public BasePusher_kernel<M> {
    using BasePusher_kernel<M>::D;
    using EmissionPolicy             = typename P::EmissionPolicy;
    using CustomParticleUpdatePolicy = typename P::CustomParticleUpdatePolicy;
    using ExternalFieldsPolicy       = typename P::ExternalFieldsPolicy;
    static constexpr auto Atm        = P::ApplyAtmosphere;

    static constexpr auto HasExtForce =
      ::kernel::traits::pusher::external::HasExternalF<ExternalFieldsPolicy, D>;
    static constexpr auto HasExtEfield =
      ::kernel::traits::pusher::external::HasExternalE<ExternalFieldsPolicy, D>;
    static constexpr auto HasExtBfield =
      ::kernel::traits::pusher::external::HasExternalB<ExternalFieldsPolicy, D>;
    static constexpr auto HasEmission =
      ::kernel::traits::pusher::emission::IsValid<EmissionPolicy, M>;

    static constexpr auto HasCustomPrtlUpdate =
      not ::kernel::traits::pusher::custom_particle_update::HasNoPolicy<CustomParticleUpdatePolicy>;

    static_assert(
      HasExtForce or HasExtEfield or HasExtBfield or
        ::kernel::traits::pusher::external::HasNoPolicy<ExternalFieldsPolicy>,
      "Invalid emission policy E for Pusher_kernel");
    static_assert(HasEmission or
                    ::kernel::traits::pusher::emission::HasNoPolicy<EmissionPolicy>,
                  "Invalid emission policy E for Pusher_kernel");

    using BasePusher_kernel<M>::particles;
    using BasePusher_kernel<M>::metric;

    using BasePusher_kernel<M>::params;
    using BasePusher_kernel<M>::gca_params;
    using BasePusher_kernel<M>::atmosphere_params;

    EmissionPolicy             emission_policy;
    CustomParticleUpdatePolicy custom_particle_update_policy;
    ExternalFieldsPolicy       external_fields_policy;

    Pusher_kernel(const PusherParams&            pusher_params,
                  PusherArrays&                  pusher_arrays,
                  const randacc_ndfield_t<D, 6>& EB,
                  const M&                       metric,
                  const P&                       policies)
      : BasePusher_kernel<M> { pusher_params, pusher_arrays, EB, metric }
      , external_fields_policy { policies.external_fields_policy }
      , emission_policy { policies.emission_policy }
      , custom_particle_update_policy { policies.custom_particle_update_policy } {}

    Inline void operator()(index_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        if (particles.tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      this->getParticlePosition(p, xp_Cd);
      if (params.pusher_flags == ParticlePusher::PHOTON) {
        /**
         * Procedure for massless particles
         */
        if constexpr (HasEmission) {
          // get Cartesian position
          coord_t<M::PrtlDim> xp_Ph { ZERO };
          if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                        M::PrtlDim == Dim::_3D) {
            xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
          }
          if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
            xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
          }
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
          }

          // get Cartesian velocity
          vec_t<Dim::_3D> u_prime { ZERO };
          u_prime[0] = particles.ux1(p);
          u_prime[1] = particles.ux2(p);
          u_prime[2] = particles.ux3(p);

          // get Cartesian fields
          vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
          vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
          this->template getInterpolatedEMFields<SHAPE_ORDER>(p, ei, bi);

          metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
          metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);

          processEmission(p, u_prime, xp_Cd, xp_Ph, ei_Cart, bi_Cart);
        }
        // update the position
        this->positionPush(false, p, xp_Cd);

        // call custom particle position update
        if constexpr (HasCustomPrtlUpdate) {
          custom_prtl_update(p, xp_Cd, *this);
        }

        // apply boundary conditions
        this->boundaryConditions(p, xp_Cd);
      } else {
        /**
         * Procedure for massive particles
         */
        // update cartesian velocity
        vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
        vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
        vec_t<Dim::_3D> external_force_Cart { ZERO };
        vec_t<Dim::_3D> u_prime { ZERO };
        vec_t<Dim::_3D> ei_Cart_rad { ZERO }, bi_Cart_rad { ZERO };
        bool            is_gca { false };

        // field interpolation 0th-11th order
        this->template getInterpolatedEMFields<SHAPE_ORDER>(p, ei, bi);

        metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
        metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);

        coord_t<M::PrtlDim> xp_Ph { ZERO };

        if constexpr ((HasExtForce or Atm) or (HasExtEfield or HasExtBfield) or
                      HasEmission) {
          if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                        M::PrtlDim == Dim::_3D) {
            xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
          }
          if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
            xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
          }
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
          }
        }

        // add the user-provided external fields
        if constexpr (HasExtEfield) {
          vec_t<Dim::_3D> ext_e_Ph { ZERO };
          vec_t<Dim::_3D> ext_e_Cart { ZERO };
          if constexpr (
            ::kernel::traits::pusher::external::HasEx1<ExternalFieldsPolicy, D>) {
            ext_e_Ph[0] = external_fields_policy.ex1(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasEx2<ExternalFieldsPolicy, D>) {
            ext_e_Ph[1] = external_fields_policy.ex2(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasEx3<ExternalFieldsPolicy, D>) {
            ext_e_Ph[2] = external_fields_policy.ex3(xp_Ph);
          }
          metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd, ext_e_Ph, ext_e_Cart);
          ei_Cart[0] += ext_e_Cart[0];
          ei_Cart[1] += ext_e_Cart[1];
          ei_Cart[2] += ext_e_Cart[2];
        }

        if constexpr (HasExtBfield) {
          vec_t<Dim::_3D> ext_b_Ph { ZERO };
          vec_t<Dim::_3D> ext_b_Cart { ZERO };
          if constexpr (
            ::kernel::traits::pusher::external::HasBx1<ExternalFieldsPolicy, D>) {
            ext_b_Ph[0] = external_fields_policy.bx1(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasBx2<ExternalFieldsPolicy, D>) {
            ext_b_Ph[1] = external_fields_policy.bx2(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasBx3<ExternalFieldsPolicy, D>) {
            ext_b_Ph[2] = external_fields_policy.bx3(xp_Ph);
          }
          metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd, ext_b_Ph, ext_b_Cart);
          bi_Cart[0] += ext_b_Cart[0];
          bi_Cart[1] += ext_b_Cart[1];
          bi_Cart[2] += ext_b_Cart[2];
        }

        // backup fields & velocities to use later in radiative drag
        if ((params.radiative_drag_flags != RadiativeDrag::NONE) or HasEmission) {
          ei_Cart_rad[0] = ei_Cart[0];
          ei_Cart_rad[1] = ei_Cart[1];
          ei_Cart_rad[2] = ei_Cart[2];
          bi_Cart_rad[0] = bi_Cart[0];
          bi_Cart_rad[1] = bi_Cart[1];
          bi_Cart_rad[2] = bi_Cart[2];
          u_prime[0]     = particles.ux1(p);
          u_prime[1]     = particles.ux2(p);
          u_prime[2]     = particles.ux3(p);
        }

        // compute the external force either user-provided or from the atmosphere model
        if constexpr (HasExtForce or Atm) {
          real_t f_x1 = ZERO, f_x2 = ZERO, f_x3 = ZERO;
          if constexpr (
            ::kernel::traits::pusher::external::HasFx1<ExternalFieldsPolicy, D>) {
            f_x1 = external_fields_policy.fx1(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasFx2<ExternalFieldsPolicy, D>) {
            f_x2 = external_fields_policy.fx2(xp_Ph);
          }
          if constexpr (
            ::kernel::traits::pusher::external::HasFx3<ExternalFieldsPolicy, D>) {
            f_x3 = external_fields_policy.fx3(xp_Ph);
          }
          if constexpr (Atm) {
            if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
              if (not cmp::AlmostZero(atmosphere_params.gx1) and
                  ((atmosphere_params.ds < ZERO or
                    xp_Ph[0] <= atmosphere_params.x_surf + atmosphere_params.ds) and
                   (atmosphere_params.ds > ZERO or
                    xp_Ph[0] >= atmosphere_params.x_surf + atmosphere_params.ds))) {
                if constexpr (M::CoordType == Coord::Cart) {
                  f_x1 += atmosphere_params.gx1;
                } else {
                  f_x1 += atmosphere_params.gx1 *
                          SQR(atmosphere_params.x_surf / xp_Ph[0]);
                }
              }
            }
            if constexpr (D == Dim::_2D or D == Dim::_3D) {
              if (not cmp::AlmostZero(atmosphere_params.gx2) and
                  ((atmosphere_params.ds < ZERO or
                    xp_Ph[1] <= atmosphere_params.x_surf + atmosphere_params.ds) and
                   (atmosphere_params.ds > ZERO or
                    xp_Ph[1] >= atmosphere_params.x_surf + atmosphere_params.ds))) {
                if constexpr (M::CoordType == Coord::Cart) {
                  f_x2 += atmosphere_params.gx2;
                } else {
                  raise::KernelError(HERE, "Invalid force for coordinate system");
                }
              }
            }
            if constexpr (D == Dim::_3D) {
              if (not cmp::AlmostZero(atmosphere_params.gx3) and
                  ((atmosphere_params.ds < ZERO or
                    xp_Ph[2] <= atmosphere_params.x_surf + atmosphere_params.ds) and
                   (atmosphere_params.ds > ZERO or
                    xp_Ph[2] >= atmosphere_params.x_surf + atmosphere_params.ds))) {
                if constexpr (M::CoordType == Coord::Cart) {
                  f_x3 += atmosphere_params.gx3;
                } else {
                  raise::KernelError(HERE, "Invalid force for coordinate system");
                }
              }
            }
          }
          metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd,
                                                          { f_x1, f_x2, f_x3 },
                                                          external_force_Cart);
        }
        if (params.pusher_flags & ParticlePusher::GCA) {
          /* hybrid GCA/conventional mode --------------------------------- */
          const auto E2 { NORM_SQR(ei_Cart[0], ei_Cart[1], ei_Cart[2]) };
          const auto B2 { NORM_SQR(bi_Cart[0], bi_Cart[1], bi_Cart[2]) };
          const auto rL { math::sqrt(ONE + NORM_SQR(particles.ux1(p),
                                                    particles.ux2(p),
                                                    particles.ux3(p))) *
                          params.dt /
                          (TWO * math::abs(this->coeff) * math::sqrt(B2)) };
          if ((B2 > ZERO) and (rL < gca_params.larmor) and
              ((E2 / B2) < gca_params.EovrB_sqr)) {
            is_gca = true;
            // update with GCA
            if constexpr (HasExtForce or Atm) {
              this->velocityEMPush_GCA_ExtForce(p, external_force_Cart, ei_Cart, bi_Cart);
            } else {
              this->velocityEMPush_GCA(p, ei_Cart, bi_Cart);
            }
          } else {
            // update with conventional pusher
            if constexpr (HasExtForce or Atm) {
              particles.ux1(p) += HALF * params.dt * external_force_Cart[0];
              particles.ux2(p) += HALF * params.dt * external_force_Cart[1];
              particles.ux3(p) += HALF * params.dt * external_force_Cart[2];
            }
            if (params.pusher_flags & ParticlePusher::BORIS) {
              this->velocityEMPush_Boris(p, ei_Cart, bi_Cart);
            } else if (params.pusher_flags & ParticlePusher::VAY) {
              this->velocityEMPush_Vay(p, ei_Cart, bi_Cart);
            } else {
              raise::KernelError(HERE, "Invalid pusher algorithm for GCA mode");
            }
            if constexpr (HasExtForce or Atm) {
              particles.ux1(p) += HALF * params.dt * external_force_Cart[0];
              particles.ux2(p) += HALF * params.dt * external_force_Cart[1];
              particles.ux3(p) += HALF * params.dt * external_force_Cart[2];
            }
          }
        } else {
          /* conventional pusher mode ------------------------------------- */
          // update with conventional pusher
          if constexpr (HasExtForce or Atm) {
            particles.ux1(p) += HALF * params.dt * external_force_Cart[0];
            particles.ux2(p) += HALF * params.dt * external_force_Cart[1];
            particles.ux3(p) += HALF * params.dt * external_force_Cart[2];
          }
          if (params.pusher_flags & ParticlePusher::BORIS) {
            this->velocityEMPush_Boris(p, ei_Cart, bi_Cart);
          } else if (params.pusher_flags & ParticlePusher::VAY) {
            this->velocityEMPush_Vay(p, ei_Cart, bi_Cart);
          } else {
            raise::KernelError(HERE, "Invalid pusher algorithm for GCA mode");
          }
          if constexpr (HasExtForce or Atm) {
            particles.ux1(p) += HALF * params.dt * external_force_Cart[0];
            particles.ux2(p) += HALF * params.dt * external_force_Cart[1];
            particles.ux3(p) += HALF * params.dt * external_force_Cart[2];
          }
        }
        // radiative drag
        if constexpr (not HasEmission) {
          if ((not is_gca) and
              (params.radiative_drag_flags != RadiativeDrag::NONE)) {
            u_prime[0] = HALF * (u_prime[0] + particles.ux1(p));
            u_prime[1] = HALF * (u_prime[1] + particles.ux2(p));
            u_prime[2] = HALF * (u_prime[2] + particles.ux3(p));
            if (params.radiative_drag_flags & RadiativeDrag::SYNCHROTRON) {
              this->synchrotronDrag(p, u_prime, ei_Cart_rad, bi_Cart_rad);
            }
            if (params.radiative_drag_flags & RadiativeDrag::COMPTON) {
              this->inverseComptonDrag(p, u_prime);
            }
          }
        } else {
          u_prime[0] = HALF * (u_prime[0] + particles.ux1(p));
          u_prime[1] = HALF * (u_prime[1] + particles.ux2(p));
          u_prime[2] = HALF * (u_prime[2] + particles.ux3(p));
          processEmission(p, u_prime, xp_Cd, xp_Ph, ei_Cart_rad, bi_Cart_rad);
        }
        // update position
        this->positionPush(true, p, xp_Cd);

        // call custom particle position update
        if constexpr (HasCustomPrtlUpdate) {
          custom_prtl_update(p, xp_Cd, *this);
        }

        // apply boundary conditions
        this->boundaryConditions(p, xp_Cd);
      }
    }

    // Extra
    Inline void processEmission(index_t                    p,
                                vec_t<Dim::_3D>&           u_prime,
                                const coord_t<M::PrtlDim>& xp_Cd,
                                const coord_t<M::PrtlDim>& xp_Ph,
                                const vec_t<Dim::_3D>&     ep_Cart,
                                const vec_t<Dim::_3D>&     bp_Cart) const
      requires ::kernel::traits::pusher::emission::IsValid<EmissionPolicy, M>
    {
      typename EmissionPolicy::Payload payload;
      vec_t<Dim::_3D>                  delta_u_Ph { ZERO };
      const auto emission_response = emission_policy.shouldEmit(xp_Cd,
                                                                xp_Ph,
                                                                u_prime,
                                                                ep_Cart,
                                                                bp_Cart,
                                                                delta_u_Ph,
                                                                payload);

      if (emission_response.second) {
        particles.ux1(p) += delta_u_Ph[0];
        particles.ux2(p) += delta_u_Ph[1];
        particles.ux3(p) += delta_u_Ph[2];
      }

      if (emission_response.first) {
        vec_t<Dim::_3D> direction { ZERO };
        const auto      delta_u_Ph_mag = NORM(delta_u_Ph[0],
                                         delta_u_Ph[1],
                                         delta_u_Ph[2]);
        direction[0]                   = -delta_u_Ph[0] / delta_u_Ph_mag;
        direction[1]                   = -delta_u_Ph[1] / delta_u_Ph_mag;
        direction[2]                   = -delta_u_Ph[2] / delta_u_Ph_mag;
        tuple_t<int, M::Dim>      xi_Cd { 0 };
        tuple_t<prtldx_t, M::Dim> dxi_Cd { static_cast<prtldx_t>(0) };
        real_t                    prtl_phi = ZERO;
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          xi_Cd[0]  = particles.i1(p);
          dxi_Cd[0] = particles.dx1(p);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          xi_Cd[1]  = particles.i2(p);
          dxi_Cd[1] = particles.dx2(p);
          if constexpr (M::CoordType != Coord::Cart) {
            prtl_phi = particles.phi(p);
          }
        }
        if constexpr (M::Dim == Dim::_3D) {
          xi_Cd[2]  = particles.i3(p);
          dxi_Cd[2] = particles.dx3(p);
        }
        emission_policy
          .emit(xi_Cd, dxi_Cd, direction, particles.weight(p), prtl_phi, payload);
      }
    }
  };

} // namespace pusher::sr

#endif // KERNELS_PARTICLE_PUSHER_SR_HPP
