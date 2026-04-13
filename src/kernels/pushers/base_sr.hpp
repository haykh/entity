/**
 * @file kernels/pushers/base_sr.h
 * @brief Base classes for the particle pusher in SR
 * @implements
 *   - pusher::sr::Context
 *   - pusher::sr::ContextArrays
 *   - pusher::sr::GCAContext
 *   - pusher::sr::RadiativeDragContext
 *   - pusher::sr::AtmosphereContext
 *   - pusher::sr::BasePusher_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * @macros:
 *   - MPI_ENABLED
 */
#ifndef KERNELS_PUSHERS_BASE_SR_HPP
#define KERNELS_PUSHERS_BASE_SR_HPP

#include "enums.h"
#include "global.h"

#include "utils/numeric.h"
#include "utils/param_container.h"

#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/emission/compton.hpp"
#include "kernels/pushers/emission/synchrotron.hpp"
#include "kernels/pushers/policies_sr.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    I = static_cast<int>((XI + 1)) - 1;                                        \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

/* -------------------------------------------------------------------------- */

namespace pusher::sr {
  using namespace ntt;

  struct Context {
    // species index
    spidx_t             species_index;
    // pusher algorithm(s) assigned to the species
    ParticlePusherFlags pusher_flags { ParticlePusher::NONE };
    // radiative drag force(s) enabled for the species
    RadiativeDragFlags  radiative_drag_flags { RadiativeDrag::NONE };

    // species parameters
    float mass, charge;

    // time variable
    simtime_t time;

    // global constants
    real_t dt, omegaB0;

    // grid parameters
    int                  ni1, ni2, ni3;
    boundaries_t<PrtlBC> boundaries;

    // parameters for the advanced features
    prm::Parameters gca_params;
    prm::Parameters radiative_drag_params;
    prm::Parameters atmosphere_params;
  };

  struct ContextArrays {
    spidx_t            sp;
    array_t<int*>      i1, i2, i3;
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi;
    array_t<real_t*>   weight;
    array_t<short*>    tag;
  };

  struct GCAContext {
    const real_t larmor;
    const real_t EovrB_sqr;

    GCAContext(const Context& ctx)
      : larmor { (ctx.pusher_flags & ParticlePusher::GCA)
                   ? ctx.gca_params.get<real_t>("larmor_max")
                   : ZERO }
      , EovrB_sqr { (ctx.pusher_flags & ParticlePusher::GCA)
                      ? SQR(ctx.gca_params.get<real_t>("e_ovr_b_max"))
                      : ZERO } {}
  };

  struct RadiativeDragContext {
    const real_t synchrotron_coeff;
    const real_t compton_coeff;

    RadiativeDragContext(const Context& ctx)
      : synchrotron_coeff { (ctx.radiative_drag_flags & RadiativeDrag::SYNCHROTRON)
                              ? (static_cast<real_t>(0.1) * ctx.dt * ctx.omegaB0 /
                                 (SQR(ctx.radiative_drag_params.get<real_t>(
                                    "synchrotron_gamma_rad")) *
                                  ctx.mass))
                              : ZERO }
      , compton_coeff {
        (ctx.radiative_drag_flags & RadiativeDrag::COMPTON)
          ? (static_cast<real_t>(0.1) * ctx.dt * ctx.omegaB0 /
             (SQR(ctx.radiative_drag_params.get<real_t>("compton_gamma_rad")) *
              ctx.mass))
          : ZERO
      } {}
  };

  struct AtmosphereContext {
    const real_t gx1, gx2, gx3;
    const real_t x_surf, ds;

    AtmosphereContext(const Context& ctx)
      : gx1 { ctx.atmosphere_params.get<real_t>("gx1") }
      , gx2 { ctx.atmosphere_params.get<real_t>("gx2") }
      , gx3 { ctx.atmosphere_params.get<real_t>("gx3") }
      , x_surf { ctx.atmosphere_params.get<real_t>("x_surf") }
      , ds { ctx.atmosphere_params.get<real_t>("ds") } {}
  };

  struct BoundariesContext {
    bool is_absorb_i1min { false }, is_absorb_i1max { false };
    bool is_absorb_i2min { false }, is_absorb_i2max { false };
    bool is_absorb_i3min { false }, is_absorb_i3max { false };
    bool is_periodic_i1min { false }, is_periodic_i1max { false };
    bool is_periodic_i2min { false }, is_periodic_i2max { false };
    bool is_periodic_i3min { false }, is_periodic_i3max { false };
    bool is_reflect_i1min { false }, is_reflect_i1max { false };
    bool is_reflect_i2min { false }, is_reflect_i2max { false };
    bool is_reflect_i3min { false }, is_reflect_i3max { false };
    bool is_axis_i2min { false }, is_axis_i2max { false };
  };

  template <class M>
  struct BaseKernel {
    static constexpr auto D = M::Dim;

    const M                       metric;
    const randacc_ndfield_t<D, 6> EB;

    ContextArrays particles;

    const Context              ctx;
    const GCAContext           gca_ctx;
    const RadiativeDragContext radiative_drag_ctx;
    const AtmosphereContext    atmosphere_ctx;
    BoundariesContext          boundaries;

    const real_t coeff;

    BaseKernel(const Context&                      ctx,
               ContextArrays&                      pusher_arrays,
               const randacc_ndfield_t<M::Dim, 6>& EB,
               const M&                            metric)
      : metric { metric }
      , EB { EB }
      , particles { pusher_arrays }
      , ctx { ctx }
      , gca_ctx { ctx }
      , radiative_drag_ctx { ctx }
      , atmosphere_ctx { ctx }
      , coeff { HALF * (ctx.charge / ctx.mass) * ctx.omegaB0 * ctx.dt } {
      raise::ErrorIf(ctx.pusher_flags == ParticlePusher::NONE,
                     "No particle pusher specified",
                     HERE);
      raise::ErrorIf(ctx.boundaries.size() < 1,
                     "ctx.boundaries defined incorrectly",
                     HERE);
      boundaries.is_absorb_i1min = (ctx.boundaries[0].first == PrtlBC::ATMOSPHERE) ||
                                   (ctx.boundaries[0].first == PrtlBC::ABSORB);
      boundaries.is_absorb_i1max = (ctx.boundaries[0].second == PrtlBC::ATMOSPHERE) ||
                                   (ctx.boundaries[0].second == PrtlBC::ABSORB);
      boundaries.is_periodic_i1min = (ctx.boundaries[0].first == PrtlBC::PERIODIC);
      boundaries.is_periodic_i1max = (ctx.boundaries[0].second == PrtlBC::PERIODIC);
      boundaries.is_reflect_i1min = (ctx.boundaries[0].first == PrtlBC::REFLECT);
      boundaries.is_reflect_i1max = (ctx.boundaries[0].second == PrtlBC::REFLECT);
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(ctx.boundaries.size() < 2,
                       "ctx.boundaries defined incorrectly",
                       HERE);
        boundaries.is_absorb_i2min = (ctx.boundaries[1].first ==
                                      PrtlBC::ATMOSPHERE) ||
                                     (ctx.boundaries[1].first == PrtlBC::ABSORB);
        boundaries.is_absorb_i2max = (ctx.boundaries[1].second ==
                                      PrtlBC::ATMOSPHERE) ||
                                     (ctx.boundaries[1].second == PrtlBC::ABSORB);
        boundaries.is_periodic_i2min = (ctx.boundaries[1].first == PrtlBC::PERIODIC);
        boundaries.is_periodic_i2max = (ctx.boundaries[1].second ==
                                        PrtlBC::PERIODIC);
        boundaries.is_reflect_i2min = (ctx.boundaries[1].first == PrtlBC::REFLECT);
        boundaries.is_reflect_i2max = (ctx.boundaries[1].second == PrtlBC::REFLECT);
        boundaries.is_axis_i2min = (ctx.boundaries[1].first == PrtlBC::AXIS);
        boundaries.is_axis_i2max = (ctx.boundaries[1].second == PrtlBC::AXIS);
      }
      if constexpr (D == Dim::_3D) {
        raise::ErrorIf(ctx.boundaries.size() < 3,
                       "ctx.boundaries defined incorrectly",
                       HERE);
        boundaries.is_absorb_i3min = (ctx.boundaries[2].first ==
                                      PrtlBC::ATMOSPHERE) ||
                                     (ctx.boundaries[2].first == PrtlBC::ABSORB);
        boundaries.is_absorb_i3max = (ctx.boundaries[2].second ==
                                      PrtlBC::ATMOSPHERE) ||
                                     (ctx.boundaries[2].second == PrtlBC::ABSORB);
        boundaries.is_periodic_i3min = (ctx.boundaries[2].first == PrtlBC::PERIODIC);
        boundaries.is_periodic_i3max = (ctx.boundaries[2].second ==
                                        PrtlBC::PERIODIC);
        boundaries.is_reflect_i3min = (ctx.boundaries[2].first == PrtlBC::REFLECT);
        boundaries.is_reflect_i3max = (ctx.boundaries[2].second == PrtlBC::REFLECT);
      }
    }

    Inline void getParticlePosition(index_t p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xp[0] = i_di_to_Xi(particles.i1(p), particles.dx1(p));
      }
      if constexpr (D == Dim::_2D) {
        xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp[2] = particles.phi(p);
        }
      }
      if constexpr (D == Dim::_3D) {
        xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));
        xp[2] = i_di_to_Xi(particles.i3(p), particles.dx3(p));
      }
    }

    Inline void positionPush(bool massive, index_t p, coord_t<M::PrtlDim>& xp) const {
      // get cartesian velocity
      if constexpr (M::CoordType == Coord::Cart) {
        // i+di push for Cartesian basis
        const real_t dt_inv_energy {
          massive
            ? (ctx.dt / math::sqrt(ONE + SQR(particles.ux1(p)) +
                                   SQR(particles.ux2(p)) + SQR(particles.ux3(p))))
            : (ctx.dt / math::sqrt(SQR(particles.ux1(p)) + SQR(particles.ux2(p)) +
                                   SQR(particles.ux3(p))))
        };
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          particles.i1_prev(p)  = particles.i1(p);
          particles.dx1_prev(p) = particles.dx1(p);
          particles.dx1(p) += metric.template transform<1, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux1(p)) *
                              dt_inv_energy;
          particles.i1(p) += static_cast<int>(particles.dx1(p) >= ONE) -
                             static_cast<int>(particles.dx1(p) < ZERO);
          particles.dx1(p) -= (particles.dx1(p) >= ONE);
          particles.dx1(p) += (particles.dx1(p) < ZERO);
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          particles.i2_prev(p)  = particles.i2(p);
          particles.dx2_prev(p) = particles.dx2(p);
          particles.dx2(p) += metric.template transform<2, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux2(p)) *
                              dt_inv_energy;
          particles.i2(p) += static_cast<int>(particles.dx2(p) >= ONE) -
                             static_cast<int>(particles.dx2(p) < ZERO);
          particles.dx2(p) -= (particles.dx2(p) >= ONE);
          particles.dx2(p) += (particles.dx2(p) < ZERO);
        }
        if constexpr (D == Dim::_3D) {
          particles.i3_prev(p)  = particles.i3(p);
          particles.dx3_prev(p) = particles.dx3(p);
          particles.dx3(p) += metric.template transform<3, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux3(p)) *
                              dt_inv_energy;
          particles.i3(p) += static_cast<int>(particles.dx3(p) >= ONE) -
                             static_cast<int>(particles.dx3(p) < ZERO);
          particles.dx3(p) -= (particles.dx3(p) >= ONE);
          particles.dx3(p) += (particles.dx3(p) < ZERO);
        }
      } else {
        // full Cartesian coordinate push in non-Cartesian basis
        const real_t inv_energy {
          massive
            ? ONE / math::sqrt(ONE + SQR(particles.ux1(p)) +
                               SQR(particles.ux2(p)) + SQR(particles.ux3(p)))
            : ONE / math::sqrt(SQR(particles.ux1(p)) + SQR(particles.ux2(p)) +
                               SQR(particles.ux3(p)))
        };
        vec_t<Dim::_3D>     vp_Cart { particles.ux1(p) * inv_energy,
                                  particles.ux2(p) * inv_energy,
                                  particles.ux3(p) * inv_energy };
        // get cartesian position
        coord_t<M::PrtlDim> xp_Cart { ZERO };
        metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xp, xp_Cart);
        // update cartesian position
        for (auto d = 0u; d < M::PrtlDim; ++d) {
          xp_Cart[d] += vp_Cart[d] * ctx.dt;
        }
        // transform back to code
        metric.template convert_xyz<Crd::XYZ, Crd::Cd>(xp_Cart, xp);

        // update x1
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          particles.i1_prev(p)  = particles.i1(p);
          particles.dx1_prev(p) = particles.dx1(p);
          from_Xi_to_i_di(xp[0], particles.i1(p), particles.dx1(p));
        }

        // update x2 & phi
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          particles.i2_prev(p)  = particles.i2(p);
          particles.dx2_prev(p) = particles.dx2(p);
          from_Xi_to_i_di(xp[1], particles.i2(p), particles.dx2(p));
          if constexpr (D == Dim::_2D && M::PrtlDim == Dim::_3D) {
            particles.phi(p) = xp[2];
          }
        }

        // update x3
        if constexpr (D == Dim::_3D) {
          particles.i3_prev(p)  = particles.i3(p);
          particles.dx3_prev(p) = particles.dx3(p);
          from_Xi_to_i_di(xp[2], particles.i3(p), particles.dx3(p));
        }
      }
    }

    /**
     * @brief update particle velocities
     * @param p, e0, b0 index & interpolated fields
     */
    Inline void velocityEMPush_Boris(index_t          p,
                                     vec_t<Dim::_3D>& e0,
                                     vec_t<Dim::_3D>& b0) const {
      real_t COEFF { coeff };

      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;
      vec_t<Dim::_3D> u0 { particles.ux1(p) + e0[0],
                           particles.ux2(p) + e0[1],
                           particles.ux3(p) + e0[2] };

      COEFF *= ONE / math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
      b0[0] *= COEFF;
      b0[1] *= COEFF;
      b0[2] *= COEFF;
      COEFF  = TWO / (ONE + NORM_SQR(b0[0], b0[1], b0[2]));

      vec_t<Dim::_3D> u1 {
        (u0[0] + CROSS_x1(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
        (u0[1] + CROSS_x2(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
        (u0[2] + CROSS_x3(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF
      };

      u0[0] += CROSS_x1(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[0];
      u0[1] += CROSS_x2(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[1];
      u0[2] += CROSS_x3(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[2];

      particles.ux1(p) = u0[0];
      particles.ux2(p) = u0[1];
      particles.ux3(p) = u0[2];
    }

    Inline void velocityEMPush_Vay(index_t          p,
                                   vec_t<Dim::_3D>& e0,
                                   vec_t<Dim::_3D>& b0) const {
      auto COEFF { coeff };
      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;

      b0[0] *= COEFF;
      b0[1] *= COEFF;
      b0[2] *= COEFF;

      COEFF = ONE / math::sqrt(ONE + NORM_SQR(particles.ux1(p),
                                              particles.ux2(p),
                                              particles.ux3(p)));

      vec_t<Dim::_3D> u1 { (particles.ux1(p) + TWO * e0[0] +
                            CROSS_x1(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF),
                           (particles.ux2(p) + TWO * e0[1] +
                            CROSS_x2(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF),
                           (particles.ux3(p) + TWO * e0[2] +
                            CROSS_x3(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF) };
      COEFF = DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]);
      auto COEFF2 { ONE + NORM_SQR(u1[0], u1[1], u1[2]) -
                    NORM_SQR(b0[0], b0[1], b0[2]) };

      COEFF = ONE /
              math::sqrt(
                INV_2 * (COEFF2 + math::sqrt(SQR(COEFF2) +
                                             FOUR * (SQR(b0[0]) + SQR(b0[1]) +
                                                     SQR(b0[2]) + SQR(COEFF)))));
      COEFF2 = ONE / (ONE + SQR(b0[0] * COEFF) + SQR(b0[1] * COEFF) +
                      SQR(b0[2] * COEFF));

      particles.ux1(p) = COEFF2 *
                         (u1[0] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[0] * COEFF) +
                          u1[1] * b0[2] * COEFF - u1[2] * b0[1] * COEFF);
      particles.ux2(p) = COEFF2 *
                         (u1[1] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[1] * COEFF) +
                          u1[2] * b0[0] * COEFF - u1[0] * b0[2] * COEFF);
      particles.ux3(p) = COEFF2 *
                         (u1[2] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[2] * COEFF) +
                          u1[0] * b0[1] * COEFF - u1[1] * b0[0] * COEFF);
    }

    Inline void velocityEMPush_GCA(index_t          p,
                                   vec_t<Dim::_3D>& e0,
                                   vec_t<Dim::_3D>& b0) const {
      const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                          NORM_SQR(b0[0], b0[1], b0[2]) };

      const vec_t<Dim::_3D> wE {
        CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
      };

      {
        const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
        b0[0] *= b_norm_inv;
        b0[1] *= b_norm_inv;
        b0[2] *= b_norm_inv;
      }
      auto upar {
        DOT(particles.ux1(p), particles.ux2(p), particles.ux3(p), b0[0], b0[1], b0[2]) +
        coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2])
      };

      real_t factor;
      {
        const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
        if (wE_sqr < static_cast<real_t>(0.01)) {
          factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
        } else {
          factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
        }
      }
      const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
      const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                         math::sqrt(
                           ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
      particles.ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      particles.ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      particles.ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    /**
     * @brief velocity push with external force & EM fields in GCA mode
     */
    Inline void velocityEMPush_GCA_ExtForce(index_t          p,
                                            vec_t<Dim::_3D>& f0,
                                            vec_t<Dim::_3D>& e0,
                                            vec_t<Dim::_3D>& b0) const {
      const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                          NORM_SQR(b0[0], b0[1], b0[2]) };

      const vec_t<Dim::_3D> wE {
        CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
      };

      {
        const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
        b0[0] *= b_norm_inv;
        b0[1] *= b_norm_inv;
        b0[2] *= b_norm_inv;
      }
      auto upar {
        DOT(particles.ux1(p), particles.ux2(p), particles.ux3(p), b0[0], b0[1], b0[2]) +
        coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) +
        ctx.dt * DOT(f0[0], f0[1], f0[2], b0[0], b0[1], b0[2])
      };

      real_t factor;
      {
        const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
        if (wE_sqr < static_cast<real_t>(0.01)) {
          factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
        } else {
          factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
        }
      }
      const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
      const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                         math::sqrt(
                           ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
      particles.ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      particles.ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      particles.ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    // Getters

    template <unsigned short O>
    Inline void getInterpolatedEMFields(index_t          p,
                                        vec_t<Dim::_3D>& e0,
                                        vec_t<Dim::_3D>& b0) const {

      // Zig-zag interpolation
      if constexpr (O == 0u) {

        if constexpr (D == Dim::_1D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);

          // first order
          real_t c0, c1;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;

          // Ex1
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::ex1);
          c1    = EB(i + indx, em::ex1);
          e0[0] = c0 * pondmx + c1 * pondpx;
          // Ex2
          // Interpolate --- (primal)
          c0    = EB(i, em::ex2);
          c1    = EB(i + 1, em::ex2);
          e0[1] = c0 * ponpmx + c1 * ponppx;
          // Ex3
          // Interpolate --- (primal)
          c0    = EB(i, em::ex3);
          c1    = EB(i + 1, em::ex3);
          e0[2] = c0 * ponpmx + c1 * ponppx;
          // Bx1
          // Interpolate --- (primal)
          c0    = EB(i, em::bx1);
          c1    = EB(i + 1, em::bx1);
          b0[0] = c0 * ponpmx + c1 * ponppx;
          // Bx2
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::bx2);
          c1    = EB(i + indx, em::bx2);
          b0[1] = c0 * pondmx + c1 * pondpx;
          // Bx3
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::bx3);
          c1    = EB(i + indx, em::bx3);
          b0[2] = c0 * pondmx + c1 * pondpx;
        } else if constexpr (D == Dim::_2D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);
          int indy = static_cast<int>(dx2_ + HALF);

          // first order
          real_t c000, c100, c010, c110, c00, c10;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;
          real_t ponpmy = ONE - dx2_;
          real_t ponppy = dx2_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;
          real_t pondmy = static_cast<real_t>(indy + ONE) - (dx2_ + HALF);
          real_t pondpy = ONE - pondmy;

          // Ex1
          // Interpolate --- (dual, primal)
          c000  = EB(i - 1 + indx, j, em::ex1);
          c100  = EB(i + indx, j, em::ex1);
          c010  = EB(i - 1 + indx, j + 1, em::ex1);
          c110  = EB(i + indx, j + 1, em::ex1);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          e0[0] = c00 * ponpmy + c10 * ponppy;
          // Ex2
          // Interpolate -- (primal, dual)
          c000  = EB(i, j - 1 + indy, em::ex2);
          c100  = EB(i + 1, j - 1 + indy, em::ex2);
          c010  = EB(i, j + indy, em::ex2);
          c110  = EB(i + 1, j + indy, em::ex2);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          e0[1] = c00 * pondmy + c10 * pondpy;
          // Ex3
          // Interpolate -- (primal, primal)
          c000  = EB(i, j, em::ex3);
          c100  = EB(i + 1, j, em::ex3);
          c010  = EB(i, j + 1, em::ex3);
          c110  = EB(i + 1, j + 1, em::ex3);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          e0[2] = c00 * ponpmy + c10 * ponppy;

          // Bx1
          // Interpolate -- (primal, dual)
          c000  = EB(i, j - 1 + indy, em::bx1);
          c100  = EB(i + 1, j - 1 + indy, em::bx1);
          c010  = EB(i, j + indy, em::bx1);
          c110  = EB(i + 1, j + indy, em::bx1);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          b0[0] = c00 * pondmy + c10 * pondpy;
          // Bx2
          // Interpolate -- (dual, primal)
          c000  = EB(i - 1 + indx, j, em::bx2);
          c100  = EB(i + indx, j, em::bx2);
          c010  = EB(i - 1 + indx, j + 1, em::bx2);
          c110  = EB(i + indx, j + 1, em::bx2);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          b0[1] = c00 * ponpmy + c10 * ponppy;
          // Bx3
          // Interpolate -- (dual, dual)
          c000  = EB(i - 1 + indx, j - 1 + indy, em::bx3);
          c100  = EB(i + indx, j - 1 + indy, em::bx3);
          c010  = EB(i - 1 + indx, j + indy, em::bx3);
          c110  = EB(i + indx, j + indy, em::bx3);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          b0[2] = c00 * pondmy + c10 * pondpy;
        } else if constexpr (D == Dim::_3D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const int  k { particles.i3(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };
          const auto dx3_ { static_cast<real_t>(particles.dx3(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);
          int indy = static_cast<int>(dx2_ + HALF);
          int indz = static_cast<int>(dx3_ + HALF);

          // first order
          real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01,
            c11, c0, c1;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;
          real_t ponpmy = ONE - dx2_;
          real_t ponppy = dx2_;
          real_t ponpmz = ONE - dx3_;
          real_t ponppz = dx3_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;
          real_t pondmy = static_cast<real_t>(indy + ONE) - (dx2_ + HALF);
          real_t pondpy = ONE - pondmy;
          real_t pondmz = static_cast<real_t>(indz + ONE) - (dx3_ + HALF);
          real_t pondpz = ONE - pondmz;

          // Ex1
          // Interpolate --- (dual, primal, primal)
          c000  = EB(i - 1 + indx, j, k, em::ex1);
          c100  = EB(i + indx, j, k, em::ex1);
          c010  = EB(i - 1 + indx, j + 1, k, em::ex1);
          c110  = EB(i + indx, j + 1, k, em::ex1);
          c001  = EB(i - 1 + indx, j, k + 1, em::ex1);
          c101  = EB(i + indx, j, k + 1, em::ex1);
          c011  = EB(i - 1 + indx, j + 1, k + 1, em::ex1);
          c111  = EB(i + indx, j + 1, k + 1, em::ex1);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          e0[0] = c0 * ponpmz + c1 * ponppz;
          // Ex2
          // Interpolate -- (primal, dual, primal)
          c000  = EB(i, j - 1 + indy, k, em::ex2);
          c100  = EB(i + 1, j - 1 + indy, k, em::ex2);
          c010  = EB(i, j + indy, k, em::ex2);
          c110  = EB(i + 1, j + indy, k, em::ex2);
          c001  = EB(i, j - 1 + indy, k + 1, em::ex2);
          c101  = EB(i + 1, j - 1 + indy, k + 1, em::ex2);
          c011  = EB(i, j + indy, k + 1, em::ex2);
          c111  = EB(i + 1, j + indy, k + 1, em::ex2);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * pondmy + c10 * pondpy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * pondmy + c11 * pondpy;
          e0[1] = c0 * ponpmz + c1 * ponppz;
          // Ex3
          // Interpolate -- (primal, primal, dual)
          c000  = EB(i, j, k - 1 + indz, em::ex3);
          c100  = EB(i + 1, j, k - 1 + indz, em::ex3);
          c010  = EB(i, j + 1, k - 1 + indz, em::ex3);
          c110  = EB(i + 1, j + 1, k - 1 + indz, em::ex3);
          c001  = EB(i, j, k + indz, em::ex3);
          c101  = EB(i + 1, j, k + indz, em::ex3);
          c011  = EB(i, j + 1, k + indz, em::ex3);
          c111  = EB(i + 1, j + 1, k + indz, em::ex3);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * ponpmy + c11 * ponppy;
          e0[2] = c0 * pondmz + c1 * pondpz;

          // Bx1
          // Interpolate -- (primal, dual, dual)
          c000  = EB(i, j - 1 + indy, k - 1 + indz, em::bx1);
          c100  = EB(i + 1, j - 1 + indy, k - 1 + indz, em::bx1);
          c010  = EB(i, j + indy, k - 1 + indz, em::bx1);
          c110  = EB(i + 1, j + indy, k - 1 + indz, em::bx1);
          c001  = EB(i, j - 1 + indy, k + indz, em::bx1);
          c101  = EB(i + 1, j - 1 + indy, k + indz, em::bx1);
          c011  = EB(i, j + indy, k + indz, em::bx1);
          c111  = EB(i + 1, j + indy, k + indz, em::bx1);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * pondmy + c10 * pondpy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * pondmy + c11 * pondpy;
          b0[0] = c0 * pondmz + c1 * pondpz;
          // Bx2
          // Interpolate -- (dual, primal, dual)
          c000  = EB(i - 1 + indx, j, k - 1 + indz, em::bx2);
          c100  = EB(i + indx, j, k - 1 + indz, em::bx2);
          c010  = EB(i - 1 + indx, j + 1, k - 1 + indz, em::bx2);
          c110  = EB(i + indx, j + 1, k - 1 + indz, em::bx2);
          c001  = EB(i - 1 + indx, j, k + indz, em::bx2);
          c101  = EB(i + indx, j, k + indz, em::bx2);
          c011  = EB(i - 1 + indx, j + 1, k + indz, em::bx2);
          c111  = EB(i + indx, j + 1, k + indz, em::bx2);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          b0[1] = c0 * pondmz + c1 * pondpz;
          // Bx3
          // Interpolate -- (dual, dual, primal)
          c000  = EB(i - 1 + indx, j - 1 + indy, k, em::bx3);
          c100  = EB(i + indx, j - 1 + indy, k, em::bx3);
          c010  = EB(i - 1 + indx, j + indy, k, em::bx3);
          c110  = EB(i + indx, j + indy, k, em::bx3);
          c001  = EB(i - 1 + indx, j - 1 + indy, k + 1, em::bx3);
          c101  = EB(i + indx, j - 1 + indy, k + 1, em::bx3);
          c011  = EB(i - 1 + indx, j + indy, k + 1, em::bx3);
          c111  = EB(i + indx, j + indy, k + 1, em::bx3);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          b0[2] = c0 * ponpmz + c1 * ponppz;
        }
      } else if constexpr (O >= 1u) {

        if constexpr (D == Dim::_1D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          // primal and dual shape function
          real_t     Sp[O + 1], Sd[O + 1];
          // minimum contributing cells
          int        ip_min, id_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, Sp);

          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, Sd);

          // Ex1 -- dual
          e0[0] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[0] += Sd[idx1] * EB(id_min + idx1, em::ex1);
          }

          // Ex2 -- primal
          e0[1] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[1] += Sp[idx1] * EB(ip_min + idx1, em::ex2);
          }

          // Ex3 -- primal
          e0[2] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[2] += Sp[idx1] * EB(ip_min + idx1, em::ex3);
          }

          // Bx1 -- primal
          b0[0] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[0] += Sp[idx1] * EB(ip_min + idx1, em::bx1);
          }

          // Bx2 -- dual
          b0[1] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[1] += Sd[idx1] * EB(id_min + idx1, em::bx2);
          }

          // Bx3 -- dual
          b0[2] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[2] += Sd[idx1] * EB(id_min + idx1, em::bx3);
          }

        } else if constexpr (D == Dim::_2D) {

          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };

          // primal and dual shape function
          real_t S1p[O + 1], S1d[O + 1];
          real_t S2p[O + 1], S2d[O + 1];
          // minimum contributing cells
          int    ip_min, id_min;
          int    jp_min, jd_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, S1p);
          prtl_shape::order<false, O>(j, dx2_, jp_min, S2p);
          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, S1d);
          prtl_shape::order<true, O>(j, dx2_, jd_min, S2d);

          // Ex1 -- dual, primal
          e0[0] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::ex1);
            }
            e0[0] += c0 * S2p[idx2];
          }

          // Ex2 -- primal, dual
          e0[1] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::ex2);
            }
            e0[1] += c0 * S2d[idx2];
          }

          // Ex3 -- primal, primal
          e0[2] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jp_min + idx2, em::ex3);
            }
            e0[2] += c0 * S2p[idx2];
          }

          // Bx1 -- primal, dual
          b0[0] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::bx1);
            }
            b0[0] += c0 * S2d[idx2];
          }

          // Bx2 -- dual, primal
          b0[1] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::bx2);
            }
            b0[1] += c0 * S2p[idx2];
          }

          // Bx3 -- dual, dual
          b0[2] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jd_min + idx2, em::bx3);
            }
            b0[2] += c0 * S2d[idx2];
          }

        } else if constexpr (D == Dim::_3D) {

          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const int  k { particles.i3(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };
          const auto dx3_ { static_cast<real_t>(particles.dx3(p)) };

          // primal and dual shape function
          real_t S1p[O + 1], S1d[O + 1];
          real_t S2p[O + 1], S2d[O + 1];
          real_t S3p[O + 1], S3d[O + 1];

          // minimum contributing cells
          int ip_min, id_min;
          int jp_min, jd_min;
          int kp_min, kd_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, S1p);
          prtl_shape::order<false, O>(j, dx2_, jp_min, S2p);
          prtl_shape::order<false, O>(k, dx3_, kp_min, S3p);
          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, S1d);
          prtl_shape::order<true, O>(j, dx2_, jd_min, S2d);
          prtl_shape::order<true, O>(k, dx3_, kd_min, S3d);

          // Ex1 -- dual, primal, primal
          e0[0] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jp_min + idx2, kp_min + idx3, em::ex1);
              }
              c0 += c00 * S2p[idx2];
            }
            e0[0] += c0 * S3p[idx3];
          }

          // Ex2 -- primal, dual, primal
          e0[1] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jd_min + idx2, kp_min + idx3, em::ex2);
              }
              c0 += c00 * S2d[idx2];
            }
            e0[1] += c0 * S3p[idx3];
          }

          // Ex3 -- primal, primal, dual
          e0[2] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jp_min + idx2, kd_min + idx3, em::ex3);
              }
              c0 += c00 * S2p[idx2];
            }
            e0[2] += c0 * S3d[idx3];
          }

          // Bx1 -- primal, dual, dual
          b0[0] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jd_min + idx2, kd_min + idx3, em::bx1);
              }
              c0 += c00 * S2d[idx2];
            }
            b0[0] += c0 * S3d[idx3];
          }

          // Bx2 -- dual, primal, dual
          b0[1] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jp_min + idx2, kd_min + idx3, em::bx2);
              }
              c0 += c00 * S2p[idx2];
            }
            b0[1] += c0 * S3d[idx3];
          }

          // Bx3 -- dual, dual, primal
          b0[2] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jd_min + idx2, kp_min + idx3, em::bx3);
              }
              c0 += c00 * S2d[idx2];
            }
            b0[2] += c0 * S3p[idx3];
          }
        }
      }
    }

    Inline void boundaryConditions(index_t p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i1(p) < 0) {
          if (boundaries.is_periodic_i1min) {
            particles.i1(p)      += ctx.ni1;
            particles.i1_prev(p) += ctx.ni1;
          } else if (boundaries.is_absorb_i1min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i1min) {
            particles.i1(p)  = 0;
            particles.dx1(p) = ONE - particles.dx1(p);
            invert_vel       = true;
          }
        } else if (particles.i1(p) >= ctx.ni1) {
          if (boundaries.is_periodic_i1max) {
            particles.i1(p)      -= ctx.ni1;
            particles.i1_prev(p) -= ctx.ni1;
          } else if (boundaries.is_absorb_i1max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i1max) {
            particles.i1(p)  = ctx.ni1 - 1;
            particles.dx1(p) = ONE - particles.dx1(p);
            invert_vel       = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            particles.ux1(p) = -particles.ux1(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[0] = -v[0];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i2(p) < 0) {
          if (boundaries.is_periodic_i2min) {
            particles.i2(p)      += ctx.ni2;
            particles.i2_prev(p) += ctx.ni2;
          } else if (boundaries.is_absorb_i2min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i2min) {
            particles.i2(p)  = 0;
            particles.dx2(p) = ONE - particles.dx2(p);
            invert_vel       = true;
          } else if (boundaries.is_axis_i2min) {
            particles.i2(p)  = 0;
            particles.dx2(p) = ONE - particles.dx2(p);
          }
        } else if (particles.i2(p) >= ctx.ni2) {
          if (boundaries.is_periodic_i2max) {
            particles.i2(p)      -= ctx.ni2;
            particles.i2_prev(p) -= ctx.ni2;
          } else if (boundaries.is_absorb_i2max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i2max) {
            particles.i2(p)  = ctx.ni2 - 1;
            particles.dx2(p) = ONE - particles.dx2(p);
            invert_vel       = true;
          } else if (boundaries.is_axis_i2max) {
            particles.i2(p)  = ctx.ni2 - 1;
            particles.dx2(p) = ONE - particles.dx2(p);
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            particles.ux2(p) = -particles.ux2(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[1] = -v[1];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i3(p) < 0) {
          if (boundaries.is_periodic_i3min) {
            particles.i3(p)      += ctx.ni3;
            particles.i3_prev(p) += ctx.ni3;
          } else if (boundaries.is_absorb_i3min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i3min) {
            particles.i3(p)  = 0;
            particles.dx3(p) = ONE - particles.dx3(p);
            invert_vel       = true;
          }
        } else if (particles.i3(p) >= ctx.ni3) {
          if (boundaries.is_periodic_i3max) {
            particles.i3(p)      -= ctx.ni3;
            particles.i3_prev(p) -= ctx.ni3;
          } else if (boundaries.is_absorb_i3max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (boundaries.is_reflect_i3max) {
            particles.i3(p)  = ctx.ni3 - 1;
            particles.dx3(p) = ONE - particles.dx3(p);
            invert_vel       = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            particles.ux3(p) = -particles.ux3(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[2] = -v[2];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
#if defined(MPI_ENABLED)
      if constexpr (D == Dim::_1D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1);
      } else if constexpr (D == Dim::_2D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1,
                                        particles.i2(p) < 0,
                                        particles.i2(p) >= ctx.ni2);
      } else if constexpr (D == Dim::_3D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1,
                                        particles.i2(p) < 0,
                                        particles.i2(p) >= ctx.ni2,
                                        particles.i3(p) < 0,
                                        particles.i3(p) >= ctx.ni3);
      }
#endif
    }

    Inline void synchrotronDrag(index_t                p,
                                vec_t<Dim::_3D>&       u_prime,
                                const vec_t<Dim::_3D>& e0,
                                const vec_t<Dim::_3D>& b0) const {
      real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
      u_prime[0]             *= gamma_prime_sqr;
      u_prime[1]             *= gamma_prime_sqr;
      u_prime[2]             *= gamma_prime_sqr;
      gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);
      const real_t beta_dot_e {
        DOT(u_prime[0], u_prime[1], u_prime[2], e0[0], e0[1], e0[2])
      };
      vec_t<Dim::_3D> e_plus_beta_cross_b {
        e0[0] + CROSS_x1(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[1] + CROSS_x2(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[2] + CROSS_x3(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2])
      };
      vec_t<Dim::_3D> kappaR {
        CROSS_x1(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[0],
        CROSS_x2(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[1],
        CROSS_x3(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[2],
      };
      const real_t chiR_sqr { NORM_SQR(e_plus_beta_cross_b[0],
                                       e_plus_beta_cross_b[1],
                                       e_plus_beta_cross_b[2]) -
                              SQR(beta_dot_e) };
      particles.ux1(p) += radiative_drag_ctx.synchrotron_coeff *
                          (kappaR[0] - gamma_prime_sqr * u_prime[0] * chiR_sqr);
      particles.ux2(p) += radiative_drag_ctx.synchrotron_coeff *
                          (kappaR[1] - gamma_prime_sqr * u_prime[1] * chiR_sqr);
      particles.ux3(p) += radiative_drag_ctx.synchrotron_coeff *
                          (kappaR[2] - gamma_prime_sqr * u_prime[2] * chiR_sqr);
    }

    Inline void inverseComptonDrag(index_t p, vec_t<Dim::_3D>& u_prime) const {
      real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
      u_prime[0]             *= gamma_prime_sqr;
      u_prime[1]             *= gamma_prime_sqr;
      u_prime[2]             *= gamma_prime_sqr;
      gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);

      particles.ux1(p) -= radiative_drag_ctx.compton_coeff * gamma_prime_sqr *
                          u_prime[0];
      particles.ux2(p) -= radiative_drag_ctx.compton_coeff * gamma_prime_sqr *
                          u_prime[1];
      particles.ux3(p) -= radiative_drag_ctx.compton_coeff * gamma_prime_sqr *
                          u_prime[2];
    }
  };

  template <class M, class D, ntt::EmissionTypeFlag E>
  auto MakeEmissionPolicy(D&                     domain,
                          const prm::Parameters& params,
                          const Context&         pusher_ctx) {
    if constexpr (E == ntt::EmissionType::SYNCHROTRON) {
      const auto photon_species = params.get<spidx_t>(
        "radiation.emission.synchrotron.photon_species");
      raise::ErrorIf(photon_species > domain.species.size(),
                     "Invalid photon_species for Synchrotron emission",
                     HERE);
      auto& emitted_species = domain.species[photon_species - 1];
      raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.mass()),
                     "Emitted photon species must have zero mass",
                     HERE);
      raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.charge()),
                     "Emitted photon species must have zero charge",
                     HERE);
      return pusher::sr::emission::Synchrotron<M>(emitted_species,
                                                  photon_species,
                                                  pusher_ctx.mass,
                                                  pusher_ctx.charge,
                                                  pusher_ctx.radiative_drag_flags,
                                                  domain.index(),
                                                  params,
                                                  domain.random_pool());
    } else if constexpr (E == ntt::EmissionType::COMPTON) {
      const auto photon_species = params.get<spidx_t>(
        "radiation.emission.compton.photon_species");
      raise::ErrorIf(photon_species > domain.species.size(),
                     "Invalid photon_species for Compton emission",
                     HERE);
      auto& emitted_species = domain.species[photon_species - 1];
      raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.mass()),
                     "Emitted photon species must have zero mass",
                     HERE);
      raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.charge()),
                     "Emitted photon species must have zero charge",
                     HERE);
      return pusher::sr::emission::Compton<M>(emitted_species,
                                              photon_species,
                                              pusher_ctx.mass,
                                              pusher_ctx.charge,
                                              pusher_ctx.radiative_drag_flags,
                                              domain.index(),
                                              params,
                                              domain.random_pool());
    } else {
      raise::Error("Invalid emission type for MakeEmissionPolicy", HERE);
      return traits::emission::NoPolicy_t {};
    }
  }

  template <class M, class D, class PGen, class F>
  void MakePolicy(const PGen&                  pgen,
                  D&                           domain,
                  const ntt::SimulationParams& params,
                  const Context&               pusher_ctx,
                  ntt::EmissionTypeFlag        emission_type,
                  bool                         atm,
                  F&&                          callback) {
    auto with_emission = [&](auto next) {
      switch (emission_type) {
        case ntt::EmissionType::SYNCHROTRON:
          next(MakeEmissionPolicy<M, ntt::EmissionType::SYNCHROTRON>(domain,
                                                                     params,
                                                                     pusher_ctx));
          break;
        case ntt::EmissionType::COMPTON:
          next(MakeEmissionPolicy<M, ntt::EmissionType::COMPTON>(domain,
                                                                 params,
                                                                 pusher_ctx));
          break;
        case ntt::EmissionType::CUSTOM:
          if constexpr (
            arch::traits::pgen::HasEmissionPolicy<PGen, decltype(domain)>) {
            next(pgen.EmissionPolicy(pusher_ctx.time,
                                     pusher_ctx.species_index,
                                     domain));
          } else {
            raise::Error("Custom emission policy flag is set but problem "
                         "generator does not define an emission policy",
                         HERE);
          }
          break;
        case ntt::EmissionType::NONE:
        default:
          next(traits::emission::NoPolicy_t {});
          break;
      }
    };

    auto with_custom_prtl_upd = [&](auto next) {
      if constexpr (traits::pgen::DefinesCustomParticleUpdate<PGen>) {
        next(pgen.CustomParticleUpdate(pusher_ctx.time,
                                       pusher_ctx.species_index,
                                       domain));
      } else {
        next(traits::custom_particle_update::NoPolicy_t {});
      }
    };

    auto with_ext_fields = [&](auto next) {
      if constexpr (traits::pgen::DefinesExternalFields<PGen>) {
        next(pgen.ExternalFields(pusher_ctx.time, pusher_ctx.species_index, domain));
      } else {
        next(traits::external_fields::NoPolicy_t {});
      }
    };

    with_emission([&](auto ep) {
      with_custom_prtl_upd([&](auto cpu) {
        with_ext_fields([&](auto ef) {
          using E   = decltype(ep);
          using CPU = decltype(cpu);
          using EF  = decltype(ef);
          if (atm) {
            callback(Policy<E, CPU, EF, true> { ep, cpu, ef });
          } else {
            callback(Policy<E, CPU, EF, false> { ep, cpu, ef });
          }
        });
      });
    });
  }

} // namespace pusher::sr

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // KERNELS_PUSHERS_BASE_SR_HPP
