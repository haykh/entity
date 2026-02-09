/**
 * @file kernels/current_deposit.hpp
 * @brief Covariant algorithms for the current deposition
 * @implements
 *   - kernel::DepositCurrents_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_CURRENTS_DEPOSIT_HPP
#define KERNELS_CURRENTS_DEPOSIT_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include "particle_shapes.hpp"

#include <Kokkos_Core.hpp>

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

namespace kernel {
  using namespace ntt;

  /**
   * @brief Algorithm for the current deposition
   */
  template <SimEngine::type S, class M, unsigned short O = 1u>
    requires metric::traits::HasD<M> &&
             ((S == SimEngine::SRPIC && metric::traits::HasTransformXYZ<M>) ||
              (S == SimEngine::GRPIC && metric::traits::HasTransform<M> &&
               metric::traits::HasConvert_i<M> && metric::traits::HasAlpha<M>))
  class DepositCurrents_kernel {
    static_assert(O <= 11u, "Shape function order O must be <= 11");
    static constexpr auto D = M::Dim;

    scatter_ndfield_t<D, 3>  J;
    const array_t<int*>      i1, i2, i3;
    const array_t<int*>      i1_prev, i2_prev, i3_prev;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const M                  metric;
    const real_t             charge, inv_dt;

  public:
    /**
     * @brief explicit constructor.
     */
    DepositCurrents_kernel(const scatter_ndfield_t<D, 3>& scatter_cur,
                           const array_t<int*>&           i1,
                           const array_t<int*>&           i2,
                           const array_t<int*>&           i3,
                           const array_t<int*>&           i1_prev,
                           const array_t<int*>&           i2_prev,
                           const array_t<int*>&           i3_prev,
                           const array_t<prtldx_t*>&      dx1,
                           const array_t<prtldx_t*>&      dx2,
                           const array_t<prtldx_t*>&      dx3,
                           const array_t<prtldx_t*>&      dx1_prev,
                           const array_t<prtldx_t*>&      dx2_prev,
                           const array_t<prtldx_t*>&      dx3_prev,
                           const array_t<real_t*>&        ux1,
                           const array_t<real_t*>&        ux2,
                           const array_t<real_t*>&        ux3,
                           const array_t<real_t*>&        phi,
                           const array_t<real_t*>&        weight,
                           const array_t<short*>&         tag,
                           const M&                       metric,
                           real_t                         charge,
                           const real_t                   dt)
      : J { scatter_cur }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {
      raise::ErrorIf(
        (O == 2u and N_GHOSTS < 2),
        "Order of interpolation is 2, but number of ghost cells is < 2",
        HERE);
    }

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline auto operator()(index_t p) const -> void {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      // recover particle velocity to deposit in unsimulated direction
      vec_t<Dim::_3D> vp { ZERO };
      {
        coord_t<M::PrtlDim> xp { ZERO };
        if constexpr (D == Dim::_1D) {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
        } else if constexpr (D == Dim::_2D) {
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
            xp[2] = phi(p);
          } else {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
          }
        } else {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
          xp[1] = i_di_to_Xi(i2(p), dx2(p));
          xp[2] = i_di_to_Xi(i3(p), dx3(p));
        }
        auto inv_energy { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::U>(xp,
                                                          { ux1(p), ux2(p), ux3(p) },
                                                          vp);
          inv_energy = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
        } else {
          coord_t<Dim::_2D> xp_ { ZERO };
          xp_[0] = xp[0];
          real_t     theta_Cd { xp[1] };
          const auto theta_Ph { metric.template convert<2, Crd::Cd, Crd::Ph>(
            theta_Cd) };
          const auto small_angle { static_cast<real_t>(constant::SMALL_ANGLE_GR) };
          const auto large_angle { static_cast<real_t>(
            constant::PI - constant::SMALL_ANGLE_GR) };
          if (theta_Ph < small_angle) {
            theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(small_angle);
          } else if (theta_Ph >= large_angle) {
            theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(large_angle);
          }
          xp_[1] = theta_Cd;
          metric.template transform<Idx::D, Idx::U>(xp_,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    vp);
          inv_energy = metric.alpha(xp_) /
                       math::sqrt(ONE + ux1(p) * vp[0] + ux2(p) * vp[1] +
                                  ux3(p) * vp[2]);
        }
        if (Kokkos::isnan(vp[2]) || Kokkos::isinf(vp[2])) {
          vp[2] = ZERO;
        }
        vp[0] *= inv_energy;
        vp[1] *= inv_energy;
        vp[2] *= inv_energy;
      }

      const real_t coeff { weight(p) * charge };

      // ToDo: interpolation_order as parameter
      if constexpr (O == 0u) {
        /*
          Zig-zag deposit
        */
        const auto dxp_r_1 { static_cast<prtldx_t>(i1(p) == i1_prev(p)) *
                             (dx1(p) + dx1_prev(p)) *
                             static_cast<prtldx_t>(INV_2) };

        const real_t Wx1_1 { INV_2 * (dxp_r_1 + dx1_prev(p) +
                                      static_cast<real_t>(i1(p) > i1_prev(p))) };
        const real_t Wx1_2 { INV_2 * (dx1(p) + dxp_r_1 +
                                      static_cast<real_t>(
                                        static_cast<int>(i1(p) > i1_prev(p)) +
                                        i1_prev(p) - i1(p))) };
        const real_t Fx1_1 { (static_cast<real_t>(i1(p) > i1_prev(p)) +
                              dxp_r_1 - dx1_prev(p)) *
                             coeff * inv_dt };
        const real_t Fx1_2 { (static_cast<real_t>(
                                i1(p) - i1_prev(p) -
                                static_cast<int>(i1(p) > i1_prev(p))) +
                              dx1(p) - dxp_r_1) *
                             coeff * inv_dt };

        auto J_acc = J.access();

        if constexpr (D == Dim::_1D) {
          const real_t Fx2_1 { HALF * vp[1] * coeff };
          const real_t Fx2_2 { HALF * vp[1] * coeff };

          const real_t Fx3_1 { HALF * vp[2] * coeff };
          const real_t Fx3_2 { HALF * vp[2] * coeff };

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx1) += Fx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx1)      += Fx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx2)     += Fx2_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx2) += Fx2_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx2)          += Fx2_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx2)      += Fx2_2 * Wx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx3)     += Fx3_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx3) += Fx3_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx3)          += Fx3_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx3)      += Fx3_2 * Wx1_2;
        } else if constexpr (D == Dim::_2D || D == Dim::_3D) {
          const auto dxp_r_2 { static_cast<prtldx_t>(i2(p) == i2_prev(p)) *
                               (dx2(p) + dx2_prev(p)) *
                               static_cast<prtldx_t>(INV_2) };

          const real_t Wx2_1 { INV_2 * (dxp_r_2 + dx2_prev(p) +
                                        static_cast<real_t>(i2(p) > i2_prev(p))) };
          const real_t Wx2_2 { INV_2 * (dx2(p) + dxp_r_2 +
                                        static_cast<real_t>(
                                          static_cast<int>(i2(p) > i2_prev(p)) +
                                          i2_prev(p) - i2(p))) };
          const real_t Fx2_1 { (static_cast<real_t>(i2(p) > i2_prev(p)) +
                                dxp_r_2 - dx2_prev(p)) *
                               coeff * inv_dt };
          const real_t Fx2_2 { (static_cast<real_t>(
                                  i2(p) - i2_prev(p) -
                                  static_cast<int>(i2(p) > i2_prev(p))) +
                                dx2(p) - dxp_r_2) *
                               coeff * inv_dt };

          if constexpr (D == Dim::_2D) {
            const real_t Fx3_1 { HALF * vp[2] * coeff };
            const real_t Fx3_2 { HALF * vp[2] * coeff };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx1) += Fx1_2 *
                                                                   (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS + 1, cur::jx1) += Fx1_2 * Wx2_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                                   (ONE - Wx1_2);
            J_acc(i1(p) + N_GHOSTS + 1, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 * Wx1_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          } else {
            const auto   dxp_r_3 { static_cast<prtldx_t>(i3(p) == i3_prev(p)) *
                                 (dx3(p) + dx3_prev(p)) *
                                 static_cast<prtldx_t>(INV_2) };
            const real_t Wx3_1 { INV_2 * (dxp_r_3 + dx3_prev(p) +
                                          static_cast<real_t>(i3(p) > i3_prev(p))) };
            const real_t Wx3_2 { INV_2 * (dx3(p) + dxp_r_3 +
                                          static_cast<real_t>(
                                            static_cast<int>(i3(p) > i3_prev(p)) +
                                            i3_prev(p) - i3(p))) };
            const real_t Fx3_1 { (static_cast<real_t>(i3(p) > i3_prev(p)) +
                                  dxp_r_3 - dx3_prev(p)) *
                                 coeff * inv_dt };
            const real_t Fx3_2 { (static_cast<real_t>(
                                    i3(p) - i3_prev(p) -
                                    static_cast<int>(i3(p) > i3_prev(p))) +
                                  dx3(p) - dxp_r_3) *
                                 coeff * inv_dt };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * Wx2_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * Wx2_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * Wx2_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * Wx1_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * Wx1_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * Wx1_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          }
        }
      } else if constexpr ((O >= 1u) and (O <= 11u)) {

        // shape function in dim1 -> always required
        real_t iS_x1[O + 2], fS_x1[O + 2];
        // indices of the shape function
        int    i1_min, i1_max;

        // call shape function
        prtl_shape::for_deposit<O>(i1_prev(p),
                                   static_cast<real_t>(dx1_prev(p)),
                                   i1(p),
                                   static_cast<real_t>(dx1(p)),
                                   i1_min,
                                   i1_max,
                                   iS_x1,
                                   fS_x1);

        if constexpr (D == Dim::_1D) {
          // define weight vectors
          real_t Wx1[O + 2];
          real_t Wx23[O + 2];

          // Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
            // Esirkepov 2001, Eq. 38 for 1D case
            Wx1[i]  = fS_x1[i] - iS_x1[i];
            Wx23[i] = HALF * (fS_x1[i] + iS_x1[i]);
          }

          // contribution within the shape function stencil
          real_t jx1[O + 2];

          // prefactors for j update
          const real_t Qdx1dt = coeff * inv_dt;
          const real_t QVx2   = coeff * vp[1];
          const real_t QVx3   = coeff * vp[2];

          // Calculate current contribution
          jx1[0] = -Qdx1dt * Wx1[0];
#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
            jx1[i] = jx1[i - 1] - Qdx1dt * Wx1[i];
          }

          // account for ghost cells
          i1_min += N_GHOSTS;
          i1_max += N_GHOSTS;

          // get number of update indices for asymmetric movement
          const int di_x1 = i1_max - i1_min;

          /*
              Current update
          */
          auto J_acc = J.access();

          for (int i = 0; i < di_x1; ++i) {
            J_acc(i1_min + i, cur::jx1) += jx1[i];
          }

          for (int i = 0; i <= di_x1; ++i) {
            J_acc(i1_min + i, cur::jx2) += QVx2 * Wx23[i];
          }

          for (int i = 0; i <= di_x1; ++i) {
            J_acc(i1_min + i, cur::jx3) += QVx3 * Wx23[i];
          }

        } else if constexpr (D == Dim::_2D) {

          // shape function in dim1 -> always required
          real_t iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          int    i2_min, i2_max;

          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                     static_cast<real_t>(dx2_prev(p)),
                                     i2(p),
                                     static_cast<real_t>(dx2(p)),
                                     i2_min,
                                     i2_max,
                                     iS_x2,
                                     fS_x2);

          // define weight tensors
          real_t Wx1[O + 2][O + 2];
          real_t Wx2[O + 2][O + 2];
          real_t Wx3[O + 2][O + 2];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              // Esirkepov 2001, Eq. 38 (simplified)
              Wx1[i][j] = HALF * (fS_x1[i] - iS_x1[i]) * (fS_x2[j] + iS_x2[j]);

              Wx2[i][j] = HALF * (fS_x1[i] + iS_x1[i]) * (fS_x2[j] - iS_x2[j]);

              Wx3[i][j] = THIRD * (fS_x2[j] * (HALF * iS_x1[i] + fS_x1[i]) +
                                   iS_x2[j] * (HALF * fS_x1[i] + iS_x1[i]));
            }
          }

          // contribution within the shape function stencil
          real_t jx1[O + 2][O + 2], jx2[O + 2][O + 2];

          // prefactors for j update
          const real_t Qdx1dt = coeff * inv_dt;
          const real_t Qdx2dt = coeff * inv_dt;
          const real_t QVx3   = coeff * vp[2];

          // Calculate current contribution

          // jx1
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
            jx1[0][j] = -Qdx1dt * Wx1[0][j];
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx1[i][j] = jx1[i - 1][j] - Qdx1dt * Wx1[i][j];
            }
          }

          // jx2
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
            jx2[i][0] = -Qdx2dt * Wx2[i][0];
          }

#pragma unroll
          for (int j = 1; j < O + 2; ++j) {
#pragma unroll
            for (int i = 0; i < O + 2; ++i) {
              jx2[i][j] = jx2[i][j - 1] - Qdx2dt * Wx2[i][j];
            }
          }

          // account for ghost cells
          i1_min += N_GHOSTS;
          i2_min += N_GHOSTS;
          i1_max += N_GHOSTS;
          i2_max += N_GHOSTS;

          // get number of update indices for asymmetric movement
          const int di_x1 = i1_max - i1_min;
          const int di_x2 = i2_max - i2_min;

          /*
              Current update
          */
          auto J_acc = J.access();

          for (int i = 0; i < di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx1) += jx1[i][j];
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j < di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx2) += jx2[i][j];
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx3) += QVx3 * Wx3[i][j];
            }
          }

        } else if constexpr (D == Dim::_3D) {
          // shape function in dim2
          real_t iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          int    i2_min, i2_max;
          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                     static_cast<real_t>(dx2_prev(p)),
                                     i2(p),
                                     static_cast<real_t>(dx2(p)),
                                     i2_min,
                                     i2_max,
                                     iS_x2,
                                     fS_x2);

          // shape function in dim3
          real_t iS_x3[O + 2], fS_x3[O + 2];
          // indices of the shape function
          int    i3_min, i3_max;

          // call shape function
          prtl_shape::for_deposit<O>(i3_prev(p),
                                     static_cast<real_t>(dx3_prev(p)),
                                     i3(p),
                                     static_cast<real_t>(dx3(p)),
                                     i3_min,
                                     i3_max,
                                     iS_x3,
                                     fS_x3);

          // define weight tensors
          real_t Wx1[O + 2][O + 2][O + 2];
          real_t Wx2[O + 2][O + 2][O + 2];
          real_t Wx3[O + 2][O + 2][O + 2];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                // Esirkepov 2001, Eq. 31
                Wx1[i][j][k] = THIRD * (fS_x1[i] - iS_x1[i]) *
                               ((iS_x2[j] * iS_x3[k] + fS_x2[j] * fS_x3[k]) +
                                HALF * (iS_x3[k] * fS_x2[j] + iS_x2[j] * fS_x3[k]));

                Wx2[i][j][k] = THIRD * (fS_x2[j] - iS_x2[j]) *
                               (iS_x1[i] * iS_x3[k] + fS_x1[i] * fS_x3[k] +
                                HALF * (iS_x3[k] * fS_x1[i] + iS_x1[i] * fS_x3[k]));

                Wx3[i][j][k] = THIRD * (fS_x3[k] - iS_x3[k]) *
                               (iS_x1[i] * iS_x2[j] + fS_x1[i] * fS_x2[j] +
                                HALF * (iS_x1[i] * fS_x2[j] + iS_x2[j] * fS_x1[i]));
              }
            }
          }

          // contribution within the shape function stencil
          real_t jx1[O + 2][O + 2][O + 2], jx2[O + 2][O + 2][O + 2],
            jx3[O + 2][O + 2][O + 2];

          // prefactors to j update
          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t Qdzdt = coeff * inv_dt;

          // Calculate current contribution

          // jx1
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx1[0][j][k] = -Qdxdt * Wx1[0][j][k];
            }
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                jx1[i][j][k] = jx1[i - 1][j][k] - Qdxdt * Wx1[i][j][k];
              }
            }
          }

          // jx2
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx2[i][0][k] = -Qdydt * Wx2[i][0][k];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 1; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                jx2[i][j][k] = jx2[i][j - 1][k] - Qdydt * Wx2[i][j][k];
              }
            }
          }

          // jx3
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx3[i][j][0] = -Qdydt * Wx3[i][j][0];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                jx3[i][j][k] = jx3[i][j][k - 1] - Qdzdt * Wx3[i][j][k];
              }
            }
          }

          // account for ghost cells
          i1_min += N_GHOSTS;
          i2_min += N_GHOSTS;
          i3_min += N_GHOSTS;
          i1_max += N_GHOSTS;
          i2_max += N_GHOSTS;
          i3_max += N_GHOSTS;

          // get number of update indices for asymmetric movement
          const int di_x1 = i1_max - i1_min;
          const int di_x2 = i2_max - i2_min;
          const int di_x3 = i3_max - i3_min;

          /*
            Current update
          */
          auto J_acc = J.access();

          for (int i = 0; i < di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              for (int k = 0; k <= di_x3; ++k) {
                J_acc(i1_min + i, i2_min + j, i3_min + k, cur::jx1) += jx1[i][j][k];
              }
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j < di_x2; ++j) {
              for (int k = 0; k <= di_x3; ++k) {
                J_acc(i1_min + i, i2_min + j, i3_min + k, cur::jx2) += jx2[i][j][k];
              }
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              for (int k = 0; k < di_x3; ++k) {
                J_acc(i1_min + i, i2_min + j, i3_min + k, cur::jx3) += jx3[i][j][k];
              }
            }
          }

        } // dim
      } else { // order
        raise::KernelError(
          HERE,
          "Unsupported interpolation order. O > 11 not supported. Seriously. "
          "What are you even doing here? Entity already goes to 11!");
      }
    }
  };
} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
