/**
 * @file kernels/reduced_stats.hpp
 * @brief Compute reduced field/moment quantities for stats output
 * @implements
 *   - kernel::ReducedFields_kernel<>
 *   - kernel::ReducedParticleMoments_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_REDUCED_STATS_HPP
#define KERNELS_REDUCED_STATS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine::type S, class M, StatsID::type F, unsigned short I = 0>
    requires metric::traits::HasD<M> &&
             (metric::traits::HasTransform_i<M> || I == 0) &&
             metric::traits::HasTransform<M> && metric::traits::HasSqrtDetH<M> &&
             (I <= 3)
  class ReducedFields_kernel {
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6> EM;
    ndfield_t<D, 3> J;
    const M         metric;

  public:
    ReducedFields_kernel(const ndfield_t<D, 6>& EM,
                         const ndfield_t<D, 3>& J,
                         const M&               metric)
      : EM { EM }
      , J { J }
      , metric { metric } {}

    Inline void operator()(index_t i1, real_t& buff) const {
      const auto i1_ = COORD(i1);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>({ i1_ },
                                                                         b1_u);
          buff += b1_u * b1_d * metric.sqrt_det_h({ i1_ });
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF },
            b2_u);
          buff += b2_u * b2_d * metric.sqrt_det_h({ i1_ + HALF });
        } else {
          const auto b3_u = EM(i1, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF },
            b3_u);
          buff += b3_u * b3_d * metric.sqrt_det_h({ i1_ + HALF });
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF },
            e1_u);
          buff += e1_u * e1_d * metric.sqrt_det_h({ i1_ + HALF });
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>({ i1_ },
                                                                         e2_u);
          buff += e2_u * e2_d * metric.sqrt_det_h({ i1_ });
        } else {
          const auto e3_u = EM(i1, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>({ i1_ },
                                                                         e3_u);
          buff += e3_u * e3_d * metric.sqrt_det_h({ i1_ });
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx2));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx3));
          buff += (e2_t * b3_t - e3_t * b2_t) * metric.sqrt_det_h({ i1_ + HALF });
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::ex1));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::bx1) + EM(i1 + 1, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx3));
          buff += (e3_t * b1_t - e1_t * b3_t) * metric.sqrt_det_h({ i1_ + HALF });
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::ex1));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::bx1) + EM(i1 + 1, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx2));
          buff += (e1_t * b2_t - e2_t * b1_t) * metric.sqrt_det_h({ i1_ + HALF });
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF },
          { EM(i1, em::ex1),
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)),
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF },
          { J(i1, cur::jx1),
            HALF * (J(i1, cur::jx2) + J(i1 + 1, cur::jx2)),
            HALF * (J(i1, cur::jx3) + J(i1 + 1, cur::jx3)) },
          j_t);
        buff += (e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2]) *
                metric.sqrt_det_h({ i1_ + HALF });
      }
    }

    Inline void operator()(index_t i1, index_t i2, real_t& buff) const {
      const auto i1_ = COORD(i1);
      const auto i2_ = COORD(i2);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, i2, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_, i2_ + HALF },
            b1_u);
          buff += b1_u * b1_d * metric.sqrt_det_h({ i1_, i2_ + HALF });
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, i2, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ },
            b2_u);
          buff += b2_u * b2_d * metric.sqrt_det_h({ i1_ + HALF, i2_ });
        } else {
          const auto b3_u = EM(i1, i2, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ + HALF },
            b3_u);
          buff += b3_u * b3_d * metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, i2, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ },
            e1_u);
          buff += e1_u * e1_d * metric.sqrt_det_h({ i1_ + HALF, i2_ });
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, i2, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_, i2_ + HALF },
            e2_u);
          buff += e2_u * e2_d * metric.sqrt_det_h({ i1_, i2_ + HALF });
        } else {
          const auto e3_u = EM(i1, i2, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_, i2_ },
            e3_u);
          buff += e3_u * e3_d * metric.sqrt_det_h({ i1_, i2_ });
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx2) + EM(i1, i2 + 1, em::bx2)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            EM(i1, i2, em::bx3));
          buff += (e2_t * b3_t - e3_t * b2_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx1) + EM(i1 + 1, i2, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            EM(i1, i2, em::bx3));
          buff += (e3_t * b1_t - e1_t * b3_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx1) + EM(i1 + 1, i2, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx2) + EM(i1, i2 + 1, em::bx2)));
          buff += (e1_t * b2_t - e2_t * b1_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF },
          { HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)),
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)),
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF },
          { HALF * (J(i1, i2, cur::jx1) + J(i1, i2 + 1, cur::jx1)),
            HALF * (J(i1, i2, cur::jx2) + J(i1 + 1, i2, cur::jx2)),
            INV_4 * (J(i1, i2, cur::jx3) + J(i1 + 1, i2, cur::jx3) +
                     J(i1, i2 + 1, cur::jx3) + J(i1 + 1, i2 + 1, cur::jx3)) },
          j_t);
        buff += (e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2]) *
                metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3, real_t& buff) const {
      const auto i1_ = COORD(i1);
      const auto i2_ = COORD(i2);
      const auto i3_ = COORD(i3);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, i2, i3, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_, i2_ + HALF, i3_ + HALF },
            b1_u);
          buff += b1_u * b1_d * metric.sqrt_det_h({ i1_, i2_ + HALF, i3_ + HALF });
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, i2, i3, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF, i2_, i3_ + HALF },
            b2_u);
          buff += b2_u * b2_d * metric.sqrt_det_h({ i1_ + HALF, i2_, i3_ + HALF });
        } else {
          const auto b3_u = EM(i1, i2, i3, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ + HALF, i3_ },
            b3_u);
          buff += b3_u * b3_d * metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ });
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, i2, i3, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF, i2_, i3_ },
            e1_u);
          buff += e1_u * e1_d * metric.sqrt_det_h({ i1_ + HALF, i2_, i3_ });
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, i2, i3, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_, i2_ + HALF, i3_ },
            e2_u);
          buff += e2_u * e2_d * metric.sqrt_det_h({ i1_, i2_ + HALF, i3_ });
        } else {
          const auto e3_u = EM(i1, i2, i3, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_, i2_, i3_ + HALF },
            e3_u);
          buff += e3_u * e3_d * metric.sqrt_det_h({ i1_, i2_, i3_ + HALF });
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
               EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
               EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx2) + EM(i1, i2 + 1, i3, em::bx2)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx3) + EM(i1, i2, i3 + 1, em::bx3)));
          buff += (e2_t * b3_t - e3_t * b2_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ + HALF });
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
               EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
               EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx1) + EM(i1 + 1, i2, i3, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx3) + EM(i1, i2, i3 + 1, em::bx3)));
          buff += (e3_t * b1_t - e1_t * b3_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ + HALF });
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
               EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
               EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx1) + EM(i1 + 1, i2, i3, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx2) + EM(i1, i2 + 1, i3, em::bx2)));
          buff += (e1_t * b2_t - e2_t * b1_t) *
                  metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ + HALF });
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF, i3_ + HALF },
          { INV_4 * (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
                     EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)),
            INV_4 * (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
                     EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)),
            INV_4 * (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
                     EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF, i3_ + HALF },
          { INV_4 * (J(i1, i2, i3, cur::jx1) + J(i1, i2 + 1, i3, cur::jx1) +
                     J(i1, i2, i3 + 1, cur::jx1) + J(i1, i2 + 1, i3 + 1, cur::jx1)),
            INV_4 * (J(i1, i2, i3, cur::jx2) + J(i1 + 1, i2, i3, cur::jx2) +
                     J(i1, i2, i3 + 1, cur::jx2) + J(i1 + 1, i2, i3 + 1, cur::jx2)),
            INV_4 * (J(i1, i2, i3, cur::jx3) + J(i1 + 1, i2, i3, cur::jx3) +
                     J(i1, i2 + 1, i3, cur::jx3) + J(i1 + 1, i2 + 1, i3, cur::jx3)) },
          j_t);
        buff += (e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2]) *
                metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ + HALF });
      }
    }
  };

  template <StatsID::type P>
  auto get_contrib(float mass, float charge) -> real_t {
    if constexpr (P == StatsID::Rho) {
      return mass;
    } else if constexpr (P == StatsID::Charge) {
      return charge;
    } else {
      return ONE;
    }
  }

  template <SimEngine::type S, class M, StatsID::type P>
    requires metric::traits::HasD<M> &&
             ((S == SimEngine::SRPIC && metric::traits::HasTransformXYZ<M>) ||
              S == SimEngine::GRPIC && metric::traits::HasTransform<M>) &&
             ((P == StatsID::Rho) || (P == StatsID::Charge) ||
              (P == StatsID::N) || (P == StatsID::Npart) || (P == StatsID::T))
  class ReducedParticleMoments_kernel {
    static constexpr auto D = M::Dim;

    const unsigned short     c1, c2;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const float              charge;
    const bool               use_weights;
    const M                  metric;

    const real_t contrib;

  public:
    ReducedParticleMoments_kernel(const std::vector<unsigned short>& components,
                                  const array_t<int*>&               i1,
                                  const array_t<int*>&               i2,
                                  const array_t<int*>&               i3,
                                  const array_t<prtldx_t*>&          dx1,
                                  const array_t<prtldx_t*>&          dx2,
                                  const array_t<prtldx_t*>&          dx3,
                                  const array_t<real_t*>&            ux1,
                                  const array_t<real_t*>&            ux2,
                                  const array_t<real_t*>&            ux3,
                                  const array_t<real_t*>&            phi,
                                  const array_t<real_t*>&            weight,
                                  const array_t<short*>&             tag,
                                  float                              mass,
                                  float                              charge,
                                  bool     use_weights,
                                  const M& metric)
      : c1 { (components.size() > 0) ? components[0]
                                     : static_cast<unsigned short>(0) }
      , c2 { (components.size() == 2) ? components[1]
                                      : static_cast<unsigned short>(0) }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , charge { charge }
      , use_weights { use_weights }
      , metric { metric }
      , contrib { get_contrib<P>(mass, charge) } {
      raise::ErrorIf(((P == StatsID::Rho) || (P == StatsID::Charge)) &&
                       (mass == ZERO),
                     "Rho & Charge for massless particles not defined",
                     HERE);
    }

    Inline void operator()(index_t p, real_t& buff) const {
      if (tag(p) != ParticleTag::alive) {
        return;
      }
      auto dV = ONE;

      if constexpr (P == StatsID::Npart) {
        buff += ONE;
        return;
      } else {
        coord_t<D> x_Code { ZERO };
        if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        }
        if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        }
        if constexpr (D == Dim::_3D) {
          x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
        }
        dV = metric.sqrt_det_h(x_Code);
        if constexpr (P == StatsID::N or P == StatsID::Rho or P == StatsID::Charge) {
          buff += dV * (use_weights ? weight(p) : contrib);
        } else {
          // for stress-energy tensor
          real_t          energy { ZERO };
          vec_t<Dim::_3D> u_Phys { ZERO };
          if constexpr (S == SimEngine::SRPIC) {
            // SR
            // stress-energy tensor for SR is computed in the tetrad (hatted) basis
            if constexpr (M::CoordType == Coord::Cart) {
              u_Phys[0] = ux1(p);
              u_Phys[1] = ux2(p);
              u_Phys[2] = ux3(p);
            } else {
              static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
              coord_t<M::PrtlDim> x_Code { ZERO };
              x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
              x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
              if constexpr (D == Dim::_3D) {
                x_Code[2] = static_cast<real_t>(i3(p)) +
                            static_cast<real_t>(dx3(p));
              } else {
                x_Code[2] = phi(p);
              }
              metric.template transform_xyz<Idx::XYZ, Idx::T>(
                x_Code,
                { ux1(p), ux2(p), ux3(p) },
                u_Phys);
            }
            if (mass == ZERO) {
              energy = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
            } else {
              energy = mass * math::sqrt(
                                ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
            }
          } else {
            // GR
            // stress-energy tensor for GR is computed in contravariant basis
            static_assert(D != Dim::_1D, "GRPIC 1D");
            coord_t<D> x_Code { ZERO };
            x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            if constexpr (D == Dim::_3D) {
              x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
            }
            vec_t<Dim::_3D> u_Cntrv { ZERO };
            // compute u_i u^i for energy
            metric.template transform<Idx::D, Idx::U>(x_Code,
                                                      { ux1(p), ux2(p), ux3(p) },
                                                      u_Cntrv);
            energy = u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) +
                     u_Cntrv[2] * ux3(p);
            if (mass == ZERO) {
              energy = math::sqrt(energy);
            } else {
              energy = mass * math::sqrt(ONE + energy);
            }
            metric.template transform<Idx::U, Idx::PU>(x_Code, u_Cntrv, u_Phys);
          }
          // compute the corresponding moment
          real_t coeff = ONE;
#pragma unroll
          for (const auto& c : { c1, c2 }) {
            if (c == 0) {
              coeff *= energy;
            } else {
              coeff *= u_Phys[c - 1];
            }
          }
          buff += dV * coeff / energy;
        }
      }
    }
  };

} // namespace kernel

#endif
