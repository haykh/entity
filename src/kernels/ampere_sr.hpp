/**
 * @file kernels/ampere_gr.hpp
 * @brief Algorithms for Ampere's law in curvilinear SR
 * @implements
 *   - kernel::sr::Ampere_kernel<>
 *   - kernel::sr::CurrentsAmpere_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_AMPERE_SR_HPP
#define KERNELS_AMPERE_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel::sr {
  using namespace ntt;

  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear space
   * @tparam M Metric
   */
  template <class M>
    requires metric::traits::HasD<M> && metric::traits::HasH_ij<M> &&
             metric::traits::HasSqrtDetH<M> && metric::traits::HasPolarArea<M>
  class Ampere_kernel {
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6> EB;
    const M         metric;
    const ncells_t  i2max;
    const real_t    coeff;
    bool            is_axis_i2min { false }, is_axis_i2max { false };

  public:
    Ampere_kernel(const ndfield_t<D, 6>&      EB,
                  const M&                    metric,
                  real_t                      coeff,
                  ncells_t                    ni2,
                  const boundaries_t<FldsBC>& boundaries)
      : EB { EB }
      , metric { metric }
      , i2max { ni2 + N_GHOSTS }
      , coeff { coeff } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr ncells_t i2min { N_GHOSTS };
        const real_t       i1_ { COORD(i1) };
        const real_t       i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t h3_mHpH { metric.template h_<3, 3>({ i1_ - HALF, i2_ + HALF }) };
        const real_t h3_pHpH { metric.template h_<3, 3>({ i1_ + HALF, i2_ + HALF }) };

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0
          const real_t inv_polar_area_pH0 { ONE / metric.polar_area(i1_ + HALF) };
          EB(i1, i2, em::ex1) += inv_polar_area_pH0 * coeff *
                                 (h3_pHpH * EB(i1, i2, em::bx3));
          EB(i1, i2, em::ex2) += coeff * inv_sqrt_detH_0pH *
                                 (h3_mHpH * EB(i1 - 1, i2, em::bx3) -
                                  h3_pHpH * EB(i1, i2, em::bx3));
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi
          const real_t inv_polar_area_pH0 { ONE / metric.polar_area(i1_ + HALF) };
          const real_t h3_pHmH { metric.template h_<3, 3>(
            { i1_ + HALF, i2_ - HALF }) };
          EB(i1, i2, em::ex1) -= inv_polar_area_pH0 * coeff *
                                 (h3_pHmH * EB(i1, i2, em::bx3));
        } else {
          const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          const real_t h1_0mH { metric.template h_<1, 1>({ i1_, i2_ - HALF }) };
          const real_t h1_0pH { metric.template h_<1, 1>({ i1_, i2_ + HALF }) };
          const real_t h2_pH0 { metric.template h_<2, 2>({ i1_ + HALF, i2_ }) };
          const real_t h2_mH0 { metric.template h_<2, 2>({ i1_ - HALF, i2_ }) };
          const real_t h3_pHmH { metric.template h_<3, 3>(
            { i1_ + HALF, i2_ - HALF }) };
          EB(i1, i2, em::ex1) += coeff * inv_sqrt_detH_pH0 *
                                 (h3_pHpH * EB(i1, i2, em::bx3) -
                                  h3_pHmH * EB(i1, i2 - 1, em::bx3));
          EB(i1, i2, em::ex2) += coeff * inv_sqrt_detH_0pH *
                                 (h3_mHpH * EB(i1 - 1, i2, em::bx3) -
                                  h3_pHpH * EB(i1, i2, em::bx3));
          EB(i1, i2, em::ex3) += coeff * inv_sqrt_detH_00 *
                                 (h1_0mH * EB(i1, i2 - 1, em::bx1) -
                                  h1_0pH * EB(i1, i2, em::bx1) +
                                  h2_pH0 * EB(i1, i2, em::bx2) -
                                  h2_mH0 * EB(i1 - 1, i2, em::bx2));
        }
      } else {
        raise::KernelError(HERE, "Ampere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(HERE, "Ampere_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Add the currents to the E field with the appropriate conversion
   */
  template <class M>
    requires metric::traits::HasD<M> && metric::traits::HasH_ij<M> &&
             metric::traits::HasSqrtDetH<M> && metric::traits::HasPolarArea<M>
  class CurrentsAmpere_kernel {
    static constexpr auto     D     = M::Dim;
    static constexpr ncells_t i2min = N_GHOSTS;

    ndfield_t<D, 6> E;
    ndfield_t<D, 3> J;
    const M         metric;
    const ncells_t  i2max;
    // coeff = -dt * q0 * n0 / B0
    const real_t    coeff;
    const real_t    inv_n0;
    bool            is_axis_i2min { false };
    bool            is_axis_i2max { false };

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpere_kernel(const ndfield_t<D, 6>&      E,
                          const ndfield_t<D, 3>&      J,
                          const M&                    metric,
                          real_t                      coeff,
                          real_t                      inv_n0,
                          ncells_t                    ni2,
                          const boundaries_t<FldsBC>& boundaries)
      : E { E }
      , J { J }
      , metric { metric }
      , i2max { ni2 + N_GHOSTS }
      , coeff { coeff }
      , inv_n0 { inv_n0 } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const real_t i1_ { COORD(i1) };
        const real_t i2_ { COORD(i2) };

        // convert the "curly" current, to contravariant, normalized to
        // `J0=n0*q0` then add "curly" current with the right coefficient

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0 (first active cell)
          // r
          J(i1, i2, cur::jx1) *= inv_n0 * HALF / metric.polar_area(i1_ + HALF);

          // theta
          J(i1, i2, cur::jx2) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ + HALF });
          E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;

          // phi
          J(i1, i2, cur::jx3) = ZERO;
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi (first ghost cell from end)
          // r
          J(i1, i2, cur::jx1) *= inv_n0 * HALF / metric.polar_area(i1_ + HALF);

          // phi
          J(i1, i2, cur::jx3) = ZERO;
        } else {
          // 0 < theta < pi
          // r
          J(i1, i2, cur::jx1) *= inv_n0 / metric.sqrt_det_h({ i1_ + HALF, i2_ });

          // theta
          J(i1, i2, cur::jx2) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ + HALF });
          E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;

          // phi
          J(i1, i2, cur::jx3) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ });
          E(i1, i2, em::ex3)  += J(i1, i2, cur::jx3) * coeff;
        }

        E(i1, i2, em::ex1) += J(i1, i2, cur::jx1) * coeff;
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::sr

#endif // KERNELS_AMPERE_SR_HPP
