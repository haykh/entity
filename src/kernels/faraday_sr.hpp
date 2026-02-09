/**
 * @file kernels/faraday_sr.hpp
 * @brief Algorithms for Faraday's law in curvilinear SR
 * @implements
 *   - kernel::sr::Faraday_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_FARADAY_SR_HPP
#define KERNELS_FARADAY_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel::sr {
  using namespace ntt;

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Curvilinear
   * space (diagonal metric)
   */
  template <class M>
    requires metric::traits::HasD<M> && metric::traits::HasH_ij<M> &&
             metric::traits::HasSqrtDetH<M>
  class Faraday_kernel {
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6> EB;
    const M         metric;
    const real_t    coeff;
    bool            is_axis_i2min { false };

  public:
    Faraday_kernel(const ndfield_t<D, 6>&      EB,
                   const M&                    metric,
                   real_t                      coeff,
                   const boundaries_t<FldsBC>& boundaries)
      : EB { EB }
      , metric { metric }
      , coeff { coeff } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr ncells_t i2min { N_GHOSTS };
        const real_t       i1_ { COORD(i1) };
        const real_t       i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
                                                  { i1_ + HALF, i2_ + HALF }) };
        const real_t h1_pHp1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + ONE }) };
        const real_t h1_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
        const real_t h2_p1pH { metric.template h_<2, 2>({ i1_ + ONE, i2_ + HALF }) };
        const real_t h2_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };
        const real_t h3_00 { metric.template h_<3, 3>({ i1_, i2_ }) };
        const real_t h3_0p1 { metric.template h_<3, 3>({ i1_, i2_ + ONE }) };

        EB(i1, i2, em::bx1) += coeff * inv_sqrt_detH_0pH *
                               (h3_00 * EB(i1, i2, em::ex3) -
                                h3_0p1 * EB(i1, i2 + 1, em::ex3));
        if ((i2 != i2min) || !is_axis_i2min) {
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          const real_t h3_p10 { metric.template h_<3, 3>({ i1_ + ONE, i2_ }) };
          EB(i1, i2, em::bx2) += coeff * inv_sqrt_detH_pH0 *
                                 (h3_p10 * EB(i1 + 1, i2, em::ex3) -
                                  h3_00 * EB(i1, i2, em::ex3));
        }
        EB(i1, i2, em::bx3) += coeff * inv_sqrt_detH_pHpH *
                               (h1_pHp1 * EB(i1, i2 + 1, em::ex1) -
                                h1_pH0 * EB(i1, i2, em::ex1) +
                                h2_0pH * EB(i1, i2, em::ex2) -
                                h2_p1pH * EB(i1 + 1, i2, em::ex2));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 3D implementation called for D != 3");
      }
    }
  };
} // namespace kernel::sr

#endif // KERNELS_FARADAY_SR_HPP
