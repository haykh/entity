/**
 * @file kernels/faraday_gr.hpp
 * @brief Algorithms for Faraday's law in GR
 * @implements
 *   - kernel::gr::Faraday_kernel<>
 * @namespaces:
 *   - kernel::gr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_FARADAY_GR_HPP
#define KERNELS_FARADAY_GR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::gr {
  using namespace ntt;

  /**
   * @brief Algorithms for Faraday's law
   * @brief `d(Bin)^i / dt = -curl E_j`, `Bout += dt * d(Bin)/dt`
   * @tparam M Metric
   */
  template <class M>
  class Faraday_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const ndfield_t<D, 6> Bin;
    ndfield_t<D, 6>       Bout;
    const ndfield_t<D, 6> E;
    const M               metric;
    const std::size_t     i2max;
    const real_t          coeff;
    bool                  is_axis_i2min { false };

  public:
    Faraday_kernel(const ndfield_t<D, 6>&      Bin,
                   const ndfield_t<D, 6>&      Bout,
                   const ndfield_t<D, 6>&      E,
                   const M&                    metric,
                   real_t                      coeff,
                   std::size_t                 ni2,
                   const boundaries_t<FldsBC>& boundaries)
      : Bin { Bin }
      , Bout { Bout }
      , E { E }
      , metric { metric }
      , i2max { ni2 + N_GHOSTS }
      , coeff { coeff } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr std::size_t i2min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };
        const real_t          inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t          inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
                                                  { i1_ + HALF, i2_ + HALF }) };

        Bout(i1, i2, em::bx1) = Bin(i1, i2, em::bx1) +
                                coeff * inv_sqrt_detH_0pH *
                                  (E(i1, i2, em::ex3) - E(i1, i2 + 1, em::ex3));
        if ((i2 == i2min) && is_axis_i2min) {
          Bout(i1, i2, em::bx2) = ZERO;
        } else {
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          Bout(i1, i2, em::bx2) = Bin(i1, i2, em::bx2) +
                                  coeff * inv_sqrt_detH_pH0 *
                                    (E(i1 + 1, i2, em::ex3) - E(i1, i2, em::ex3));
        }

        Bout(i1, i2, em::bx3) = Bin(i1, i2, em::bx3) +
                                coeff * inv_sqrt_detH_pHpH *
                                  (E(i1, i2 + 1, em::ex1) - E(i1, i2, em::ex1) +
                                   E(i1, i2, em::ex2) - E(i1 + 1, i2, em::ex2));
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

} // namespace kernel::gr

#endif // KERNELS_FARADAY_GR_HPP
