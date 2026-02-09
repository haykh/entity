/**
 * @file kernels/divergences.hpp
 * @brief Compute covariant divergences of fields
 * @implements
 *   - kernel::ComputeDivergence_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_DIVERGENCES_HPP
#define KERNELS_DIVERGENCES_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel {
  using namespace ntt;

  // @TODO: take care of boundaries
  template <class M, unsigned short N>
    requires metric::traits::HasD<M> && metric::traits::HasSqrtDetH<M>
  class ComputeDivergence_kernel {
    const M metric;

    const ndfield_t<M::Dim, 6> fields;
    ndfield_t<M::Dim, N>       buff;
    const idx_t                buff_idx;

  public:
    ComputeDivergence_kernel(const M&                    metric,
                             const ndfield_t<M::Dim, 6>& fields,
                             ndfield_t<M::Dim, N>&       buff,
                             idx_t                       buff_idx)
      : metric { metric }
      , fields { fields }
      , buff { buff }
      , buff_idx { buff_idx } {
      raise::ErrorIf(buff_idx >= N, "Invalid component index", HERE);
    }

    Inline void operator()(index_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        if constexpr (M::CoordType == Coord::Cart) {
          buff(i1, buff_idx) = fields(i1, em::ex1) - fields(i1 - 1, em::ex1);
        } else {
          const auto i1_     = COORD(i1);
          buff(i1, buff_idx) = (fields(i1, em::ex1) *
                                  metric.sqrt_det_h({ i1_ + HALF }) -
                                fields(i1 - 1, em::ex1) *
                                  metric.sqrt_det_h({ i1_ - HALF })) /
                               metric.sqrt_det_h({ i1_ });
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of ComputeDivergence_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        if constexpr (M::CoordType == Coord::Cart) {
          buff(i1, i2, buff_idx) = fields(i1, i2, em::ex1) -
                                   fields(i1 - 1, i2, em::ex1) +
                                   fields(i1, i2, em::ex2) -
                                   fields(i1, i2 - 1, em::ex2);
        } else {
          const auto i1_ = COORD(i1);
          const auto i2_ = COORD(i2);
          buff(i1, i2, buff_idx) =
            (fields(i1, i2, em::ex1) * metric.sqrt_det_h({ i1_ + HALF, i2_ }) -
             fields(i1 - 1, i2, em::ex1) * metric.sqrt_det_h({ i1_ - HALF, i2_ }) +
             fields(i1, i2, em::ex2) * metric.sqrt_det_h({ i1_, i2_ + HALF }) -
             fields(i1, i2 - 1, em::ex2) * metric.sqrt_det_h({ i1_, i2_ - HALF })) /
            metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of ComputeDivergence_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        if constexpr (M::CoordType == Coord::Cart) {
          buff(i1, i2, i3, buff_idx) = fields(i1, i2, i3, em::ex1) -
                                       fields(i1 - 1, i2, i3, em::ex1) +
                                       fields(i1, i2, i3, em::ex2) -
                                       fields(i1, i2 - 1, i3, em::ex2) +
                                       fields(i1, i2, i3, em::ex3) -
                                       fields(i1, i2, i3 - 1, em::ex3);
        } else {
          const auto i1_ = COORD(i1);
          const auto i2_ = COORD(i2);
          const auto i3_ = COORD(i3);
          buff(i1, i2, i3, buff_idx) =
            (fields(i1, i2, i3, em::ex1) *
               metric.sqrt_det_h({ i1_ + HALF, i2_, i3_ }) -
             fields(i1 - 1, i2, i3, em::ex1) *
               metric.sqrt_det_h({ i1_ - HALF, i2_, i3_ }) +
             fields(i1, i2, i3, em::ex2) *
               metric.sqrt_det_h({ i1_, i2_ + HALF, i3_ }) -
             fields(i1, i2 - 1, i3, em::ex2) *
               metric.sqrt_det_h({ i1_, i2_ - HALF, i3_ }) +
             fields(i1, i2, i3, em::ex3) *
               metric.sqrt_det_h({ i1_, i2_, i3_ + HALF }) -
             fields(i1, i2, i3 - 1, em::ex3) *
               metric.sqrt_det_h({ i1_, i2_, i3_ - HALF })) /
            metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF, i3_ + HALF });
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of ComputeDivergence_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_DIVERGENCES_HPP
