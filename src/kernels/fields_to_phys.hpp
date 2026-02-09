/**
 * @file kernels/fields_to_phys.hpp
 * @brief Algorithms to interpolate staggered fields and convert to a different basis
 * @implements
 *   - kernel::FieldsToPhys_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 * The field `from(*, cfi)` is manipulated and saved to the
 * corresponding `to(*, cti)` field
 * @note
 * The behavior of this function is determined with the
 * PrepareOutput:: flags, which can be chained together to
 * perform multiple operations.
 * @example
 * PrepareOutput::InterpToCellCenterFromEdges | PrepareOutput::ConvertToHat
 * will both interpolate the fields to the cell center and convert them
 * to the tetrad basis
 */

#ifndef KERNELS_FIELDS_TO_PHYS_HPP
#define KERNELS_FIELDS_TO_PHYS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel {
  using namespace ntt;

  template <class M, int N1, int N2>
    requires metric::traits::HasD<M> && metric::traits::HasTransform<M> &&
             (N1 >= 3) && (N2 >= 3)
  class FieldsToPhys_kernel {
    static constexpr auto D = M::Dim;

    const ndfield_t<D, N1>   Ffrom;
    ndfield_t<D, N2>         Fto;
    const PrepareOutputFlags flags;
    const M                  metric;

    const unsigned short cf1, cf2, cf3;
    const unsigned short ct1, ct2, ct3;

  public:
    FieldsToPhys_kernel(const ndfield_t<D, N1>&   from,
                        ndfield_t<D, N2>&         to,
                        list_t<unsigned short, 3> comps_from,
                        list_t<unsigned short, 3> comps_to,
                        const PrepareOutputFlags& flags,
                        const M&                  metric)
      : Ffrom { from }
      , Fto { to }
      , flags { flags }
      , metric { metric }
      , cf1 { comps_from[0] }
      , cf2 { comps_from[1] }
      , cf3 { comps_from[2] }
      , ct1 { comps_to[0] }
      , ct2 { comps_to[1] }
      , ct3 { comps_to[2] } {
      raise::ErrorIf((cf1 >= N1) || (cf2 >= N1) || (cf3 >= N1),
                     "FieldsToPhys_kernel: Invalid component index",
                     HERE);
      raise::ErrorIf((ct1 >= N2) || (ct2 >= N2) || (ct3 >= N2),
                     "FieldsToPhys_kernel: Invalid component index",
                     HERE);
    }

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        real_t          i1_ { COORD(i1) };
        vec_t<Dim::_3D> f_int { ZERO }, f_fin { ZERO };
        auto            cell_center = false;
        if (flags & PrepareOutput::InterpToCellCenterFromEdges) {
          f_int[0]    = Ffrom(i1, cf1);
          f_int[1]    = INV_2 * (Ffrom(i1, cf2) + Ffrom(i1 + 1, cf2));
          f_int[2]    = INV_2 * (Ffrom(i1, cf3) + Ffrom(i1 + 1, cf3));
          cell_center = true;
        } else if (flags & PrepareOutput::InterpToCellCenterFromFaces) {
          f_int[0]    = INV_2 * (Ffrom(i1, cf1) + Ffrom(i1 + 1, cf1));
          f_int[1]    = Ffrom(i1, cf2);
          f_int[2]    = Ffrom(i1, cf3);
          cell_center = true;
        } else {
          f_int[0] = Ffrom(i1, cf1);
          f_int[1] = Ffrom(i1, cf2);
          f_int[2] = Ffrom(i1, cf3);
        }

        coord_t<Dim::_1D> xi_field { ZERO };
        if (cell_center) {
          xi_field[0] = i1_ + HALF;
        } else {
          xi_field[0] = i1_;
        }

        if (flags & PrepareOutput::ConvertToHat) {
          metric.template transform<Idx::U, Idx::T>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCntrv) {
          metric.template transform<Idx::U, Idx::PU>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCov) {
          metric.template transform<Idx::D, Idx::PD>(xi_field, f_int, f_fin);
        } else {
          f_fin[0] = f_int[0];
          f_fin[1] = f_int[1];
          f_fin[2] = f_int[2];
        }
        Fto(i1, ct1) = f_fin[0];
        Fto(i1, ct2) = f_fin[1];
        Fto(i1, ct3) = f_fin[2];
      } else {
        raise::KernelError(
          HERE,
          "FieldsToPhys_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        real_t i1_ { COORD(i1) };
        real_t i2_ { COORD(i2) };

        vec_t<Dim::_3D> f_int { ZERO }, f_fin { ZERO };
        auto            cell_center = false;
        if (flags & PrepareOutput::InterpToCellCenterFromEdges) {
          f_int[0]    = INV_2 * (Ffrom(i1, i2, cf1) + Ffrom(i1, i2 + 1, cf1));
          f_int[1]    = INV_2 * (Ffrom(i1, i2, cf2) + Ffrom(i1 + 1, i2, cf2));
          f_int[2]    = INV_4 * (Ffrom(i1, i2, cf3) + Ffrom(i1 + 1, i2, cf3) +
                              Ffrom(i1, i2 + 1, cf3) + Ffrom(i1 + 1, i2 + 1, cf3));
          cell_center = true;
        } else if (flags & PrepareOutput::InterpToCellCenterFromFaces) {
          f_int[0]    = INV_2 * (Ffrom(i1, i2, cf1) + Ffrom(i1 + 1, i2, cf1));
          f_int[1]    = INV_2 * (Ffrom(i1, i2, cf2) + Ffrom(i1, i2 + 1, cf2));
          f_int[2]    = Ffrom(i1, i2, cf3);
          cell_center = true;
        } else {
          f_int[0] = Ffrom(i1, i2, cf1);
          f_int[1] = Ffrom(i1, i2, cf2);
          f_int[2] = Ffrom(i1, i2, cf3);
        }

        coord_t<Dim::_2D> xi_field { ZERO };
        if (cell_center) {
          xi_field[0] = i1_ + HALF;
          xi_field[1] = i2_ + HALF;
        } else {
          xi_field[0] = i1_;
          xi_field[1] = i2_;
        }

        if (flags & PrepareOutput::ConvertToHat) {
          metric.template transform<Idx::U, Idx::T>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCntrv) {
          metric.template transform<Idx::U, Idx::PU>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCov) {
          metric.template transform<Idx::D, Idx::PD>(xi_field, f_int, f_fin);
        } else {
          f_fin[0] = f_int[0];
          f_fin[1] = f_int[1];
          f_fin[2] = f_int[2];
        }
        Fto(i1, i2, ct1) = f_fin[0];
        Fto(i1, i2, ct2) = f_fin[1];
        Fto(i1, i2, ct3) = f_fin[2];
      } else {
        raise::KernelError(
          HERE,
          "FieldsToPhys_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        real_t i1_ { COORD(i1) };
        real_t i2_ { COORD(i2) };
        real_t i3_ { COORD(i3) };

        vec_t<Dim::_3D> f_int { ZERO }, f_fin { ZERO };
        auto            cell_center = false;
        if (flags & PrepareOutput::InterpToCellCenterFromEdges) {
          f_int[0] = INV_4 * (Ffrom(i1, i2, i3, cf1) + Ffrom(i1, i2 + 1, i3, cf1) +
                              Ffrom(i1, i2, i3 + 1, cf1) +
                              Ffrom(i1, i2 + 1, i3 + 1, cf1));
          f_int[1] = INV_4 * (Ffrom(i1, i2, i3, cf2) + Ffrom(i1 + 1, i2, i3, cf2) +
                              Ffrom(i1, i2, i3 + 1, cf2) +
                              Ffrom(i1 + 1, i2, i3 + 1, cf2));
          f_int[2] = INV_4 * (Ffrom(i1, i2, i3, cf3) + Ffrom(i1 + 1, i2, i3, cf3) +
                              Ffrom(i1, i2 + 1, i3, cf3) +
                              Ffrom(i1 + 1, i2 + 1, i3, cf3));
          cell_center = true;
        } else if (flags & PrepareOutput::InterpToCellCenterFromFaces) {
          f_int[0] = INV_2 * (Ffrom(i1, i2, i3, cf1) + Ffrom(i1 + 1, i2, i3, cf1));
          f_int[1] = INV_2 * (Ffrom(i1, i2, i3, cf2) + Ffrom(i1, i2 + 1, i3, cf2));
          f_int[2] = INV_2 * (Ffrom(i1, i2, i3, cf3) + Ffrom(i1, i2, i3 + 1, cf3));
          cell_center = true;
        } else {
          f_int[0] = Ffrom(i1, i2, i3, cf1);
          f_int[1] = Ffrom(i1, i2, i3, cf2);
          f_int[2] = Ffrom(i1, i2, i3, cf3);
        }

        coord_t<Dim::_3D> xi_field { ZERO };
        if (cell_center) {
          xi_field[0] = i1_ + HALF;
          xi_field[1] = i2_ + HALF;
          xi_field[2] = i3_ + HALF;
        } else {
          xi_field[0] = i1_;
          xi_field[1] = i2_;
          xi_field[2] = i3_;
        }

        if (flags & PrepareOutput::ConvertToHat) {
          metric.template transform<Idx::U, Idx::T>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCntrv) {
          metric.template transform<Idx::U, Idx::PU>(xi_field, f_int, f_fin);
        } else if (flags & PrepareOutput::ConvertToPhysCov) {
          metric.template transform<Idx::D, Idx::PD>(xi_field, f_int, f_fin);
        } else {
          f_fin[0] = f_int[0];
          f_fin[1] = f_int[1];
          f_fin[2] = f_int[2];
        }

        Fto(i1, i2, i3, ct1) = f_fin[0];
        Fto(i1, i2, i3, ct2) = f_fin[1];
        Fto(i1, i2, i3, ct3) = f_fin[2];
      } else {
        raise::KernelError(
          HERE,
          "FieldsToPhys_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_FIELDS_TO_PHYS_HPP
