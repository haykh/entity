/**
 * @file kernels/prtls_to_phys.hpp
 * @brief Convert particle positions & velocities to physical units
 * @implements
 *   - kernel::PrtlToPhys_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 * This is a kernel that converts particle positions
 * and velocities to physical units
 * @note SR : to the corresponding tetrad basis
 * @note GR : x -- coordinate basis, u -- covariant basis
 */

#ifndef KERNELS_PRTLS_TO_PHYS_HPP
#define KERNELS_PRTLS_TO_PHYS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine::type S, class M, bool T>
  class PrtlToPhys_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr Dimension D = M::Dim;

  protected:
    const npart_t            stride;
    array_t<npart_t*>        out_indices;
    array_t<real_t*>         buff_x1;
    array_t<real_t*>         buff_x2;
    array_t<real_t*>         buff_x3;
    array_t<real_t*>         buff_ux1;
    array_t<real_t*>         buff_ux2;
    array_t<real_t*>         buff_ux3;
    array_t<real_t*>         buff_wei;
    array_t<real_t**>        buff_pldr;
    array_t<npart_t**>       buff_pldi;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<real_t**>  pld_r;
    const array_t<npart_t**> pld_i;
    const M                  metric;

  public:
    PrtlToPhys_kernel(npart_t                   stride,
                      array_t<npart_t*>         out_indices,
                      array_t<real_t*>&         buff_x1,
                      array_t<real_t*>&         buff_x2,
                      array_t<real_t*>&         buff_x3,
                      array_t<real_t*>&         buff_ux1,
                      array_t<real_t*>&         buff_ux2,
                      array_t<real_t*>&         buff_ux3,
                      array_t<real_t*>&         buff_wei,
                      array_t<real_t**>&        buff_pldr,
                      array_t<npart_t**>&       buff_pldi,
                      const array_t<int*>&      i1,
                      const array_t<int*>&      i2,
                      const array_t<int*>&      i3,
                      const array_t<prtldx_t*>& dx1,
                      const array_t<prtldx_t*>& dx2,
                      const array_t<prtldx_t*>& dx3,
                      const array_t<real_t*>&   ux1,
                      const array_t<real_t*>&   ux2,
                      const array_t<real_t*>&   ux3,
                      const array_t<real_t*>&   phi,
                      const array_t<real_t*>&   weight,
                      const array_t<real_t**>&  pld_r,
                      const array_t<npart_t**>& pld_i,
                      const M&                  metric)
      : stride { stride }
      , out_indices { out_indices }
      , buff_x1 { buff_x1 }
      , buff_x2 { buff_x2 }
      , buff_x3 { buff_x3 }
      , buff_ux1 { buff_ux1 }
      , buff_ux2 { buff_ux2 }
      , buff_ux3 { buff_ux3 }
      , buff_wei { buff_wei }
      , buff_pldr { buff_pldr }
      , buff_pldi { buff_pldi }
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
      , pld_r { pld_r }
      , pld_i { pld_i }
      , metric { metric } {
      if constexpr ((D == Dim::_1D) || (D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(buff_x1.extent(0) == 0, "Invalid buffer size", HERE);
      }
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(buff_x2.extent(0) == 0, "Invalid buffer size", HERE);
      }
      if constexpr (((D == Dim::_2D) && (M::CoordType != Coord::Cart)) ||
                    (D == Dim::_3D)) {
        raise::ErrorIf(buff_x3.extent(0) == 0, "Invalid buffer size", HERE);
      }
      raise::ErrorIf(buff_ux1.extent(0) == 0, "Invalid buffer size", HERE);
      raise::ErrorIf(buff_ux2.extent(0) == 0, "Invalid buffer size", HERE);
      raise::ErrorIf(buff_ux3.extent(0) == 0, "Invalid buffer size", HERE);
    }

    Inline void operator()(index_t p) const {
      if constexpr (not T) { // no tracking enabled
        bufferX(p * stride, p);
        bufferU(p * stride, p);
        buff_wei(p) = weight(p * stride);
        bufferPlds(p * stride, p);
      } else {
        bufferX(out_indices(p), p);
        bufferU(out_indices(p), p);
        buff_wei(p) = weight(out_indices(p));
        bufferPlds(out_indices(p), p);
      }
    }

    Inline void bufferX(index_t p_from, index_t p_to) const {
      if constexpr ((D == Dim::_1D) || (D == Dim::_2D) || (D == Dim::_3D)) {
        buff_x1(p_to) = metric.template convert<1, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)));
      }
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        buff_x2(p_to) = metric.template convert<2, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)));
      }
      if constexpr ((D == Dim::_2D) && (M::CoordType != Coord::Cart)) {
        buff_x3(p_to) = phi(p_from);
      } else if constexpr (D == Dim::_3D) {
        buff_x3(p_to) = metric.template convert<3, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i3(p_from)) + static_cast<real_t>(dx3(p_from)));
      }
    }

    Inline void bufferU(index_t p_from, index_t p_to) const {
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (D == Dim::_1D) {
        if constexpr (M::CoordType == Coord::Cart) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unsupported coordinate system in 1D");
        }
      } else if constexpr (D == Dim::_2D) {
        if constexpr (M::CoordType == Coord::Cart) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)),
              static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)),
              static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)),
              phi(p_from) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else if constexpr (S == SimEngine::GRPIC) {
          metric.template transform<Idx::D, Idx::PD>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)),
              static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unrecognized simulation engine");
        }
      } else if constexpr (D == Dim::_3D) {
        if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)),
              static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)),
              static_cast<real_t>(i3(p_from)) + static_cast<real_t>(dx3(p_from)) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else if constexpr (S == SimEngine::GRPIC) {
          metric.template transform<Idx::D, Idx::PD>(
            { static_cast<real_t>(i1(p_from)) + static_cast<real_t>(dx1(p_from)),
              static_cast<real_t>(i2(p_from)) + static_cast<real_t>(dx2(p_from)),
              static_cast<real_t>(i3(p_from)) + static_cast<real_t>(dx3(p_from)) },
            { ux1(p_from), ux2(p_from), ux3(p_from) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unrecognized simulation engine");
        }
      }
      buff_ux1(p_to) = u_Phys[0];
      buff_ux2(p_to) = u_Phys[1];
      buff_ux3(p_to) = u_Phys[2];
    }

    Inline void bufferPlds(index_t p_from, index_t p_to) const {
      for (auto pr { 0u }; pr < buff_pldr.extent(1); ++pr) {
        buff_pldr(p_to, pr) = pld_r(p_from, pr);
      }
      for (auto pi { 0u }; pi < buff_pldi.extent(1); ++pi) {
        buff_pldi(p_to, pi) = pld_i(p_from, pi);
      }
    }
  };
} // namespace kernel

#endif // KERNELS_PRTLS_TO_PHYS_HPP
