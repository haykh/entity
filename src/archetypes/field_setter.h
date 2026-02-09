/**
 * @file archetypes/field_setter.hpp
 * @brief Defines a kernel which populates the EM fields with user-defined values
 * @implements
 *   - arch::SetEMFields_kernel
 * @namespaces:
 *   - arch::
 * @note
 * The functor accepts a class I as a template argument, which must contain
 * one of the ex1, ex2, ex3, bx1, bx2, bx3 methods (dx1, dx2, dx3 for GR).
 * * `coord_t<D>` --> [ I ] --> `real_t` --> [ SetEMFields_kernel ] --> ...
 * *      ^                        ^                                     ^
 * *  (physical)           (SR: hatted basis )                  (SR & GR: cntrv)
 * *                       (GR: phys. cntrv  )
 * @note
 * The functor automatically takes care of the staggering, ghost cells and
 * conversion to contravariant basis.
 */

#ifndef ARCHETYPES_FIELD_SETTER_HPP
#define ARCHETYPES_FIELD_SETTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include <Kokkos_Core.hpp>

namespace arch {
  using namespace ntt;

  template <class I, SimEngine::type S, class M>
    requires metric::traits::HasD<M> &&
             ((S == SimEngine::SRPIC && metric::traits::HasConvert<M> &&
               metric::traits::HasTransform_i<M>) ||
              (S == SimEngine::GRPIC && metric::traits::HasConvert_i<M>)) &&
             (S == SimEngine::SRPIC &&
                (::traits::fieldsetter::HasEx1<I, M::Dim> ||
                 ::traits::fieldsetter::HasEx2<I, M::Dim> ||
                 ::traits::fieldsetter::HasEx3<I, M::Dim> ||
                 ::traits::fieldsetter::HasBx1<I, M::Dim> ||
                 ::traits::fieldsetter::HasBx2<I, M::Dim> ||
                 ::traits::fieldsetter::HasBx3<I, M::Dim>) ||
              (S == SimEngine::GRPIC &&
                 (::traits::fieldsetter::HasDx1<I, M::Dim> &&
                  ::traits::fieldsetter::HasDx2<I, M::Dim> &&
                  ::traits::fieldsetter::HasDx3<I, M::Dim>) ||
               (::traits::fieldsetter::HasBx1<I, M::Dim> &&
                ::traits::fieldsetter::HasBx2<I, M::Dim> &&
                ::traits::fieldsetter::HasBx3<I, M::Dim>)))
  class SetEMFields_kernel {
    static constexpr Dimension D = M::Dim;

    ndfield_t<M::Dim, 6> EM;
    const I              finit;
    const M              metric;

  public:
    SetEMFields_kernel(ndfield_t<M::Dim, 6>& EM, const I& finit, const M& metric)
      : EM { EM }
      , finit { finit }
      , metric { metric } {}

    ~SetEMFields_kernel() = default;

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Phys { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          if constexpr (::traits::fieldsetter::HasEx1<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.ex1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx2<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ },
              finit.ex2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx3<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ },
              finit.ex3(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx1<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ },
              finit.bx1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx2<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.bx2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx3<I, Dim::_1D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.bx3(x_Phys));
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        // srpic
        if constexpr (S == SimEngine::SRPIC) {
          coord_t<Dim::_2D> x_Phys { ZERO };
          if constexpr (::traits::fieldsetter::HasEx1<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
            EM(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              finit.ex1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx2<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
            EM(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              finit.ex2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx3<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Phys);
            EM(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_ },
              finit.ex3(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx1<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
            EM(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              finit.bx1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx2<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
            EM(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              finit.bx2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx3<I, Dim::_2D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                      x_Phys);
            EM(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF },
              finit.bx3(x_Phys));
          }
        } else if constexpr (S == SimEngine::GRPIC) {
          // grpic
          if constexpr (::traits::fieldsetter::HasDx1<I, Dim::_2D> &&
                        ::traits::fieldsetter::HasDx2<I, Dim::_2D> &&
                        ::traits::fieldsetter::HasDx3<I, Dim::_2D>) {
            const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
            const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
              i1_ + HALF) };
            const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
            const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
              i2_ + HALF) };
            { // dx1
              EM(i1, i2, em::dx1) = finit.dx1({ x1_H, x2_0 });
            }
            { // dx2
              EM(i1, i2, em::dx2) = finit.dx2({ x1_0, x2_H });
            }
            { // dx3
              EM(i1, i2, em::dx3) = finit.dx3({ x1_0, x2_0 });
            }
          }
          if constexpr (::traits::fieldsetter::HasBx1<I, Dim::_2D> &&
                        ::traits::fieldsetter::HasBx2<I, Dim::_2D> &&
                        ::traits::fieldsetter::HasBx3<I, Dim::_2D>) {
            const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
            const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
              i1_ + HALF) };
            const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
            const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
              i2_ + HALF) };
            { // bx1
              EM(i1, i2, em::bx1) = finit.bx1({ x1_0, x2_H });
            }
            { // bx2
              EM(i1, i2, em::bx2) = finit.bx2({ x1_H, x2_0 });
            }
            { // bx3
              EM(i1, i2, em::bx3) = finit.bx3({ x1_H, x2_H });
            }
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        const auto        i3_ = COORD(i3);
        coord_t<Dim::_3D> x_Phys { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          // srpic
          if constexpr (::traits::fieldsetter::HasEx1<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ },
              finit.ex1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx2<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ },
              finit.ex2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasEx3<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_, i3_ + HALF },
              finit.ex3(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx1<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_, i2_ + HALF, i3_ + HALF },
              x_Phys);
            EM(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ + HALF },
              finit.bx1(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx2<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_ + HALF, i2_, i3_ + HALF },
              x_Phys);
            EM(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ + HALF },
              finit.bx2(x_Phys));
          }
          if constexpr (::traits::fieldsetter::HasBx3<I, Dim::_3D>) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              x_Phys);
            EM(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              finit.bx3(x_Phys));
          }
        } else if constexpr (S == SimEngine::GRPIC) {
          // grpic
          const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
          const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
            i1_ + HALF) };
          const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
          const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
            i2_ + HALF) };
          const real_t x3_0 { metric.template convert<3, Crd::Cd, Crd::Ph>(i3_) };
          const real_t x3_H { metric.template convert<3, Crd::Cd, Crd::Ph>(
            i3_ + HALF) };

          if constexpr (::traits::fieldsetter::HasDx1<I, Dim::_3D> &&
                        ::traits::fieldsetter::HasDx2<I, Dim::_3D> &&
                        ::traits::fieldsetter::HasDx3<I, Dim::_3D>) {
            { // dx1
              EM(i1, i2, i3, em::dx1) = finit.dx1({ x1_H, x2_0, x3_0 });
            }
            { // dx2
              EM(i1, i2, i3, em::dx2) = finit.dx2({ x1_0, x2_H, x3_0 });
            }
            { // dx3
              EM(i1, i2, i3, em::dx3) = finit.dx3({ x1_0, x2_0, x3_H });
            }
          }
          if constexpr (::traits::fieldsetter::HasBx1<I, Dim::_3D> &&
                        ::traits::fieldsetter::HasBx2<I, Dim::_3D> &&
                        ::traits::fieldsetter::HasBx3<I, Dim::_3D>) {
            { // bx1
              EM(i1, i2, i3, em::bx1) = finit.bx1({ x1_0, x2_H, x3_H });
            }
            { // bx2
              EM(i1, i2, i3, em::bx2) = finit.bx2({ x1_H, x2_0, x3_H });
            }
            { // bx3
              EM(i1, i2, i3, em::bx3) = finit.bx3({ x1_H, x2_H, x3_0 });
            }
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

} // namespace arch

#endif // ARCHETYPES_FIELD_SETTER_HPP
