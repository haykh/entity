/**
 * @file kernels/fields_bcs.hpp
 * @brief Kernels used for field boundary conditions
 * @implements
 *   - kernel::bc::MatchBoundaries_kernel<>
 *   - kernel::bc::AxisBoundaries_kernel<>
 *   - kernel::bc::EnforcedBoundaries_kernel<>
 * @namespaces:
 *   - kernel::bc::
 */

#ifndef KERNELS_FIELDS_BCS_HPP
#define KERNELS_FIELDS_BCS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::bc {
  using namespace ntt;

  /*
   * @tparam S: Simulation Engine
   * @tparam I: Field Setter class
   * @tparam M: Metric
   * @tparam o: Orientation
   *
   * @brief Applies matching boundary conditions (with a smooth profile) in a specific direction.
   * @note If a component is not specified in the field setter, it is ignored.
   * @note It is supposed to only be called on the active side of the absorbing edge (so sign is not needed).
   */
  template <SimEngine::type S, class I, class M, in o>
  struct MatchBoundaries_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(static_cast<unsigned short>(o) <
                    static_cast<unsigned short>(M::Dim),
                  "Invalid component index");
    static constexpr idx_t i = static_cast<idx_t>(o) + 1u;
    static constexpr bool defines_dx1 = traits::has_method<traits::dx1_t, I>::value;
    static constexpr bool defines_dx2 = traits::has_method<traits::dx2_t, I>::value;
    static constexpr bool defines_dx3 = traits::has_method<traits::dx3_t, I>::value;
    static constexpr bool defines_ex1 = traits::has_method<traits::ex1_t, I>::value;
    static constexpr bool defines_ex2 = traits::has_method<traits::ex2_t, I>::value;
    static constexpr bool defines_ex3 = traits::has_method<traits::ex3_t, I>::value;
    static constexpr bool defines_bx1 = traits::has_method<traits::bx1_t, I>::value;
    static constexpr bool defines_bx2 = traits::has_method<traits::bx2_t, I>::value;
    static constexpr bool defines_bx3 = traits::has_method<traits::bx3_t, I>::value;
    static_assert(
      (S == SimEngine::SRPIC and (defines_ex1 or defines_ex2 or defines_ex3 or
                                  defines_bx1 or defines_bx2 or defines_bx3)) or
        ((S == SimEngine::GRPIC) and (defines_dx1 or defines_dx2 or defines_dx3 or
                                      defines_bx1 or defines_bx2 or defines_bx3)),
      "none of the components of E/D or B are specified in PGEN");

    ndfield_t<M::Dim, 6> Fld;
    const I              fset;
    const M              metric;
    const real_t         xg_edge;
    const real_t         dx_abs;
    const BCTags         tags;

    MatchBoundaries_kernel(ndfield_t<M::Dim, 6> Fld,
                           const I&             fset,
                           const M&             metric,
                           real_t               xg_edge,
                           real_t               dx_abs,
                           BCTags               tags)
      : Fld { Fld }
      , fset { fset }
      , metric { metric }
      , xg_edge { xg_edge }
      , dx_abs { dx_abs }
      , tags { tags } {}

    Inline auto shape(const real_t& dx) const -> real_t {
      return math::tanh(dx * FOUR / dx_abs);
    }

    Inline void operator()(index_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        const auto i1_ = COORD(i1);

        if constexpr (S == SimEngine::SRPIC) {
          coord_t<Dim::_1D> x_Ph_0 { ZERO };
          coord_t<Dim::_1D> x_Ph_H { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph_0);
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph_H);

          // SRPIC
          auto ex1_U { ZERO }, ex2_U { ZERO }, ex3_U { ZERO }, bx1_U { ZERO },
            bx2_U { ZERO }, bx3_U { ZERO };
          if (tags & BC::E) {
            if constexpr (defines_ex1) {
              ex1_U = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ + HALF },
                fset.ex1(x_Ph_H));
            }
            if constexpr (defines_ex2) {
              ex2_U = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ },
                fset.ex2(x_Ph_0));
            }
            if constexpr (defines_ex3) {
              ex3_U = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ },
                fset.ex3(x_Ph_0));
            }
          }
          if (tags & BC::B) {
            if constexpr (defines_bx1) {
              bx1_U = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ },
                fset.bx1(x_Ph_0));
            }
            if constexpr (defines_bx2) {
              bx2_U = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ + HALF },
                fset.bx2(x_Ph_H));
            }
            if constexpr (defines_bx3) {
              bx3_U = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ + HALF },
                fset.bx3(x_Ph_H));
            }
          }

          if constexpr (defines_ex1 or defines_bx2 or defines_bx3) {
            const auto dx = math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(i1_ + HALF) - xg_edge);
            const auto s = shape(dx);
            if constexpr (defines_ex1) {
              if (tags & BC::E) {
                Fld(i1, em::ex1) = s * Fld(i1, em::ex1) + (ONE - s) * ex1_U;
              }
            }
            if constexpr (defines_bx2 or defines_bx3) {
              if (tags & BC::B) {
                if constexpr (defines_bx2) {
                  Fld(i1, em::bx2) = s * Fld(i1, em::bx2) + (ONE - s) * bx2_U;
                }
                if constexpr (defines_bx3) {
                  Fld(i1, em::bx3) = s * Fld(i1, em::bx3) + (ONE - s) * bx3_U;
                }
              }
            }
          }
          if constexpr (defines_bx1 or defines_ex2 or defines_ex3) {
            const auto dx = math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(i1_) - xg_edge);
            const auto s = shape(dx);
            if constexpr (defines_bx1) {
              if (tags & BC::B) {
                Fld(i1, em::bx1) = s * Fld(i1, em::bx1) + (ONE - s) * bx1_U;
              }
            }
            if constexpr (defines_ex2 or defines_ex3) {
              if (tags & BC::E) {
                if constexpr (defines_ex2) {
                  Fld(i1, em::ex2) = s * Fld(i1, em::ex2) + (ONE - s) * ex2_U;
                }
                if constexpr (defines_ex3) {
                  Fld(i1, em::ex3) = s * Fld(i1, em::ex3) + (ONE - s) * ex3_U;
                }
              }
            }
          }
        } else {
          // GRPIC
          raise::KernelError(HERE, "1D GRPIC not implemented");
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);

        if constexpr (S == SimEngine::SRPIC) {
          // SRPIC
          if constexpr (defines_ex1 or defines_bx2) {
            coord_t<Dim::_2D> x_Ph_H0 { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Ph_H0);
            // i1 + 1/2, i2
            real_t xi_Cd;
            if constexpr (o == in::x1) {
              xi_Cd = i1_ + HALF;
            } else {
              xi_Cd = i2_;
            }

            const auto dx = math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
            const auto s = shape(dx);

            if constexpr (defines_ex1) {
              if (tags & BC::E) {
                const auto ex1_U = metric.template transform<1, Idx::T, Idx::U>(
                  { i1_ + HALF, i2_ },
                  fset.ex1(x_Ph_H0));
                Fld(i1, i2, em::ex1) = s * Fld(i1, i2, em::ex1) + (ONE - s) * ex1_U;
              }
            }
            if constexpr (defines_bx2) {
              if (tags & BC::B) {
                const auto bx2_U = metric.template transform<2, Idx::T, Idx::U>(
                  { i1_ + HALF, i2_ },
                  fset.bx2(x_Ph_H0));
                Fld(i1, i2, em::bx2) = s * Fld(i1, i2, em::bx2) + (ONE - s) * bx2_U;
              }
            }
          }

          if constexpr (defines_ex2 or defines_bx1) {
            coord_t<Dim::_2D> x_Ph_0H { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Ph_0H);
            // i1, i2 + 1/2
            real_t xi_Cd;
            if constexpr (o == in::x1) {
              xi_Cd = i1_;
            } else {
              xi_Cd = i2_ + HALF;
            }

            const auto dx = math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
            const auto s = shape(dx);
            if constexpr (defines_ex2) {
              if (tags & BC::E) {
                auto ex2_U { ZERO };
                ex2_U = metric.template transform<2, Idx::T, Idx::U>(
                  { i1_, i2_ + HALF },
                  fset.ex2(x_Ph_0H));
                Fld(i1, i2, em::ex2) = s * Fld(i1, i2, em::ex2) + (ONE - s) * ex2_U;
              }
            }
            if constexpr (defines_bx1) {
              if (tags & BC::B) {
                auto bx1_U { ZERO };
                bx1_U = metric.template transform<1, Idx::T, Idx::U>(
                  { i1_, i2_ + HALF },
                  fset.bx1(x_Ph_0H));
                Fld(i1, i2, em::bx1) = s * Fld(i1, i2, em::bx1) + (ONE - s) * bx1_U;
              }
            }
          }

          if constexpr (defines_ex3) {
            if (tags & BC::E) {
              auto              ex3_U { ZERO };
              coord_t<Dim::_2D> x_Ph_00 { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Ph_00);
              ex3_U = metric.template transform<3, Idx::T, Idx::U>(
                { i1_, i2_ },
                fset.ex3(x_Ph_00));
              // i1, i2
              real_t xi_Cd;
              if constexpr (o == in::x1) {
                xi_Cd = i1_;
              } else {
                xi_Cd = i2_;
              }
              const auto dx = math::abs(
                metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
              const auto s = shape(dx);
              Fld(i1, i2, em::ex3) = s * Fld(i1, i2, em::ex3) + (ONE - s) * ex3_U;
            }
          }

          if constexpr (defines_bx3) {
            if (tags & BC::B) {
              auto              bx3_U { ZERO };
              coord_t<Dim::_2D> x_Ph_HH { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                        x_Ph_HH);
              bx3_U = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ + HALF, i2_ + HALF },
                fset.bx3(x_Ph_HH));
              // i1 + 1/2, i2 + 1/2
              real_t xi_Cd;
              if constexpr (o == in::x1) {
                xi_Cd = i1_ + HALF;
              } else {
                xi_Cd = i2_ + HALF;
              }
              const auto dx = math::abs(
                metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
              const auto s = shape(dx);
              // bx3
              Fld(i1, i2, em::bx3) = s * Fld(i1, i2, em::bx3) + (ONE - s) * bx3_U;
            }
          }
        } else {
          // GRPIC
          raise::KernelError(HERE, "GRPIC not implemented");
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        const auto i3_ = COORD(i3);

        if constexpr (S == SimEngine::SRPIC) {
          // SRPIC
          if constexpr (defines_ex1 or defines_ex2 or defines_ex3) {
            if (tags & BC::E) {
              if constexpr (defines_ex1) {
                // i1 + 1/2, i2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto        s = shape(dx);
                auto              ex1_U { ZERO };
                coord_t<Dim::_3D> x_Ph_H00 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                          x_Ph_H00);
                ex1_U = metric.template transform<1, Idx::T, Idx::U>(
                  { i1_ + HALF, i2_, i3_ },
                  fset.ex1(x_Ph_H00));
                Fld(i1, i2, i3, em::ex1) = s * Fld(i1, i2, i3, em::ex1) +
                                           (ONE - s) * ex1_U;
              }

              if constexpr (defines_ex2) {
                // i1, i2 + 1/2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto        s = shape(dx);
                auto              ex2_U { ZERO };
                coord_t<Dim::_3D> x_Ph_0H0 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                          x_Ph_0H0);
                ex2_U = metric.template transform<2, Idx::T, Idx::U>(
                  { i1_, i2_ + HALF, i3_ },
                  fset.ex2(x_Ph_0H0));
                Fld(i1, i2, i3, em::ex2) = s * Fld(i1, i2, i3, em::ex2) +
                                           (ONE - s) * ex2_U;
              }

              if constexpr (defines_ex3) {
                // i1, i2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto        s = shape(dx);
                auto              ex3_U { ZERO };
                coord_t<Dim::_3D> x_Ph_00H { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                          x_Ph_00H);
                ex3_U = metric.template transform<3, Idx::T, Idx::U>(
                  { i1_, i2_, i3_ + HALF },
                  fset.ex3(x_Ph_00H));
                Fld(i1, i2, i3, em::ex3) = s * Fld(i1, i2, i3, em::ex3) +
                                           (ONE - s) * ex3_U;
              }
            }
          }

          if constexpr (defines_bx1 or defines_bx2 or defines_bx3) {
            if (tags & BC::B) {
              if constexpr (defines_bx1) {
                // i1, i2 + 1/2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto s = shape(dx);
                auto       bx1_U { ZERO };
                if constexpr (defines_bx1) {
                  coord_t<Dim::_3D> x_Ph_0HH { ZERO };
                  metric.template convert<Crd::Cd, Crd::Ph>(
                    { i1_, i2_ + HALF, i3_ + HALF },
                    x_Ph_0HH);
                  bx1_U = metric.template transform<1, Idx::T, Idx::U>(
                    { i1_, i2_ + HALF, i3_ + HALF },
                    fset.bx1(x_Ph_0HH));
                }
                // bx1
                Fld(i1, i2, i3, em::bx1) = s * Fld(i1, i2, i3, em::bx1) +
                                           (ONE - s) * bx1_U;
              }

              if constexpr (defines_bx2) {
                // i1 + 1/2, i2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto        s = shape(dx);
                auto              bx2_U { ZERO };
                coord_t<Dim::_3D> x_Ph_H0H { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>(
                  { i1_ + HALF, i2_, i3_ + HALF },
                  x_Ph_H0H);
                bx2_U = metric.template transform<2, Idx::T, Idx::U>(
                  { i1_ + HALF, i2_, i3_ + HALF },
                  fset.bx2(x_Ph_H0H));
                Fld(i1, i2, i3, em::bx2) = s * Fld(i1, i2, i3, em::bx2) +
                                           (ONE - s) * bx2_U;
              }

              if constexpr (defines_bx3) {
                // i1 + 1/2, i2 + 1/2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_;
                }
                const auto dx = math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge);
                const auto        s = shape(dx);
                auto              bx3_U { ZERO };
                coord_t<Dim::_3D> x_Ph_HH0 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>(
                  { i1_ + HALF, i2_ + HALF, i3_ },
                  x_Ph_HH0);
                bx3_U = metric.template transform<3, Idx::T, Idx::U>(
                  { i1_ + HALF, i2_ + HALF, i3_ },
                  fset.bx3(x_Ph_HH0));
                Fld(i1, i2, i3, em::bx3) = s * Fld(i1, i2, i3, em::bx3) +
                                           (ONE - s) * bx3_U;
              }
            }
          }
        } else {
          // GRPIC
          raise::KernelError(HERE, "GRPIC not implemented");
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /*
   * @tparam D: Dimension
   * @tparam P: Positive/Negative direction
   *
   * @brief Applies boundary conditions near the polar axis
   */
  template <Dimension D, bool P>
  struct AxisBoundaries_kernel {
    ndfield_t<D, 6>   Fld;
    const std::size_t i_edge;
    const bool        setE, setB;

    AxisBoundaries_kernel(ndfield_t<D, 6> Fld, std::size_t i_edge, BCTags tags)
      : Fld { Fld }
      , i_edge { i_edge }
      , setE { tags & BC::Ex1 or tags & BC::Ex2 or tags & BC::Ex3 }
      , setB { tags & BC::Bx1 or tags & BC::Bx2 or tags & BC::Bx3 } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (not P) {
          if (setE) {
            Fld(i1, i_edge - 1, em::ex2) = -Fld(i1, i_edge, em::ex2);
            Fld(i1, i_edge, em::ex3)     = ZERO;
          }
          if (setB) {
            Fld(i1, i_edge - 1, em::bx1) = Fld(i1, i_edge, em::bx1);
            Fld(i1, i_edge, em::bx2)     = ZERO;
            Fld(i1, i_edge - 1, em::bx3) = Fld(i1, i_edge, em::bx3);
          }
        } else {
          if (setE) {
            Fld(i1, i_edge, em::ex2) = -Fld(i1, i_edge - 1, em::ex2);
            Fld(i1, i_edge, em::ex3) = ZERO;
          }
          if (setB) {
            Fld(i1, i_edge, em::bx1) = Fld(i1, i_edge - 1, em::bx1);
            Fld(i1, i_edge, em::bx2) = ZERO;
            Fld(i1, i_edge, em::bx3) = Fld(i1, i_edge - 1, em::bx3);
          }
        }
      } else {
        raise::KernelError(HERE, "AxisBoundaries_kernel: D != 2");
      }
    }
  };

  /*
   * @tparam I: Field Setter class
   * @tparam M: Metric
   * @tparam P: Positive/Negative direction
   * @tparam O: Orientation
   *
   * @brief Applies enforced boundary conditions (fixed value)
   */
  template <class I, class M, bool P, in O>
  struct EnforcedBoundaries_kernel {
    static constexpr Dimension D = M::Dim;
    static constexpr bool defines_ex1 = traits::has_method<traits::ex1_t, I>::value;
    static constexpr bool defines_ex2 = traits::has_method<traits::ex2_t, I>::value;
    static constexpr bool defines_ex3 = traits::has_method<traits::ex3_t, I>::value;
    static constexpr bool defines_bx1 = traits::has_method<traits::bx1_t, I>::value;
    static constexpr bool defines_bx2 = traits::has_method<traits::bx2_t, I>::value;
    static constexpr bool defines_bx3 = traits::has_method<traits::bx3_t, I>::value;

    static_assert(defines_ex1 or defines_ex2 or defines_ex3 or defines_bx1 or
                    defines_bx2 or defines_bx3,
                  "none of the components of E or B are specified in PGEN");
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(static_cast<unsigned short>(O) <
                    static_cast<unsigned short>(M::Dim),
                  "Invalid Orientation");

    ndfield_t<D, 6>   Fld;
    const I           fset;
    const M           metric;
    const std::size_t i_edge;

    EnforcedBoundaries_kernel(ndfield_t<M::Dim, 6>& Fld,
                              const I&              fset,
                              const M&              metric,
                              std::size_t           i_edge,
                              BCTags                tags)
      : Fld { Fld }
      , fset { fset }
      , metric { metric }
      , i_edge { i_edge + N_GHOSTS } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Ph_0 { ZERO };
        coord_t<Dim::_1D> x_Ph_H { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph_0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph_H);
        bool setEx1 = defines_ex1, setEx2 = defines_ex2, setEx3 = defines_ex3,
             setBx1 = defines_bx1, setBx2 = defines_bx2, setBx3 = defines_bx3;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if constexpr (defines_ex1) {
          if (setEx1) {
            Fld(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.ex1(x_Ph_H));
          }
        }
        if constexpr (defines_ex2) {
          if (setEx2) {
            Fld(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ },
              fset.ex2(x_Ph_0));
          }
        }
        if constexpr (defines_ex3) {
          if (setEx3) {
            Fld(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ },
              fset.ex3(x_Ph_0));
          }
        }
        if constexpr (defines_bx1) {
          if (setBx1) {
            Fld(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ },
              fset.bx1(x_Ph_0));
          }
        }
        if constexpr (defines_bx2) {
          if (setBx2) {
            Fld(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.bx2(x_Ph_H));
          }
        }
        if constexpr (defines_bx3) {
          if (setBx3) {
            Fld(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.bx3(x_Ph_H));
          }
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        coord_t<Dim::_2D> x_Ph_00 { ZERO };
        coord_t<Dim::_2D> x_Ph_0H { ZERO };
        coord_t<Dim::_2D> x_Ph_H0 { ZERO };
        coord_t<Dim::_2D> x_Ph_HH { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Ph_00);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Ph_0H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Ph_H0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                  x_Ph_HH);
        bool setEx1 = defines_ex1, setEx2 = defines_ex2, setEx3 = defines_ex3,
             setBx1 = defines_bx1, setBx2 = defines_bx2, setBx3 = defines_bx3;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else if (O == in::x2) {
          // x2 -- normal
          // x1,x3 -- tangential
          if constexpr (P) {
            setEx2 &= (i2 >= i_edge);
            setBx1 &= (i2 >= i_edge);
            setBx3 &= (i2 >= i_edge);
          } else {
            setEx2 &= (i2 < i_edge);
            setBx1 &= (i2 < i_edge);
            setBx3 &= (i2 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if constexpr (defines_ex1) {
          if (setEx1) {
            Fld(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              fset.ex1(x_Ph_H0));
          }
        }
        if constexpr (defines_ex2) {
          if (setEx2) {
            Fld(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              fset.ex2(x_Ph_0H));
          }
        }
        if constexpr (defines_ex3) {
          if (setEx3) {
            Fld(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_ },
              fset.ex3(x_Ph_00));
          }
        }
        if constexpr (defines_bx1) {
          if (setBx1) {
            Fld(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              fset.bx1(x_Ph_0H));
          }
        }
        if constexpr (defines_bx2) {
          if (setBx2) {
            Fld(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              fset.bx2(x_Ph_H0));
          }
        }
        if constexpr (defines_bx3) {
          if (setBx3) {
            Fld(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF },
              fset.bx3(x_Ph_HH));
          }
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
        coord_t<Dim::_3D> x_Ph_00H { ZERO };
        coord_t<Dim::_3D> x_Ph_0H0 { ZERO };
        coord_t<Dim::_3D> x_Ph_H00 { ZERO };
        coord_t<Dim::_3D> x_Ph_HH0 { ZERO };
        coord_t<Dim::_3D> x_Ph_H0H { ZERO };
        coord_t<Dim::_3D> x_Ph_0HH { ZERO };

        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                  x_Ph_00H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                  x_Ph_0H0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                  x_Ph_H00);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                  x_Ph_HH0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ + HALF },
                                                  x_Ph_H0H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ + HALF },
                                                  x_Ph_0HH);
        bool setEx1 = defines_ex1, setEx2 = defines_ex2, setEx3 = defines_ex3,
             setBx1 = defines_bx1, setBx2 = defines_bx2, setBx3 = defines_bx3;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else if (O == in::x2) {
          // x2 -- normal
          // x1,x3 -- tangential
          if constexpr (P) {
            setEx2 &= (i2 >= i_edge);
            setBx1 &= (i2 >= i_edge);
            setBx3 &= (i2 >= i_edge);
          } else {
            setEx2 &= (i2 < i_edge);
            setBx1 &= (i2 < i_edge);
            setBx3 &= (i2 < i_edge);
          }
        } else if (O == in::x3) {
          // x3 -- normal
          // x1,x2 -- tangential
          if constexpr (P) {
            setEx3 &= (i3 >= i_edge);
            setBx1 &= (i3 >= i_edge);
            setBx2 &= (i3 >= i_edge);
          } else {
            setEx3 &= (i3 < i_edge);
            setBx1 &= (i3 < i_edge);
            setBx2 &= (i3 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if constexpr (defines_ex1) {
          if (setEx1) {
            Fld(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ },
              fset.ex1(x_Ph_H00));
          }
        }
        if constexpr (defines_ex2) {
          if (setEx2) {
            Fld(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ },
              fset.ex2(x_Ph_0H0));
          }
        }
        if constexpr (defines_ex3) {
          if (setEx3) {
            Fld(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_, i3_ + HALF },
              fset.ex3(x_Ph_00H));
          }
        }
        if constexpr (defines_bx1) {
          if (setBx1) {
            Fld(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ + HALF },
              fset.bx1(x_Ph_0HH));
          }
        }
        if constexpr (defines_bx2) {
          if (setBx2) {
            Fld(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ + HALF },
              fset.bx2(x_Ph_H0H));
          }
        }
        if constexpr (defines_bx3) {
          if (setBx3) {
            Fld(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              fset.bx3(x_Ph_HH0));
          }
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

} // namespace kernel::bc

#endif // KERNELS_FIELDS_BCS_HPP
