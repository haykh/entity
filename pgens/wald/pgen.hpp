#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

#include <string>

enum InitFieldGeometry {
  Wald,
  Vertical,
};

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_, const std::string& init_field_geometry)
      : metric { metric_ } {
      if (init_field_geometry == "wald") {
        field_geometry = InitFieldGeometry::Wald;
      } else if (init_field_geometry == "vertical") {
        field_geometry = InitFieldGeometry::Vertical;
      } else {
        raise::Error(fmt::format("Unrecognized field geometry: %s",
                                 init_field_geometry.c_str()),
                     HERE);
      }
    }

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_1(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<1, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 1>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_0(const coord_t<D>& x_Cd) const -> real_t {
      real_t g_00 { -metric.alpha(x_Cd) * metric.alpha(x_Cd) +
                    metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd) *
                      metric.beta1(x_Cd) };
      return HALF * (metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd) +
                     TWO * metric.spin() * g_00);
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1])) {
        return ONE;
      } else {
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1])) {
        return ZERO;
      } else {
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
      }
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j + HALF )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

        x0m[0] = xi[0];
        x0m[1] = xi[1] - HALF;
        x0p[0] = xi[0];
        x0p[1] = xi[1] + HALF;

        real_t inv_sqrt_detH_iPjP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        return -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_iPjP;
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

        real_t alpha_iPj { metric.alpha({ xi[0], xi[1] }) };
        real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
        real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
        real_t beta_ij { metric.beta1({ xi[0] - HALF, xi[1] }) };
        real_t alpha_ij { metric.alpha({ xi[0] - HALF, xi[1] }) };

        // D1 at ( i + HALF , j )
        x0m[0] = xi[0] - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF;
        x0p[1] = xi[1];
        real_t E1d { (A_0(x0p) - A_0(x0m)) };
        real_t D1d { E1d / alpha_iPj };

        // D3 at ( i , j )
        x0m[0] = xi[0] - HALF - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] - HALF + HALF;
        x0p[1] = xi[1];
        real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij };

        real_t D1u { metric.template h<1, 1>({ xi[0], xi[1] }) * D1d +
                     metric.template h<1, 3>({ xi[0], xi[1] }) * D3d };

        return D1u;
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        x0m[0] = xi[0];
        x0m[1] = xi[1] - HALF;
        x0p[0] = xi[0];
        x0p[1] = xi[1] + HALF;
        real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t sqrt_detH_ijP { metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t alpha_ijP { metric.alpha({ xi[0], xi[1] }) };
        real_t beta_ijP { metric.beta1({ xi[0], xi[1] }) };

        real_t E2d { (A_0(x0p) - A_0(x0m)) };
        real_t D2d { E2d / alpha_ijP -
                     (A_1(x0p) - A_1(x0m)) * beta_ijP / alpha_ijP };
        real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

        return D2u;
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t beta_ij { metric.beta1({ xi[0], xi[1] }) };
        real_t alpha_ij { metric.alpha({ xi[0], xi[1] }) };
        real_t alpha_iPj { metric.alpha({ xi[0] + HALF, xi[1] }) };

        // D3 at ( i , j )
        x0m[0] = xi[0] - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF;
        x0p[1] = xi[1];
        real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij };

        // D1 at ( i + HALF , j )
        x0m[0] = xi[0] + HALF - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF + HALF;
        x0p[1] = xi[1];
        real_t E1d { (A_0(x0p) - A_0(x0m)) };
        real_t D1d { E1d / alpha_iPj };

        if (cmp::AlmostZero(x_Ph[1])) {
          return metric.template h<1, 3>({ xi[0], xi[1] }) * D1d;
        } else {
          return metric.template h<3, 3>({ xi[0], xi[1] }) * D3d +
                 metric.template h<1, 3>({ xi[0], xi[1] }) * D1d;
        }
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

  private:
    const M           metric;
    InitFieldGeometry field_geometry;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::GRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Kerr_Schild,
                                          Metric::QKerr_Schild,
                                          Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_2D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    InitFields<M, D>        init_flds;
    const Metadomain<S, M>& global_domain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { m }
      , init_flds { m.mesh().metric,
                    p.template get<std::string>("setup.init_field", "wald") } {}
  };

} // namespace user

#endif
