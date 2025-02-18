#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "framework/domain/domain.h"
#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"

#include <vector>

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_) : metric { metric_ } {}

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) 
             + TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd)
      );
      // coord_t<D> x_Ph { ZERO };
      // metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      // return HALF * SQR(x_Ph[0]) * SQR(math::sin(x_Ph[1]));
    }

    Inline auto A_1(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<1, 3>(x_Cd) 
             + TWO * metric.spin() * metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd)
      );
    }

    Inline auto A_0(const coord_t<D>& x_Cd) const -> real_t {
      real_t g_00 { -metric.alpha(x_Cd) * metric.alpha(x_Cd) 
                   + metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd) * metric.beta1(x_Cd) 
                  };
      return HALF * (metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd) 
             + TWO * metric.spin() * g_00);
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j + HALF )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;
      
      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1]))
        return ONE;
      else
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_iPj { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1]))
        return ZERO;
      else
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_iPj;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j + HALF )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;

      real_t inv_sqrt_detH_iPjP { ONE / metric.sqrt_det_h({ xi[0], xi[1]}) };
      return -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_iPjP;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
      
      real_t alpha_iPj { metric.alpha({ xi[0], xi[1] }) };
      real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
      real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
      real_t beta_ij { metric.beta1({ xi[0] - HALF, xi[1]}) };
      real_t alpha_ij { metric.alpha({ xi[0] - HALF, xi[1]}) };
      
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
      real_t B2_aux { -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ij };
      real_t D3d { -sqrt_detH_ij * beta_ij * B2_aux / alpha_ij };

      real_t D1u { metric.template h<1, 1>({ xi[0], xi[1] }) * D1d + metric.template h<1, 3>({ xi[0], xi[1] }) * D3d };

      return D1u;
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j + HALF )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
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
      real_t B3_aux { -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_ijP };
      real_t D2d { E2d / alpha_ijP + sqrt_detH_ijP * beta_ijP * B3_aux / alpha_ijP };
      real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

      return D2u;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j )
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
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
      real_t B2_aux { -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ij};
      real_t D3d { -sqrt_detH_ij * beta_ij * B2_aux / alpha_ij };

      // D1 at ( i + HALF , j )
      x0m[0] = xi[0] + HALF - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF + HALF;
      x0p[1] = xi[1];
      real_t E1d { (A_0(x0p) - A_0(x0m)) };
      real_t D1d { E1d / alpha_iPj };

      return metric.template h<3, 3>({ xi[0], xi[1] }) * D3d + metric.template h<1, 3>({ xi[0], xi[1] }) * D1d;
    }

  private:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::GRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    InitFields<M, D> init_flds;
    const Metadomain<S, M>& global_domain;
    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { m }
      , init_flds { m.mesh().metric } {}
    
    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto empty = std::vector<real_t> {};
      const auto x1s = params.template get<std::vector<real_t>>("setup.x1s", empty);
      const auto y1s = params.template get<std::vector<real_t>>("setup.y1s", empty);
      const auto z1s = params.template get<std::vector<real_t>>("setup.z1s", empty);
      const auto ux1s = params.template get<std::vector<real_t>>("setup.ux1s",
                                                                 empty);
      const auto uy1s = params.template get<std::vector<real_t>>("setup.uy1s",
                                                                 empty);
      const auto uz1s = params.template get<std::vector<real_t>>("setup.uz1s",
                                                                 empty);

      const auto x2s = params.template get<std::vector<real_t>>("setup.x2s", empty);
      const auto y2s = params.template get<std::vector<real_t>>("setup.y2s", empty);
      const auto z2s = params.template get<std::vector<real_t>>("setup.z2s", empty);
      const auto ux2s = params.template get<std::vector<real_t>>("setup.ux2s",
                                                                 empty);
      const auto uy2s = params.template get<std::vector<real_t>>("setup.uy2s",
                                                                 empty);
      const auto uz2s = params.template get<std::vector<real_t>>("setup.uz2s",
                                                                 empty);
      const std::map<std::string, std::vector<real_t>> data_1 {
        { "x1",  x1s},
        { "x2",  y1s},
        { "phi", z1s},
        {"ux1", ux1s},
        {"ux2", uy1s},
        {"ux3", uz1s}
      };
      const std::map<std::string, std::vector<real_t>> data_2 {
        { "x1",  x2s},
        { "x2",  y2s},
        { "phi", z2s},
        {"ux1", ux2s},
        {"ux2", uy2s},
        {"ux3", uz2s}
      };

      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)1, data_1);
      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)2, data_2);
    }
    // inline PGen() {}
  };

} // namespace user

#endif
