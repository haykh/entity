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

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
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
