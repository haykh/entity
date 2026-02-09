#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

#include <plog/Log.h>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC, SimEngine::GRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski,
                                          Metric::Spherical,
                                          Metric::QSpherical,
                                          Metric::Kerr_Schild,
                                          Metric::QKerr_Schild,
                                          Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M> { p } {
      const auto message = fmt::format(
        "Problem generator initialized with `%s` engine and `%dD %s` metric",
        SimEngine(S).to_string(),
        D,
        Metric(M::MetricType).to_string());
      PLOGI << message;
    }
  };

} // namespace user

#endif
