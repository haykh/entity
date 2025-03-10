#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "archetypes/problem_generator.h"

namespace user {
  using namespace ntt;

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

    inline PGen(SimulationParams& p, const Metadomain<S, M>&) :
      arch::ProblemGenerator<S, M>(p) {}

    inline PGen() {}
  };

} // namespace user

#endif
