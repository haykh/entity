#include "enums.h"

#include "arch/traits.h"
#include "utils/error.h"

#include "framework/simulation.h"
#include "framework/specialization_registry.h"

#include "engines/engine_traits.h"

#include "pgen.hpp"

#include <type_traits>

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
static constexpr bool should_compile {
  traits::check_compatibility<S>::value(user::PGen<S, M<D>>::engines) &&
  traits::check_compatibility<M<D>::MetricType>::value(user::PGen<S, M<D>>::metrics) &&
  traits::check_compatibility<D>::value(user::PGen<S, M<D>>::dimensions)
};

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
void dispatch_engine(ntt::Simulation& sim) {
  if constexpr (S == SimEngine::SRPIC) {
    sim.run<ntt::EngineSelector<S>::template type, M, D>();
  } else if constexpr (S == SimEngine::GRPIC) {
    sim.run<ntt::EngineSelector<S>::template type, M, D>();
  } else {
    static_assert(traits::always_false<std::integral_constant<SimEngine::type, S>>::value,
                  "Unsupported engine");
  }
}

auto main(int argc, char* argv[]) -> int {
  ntt::Simulation sim { argc, argv };

  auto matched  = false;
  auto launched = false;

  ntt::for_each_specialization([&](auto spec) {
    using Spec              = decltype(spec);
    const auto requested_e  = sim.requested_engine();
    const auto requested_m  = sim.requested_metric();
    const auto requested_d  = sim.requested_dimension();

    if (requested_e == Spec::engine && requested_m == Spec::metric &&
        requested_d == Spec::dimension) {
      matched = true;
      if constexpr (should_compile<Spec::engine, Spec::MetricTemplateType, Spec::dimension>) {
        dispatch_engine<Spec::engine, Spec::MetricTemplateType, Spec::dimension>(sim);
        launched = true;
      } else {
        raise::Fatal("Requested configuration is not available for this problem generator", HERE);
      }
    }
  });

  if (not matched) {
    raise::Fatal("Invalid engine, metric, or dimension combination", HERE);
  }

  if (not launched) {
    raise::Fatal("Requested combination is not enabled in this build", HERE);
  }

  return 0;
}
