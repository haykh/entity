#include "enums.h"

#include "arch/traits.h"
#include "utils/error.h"

#include "framework/simulation.h"
#include "framework/specialization_registry.h"

#include "pgen.hpp"

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
static constexpr bool should_compile {
  traits::check_compatibility<S>::value(user::PGen<S, M<D>>::engines) &&
  traits::check_compatibility<M<D>::MetricType>::value(user::PGen<S, M<D>>::metrics) &&
  traits::check_compatibility<D>::value(user::PGen<S, M<D>>::dimensions)
};

template <template <class> class E, template <Dimension> class M, Dimension D>
void shouldCompile(ntt::Simulation& sim) {
  if constexpr (should_compile<E<M<D>>::S, M, D>) {
    sim.run<E, M, D>();
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
        sim.run<Spec::EngineTemplateType, Spec::MetricTemplateType, Spec::dimension>();
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
