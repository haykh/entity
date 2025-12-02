#include "enums.h"

#include "arch/traits.h"
#include "utils/error.h"

#include "engine_registry.h"

#include "framework/simulation.h"

#include "pgen.hpp"

#include <utility>

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

template <ntt::SimEngine::type S, ntt::Metric::type M, Dimension D>
[[nodiscard]] constexpr bool registryContainsSupportedCombo() {
  bool is_supported = false;

#define NTT_CHECK_REGISTRY_ENTRY(E, METRIC, DIM)                                           \
  if (!is_supported && E<METRIC<DIM>>::S == S && METRIC<DIM>::MetricType == M && DIM == D) \
    is_supported = should_compile<E<METRIC<DIM>>::S, METRIC, DIM>;

  NTT_ENGINE_METRIC_DIMENSION_REGISTRY(NTT_CHECK_REGISTRY_ENTRY)

#undef NTT_CHECK_REGISTRY_ENTRY

  return is_supported;
}

template <ntt::SimEngine::type S, ntt::Metric::type M, int... Ds>
[[nodiscard]] constexpr bool registryCoversMetric(std::integer_sequence<int, Ds...>) {
  bool covered = true;

  ((covered =
      covered && registryContainsSupportedCombo<S, M, static_cast<Dimension>(Ds)>()),
   ...);

  return covered;
}

template <ntt::SimEngine::type S, int... Ms, int... Ds>
[[nodiscard]] constexpr bool registryCoversEngine(
  std::integer_sequence<int, Ms...> metrics, std::integer_sequence<int, Ds...> dimensions) {
  bool covered = true;

  ((covered = covered &&
                registryCoversMetric<S, static_cast<ntt::Metric::type>(Ms)>(dimensions)),
   ...);

  return covered;
}

template <class PGen, int... Es, int... Ms, int... Ds>
[[nodiscard]] constexpr bool registryCoversPGen(std::integer_sequence<int, Es...> engines,
                                                std::integer_sequence<int, Ms...> metrics,
                                                std::integer_sequence<int, Ds...> dimensions) {
  bool covered = true;

  (void)sizeof(PGen);

  ((covered = covered && registryCoversEngine<static_cast<ntt::SimEngine::type>(Es)>(
                    metrics, dimensions)),
   ...);

  return covered;
}

using ActivePGen =
  user::PGen<ntt::SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>; // type only used for traits

static_assert(
  registryCoversPGen<ActivePGen>(
    ActivePGen::engines, ActivePGen::metrics, ActivePGen::dimensions),
  "Problem generator compatibility lists include combinations missing from the registry. "
  "Add the new engine/metric/dimension triplets to NTT_ENGINE_METRIC_DIMENSION_REGISTRY.");

auto main(int argc, char* argv[]) -> int {
  ntt::Simulation sim { argc, argv };
  bool dispatched = false;

#define NTT_DISPATCH_REGISTRY_ENTRY(E, M, D)                                                 \
  if (!dispatched && sim.requested_engine() == E<M<D>>::S &&                                 \
      sim.requested_metric() == M<D>::MetricType &&                                         \
      sim.requested_dimension() == D) {                                                      \
    shouldCompile<E, M, D>(sim);                                                             \
    dispatched = true;                                                                       \
  }

  NTT_ENGINE_METRIC_DIMENSION_REGISTRY(NTT_DISPATCH_REGISTRY_ENTRY)
#undef NTT_DISPATCH_REGISTRY_ENTRY

  if (!dispatched) {
    raise::Fatal("Invalid engine/metric/dimension combination", HERE);
  }

  return 0;
}