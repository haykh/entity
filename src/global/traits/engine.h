/**
 * @file traits/engine.h
 * @brief Defines a set of traits to check if an engine class satisfies certain
 * conditions
 * @implements
 *  - traits::engineHasRun<> - checks if an engine has a run() method
 *  - traits::engineIsCompatibleWithEngine<> - checks if a metric and
 * pgen are compatible with a given simulation engine
 *  - traits::engineIsCompatibleWithSRPICEngine<> - checks if a metric
 * and pgen are compatible with the SRPIC engine
 *  - traits::engine::IsCompatibleWithGRPICEngine<> - checks if a metric
 * and pgen are compatible with the GRPIC engine
 * @namespaces:
 *   - traits::engine::
 */
#ifndef GLOBAL_TRAITS_ENGINES_H
#define GLOBAL_TRAITS_ENGINES_H

#include <concepts>

#include "traits/archetypes.h"
#include "traits/metric.h"

namespace traits::engine {

  template <ntt::SimEngine::type S, class M, template <ntt::SimEngine::type, class> class PG>
  concept IsCompatibleWithEngine =
    traits::metric::HasD<M> and
    traits::pgen::check_compatibility<S>::value(PG<S, M>::engines) and
    traits::pgen::check_compatibility<M::MetricType>::value(PG<S, M>::metrics) and
    traits::pgen::check_compatibility<M::Dim>::value(PG<S, M>::dimensions);

  template <class M, template <ntt::SimEngine::type, class> class PG>
  concept IsCompatibleWithSRPICEngine =
    IsCompatibleWithEngine<ntt::SimEngine::SRPIC, M, PG> &&
    traits::metric::HasH_ij<M> && traits::metric::HasConvert_i<M> &&
    traits::metric::HasSqrtH_ij<M>;

  template <class M, template <ntt::SimEngine::type, class> class PG>
  concept IsCompatibleWithGRPICEngine =
    IsCompatibleWithEngine<ntt::SimEngine::GRPIC, M, PG>;

  template <class E>
  concept HasRun = requires(E& engine) {
    { engine.run() } -> std::same_as<void>;
  };

} // namespace traits::engine
#endif // GLOBAL_TRAITS_ENGINES_H
