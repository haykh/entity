/**
 * @file engine/traits.h
 * @brief Defines a set of traits to check if an engine class satisfies certain
 * conditions
 * @implements
 *  - ntt::traits::engine::HasRun<> - checks if an engine has a run() method
 *  - ntt::traits::engine::IsCompatibleWithEngine<> - checks if a metric and
 * pgen are compatible with a given simulation engine
 *  - ntt::traits::engine::IsCompatibleWithSRPICEngine<> - checks if a metric
 * and pgen are compatible with the SRPIC engine
 *  - ntt::traits::engine::IsCompatibleWithGRPICEngine<> - checks if a metric
 * and pgen are compatible with the GRPIC engine
 * @namespaces:
 *   - ntt::traits::engine::
 */
#ifndef ENGINES_TRAITS_H
#define ENGINES_TRAITS_H

#include "metrics/traits.h"

#include "archetypes/traits.h"

#include <concepts>

namespace ntt {
  namespace traits {
    namespace engine {

      template <SimEngine::type S, class M, template <SimEngine::type, class> class PG>
      concept IsCompatibleWithEngine =
        metric::traits::HasD<M> and
        arch::traits::pgen::check_compatibility<S>::value(PG<S, M>::engines) and
        arch::traits::pgen::check_compatibility<M::MetricType>::value(
          PG<S, M>::metrics) and
        arch::traits::pgen::check_compatibility<M::Dim>::value(PG<S, M>::dimensions);

      template <class M, template <SimEngine::type, class> class PG>
      concept IsCompatibleWithSRPICEngine =
        IsCompatibleWithEngine<SimEngine::SRPIC, M, PG> &&
        metric::traits::HasH_ij<M> && metric::traits::HasConvert_i<M> &&
        metric::traits::HasSqrtH_ij<M>;

      template <class M, template <SimEngine::type, class> class PG>
      concept IsCompatibleWithGRPICEngine =
        IsCompatibleWithEngine<SimEngine::GRPIC, M, PG>;

      template <class E>
      concept HasRun = requires(E& engine) {
        { engine.run() } -> std::same_as<void>;
      };

    } // namespace engine
  } // namespace traits
} // namespace ntt

#endif // ENGINES_TRAITS_H
