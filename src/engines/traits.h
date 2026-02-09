/**
 * @file engine/traits.h
 * @brief Defines a set of traits to check if an engine class satisfies certain conditions
 * @implements
 *  - ntt::traits::engine::HasRun<> - checks if an engine has a run() method
 * @namespaces:
 *   - ntt::traits::engine::
 */
#ifndef ENGINES_TRAITS_H
#define ENGINES_TRAITS_H

#include <concepts>

namespace ntt {

  namespace traits {
    namespace engine {
      template <class E>
      concept HasRun = requires(E& engine) {
        { engine.run() } -> std::same_as<void>;
      };
    } // namespace engine
  } // namespace traits
} // namespace ntt

#endif // ENGINES_TRAITS_H
