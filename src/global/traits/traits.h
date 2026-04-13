/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::external::HasFx1, ::HasFx2, ::HasFx3 - checks for F functions in external field class
 *   - traits::external::HasEx1, ::HasEx2, ::HasEx3 - checks for E functions in external field class
 *   - traits::external::HasBx1, ::HasBx2, ::HasBx3 - checks for B functions in external field class
 *   - traits::external::HasExternalE - checks if a class has any external E field method
 *   - traits::external::HasExternalB - checks if a class has any external B field method
 *   - traits::external::HasExternalF - checks if a class has any external force method
 *   - traits::has_method<>
 *   - traits::has_member<>
 *   - traits::run_t, traits::to_string_t
 *   - traits::check_compatibility<>
 *   - traits::compatibility<>
 *   - traits::is_pair<>
 * @namespaces:
 *   - traits::
 *   - traits::pgen::
 * @note realized with SFINAE technique
 */

#ifndef GLOBAL_TRAITS_TRAITS_H
#define GLOBAL_TRAITS_TRAITS_H

#include <Kokkos_Core.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace traits {

  template <template <typename> class Trait, typename T, typename = void>
  struct has_method : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_method<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // trivial overload of `has_method` for readability
  template <template <typename> class Trait, typename T, typename = void>
  struct has_member : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_member<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // for pgen ext_fields
  template <typename T>
  using species_t = decltype(&T::species);

  template <typename>
  struct always_false : std::false_type {};

  // generic
  template <typename T>
  struct is_pair : std::false_type {};

  template <typename T, typename U>
  struct is_pair<std::pair<T, U>> : std::true_type {};

  // c++20
  namespace params {
    template <class P>
    concept HasToString = requires(const P& params) {
      { params.to_string() } -> std::convertible_to<std::string>;
    };
  } // namespace params

} // namespace traits

#endif // GLOBAL_TRAITS_TRAITS_H
