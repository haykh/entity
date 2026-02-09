/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::fieldsetter::ex1_t, ::ex2_t, ::ex3_t - checks for special methods in field setter classes
 *   - traits::fieldsetter::bx1_t, ::bx2_t, ::bx3_t - checks for special methods in field setter classes
 *   - traits::fieldsetter::dx1_t, ::dx2_t, ::dx3_t - checks for special methods in field setter classes
 *   - traits::has_method<>
 *   - traits::has_member<>
 *   - traits::ex1_t, ::ex2_t, ::ex3_t
 *   - traits::bx1_t, ::bx2_t, ::bx3_t
 *   - traits::dx1_t, ::dx2_t, ::dx3_t
 *   - traits::run_t, traits::to_string_t
 *   - traits::check_compatibility<>
 *   - traits::compatibility<>
 *   - traits::is_pair<>
 * @namespaces:
 *   - traits::
 *   - traits::pgen::
 * @note realized with SFINAE technique
 */

#ifndef GLOBAL_ARCH_TRAITS_H
#define GLOBAL_ARCH_TRAITS_H

#include "global.h"

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace traits {

  namespace fieldsetter {
    // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3 methods
    template <class T, Dimension D>
    concept HasEx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex3(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx3(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx3(x_Ph) } -> std::convertible_to<real_t>;
    };
  } // namespace fieldsetter

  template <template <typename> class Trait, typename T, typename = void>
  struct has_method : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_method<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // trivial overload of `has_method` for readability
  template <template <typename> class Trait, typename T, typename = void>
  struct has_member : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_member<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // for pgen extforce
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

#endif // GLOBAL_ARCH_TRAITS_H
