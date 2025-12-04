/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::has_method<>
 *   - traits::has_member<>
 *   - traits::ex1_t, ::ex2_t, ::ex3_t
 *   - traits::bx1_t, ::bx2_t, ::bx3_t
 *   - traits::dx1_t, ::dx2_t, ::dx3_t
 *   - traits::run_t, traits::to_string_t
 *   - traits::pgen::init_flds_t
 *   - traits::pgen::ext_force_t
 *   - traits::pgen::ext_current_t
 *   - traits::pgen::atm_fields_t
 *   - traits::pgen::match_fields_const_t
 *   - traits::pgen::match_fields_t
 *   - traits::pgen::fix_fields_const_t
 *   - traits::pgen::fix_fields_t
 *   - traits::pgen::init_prtls_t
 *   - traits::pgen::custom_fields_t
 *   - traits::pgen::custom_field_output_t
 *   - traits::pgen::custom_poststep_t
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

  // for fieldsetter
  // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3
  template <typename T>
  using ex1_t = decltype(&T::ex1);

  template <typename T>
  using ex2_t = decltype(&T::ex2);

  template <typename T>
  using ex3_t = decltype(&T::ex3);

  template <typename T>
  using bx1_t = decltype(&T::bx1);

  template <typename T>
  using bx2_t = decltype(&T::bx2);

  template <typename T>
  using bx3_t = decltype(&T::bx3);

  template <typename T>
  using dx1_t = decltype(&T::dx1);

  template <typename T>
  using dx2_t = decltype(&T::dx2);

  template <typename T>
  using dx3_t = decltype(&T::dx3);

  // for simengine
  template <typename T>
  using run_t = decltype(&T::run);

  // for params
  template <typename T>
  using to_string_t = decltype(&T::to_string);

  // for pgen
  namespace pgen {
    template <typename T>
    using init_flds_t = decltype(&T::init_flds);

    template <typename T>
    using init_prtls_t = decltype(&T::InitPrtls);

    template <typename T>
    using ext_force_t = decltype(&T::ext_force);

    template <typename T>
    using ext_current_t = decltype(&T::ext_current);

    template <typename T>
    using atm_fields_t = decltype(&T::AtmFields);

    template <typename T>
    using match_fields_t = decltype(&T::MatchFields);

    template <typename T>
    using match_fields_in_x1_t = decltype(&T::MatchFieldsInX1);

    template <typename T>
    using match_fields_in_x2_t = decltype(&T::MatchFieldsInX2);

    template <typename T>
    using match_fields_in_x3_t = decltype(&T::MatchFieldsInX3);

    template <typename T>
    using match_fields_const_t = decltype(&T::MatchFieldsConst);

    template <typename T>
    using fix_fields_t = decltype(&T::FixFields);

    template <typename T>
    using fix_fields_const_t = decltype(&T::FixFieldsConst);

    template <typename T>
    using perfect_conductor_fields_t = decltype(&T::PerfectConductorFields);

    template <typename T>
    using perfect_conductor_fields_const_t = decltype(&T::PerfectConductorFieldsConst);

    template <typename T>
    using perfect_conductor_currents_t = decltype(&T::PerfectConductorCurrents);

    template <typename T>
    using perfect_conductor_currents_const_t = decltype(&T::PerfectConductorCurrentsConst);

    template <typename T>
    using custom_fields_t = decltype(&T::CustomFields);

    template <typename T>
    using custom_poststep_t = decltype(&T::CustomPostStep);

    template <typename T>
    using custom_field_output_t = decltype(&T::CustomFieldOutput);

    template <typename T>
    using custom_stat_t = decltype(&T::CustomStat);
  } // namespace pgen

  // for pgen extforce
  template <typename T>
  using species_t = decltype(&T::species);

  // checking compat for the problem generator + engine
  template <int N>
  struct check_compatibility {
    template <int... Is>
    static constexpr bool value(std::integer_sequence<int, Is...>) {
      return ((Is == N) || ...);
    }
  };

  template <int... Is>
  struct compatible_with {
    static constexpr auto value = std::integer_sequence<int, Is...> {};
  };

  template <typename>
  struct always_false : std::false_type {};

  // generic

  template <typename T>
  struct is_pair : std::false_type {};

  template <typename T, typename U>
  struct is_pair<std::pair<T, U>> : std::true_type {};

} // namespace traits

#endif // GLOBAL_ARCH_TRAITS_H
