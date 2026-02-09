/**
 * @file archetypes/traits.h
 * @brief Defines a set of traits to check if archetype classes satisfy certain conditions
 * @implements
 *   - arch::traits::energydist::IsValid<> - checks if energy distribution class has required operator()
 *   - arch::traits::spatialdist::IsValid<> - checks if spatial distribution class has required operator()
 *   - arch::traits::pgen::check_compatibility<> - checks if problem generator is compatible with given enums
 *   - arch::traits::pgen::compatible_with<> - defines compatible enums for problem generator
 *   - arch::traits::pgen::HasD<> - checks if problem generator has Dim static member
 *   - arch::traits::pgen::HasInitFlds<> - checks if problem generator has init_flds member
 *   - arch::traits::pgen::HasInitPrtls<> - checks if problem generator has InitPrtls method
 *   - arch::traits::pgen::HasExtForce<> - checks if problem generator has ext_force member
 *   - arch::traits::pgen::HasExtCurrent<> - checks if problem generator has ext_current member
 *   - arch::traits::pgen::HasAtmFields<> - checks if problem generator has AtmFields method
 *   - arch::traits::pgen::HasMatchFields<> - checks if problem generator has MatchFields method
 *   - arch::traits::pgen::HasMatchFieldsInX1<> - checks if problem generator has MatchFieldsInX1 method
 *   - arch::traits::pgen::HasMatchFieldsInX2<> - checks if problem generator has MatchFieldsInX2 method
 *   - arch::traits::pgen::HasMatchFieldsInX3<> - checks if problem generator has MatchFieldsInX3 method
 *   - arch::traits::pgen::HasFixFieldsConst<> - checks if problem generator has FixFieldsConst method
 *   - arch::traits::pgen::HasCustomPostStep<> - checks if problem generator has CustomPostStep method
 *   - arch::traits::pgen::HasCustomFieldOutput<> - checks if problem generator has CustomFieldOutput method
 *   - arch::traits::pgen::HasCustomStatOutput<> - checks if problem generator has CustomStat method
 * @namespaces:
 *   - arch::traits::
 */
#ifndef ARCHETYPES_TRAITS_H
#define ARCHETYPES_TRAITS_H

#include "global.h"

#include "arch/kokkos_aliases.h"

namespace arch {
  namespace traits {

    namespace energydist {

      template <class ED>
      concept IsValid = requires(const ED&             edist,
                                 const coord_t<ED::D>& x_Ph,
                                 vec_t<Dim::_3D>&      v) {
        { edist(x_Ph, v) } -> std::same_as<void>;
      };

    } // namespace energydist

    namespace spatialdist {

      template <class SD>
      concept IsValid = requires(const SD& sdist, const coord_t<SD::D>& x_Ph) {
        { sdist(x_Ph) } -> std::convertible_to<real_t>;
      };

    } // namespace spatialdist

    namespace pgen {

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

      template <class PG>
      concept HasD = requires {
        { PG::D } -> std::convertible_to<Dimension>;
      };

      template <class PG>
      concept HasInitFlds = requires(const PG& pgen) { pgen.init_flds; };

      template <class PG, class D>
      concept HasInitPrtls = requires(PG& pgen, D& domain) {
        { pgen.InitPrtls(domain) } -> std::same_as<void>;
      };

      template <class PG>
      concept HasExtForce = requires(const PG& pgen) { pgen.ext_force; };

      template <class PG>
      concept HasExtCurrent = requires(const PG& pgen) { pgen.ext_current; };

      template <class PG>
      concept HasAtmFields = requires(const PG& pgen, simtime_t time) {
        pgen.AtmFields(time);
      };

      template <class PG>
      concept HasMatchFields = requires(const PG& pgen, simtime_t time) {
        pgen.MatchFields(time);
      };

      template <class PG>
      concept HasMatchFieldsInX1 = requires(const PG& pgen, simtime_t time) {
        pgen.MatchFieldsInX1(time);
      };

      template <class PG>
      concept HasMatchFieldsInX2 = requires(const PG& pgen, simtime_t time) {
        pgen.MatchFieldsInX2(time);
      };

      template <class PG>
      concept HasMatchFieldsInX3 = requires(const PG& pgen, simtime_t time) {
        pgen.MatchFieldsInX3(time);
      };

      template <class PG>
      concept HasFixFieldsConst = requires(const PG&      pgen,
                                           const bc_in&   bc,
                                           const ntt::em& comp) {
        {
          pgen.FixFieldsConst(bc, comp)
        } -> std::convertible_to<std::pair<real_t, bool>>;
      };

      template <class PG, class D>
      concept HasCustomPostStep = requires(PG&        pgen,
                                           timestep_t s,
                                           simtime_t  t,
                                           D&         domain) {
        { pgen.CustomPostStep(s, t, domain) } -> std::same_as<void>;
      };

      template <class PG, class D>
      concept HasCustomFieldOutput = requires(PG&                  pgen,
                                              const std::string&   name,
                                              ndfield_t<PG::D, 6>& buff,
                                              index_t              idx,
                                              timestep_t           step,
                                              simtime_t            time,
                                              const D&             dom) {
        {
          pgen.CustomFieldOutput(name, buff, idx, step, time, dom)
        } -> std::same_as<void>;
      };

      template <class PG, class D>
      concept HasCustomStatOutput = requires(PG&                pgen,
                                             const std::string& name,
                                             timestep_t         s,
                                             simtime_t          t,
                                             const D&           dom) {
        { pgen.CustomStat(name, s, t, dom) } -> std::convertible_to<real_t>;
      };

    } // namespace pgen
  } // namespace traits
} // namespace arch

#endif // ARCHETYPES_TRAITS_H
