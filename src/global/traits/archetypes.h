/**
 * @file archetypes/traits.h
 * @brief Defines a set of traits to check if archetype classes satisfy certain conditions
 * @implements
 *   - traits::energydist::IsValid - checks if energy distribution class has required operator()
 *   - traits::spatialdist::IsValid - checks if spatial distribution class has required operator()
 *   - traits::pgen::check_compatibility - checks if problem generator is compatible with given enums
 *   - traits::pgen::compatible_with - defines compatible enums for problem generator
 *   - traits::pgen::HasD - checks if problem generator has Dim static member
 *   - traits::pgen::HasInitFlds - checks if problem generator has init_flds member
 *   - traits::pgen::HasInitPrtls - checks if problem generator has InitPrtls method
 *   - traits::pgen::HasExternalFields - checks if problem generator has ExternalFields method
 *   - traits::pgen::HasExtCurrent - checks if problem generator has ext_current member
 *   - traits::pgen::HasAtmFields - checks if problem generator has AtmFields method
 *   - traits::pgen::HasMatchFields - checks if problem generator has MatchFields method
 *   - traits::pgen::HasMatchFieldsInX1 - checks if problem generator has MatchFieldsInX1 method
 *   - traits::pgen::HasMatchFieldsInX2 - checks if problem generator has MatchFieldsInX2 method
 *   - traits::pgen::HasMatchFieldsInX3 - checks if problem generator has MatchFieldsInX3 method
 *   - traits::pgen::HasFixFieldsConst - checks if problem generator has FixFieldsConst method
 *   - traits::pgen::HasCustomPostStep - checks if problem generator has CustomPostStep method
 *   - traits::pgen::HasCustomFieldOutput - checks if problem generator has CustomFieldOutput method
 *   - traits::pgen::HasCustomStatOutput - checks if problem generator has CustomStat method
 *   - traits::fieldsetter::HasEx1, ::HasEx2, ::HasEx3 - checks for E functions in field setter class
 *   - traits::fieldsetter::HasBx1, ::HasBx2, ::HasBx3 - checks for B functions in field setter class
 *   - traits::fieldsetter::HasDx1, ::HasDx2, ::HasDx3 - checks for D functions in field setter class
 *   - traits::fieldsetter::HasConditionalEx1... - checks for conditional E functions in field setter class
 *   - traits::fieldsetter::HasConditionalBx1... - checks for conditional B functions in field setter class
 *   - traits::fieldsetter::HasConditionalDx1... - checks for conditional D functions in field setter class
 * @namespaces:
 *   - traits::
 */
#ifndef GLOBAL_TRAITS_ARCHETYPES_H
#define GLOBAL_TRAITS_ARCHETYPES_H

#include "global.h"

#include "arch/kokkos_aliases.h"

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
    concept HasEmissionPolicy = requires(const PG& pgen,
                                         simtime_t time,
                                         spidx_t   sp,
                                         D&        domain) {
      pgen.EmissionPolicy(time, sp, domain);
    };

    template <class PG, class M, class D>
    concept HasCustomPrtlUpdate = requires(const PG& pgen,
                                           simtime_t time,
                                           spidx_t   sp,
                                           D&        domain) {
      pgen.CustomParticleUpdate(time, sp, domain);
    };

    template <class PG, class D>
    concept HasInitPrtls = requires(PG& pgen, D& domain) {
      { pgen.InitPrtls(domain) } -> std::same_as<void>;
    };

    template <class PG, class D>
    concept HasExternalFields = requires(const PG& pgen,
                                         simtime_t time,
                                         spidx_t   sp,
                                         D&        domain) {
      requires std::same_as<bool, decltype(pgen.ExternalFields(time, sp, domain).first)>;
      pgen.ExternalFields(time, sp, domain).second;
    };

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
    concept HasFixFieldsConst = requires(const PG&    pgen,
                                         simtime_t    time,
                                         const bc_in& bc,
                                         ntt::em      comp) {
      {
        pgen.FixFieldsConst(time, bc, comp)
      } -> std::convertible_to<std::pair<real_t, bool>>;
    };

    template <class PG, class D>
    concept HasCustomPostStep = requires(PG& pgen, timestep_t s, simtime_t t, D& domain) {
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

    template <class T, Dimension D>
    concept HasConditionalEx1 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.ex1(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalEx2 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.ex2(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalEx3 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.ex3(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalBx1 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.bx1(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalBx2 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.bx2(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalBx3 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& e_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.bx3(x_Ph, e_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalDx1 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& d_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.dx1(x_Ph, d_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalDx2 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& d_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.dx2(x_Ph, d_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };

    template <class T, Dimension D>
    concept HasConditionalDx3 = requires(const T&               t,
                                         const coord_t<D>&      x_Ph,
                                         const vec_t<Dim::_3D>& d_Ph,
                                         const vec_t<Dim::_3D>& b_Ph) {
      {
        t.dx3(x_Ph, d_Ph, b_Ph)
      } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
    };
  } // namespace fieldsetter

} // namespace traits

#endif // GLOBAL_TRAITS_ARCHETYPES_H
