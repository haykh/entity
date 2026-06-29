/**
 * @file traits/pgen.h
 * @brief Traits and concepts for detecting problem generator interface methods
 * @implements
 *   - traits::pgen::compatible_with<>
 *   - traits::pgen::HasInitFlds<>
 *   - traits::pgen::HasEmissionPolicy<>
 *   - traits::pgen::HasCustomPrtlUpdate<>
 *   - traits::pgen::HasExternalFields<>
 *   - traits::pgen::HasInitPrtls<>
 *   - traits::pgen::HasExtCurrent<>
 *   - traits::pgen::HasAtmFields<>
 *   - traits::pgen::HasMatchFields<>
 *   - traits::pgen::HasMatchFieldsInX1<>
 *   - traits::pgen::HasMatchFieldsInX2<>
 *   - traits::pgen::HasMatchFieldsInX3<>
 *   - traits::pgen::HasFixFieldsConst<>
 *   - traits::pgen::HasCustomPostStep<>
 *   - traits::pgen::HasCustomFieldOutput<>
 *   - traits::pgen::HasCustomStatOutput<>
 * @namespaces:
 *   - traits::pgen::
 */

#ifndef TRAITS_PGEN_H
#define TRAITS_PGEN_H

#include "global.h"

#include "arch/kokkos_aliases.h"

namespace traits::pgen {

  template <auto... Is>
  struct compatible_with {};

  template <class PG>
  concept HasInitFlds = requires(const PG& pgen) { pgen.init_flds; };

  template <class PG, class DOM>
  concept HasEmissionPolicy = requires(const PG& pgen,
                                       simtime_t time,
                                       spidx_t   sp,
                                       DOM&      domain) {
    pgen.EmissionPolicy(time, sp, domain);
  };

  template <class PG, class DOM>
  concept HasCustomPrtlUpdate = requires(const PG& pgen,
                                         simtime_t time,
                                         spidx_t   sp,
                                         DOM&      domain) {
    pgen.CustomParticleUpdate(time, sp, domain);
  };

  template <class PG, class DOM>
  concept HasExternalFields = requires(const PG& pgen,
                                       simtime_t time,
                                       spidx_t   sp,
                                       DOM&      domain) {
    requires std::same_as<bool, decltype(pgen.ExternalFields(time, sp, domain).first)>;
    pgen.ExternalFields(time, sp, domain).second;
  };

  template <class PG, class D>
  concept HasInitPrtls = requires(PG& pgen, D& domain) {
    { pgen.InitPrtls(domain) } -> std::same_as<void>;
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

  template <class PG, class DOM>
  concept HasCustomPostStep = requires(PG& pgen, timestep_t s, simtime_t t, DOM& domain) {
    { pgen.CustomPostStep(s, t, domain) } -> std::same_as<void>;
  };

  template <class PG, Dimension D, class DOM>
  concept HasCustomFieldOutput = requires(PG&                pgen,
                                          const std::string& name,
                                          ndfield_t<D, 6>&   buff,
                                          cellidx_t          idx,
                                          timestep_t         step,
                                          simtime_t          time,
                                          const DOM&         dom) {
    {
      pgen.CustomFieldOutput(name, buff, idx, step, time, dom)
    } -> std::same_as<void>;
  };

  template <class PG, class DOM>
  concept HasCustomStatOutput = requires(PG&                pgen,
                                         const std::string& name,
                                         timestep_t         s,
                                         simtime_t          t,
                                         const DOM&         dom) {
    { pgen.CustomStat(name, s, t, dom) } -> std::convertible_to<real_t>;
  };

} // namespace traits::pgen

#endif // TRAITS_PGEN_H
