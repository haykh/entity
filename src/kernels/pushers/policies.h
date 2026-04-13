#ifndef KERNELS_PUSHERS_POLICIES_H
#define KERNELS_PUSHERS_POLICIES_H

#include "enums.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/base_sr.hpp"
#include "kernels/pushers/emission/emission.h"
#include "kernels/pushers/sr/external_fields.h"

namespace pusher {

  // namespace pgen {

  //   template <class PGen>
  //   concept DefinesCustomParticleUpdate = requires(const PGen& pgen) {
  //     pgen.CustomParticleUpdate();
  //   };

  //   template <class PGen>
  //   concept DefinesExternalFields = requires(const PGen& pgen) {
  //     pgen.ExternalFields();
  //   };

  // } // namespace pgen

  // template <class E, class M>
  // concept IsValidEmissionPolicy = emission::IsValid<E, M> or
  //                                 emission::IsNoPolicy<E>;

  // template <class F, Dimension D>
  // concept IsValidExternalFieldsPolicy =

  // template <class P, class M>
  // concept ValidPolicy = requires(const P& policy) {
  //   typename P::EmissionPolicy;
  //   typename P::CustomParticleUpdatePolicy;
  //   typename P::ExternalFieldsPolicy;
  //   // IsValidEmissionPolicy<typename P::EmissionPolicy, M>;
  //   // sr::external_fields::traits::IsValid<typename P::ExternalFieldsPolicy,
  //   M::Dim>; { P::ApplyAtmosphere } -> std::convertible_to<bool>;
  // };

  template <class M, emission::Policy<M> E, class CPU, sr::external_fields::Policy<M::Dim> F, bool Atm>
  struct Policy {
    using EmissionPolicy                  = E;
    using CustomParticleUpdatePolicy      = CPU;
    using ExternalFieldsPolicy            = F;
    static constexpr bool ApplyAtmosphere = Atm;
    E                     emission_policy;
    CPU                   custom_particle_update_policy;
    F                     external_fields_policy;

    Policy(const E&   emission_policy,
           const CPU& custom_particle_update_policy,
           const F&   external_fields_policy)
      : emission_policy(emission_policy)
      , custom_particle_update_policy(custom_particle_update_policy)
      , external_fields_policy(external_fields_policy) {}
  };

} // namespace pusher

#endif // KERNELS_PUSHERS_POLICIES_SR_H
