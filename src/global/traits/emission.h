#ifndef GLOBAL_TRAITS_EMISSION_H
#define GLOBAL_TRAITS_EMISSION_H

#include "global.h"

#include <Kokkos_Pair.hpp>

namespace traits::emission {
  template <class E>
  concept HasPayload = requires { typename E::Payload; };

  template <class E>
  concept HasNumbersInjected = requires(E& emission_policy) {
    {
      emission_policy.numbers_injected()
    } -> std::convertible_to<std::vector<npart_t>>;
  };

  template <class E>
  concept HasEmittedSpeciesIndices = requires(const E& emission_policy) {
    {
      emission_policy.emitted_species_indices()
    } -> std::convertible_to<std::vector<spidx_t>>;
  };

  template <class E, class M>
  concept HasShouldEmit = requires(const E&                   emission_policy,
                                   const coord_t<M::PrtlDim>& x_Cd,
                                   const coord_t<M::PrtlDim>& x_Ph,
                                   const vec_t<Dim::_3D>&     u_Ph,
                                   const vec_t<Dim::_3D>&     ep_Ph,
                                   const vec_t<Dim::_3D>&     bp_Ph,
                                   vec_t<Dim::_3D>&           delta_u_Ph,
                                   typename E::Payload&       payload) {
    {
      emission_policy.shouldEmit(x_Cd, x_Ph, u_Ph, ep_Ph, bp_Ph, delta_u_Ph, payload)
    } -> std::convertible_to<Kokkos::pair<bool, bool>>;
  };

  template <class E, class M>
  concept HasEmit = requires(const E&                         emission_policy,
                             const tuple_t<int, M::Dim>&      xi_Cd,
                             const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                             const vec_t<Dim::_3D>&           direction,
                             real_t                           weight,
                             real_t                           phi,
                             const typename E::Payload&       payload) {
    {
      emission_policy.emit(xi_Cd, dxi_Cd, direction, weight, phi, payload)
    } -> std::same_as<void>;
  };

  template <class E, class M>
  concept IsValidNontrivial = HasPayload<E> and HasNumbersInjected<E> and
                              HasEmittedSpeciesIndices<E> and
                              HasShouldEmit<E, M> and HasEmit<E, M>;

} // namespace traits::emission

#endif