#ifndef KERNELS_PUSHERS_POLICIES_H
#define KERNELS_PUSHERS_POLICIES_H

#include "global.h"

#include "traits/emission.h"
#include "traits/external_fields.h"

namespace pusher {

  namespace sr::external_fields {

    struct NoPolicy_t {};

    template <class F>
    concept IsNoPolicy = std::is_same<std::remove_cvref_t<F>, NoPolicy_t>::value;

    template <class F, Dimension D>
    concept Policy = traits::external_fields::HasExternalF<F, D> or
                     traits::external_fields::HasExternalE<F, D> or
                     traits::external_fields::HasExternalB<F, D> or IsNoPolicy<F>;

  } // namespace sr::external_fields

  namespace emission {

    struct NoPolicy_t {};

    template <class E>
    concept IsNoPolicy = std::is_same<std::remove_cvref_t<E>, NoPolicy_t>::value;

    template <class E, class M>
    concept Policy = traits::emission::IsValidNontrivial<E, M> or IsNoPolicy<E>;

  } // namespace emission

} // namespace pusher

#endif // KERNELS_PUSHERS_POLICIES_H