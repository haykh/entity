#ifndef KERNELS_PUSHERS_SR_CUSTOM_PARTICLE_UPDATE_H
#define KERNELS_PUSHERS_SR_CUSTOM_PARTICLE_UPDATE_H

// #include <concepts>
#include <type_traits>

namespace pusher::sr::custom_particle_update {
  struct NoPolicy_t {};

  namespace traits {
    template <class CPU>
    concept IsNoPolicy = std::is_same<std::remove_cvref_t<CPU>, NoPolicy_t>::value;

    // template <class P, class Kernel>
    // concept IsValid = requires(const P& custom_update, Kernel& pusher_kernel) {
    //   {
    //     custom_update.template operator()<Kernel>(pusher_kernel)
    //   } -> std::same_as<void>;
    // };

  } // namespace traits

} // namespace pusher::sr::custom_particle_update

#endif // KERNELS_PUSHERS_SR_CUSTOM_PARTICLE_UPDATE_H