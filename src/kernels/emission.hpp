#ifndef KERNELS_EMISSION_HPP
#define KERNELS_EMISSION_HPP

#include "enums.h"
#include "global.h"

#include "kernels/injectors.hpp"

namespace kernel {
  using namespace ntt;

  template <class M, EmissionTypeFlag E>
  struct EmissionPolicy;

  template <class M>
  struct EmissionPolicy<M, EmissionType::NONE> {};

  template <class M>
  struct EmissionPolicy<M, EmissionType::SYNCHROTRON> {};

  template <class M>
  struct EmissionPolicy<M, EmissionType::COMPTON> {};

  template <class M>
  struct EmissionPolicy<M, EmissionType::STRONGFIELDPP> {};

} // namespace kernel

#endif
