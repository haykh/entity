// SPDX-License-Identifier: BSD-3-Clause

#ifndef ENGINES_ENGINE_TRAITS_H
#define ENGINES_ENGINE_TRAITS_H

#include "enums.h"

#include "engines/grpic.hpp"
#include "engines/srpic.hpp"

namespace ntt {

  template <SimEngine::type S>
  struct EngineSelector;

  template <>
  struct EngineSelector<SimEngine::SRPIC> {
    template <class M>
    using type = SRPICEngine<M>;
  };

  template <>
  struct EngineSelector<SimEngine::GRPIC> {
    template <class M>
    using type = GRPICEngine<M>;
  };

} // namespace ntt

#endif // ENGINES_ENGINE_TRAITS_H
