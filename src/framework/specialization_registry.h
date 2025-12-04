// SPDX-License-Identifier: BSD-3-Clause

#ifndef FRAMEWORK_SPECIALIZATION_REGISTRY_H
#define FRAMEWORK_SPECIALIZATION_REGISTRY_H

#include "enums.h"

#include "arch/traits.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <tuple>

namespace ntt {

  template <SimEngine::type S,
            template <Dimension>
            class MetricTemplate,
            Dimension D>
  struct SpecializationEntry {
    using MetricType = MetricTemplate<D>;
    template <Dimension D2>
    using MetricTemplateType = MetricTemplate<D2>;

    static constexpr auto engine     = S;
    static constexpr auto metric     = MetricType::MetricType;
    static constexpr auto dimension  = D;
  };

#define NTT_FOREACH_SPECIALIZATION(MACRO)                                       \
  MACRO(SimEngine::SRPIC, metric::Minkowski, Dim::_1D)                          \
  MACRO(SimEngine::SRPIC, metric::Minkowski, Dim::_2D)                          \
  MACRO(SimEngine::SRPIC, metric::Minkowski, Dim::_3D)                          \
  MACRO(SimEngine::SRPIC, metric::Spherical, Dim::_2D)                          \
  MACRO(SimEngine::SRPIC, metric::QSpherical, Dim::_2D)                         \
  MACRO(SimEngine::GRPIC, metric::KerrSchild, Dim::_2D)                         \
  MACRO(SimEngine::GRPIC, metric::QKerrSchild, Dim::_2D)                        \
  MACRO(SimEngine::GRPIC, metric::KerrSchild0, Dim::_2D)

#define NTT_BUILD_SPECIALIZATION_ENTRY(S, M, D)                                 \
  SpecializationEntry<S, M, D> {},

  inline constexpr auto kSpecializations = std::tuple {
    NTT_FOREACH_SPECIALIZATION(NTT_BUILD_SPECIALIZATION_ENTRY)
  };

#undef NTT_BUILD_SPECIALIZATION_ENTRY

  template <class Func>
  constexpr void for_each_specialization(Func&& func) {
    std::apply([&](auto... entry) { (func(entry), ...); }, kSpecializations);
  }

} // namespace ntt

#endif // FRAMEWORK_SPECIALIZATION_REGISTRY_H
