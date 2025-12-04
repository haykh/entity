// SPDX-License-Identifier: BSD-3-Clause

#ifndef FRAMEWORK_SPECIALIZATION_REGISTRY_H
#define FRAMEWORK_SPECIALIZATION_REGISTRY_H

#include "enums.h"

#include "arch/traits.h"

#include "engines/grpic.hpp"
#include "engines/srpic.hpp"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <tuple>

namespace ntt {

  template <template <class> class EngineTemplate,
            template <Dimension>
            class MetricTemplate,
            Dimension D>
  struct SpecializationEntry {
    using MetricType = MetricTemplate<D>;
    using EngineType = EngineTemplate<MetricType>;

    using EngineTemplateType = EngineTemplate;
    using MetricTemplateType = MetricTemplate;

    static constexpr auto engine     = EngineType::S;
    static constexpr auto metric     = MetricType::MetricType;
    static constexpr auto dimension  = D;
  };

#define NTT_FOREACH_SPECIALIZATION(MACRO)                                       \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_1D)                          \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_2D)                          \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_3D)                          \
  MACRO(ntt::SRPICEngine, metric::Spherical, Dim::_2D)                          \
  MACRO(ntt::SRPICEngine, metric::QSpherical, Dim::_2D)                         \
  MACRO(ntt::GRPICEngine, metric::KerrSchild, Dim::_2D)                         \
  MACRO(ntt::GRPICEngine, metric::QKerrSchild, Dim::_2D)                        \
  MACRO(ntt::GRPICEngine, metric::KerrSchild0, Dim::_2D)

#define NTT_BUILD_SPECIALIZATION_ENTRY(E, M, D)                                 \
  SpecializationEntry<E, M, D> {},

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
