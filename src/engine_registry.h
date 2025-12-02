#ifndef ENGINE_REGISTRY_H
#define ENGINE_REGISTRY_H

#include "enums.h"

#include <cstddef>

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/grpic.hpp"
#include "engines/srpic.hpp"

namespace ntt {

  // Central registry for supported engine/metric/dimension triplets.
  // Add new combinations here to enable runtime dispatch and explicit instantiations.
#define NTT_ENGINE_METRIC_DIMENSION_REGISTRY(MACRO)                                      \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_1D)                                   \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_2D)                                   \
  MACRO(ntt::SRPICEngine, metric::Minkowski, Dim::_3D)                                   \
  MACRO(ntt::SRPICEngine, metric::Spherical, Dim::_2D)                                   \
  MACRO(ntt::SRPICEngine, metric::QSpherical, Dim::_2D)                                  \
  MACRO(ntt::GRPICEngine, metric::KerrSchild, Dim::_2D)                                  \
  MACRO(ntt::GRPICEngine, metric::KerrSchild0, Dim::_2D)                                 \
  MACRO(ntt::GRPICEngine, metric::QKerrSchild, Dim::_2D)

  namespace detail {
    template <template <class> class E, template <Dimension> class M, Dimension D>
    struct EngineRegistryEntry {
      using engine_type = E<M<D>>;
      using metric_type = M<D>;
      static constexpr auto engine { engine_type::S };
      static constexpr auto metric { metric_type::MetricType };
      static constexpr auto dimension { D };
    };

    [[maybe_unused]] constexpr std::size_t engine_registry_size {
#define NTT_COUNT_REGISTRY_ENTRY(E, M, D) +1
      0 NTT_ENGINE_METRIC_DIMENSION_REGISTRY(NTT_COUNT_REGISTRY_ENTRY)
#undef NTT_COUNT_REGISTRY_ENTRY
    };
  } // namespace detail

} // namespace ntt

#endif // ENGINE_REGISTRY_H
