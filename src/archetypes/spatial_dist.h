/**
 * @file archetypes/spatial_dist.hpp
 * @brief Spatial distribution class passed to injectors
 * @implements
 *   - arch::SpatialDistribution<>
 *   - arch::Uniform<> : arch::SpatialDistribution<>
 *   - arch::Replenish<> : arch::SpatialDistribution<>
 *   - arch::ReplenishUniform<> : arch::SpatialDistribution<>
 * @namespace
 *   - arch::
 * @note
 * Instances of these functors take coordinate position in code units
 * and return a number between 0 and 1 that represents the spatial distribution
 */

#ifndef ARCHETYPES_SPATIAL_DIST_HPP
#define ARCHETYPES_SPATIAL_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
    requires metric::traits::HasD<M>
  struct SpatialDistribution {
    static constexpr auto D = M::Dim;

    SpatialDistribution(const M& metric) : metric { metric } {}

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct Uniform : public SpatialDistribution<S, M> {
    Uniform(const M& metric) : SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>&) const -> real_t {
      return ONE;
    }
  };

  template <SimEngine::type S, class M, int N, class T>
  struct Replenish : public SpatialDistribution<S, M> {
    using SpatialDistribution<S, M>::metric;
    const ndfield_t<M::Dim, N> density;
    const idx_t                idx;

    const T      target_density;
    const real_t target_max_density;

    Replenish(const M&                    metric,
              const ndfield_t<M::Dim, N>& density,
              idx_t                       idx,
              const T&                    target_density,
              real_t                      target_max_density)
      : SpatialDistribution<S, M> { metric }
      , density { density }
      , idx { idx }
      , target_density { target_density }
      , target_max_density { target_max_density } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_1D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS, idx);
      } else if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else if constexpr (M::Dim == Dim::_3D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[2]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "Invalid dimension");
      }
      const auto target = target_density(x_Ph);
      if (0.9 * target > dens) {
        return (target - dens) / target_max_density;
      } else {
        return ZERO;
      }
    }
  };

  template <SimEngine::type S, class M, int N>
  struct ReplenishUniform : public SpatialDistribution<S, M> {
    using SpatialDistribution<S, M>::metric;
    const ndfield_t<M::Dim, N> density;
    const idx_t                idx;

    const real_t target_density;

    ReplenishUniform(const M&                    metric,
                     const ndfield_t<M::Dim, N>& density,
                     idx_t                       idx,
                     real_t                      target_density)
      : SpatialDistribution<S, M> { metric }
      , density { density }
      , idx { idx }
      , target_density { target_density } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_1D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS, idx);
      } else if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else if constexpr (M::Dim == Dim::_3D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[2]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "Invalid dimension");
      }
      if (0.9 * target_density > dens) {
        return (target_density - dens) / target_density;
      } else {
        return ZERO;
      }
    }
  };

} // namespace arch

#endif // ARCHETYPES_SPATIAL_DIST_HPP
