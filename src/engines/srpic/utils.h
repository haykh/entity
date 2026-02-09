#ifndef ENGINES_SRPIC_UTILS_H
#define ENGINES_SRPIC_UTILS_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "framework/parameters/parameters.h"

#include <tuple>

namespace ntt {
  namespace srpic {

    /**
     * @brief Get the buffer region of the atmosphere and the direction
     * @param direction direction in which the atmosphere is applied
     * @return tuple: [sign of the direction, the direction (as in::), the min and max extent
     * @note xg_min and xg_max are the extents where the fields are set, not the atmosphere itself
     * @note i.e.
     *
     *         fields set      particles injected
     * ghost zone  |               |
     *    v        v               v
     * |....|...........|*******************.....  -> x1
     * ^                ^
     * xg_min         xg_max
     * |                |                  |
     * |<--  buffer  -->|<-- atmosphere -->|
     *
     * in this case the function returns { -1, in::x1, xg_min, xg_max }
     */
    template <class M>
      requires metric::traits::HasD<M> && metric::traits::HasConvert<M>
    auto GetAtmosphereExtent(dir::direction_t<M::Dim> direction,
                             const M&                 global_metric,
                             const Grid<M::Dim>&      global_grid,
                             const SimulationParams&  params)
      -> std::tuple<short, in, real_t, real_t> {
      const auto sign     = direction.get_sign();
      const auto dim      = direction.get_dim();
      const auto min_buff = params.template get<unsigned short>(
                              "algorithms.current_filters") +
                            2;
      const auto buffer_ncells = min_buff > 5 ? min_buff : 5;
      if (M::CoordType != Coord::Cart and (dim != in::x1 or sign > 0)) {
        raise::Error("For non-cartesian coordinates atmosphere BCs is "
                     "possible only in -x1 (@ rmin)",
                     HERE);
      }
      real_t   xg_min { ZERO }, xg_max { ZERO };
      ncells_t ig_min, ig_max;
      if (sign > 0) { // + direction
        ig_min = global_grid.n_active(dim) - buffer_ncells;
        ig_max = global_grid.n_active(dim);
      } else { // - direction
        ig_min = 0;
        ig_max = buffer_ncells;
      }

      if (dim == in::x1) {
        xg_min = global_metric.template convert<1, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(ig_min));
        xg_max = global_metric.template convert<1, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(ig_max));
      } else if (dim == in::x2) {
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          xg_min = global_metric.template convert<2, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_min));
          xg_max = global_metric.template convert<2, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_max));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else if (dim == in::x3) {
        if constexpr (M::Dim == Dim::_3D) {
          xg_min = global_metric.template convert<3, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_min));
          xg_max = global_metric.template convert<3, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_max));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
      return { sign, dim, xg_min, xg_max };
    }

    template <class M>
    auto RangeWithAxisBCs(const Domain<SimEngine::SRPIC, M>& domain)
      -> range_t<M::Dim> {
      auto range = domain.mesh.rangeActiveCells();
      if constexpr (M::CoordType != Coord::Cart) {
        /**
         * @brief taking one extra cell in the x2 direction if AXIS BCs
         */
        if constexpr (M::Dim == Dim::_2D) {
          if (domain.mesh.flds_bc_in({ 0, +1 }) == FldsBC::AXIS) {
            range = CreateRangePolicy<Dim::_2D>(
              { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
              { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
          }
        } else if constexpr (M::Dim == Dim::_3D) {
          if (domain.mesh.flds_bc_in({ 0, +1, 0 }) == FldsBC::AXIS) {
            range = CreateRangePolicy<Dim::_3D>({ domain.mesh.i_min(in::x1),
                                                  domain.mesh.i_min(in::x2),
                                                  domain.mesh.i_min(in::x3) },
                                                { domain.mesh.i_max(in::x1),
                                                  domain.mesh.i_max(in::x2) + 1,
                                                  domain.mesh.i_max(in::x3) });
          }
        }
      }
      return range;
    }

  } // namespace srpic
} // namespace ntt

#endif
