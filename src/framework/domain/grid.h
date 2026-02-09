/**
 * @file framework/domain/grid.h
 * @brief Grid class containing information about the discretization
 * @implements
 *   - ntt::Grid<>
 * @cpp:
 *   - grid.cpp
 * @namespaces:
 *   - ntt::
 * @note
 * Schematics for the grid in 2D:
 *
 *                          N_GHOSTS
 *                            __
 *       |<--      Ni1    -->|  |
 *    ...........................
 *    .                         .
 *    .                         .  <-- i2_max
 *    .  ^===================^  .___________________
 *    .  |                   \  .                   |
 *    .  |                   \  .                   |
 *    .  |                   \  .                   |-- Ni2
 *    .  |                   \  .                   |
 *    .  |                   \  .  <-- i2_min       |
 *    .  ^-------------------^  . __________________|
 *    .                         .   |
 *    .                         .   |- N_GHOSTS
 *    ........................... __|
 *        ^                   ^
 *        |                   |
 *      i1_min             i1_max
 * @note
 * Grid::RangeActiveCells
 *    . . . . . . . . . . .
 *    .                   .
 *    .                   .
 *    .   ^= = = = = =^   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   ^- - - - - -^   .
 *    .                   .
 *    .                   .
 *    . . . . . . . . . . .
 *
 * Grid::RangeAllCells
 *    . . . . . . . . . .
 *    .* * * * * * * * *.
 *    .* * * * * * * * *.
 *    .* *^= = = = =^* *.
 *    .* *|* * * * *\* *.
 *    .* *|* * * * *\* *.
 *    .* *|* * * * *\* *.
 *    .* *^- - - - -^* *.
 *    .* * * * * * * * *.
 *    .* * * * * * * * *.
 *    . . . . . . . . . .
 *
 */

#ifndef FRAMEWORK_DOMAIN_GRID_H
#define FRAMEWORK_DOMAIN_GRID_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <vector>

namespace ntt {

  template <Dimension D>
  struct Grid {
    Grid(const std::vector<ncells_t>& res, const boundaries_t<real_t>& ext)
      : m_resolution { res }
      , m_extent { ext } {
      raise::ErrorIf(m_resolution.size() != D, "invalid dimension", HERE);
    }

    Grid(const std::vector<ncells_t>& res,
         const boundaries_t<real_t>&  ext,
         const boundaries_t<FldsBC>&  flds_bc,
         const boundaries_t<PrtlBC>&  prtl_bc)
      : Grid { res, ext } {
      for (auto d { 0 }; d < D; ++d) {
        dir::direction_t<D> dir_plus;
        dir_plus[d] = +1;
        dir::direction_t<D> dir_minus;
        dir_minus[d] = -1;
        set_flds_bc(dir_plus, flds_bc[d].second);
        set_flds_bc(dir_minus, flds_bc[d].first);
        set_prtl_bc(dir_plus, prtl_bc[d].second);
        set_prtl_bc(dir_minus, prtl_bc[d].first);
      }
    }

    ~Grid() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto i_min(in i) const -> ncells_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? N_GHOSTS : 0;
        case in::x2:
          return (m_resolution.size() > 1) ? N_GHOSTS : 0;
        case in::x3:
          return (m_resolution.size() > 2) ? N_GHOSTS : 0;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto i_max(in i) const -> ncells_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? (m_resolution[0] + N_GHOSTS) : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? (m_resolution[1] + N_GHOSTS) : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? (m_resolution[2] + N_GHOSTS) : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_active(in i) const -> ncells_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? m_resolution[0] : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? m_resolution[1] : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? m_resolution[2] : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_active() const -> std::vector<ncells_t> {
      return m_resolution;
    }

    [[nodiscard]]
    auto num_active() const -> ncells_t {
      ncells_t total_active = 1u;
      for (const auto& res : m_resolution) {
        total_active *= res;
      }
      return total_active;
    }

    [[nodiscard]]
    auto n_all(in i) const -> ncells_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? (m_resolution[0] + 2 * N_GHOSTS) : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? (m_resolution[1] + 2 * N_GHOSTS) : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? (m_resolution[2] + 2 * N_GHOSTS) : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_all() const -> std::vector<ncells_t> {
      std::vector<ncells_t> nall;
      for (auto i = 0u; i < D; ++i) {
        nall.push_back(m_resolution[i] + 2 * N_GHOSTS);
      }
      return nall;
    }

    [[nodiscard]]
    auto num_all() const -> ncells_t {
      ncells_t total_all = 1u;
      for (const auto& res : n_all()) {
        total_all *= res;
      }
      return total_all;
    }

    /* Ranges in the device execution space --------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells)
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeActiveCells() const -> range_t<D>;
    /**
     * @brief Loop over all cells
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeAllCells() const -> range_t<D>;

    /**
     * @brief Pick a particular region of cells
     * @param box_region_t region of cells to pick: tuple of cellLayer objects
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeCells(const box_region_t<D>&) const -> range_t<D>;
    /**
     * @brief Pick a particular region of cells
     * @overload
     * @param range tuple of respective min and max ranges
     * @example {-1, 1} converts into {i_min - 1, i_max + 1} etc
     * @example {{0, 0}, {0, 0}, {0, 0}} corresponds to allActiveLayer in all 3 dimensions
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeCells(const tuple_t<list_t<int, 2>, D>&) const -> range_t<D>;

    /* Ranges in the host execution space ----------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells)
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeActiveCellsOnHost() const -> range_h_t<D>;
    /**
     * @brief Loop over all cells
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeAllCellsOnHost() const -> range_h_t<D>;
    /**
     * @brief Pick a particular region of cells
     * @param box_region_t region of cells to pick: tuple of cellLayer objects
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeCellsOnHost(const box_region_t<D>&) const -> range_h_t<D>;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto extent(in i) const -> std::pair<real_t, real_t> {
      switch (i) {
        case in::x1:
          return (m_extent.size() > 0) ? m_extent[0]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case in::x2:
          return (m_extent.size() > 1) ? m_extent[1]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case in::x3:
          return (m_extent.size() > 2) ? m_extent[2]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto extent() const -> boundaries_t<real_t> {
      return m_extent;
    }

    [[nodiscard]]
    auto flds_bc() const -> boundaries_t<FldsBC>;

    [[nodiscard]]
    auto prtl_bc() const -> boundaries_t<PrtlBC>;

    [[nodiscard]]
    auto flds_bc_in(const dir::direction_t<D>& direction) const -> FldsBC {
      raise::ErrorIf(m_flds_bc.find(direction) == m_flds_bc.end(),
                     "direction not found",
                     HERE);
      return m_flds_bc.at(direction);
    }

    [[nodiscard]]
    auto prtl_bc_in(const dir::direction_t<D>& direction) const -> PrtlBC {
      raise::ErrorIf(m_prtl_bc.find(direction) == m_prtl_bc.end(),
                     "direction not found",
                     HERE);
      return m_prtl_bc.at(direction);
    }

    /* setters -------------------------------------------------------------- */
    inline void set_flds_bc(const dir::direction_t<D>& direction, const FldsBC& bc) {
      m_flds_bc.insert_or_assign(direction, bc);
    }

    inline void set_prtl_bc(const dir::direction_t<D>& direction, const PrtlBC& bc) {
      m_prtl_bc.insert_or_assign(direction, bc);
    }

  protected:
    std::vector<ncells_t> m_resolution;
    boundaries_t<real_t>  m_extent;
    dir::map_t<D, FldsBC> m_flds_bc;
    dir::map_t<D, PrtlBC> m_prtl_bc;
  };

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_GRID_H
