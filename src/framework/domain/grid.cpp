#include "framework/domain/grid.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace ntt {

  template <>
  auto Grid<Dim::_1D>::rangeAllCells() const -> range_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeAllCells() const -> range_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeAllCells() const -> range_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::allLayer,
                                    CellLayer::allLayer,
                                    CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_1D>::rangeActiveCells() const -> range_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeActiveCells() const -> range_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeActiveCells() const -> range_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::activeLayer,
                                    CellLayer::activeLayer,
                                    CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <Dimension D>
  auto Grid<D>::rangeCells(const box_region_t<D>& region) const -> range_t<D> {
    tuple_t<ncells_t, D> imin, imax;
    for (auto i { 0u }; i < D; i++) {
      switch (region[i]) {
        case CellLayer::allLayer:
          imin[i] = 0;
          imax[i] = n_all((in)i);
          break;
        case CellLayer::activeLayer:
          imin[i] = i_min((in)i);
          imax[i] = i_max((in)i);
          break;
        case CellLayer::minGhostLayer:
          imin[i] = 0;
          imax[i] = i_min((in)i);
          break;
        case CellLayer::minActiveLayer:
          imin[i] = i_min((in)i);
          imax[i] = i_min((in)i) + N_GHOSTS;
          break;
        case CellLayer::maxActiveLayer:
          imin[i] = i_max((in)i) - N_GHOSTS;
          imax[i] = i_max((in)i);
          break;
        case CellLayer::maxGhostLayer:
          imin[i] = i_max((in)i);
          imax[i] = n_all((in)i);
          break;
        default:
          raise::Error("Invalid cell layer", HERE);
          throw;
      }
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  template <Dimension D>
  auto Grid<D>::rangeCellsOnHost(const box_region_t<D>& region) const
    -> range_h_t<D> {
    tuple_t<ncells_t, D> imin, imax;
    for (auto i { 0u }; i < D; i++) {
      switch (region[i]) {
        case CellLayer::allLayer:
          imin[i] = 0;
          imax[i] = n_all((in)i);
          break;
        case CellLayer::activeLayer:
          imin[i] = i_min((in)i);
          imax[i] = i_max((in)i);
          break;
        case CellLayer::minGhostLayer:
          imin[i] = 0;
          imax[i] = i_min((in)i);
          break;
        case CellLayer::minActiveLayer:
          imin[i] = i_min((in)i);
          imax[i] = i_min((in)i) + N_GHOSTS;
          break;
        case CellLayer::maxActiveLayer:
          imin[i] = i_max((in)i) - N_GHOSTS;
          imax[i] = i_max((in)i);
          break;
        case CellLayer::maxGhostLayer:
          imin[i] = i_max((in)i);
          imax[i] = n_all((in)i);
          break;
        default:
          raise::Error("Invalid cell layer", HERE);
      }
    }
    return CreateRangePolicyOnHost<D>(imin, imax);
  }

  template <>
  auto Grid<Dim::_1D>::rangeAllCellsOnHost() const -> range_h_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeAllCellsOnHost() const -> range_h_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeAllCellsOnHost() const -> range_h_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::allLayer,
                                    CellLayer::allLayer,
                                    CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_1D>::rangeActiveCellsOnHost() const -> range_h_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeActiveCellsOnHost() const -> range_h_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeActiveCellsOnHost() const -> range_h_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::activeLayer,
                                    CellLayer::activeLayer,
                                    CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <Dimension D>
  auto Grid<D>::rangeCells(const tuple_t<list_t<int, 2>, D>& ranges) const
    -> range_t<D> {
    tuple_t<ncells_t, D> imin, imax;
    for (auto i { 0u }; i < D; i++) {
      raise::ErrorIf((ranges[i][0] < -(int)N_GHOSTS) ||
                       (ranges[i][1] > (int)N_GHOSTS),
                     "Invalid cell layer picked",
                     HERE);
      imin[i] = i_min((in)i) + ranges[i][0];
      imax[i] = i_max((in)i) + ranges[i][1];
      raise::ErrorIf(imin[i] >= imax[i], "Invalid cell layer picked", HERE);
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  template <>
  auto Grid<Dim::_1D>::flds_bc() const -> boundaries_t<FldsBC> {
    return {
      { flds_bc_in({ -1 }), flds_bc_in({ 1 }) }
    };
  }

  template <>
  auto Grid<Dim::_2D>::flds_bc() const -> boundaries_t<FldsBC> {
    return {
      { flds_bc_in({ -1, 0 }), flds_bc_in({ 1, 0 }) },
      { flds_bc_in({ 0, -1 }), flds_bc_in({ 0, 1 }) }
    };
  }

  template <>
  auto Grid<Dim::_3D>::flds_bc() const -> boundaries_t<FldsBC> {
    return {
      { flds_bc_in({ -1, 0, 0 }), flds_bc_in({ 1, 0, 0 }) },
      { flds_bc_in({ 0, -1, 0 }), flds_bc_in({ 0, 1, 0 }) },
      { flds_bc_in({ 0, 0, -1 }), flds_bc_in({ 0, 0, 1 }) }
    };
  }

  template <>
  auto Grid<Dim::_1D>::prtl_bc() const -> boundaries_t<PrtlBC> {
    return {
      { prtl_bc_in({ -1 }), prtl_bc_in({ 1 }) }
    };
  }

  template <>
  auto Grid<Dim::_2D>::prtl_bc() const -> boundaries_t<PrtlBC> {
    return {
      { prtl_bc_in({ -1, 0 }), prtl_bc_in({ 1, 0 }) },
      { prtl_bc_in({ 0, -1 }), prtl_bc_in({ 0, 1 }) }
    };
  }

  template <>
  auto Grid<Dim::_3D>::prtl_bc() const -> boundaries_t<PrtlBC> {
    return {
      { prtl_bc_in({ -1, 0, 0 }), prtl_bc_in({ 1, 0, 0 }) },
      { prtl_bc_in({ 0, -1, 0 }), prtl_bc_in({ 0, 1, 0 }) },
      { prtl_bc_in({ 0, 0, -1 }), prtl_bc_in({ 0, 0, 1 }) }
    };
  }

  template struct Grid<Dim::_1D>;
  template struct Grid<Dim::_2D>;
  template struct Grid<Dim::_3D>;

} // namespace ntt
