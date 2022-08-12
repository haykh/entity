#include "global.h"
#include "meshblock.h"
#include "particles.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {
  const auto Dim1 = Dimension::ONE_D;
  const auto Dim2 = Dimension::TWO_D;
  const auto Dim3 = Dimension::THREE_D;

  template <Dimension D>
  Mesh<D>::Mesh(const std::vector<unsigned int>& res,
                const std::vector<real_t>&       ext,
                const real_t*                    params)
    : m_i1min {res.size() > 0 ? N_GHOSTS : 0},
      m_i1max {res.size() > 0 ? N_GHOSTS + (int)(res[0]) : 1},
      m_i2min {res.size() > 1 ? N_GHOSTS : 0},
      m_i2max {res.size() > 1 ? N_GHOSTS + (int)(res[1]) : 1},
      m_i3min {res.size() > 2 ? N_GHOSTS : 0},
      m_i3max {res.size() > 2 ? N_GHOSTS + (int)(res[2]) : 1},
      m_Ni1 {res.size() > 0 ? (int)(res[0]) : 1},
      m_Ni2 {res.size() > 1 ? (int)(res[1]) : 1},
      m_Ni3 {res.size() > 2 ? (int)(res[2]) : 1},
      metric {res, ext, params} {}

  template <>
  auto Mesh<Dim1>::rangeAllCells() -> RangeND<Dim1> {
    boxRegion<Dim1> region {CellLayer::allLayer};
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim2>::rangeAllCells() -> RangeND<Dim2> {
    boxRegion<Dim2> region {CellLayer::allLayer, CellLayer::allLayer};
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim3>::rangeAllCells() -> RangeND<Dim3> {
    boxRegion<Dim3> region {CellLayer::allLayer, CellLayer::allLayer, CellLayer::allLayer};
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim1>::rangeActiveCells() -> RangeND<Dim1> {
    boxRegion<Dim1> region {CellLayer::allActiveLayer};
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim2>::rangeActiveCells() -> RangeND<Dim2> {
    boxRegion<Dim2> region {CellLayer::allActiveLayer, CellLayer::allActiveLayer};
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim3>::rangeActiveCells() -> RangeND<Dim3> {
    boxRegion<Dim3> region {
      CellLayer::allActiveLayer, CellLayer::allActiveLayer, CellLayer::allActiveLayer};
    return rangeCells(region);
  }

  template <Dimension D>
  auto Mesh<D>::rangeCells(const boxRegion<D>& region) -> RangeND<D> {
    tuple_t<int, D> imin, imax;
    for (short i = 0; i < (short)D; i++) {
      switch (region[i]) {
      case CellLayer::allLayer:
        imin[i] = 0;
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      case CellLayer::allActiveLayer:
        imin[i] = i_min(i);
        imax[i] = i_max(i);
        break;
      case CellLayer::minGhostLayer:
        imin[i] = 0;
        imax[i] = i_min(i);
        break;
      case CellLayer::minActiveLayer:
        imin[i] = i_min(i);
        imax[i] = i_min(i) + N_GHOSTS;
        break;
      case CellLayer::maxActiveLayer:
        imin[i] = i_max(i) - N_GHOSTS;
        imax[i] = i_max(i);
        break;
      case CellLayer::maxGhostLayer:
        imin[i] = i_max(i);
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      default:
        NTTError("Invalid cell layer");
      }
    }
    return NTTRange<D>(imin, imax);
  }

  template <Dimension D, SimulationType S>
  Meshblock<D, S>::Meshblock(const std::vector<unsigned int>&    res,
                             const std::vector<real_t>&          ext,
                             const real_t*                       params,
                             const std::vector<ParticleSpecies>& species)
    : Mesh<D>(res, ext, params), Fields<D, S>(res) {
    for (auto& part : species) {
      particles.emplace_back(part);
    }
  }
} // namespace ntt

template class ntt::Mesh<ntt::Dimension::ONE_D>;
template class ntt::Mesh<ntt::Dimension::TWO_D>;
template class ntt::Mesh<ntt::Dimension::THREE_D>;

#if SIMTYPE == PIC_SIMTYPE
template class ntt::Meshblock<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
#elif SIMTYPE == GRPIC_SIMTYPE
template class ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
template class ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;
#endif