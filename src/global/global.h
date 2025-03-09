/**
 * @file global.h
 * @brief Global constants, macros, aliases and enums
 * @implements
 *   - ntt::N_GHOSTS
 *   - macro COORD
 *   - macro HERE
 *   - enum Dimension, enum Dim
 *   - enum ntt::em
 *   - enum ntt::cur
 *   - enum ntt::ParticleTag    // dead, alive
 *   - type PrepareOutputFlags
 *   - enum PrepareOutput
 *   - enum CellLayer           // allLayer, activeLayer, minGhostLayer,
 *                                 minActiveLayer, maxActiveLayer, maxGhostLayer
 *   - enum Idx                // U, D, T, XYZ, Sph, PU, PD
 *   - enum Crd                // Cd, Ph, XYZ, Sph
 *   - enum in                 // x1, x2, x3
 *   - enum bc_in                // Px1, Mx1, Px2, Mx2, Px3, Mx3
 *   - type box_region_t
 *   - files::LogFile, files::ErrFile, files::InfoFile
 *   - type prtldx_t
 *   - type real_t
 *   - type tuple_t
 *   - type list_t
 *   - type coord_t
 *   - type vec_t
 *   - type index_t
 *   - type range_tuple_t
 *   - ntt::GlobalInitialize -> void
 *   - ntt::GlobalFinalize -> void
 *   - enum ntt::DiagFlags
 * @cpp:
 *   - global.cpp
 * @namespaces:
 *   - ntt::
 *   - files::
 * @macros:
 *   - MPI_ENABLED
 * @note
 * CellLayer enum:
 *
 * min/max layers have N_GHOSTS cells
 *
 * allLayer:                 .* *|* * * * * * * * *\* *.
 * activeLayer:              .   |* * * * * * * * *\   .
 * minGhostLayer:            .* *|                 \   .
 * minActiveLayer:           .   |* *              \   .
 * maxActiveLayer:           .   |              * *\   .
 * maxGhostLayer:            .   |                 \* *.
 *
 * @example
 * Usage of the CellLayer enum:
 *
 * 1. box_region_t<Dim2>{ CellLayer::minGhostLayer, CellLayer::maxGhostLayer }
 * results in a region [ [ i1min - N_GHOSTS, i1min ),
 *                       [ i2max, i2max + N_GHOSTS ) ]
 * i.e.,
 *    . . . . . . . . . .
 *    .* *              .
 *    .* *              .
 *    .   ^= = = = =^   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   ^- - - - -^   .
 *    .                 .
 *    .                 .
 *    . . . . . . . . . .
 *
 * 2. box_region_t<Dim3>{ CellLayer::activeLayer,
 *                     CellLayer::maxActiveLayer,
 *                     CellLayer::allLayer }
 * results in a region [ [ i1min, i1max ),
 *                       [ i2max - N_GHOSTS, i2max ),
 *                       [ i3min - N_GHOSTS, i3max + N_GHOSTS ) ]
 * i.e., in x1 & x2 slice
 *    . . . . . . . . . .
 *    .                 .
 *    .                 .
 *    .   ^= = = = =^   .
 *    .   |* * * * *\   .
 *    .   |* * * * *\   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   ^- - - - -^   .
 *    .                 .
 *    .                 .
 *    . . . . . . . . . .
 */

#ifndef GLOBAL_GLOBAL_H
#define GLOBAL_GLOBAL_H

#include <limits>
#include <utility>
#include <vector>

#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20013

#define HERE __FILE__, __func__, __LINE__

namespace files {
  enum {
    LogFile = 1,
    ErrFile,
    InfoFile
  };
} // namespace files

namespace ntt {

  inline constexpr unsigned int N_GHOSTS = 2;
// Coordinate shift to account for ghost cells
#define COORD(I)                                                               \
  (static_cast<real_t>(static_cast<int>((I)) - static_cast<int>(N_GHOSTS)))

  enum em {
    ex1 = 0,
    ex2 = 1,
    ex3 = 2,
    dx1 = 0,
    dx2 = 1,
    dx3 = 2,
    bx1 = 3,
    bx2 = 4,
    bx3 = 5,
    hx1 = 3,
    hx2 = 4,
    hx3 = 5
  };

  enum cur {
    jx1 = 0,
    jx2 = 1,
    jx3 = 2
  };

  enum ParticleTag : short {
    dead = 0,
    alive
  };

  void GlobalInitialize(int argc, char* argv[]);

  void GlobalFinalize();

} // namespace ntt

/* global scope enums & aliases --------------------------------------------- */

enum Dimension : unsigned short {
  _1D = 1,
  _2D = 2,
  _3D = 3
};

enum class CellLayer {
  allLayer,
  activeLayer,
  minGhostLayer,
  minActiveLayer,
  maxActiveLayer,
  maxGhostLayer
};

enum class Idx {
  U,   // contravariant
  D,   // covariant
  T,   // tetrad
  XYZ, // Cartesian
  Sph, // spherical
  PU,  // physical contravariant
  PD,  // physical covariant
};

enum class Crd {
  Cd,  // code units
  Ph,  // physical units
  XYZ, // Cartesian
  Sph, // spherical
};

enum class in : unsigned short {
  x1 = 0,
  x2 = 1,
  x3 = 2,
};

enum class bc_in : short {
  Mx1 = -1,
  Px1 = 1,
  Mx2 = -2,
  Px2 = 2,
  Mx3 = -3,
  Px3 = 3,
};

template <Dimension D>
using box_region_t = CellLayer[D];

/* config flags ------------------------------------------------------------- */

namespace PrepareOutput {
  enum PrepareOutputFlags_ {
    None                        = 0,
    InterpToCellCenterFromEdges = 1 << 0,
    InterpToCellCenterFromFaces = 1 << 1,
    ConvertToHat                = 1 << 2,
    ConvertToPhysCntrv          = 1 << 3,
    ConvertToPhysCov            = 1 << 4,
  };
} // namespace PrepareOutput

typedef int PrepareOutputFlags;

namespace Timer {
  enum TimerFlags_ {
    None            = 0,
    PrintTotal      = 1 << 0,
    PrintTitle      = 1 << 1,
    AutoConvert     = 1 << 2,
    PrintOutput     = 1 << 3,
    PrintPrtlClear  = 1 << 4,
    PrintCheckpoint = 1 << 5,
    PrintNormed     = 1 << 6,
    Default         = PrintNormed | PrintTotal | PrintTitle | AutoConvert,
  };
} // namespace Timer

typedef int TimerFlags;

namespace Diag {
  enum DiagFlags_ {
    None     = 0,
    Progress = 1 << 0,
    Timers   = 1 << 1,
    Species  = 1 << 2,
    Colorful = 1 << 3,
    Default  = Progress | Timers | Species | Colorful,
  };

} // namespace Diag

typedef int DiagFlags;

namespace Comm {
  enum CommTags_ {
    None = 0,
    E    = 1 << 0,
    B    = 1 << 1,
    J    = 1 << 2,
    D    = 1 << 3,
    D0   = 1 << 4,
    B0   = 1 << 5,
    H    = 1 << 6,
    Bckp = 1 << 7,
    Buff = 1 << 8,
  };
} // namespace Comm

typedef int CommTags;

namespace WriteMode {
  enum WriteModeTags_ {
    None      = 0,
    Fields    = 1 << 0,
    Particles = 1 << 1,
    Spectra   = 1 << 2,
  };
} // namespace WriteMode

typedef int WriteModeTags;

namespace BC {
  enum BCTags_ {
    None = 0,
    Ex1  = 1 << 0,
    Ex2  = 1 << 1,
    Ex3  = 1 << 2,
    Bx1  = 1 << 3,
    Bx2  = 1 << 4,
    Bx3  = 1 << 5,
    Dx1  = 1 << 0,
    Dx2  = 1 << 1,
    Dx3  = 1 << 2,
    B    = Bx1 | Bx2 | Bx3,
    E    = Ex1 | Ex2 | Ex3,
    D    = Dx1 | Dx2 | Dx3,
  };
} // namespace BC

typedef int BCTags;

namespace Inj {
  enum InjTags_ {
    None        = 0,
    AssumeEmpty = 1 << 0,
  };
} // namespace Inj

typedef int InjTags;

/* aliases ------------------------------------------------------------------ */

using Dim = Dimension;

// Defining precision-based constants and types
using prtldx_t = float;
#if defined(SINGLE_PRECISION)
using real_t = float;
#else
using real_t = double;
#endif

namespace Range {
  constexpr std::pair<real_t, real_t> All = {
    -std::numeric_limits<real_t>::infinity(),
    std::numeric_limits<real_t>::infinity()
  };
  constexpr real_t Min = -std::numeric_limits<real_t>::infinity();
  constexpr real_t Max = std::numeric_limits<real_t>::infinity();
}; // namespace Range

// ND list alias
template <typename T, Dimension D>
using tuple_t = T[D];

// list alias of size N
template <typename T, unsigned short N>
using list_t = T[N];

// ND coordinate alias
template <Dimension D>
using coord_t = tuple_t<real_t, D>;

// ND vector alias
template <Dimension D>
using vec_t = tuple_t<real_t, D>;

using index_t = const std::size_t;
using idx_t   = unsigned short;

using range_tuple_t = std::pair<std::size_t, std::size_t>;

template <typename T>
using boundaries_t = std::vector<std::pair<T, T>>;

#endif // GLOBAL_GLOBAL_H
