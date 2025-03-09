/**
 * @file arch/mpi_tags.h
 * @brief MPI tags for particle communication
 * @implements
 *   - mpi::PrtlSendTag<>
 *   - mpi::SendTag<> -> short
 * @namespaces:
 *   - mpi::
 */
#ifndef GLOBAL_ARCH_MPI_TAGS_H
#define GLOBAL_ARCH_MPI_TAGS_H

#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace mpi {
  using namespace dir;

  template <Dimension D>
  struct PrtlSendTag {
    inline static auto tag2dir(short tag) -> direction_t<D> {
      raise::ErrorIf((tag - 2 < 0) || (tag - 2 >= Directions<D>::all.size()),
                     "Invalid tag",
                     HERE);
      return Directions<D>::all[tag - 2];
    }

    inline static auto dir2tag(const direction_t<D>&) -> short;
  };

  template <>
  struct PrtlSendTag<Dim::_1D> {
    inline static constexpr short im1 { 2 };
    inline static constexpr short ip1 { 3 };

    inline static auto dir2tag(const direction_t<Dim::_1D>& dir) -> short {
      if (dir == direction_t<Dim::_1D>({ -1 })) {
        return im1;
      } else if (dir == direction_t<Dim::_1D>({ 1 })) {
        return ip1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim::_2D> {
    inline static constexpr short im1_jm1 { 2 };
    inline static constexpr short im1__j0 { 3 };
    inline static constexpr short im1_jp1 { 4 };
    inline static constexpr short i0__jm1 { 5 };
    inline static constexpr short i0__jp1 { 6 };
    inline static constexpr short ip1_jm1 { 7 };
    inline static constexpr short ip1__j0 { 8 };
    inline static constexpr short ip1_jp1 { 9 };

    inline static auto dir2tag(const direction_t<Dim::_2D>& dir) -> short {
      if (dir == direction_t<Dim::_2D>({ -1, -1 })) {
        return im1_jm1;
      } else if (dir == direction_t<Dim::_2D>({ -1, 0 })) {
        return im1__j0;
      } else if (dir == direction_t<Dim::_2D>({ -1, 1 })) {
        return im1_jp1;
      } else if (dir == direction_t<Dim::_2D>({ 0, -1 })) {
        return i0__jm1;
      } else if (dir == direction_t<Dim::_2D>({ 0, 1 })) {
        return i0__jp1;
      } else if (dir == direction_t<Dim::_2D>({ 1, -1 })) {
        return ip1_jm1;
      } else if (dir == direction_t<Dim::_2D>({ 1, 0 })) {
        return ip1__j0;
      } else if (dir == direction_t<Dim::_2D>({ 1, 1 })) {
        return ip1_jp1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim::_3D> {
    inline static constexpr short im1_jm1_km1 { 2 };
    inline static constexpr short im1_jm1__k0 { 3 };
    inline static constexpr short im1_jm1_kp1 { 4 };
    inline static constexpr short im1__j0_km1 { 5 };
    inline static constexpr short im1__j0__k0 { 6 };
    inline static constexpr short im1__j0_kp1 { 7 };
    inline static constexpr short im1_jp1_km1 { 8 };
    inline static constexpr short im1_jp1__k0 { 9 };
    inline static constexpr short im1_jp1_kp1 { 10 };
    inline static constexpr short i0__jm1_km1 { 11 };
    inline static constexpr short i0__jm1__k0 { 12 };
    inline static constexpr short i0__jm1_kp1 { 13 };
    inline static constexpr short i0___j0_km1 { 14 };
    inline static constexpr short i0___j0_kp1 { 15 };
    inline static constexpr short i0__jp1_km1 { 16 };
    inline static constexpr short i0__jp1__k0 { 17 };
    inline static constexpr short i0__jp1_kp1 { 18 };
    inline static constexpr short ip1_jm1_km1 { 19 };
    inline static constexpr short ip1_jm1__k0 { 20 };
    inline static constexpr short ip1_jm1_kp1 { 21 };
    inline static constexpr short ip1__j0_km1 { 22 };
    inline static constexpr short ip1__j0__k0 { 23 };
    inline static constexpr short ip1__j0_kp1 { 24 };
    inline static constexpr short ip1_jp1_km1 { 25 };
    inline static constexpr short ip1_jp1__k0 { 26 };
    inline static constexpr short ip1_jp1_kp1 { 27 };

    inline static auto dir2tag(const direction_t<Dim::_3D>& dir) -> short {
      if (dir == direction_t<Dim::_3D>({ -1, -1, -1 })) {
        return im1_jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, -1, 0 })) {
        return im1_jm1__k0;
      } else if (dir == direction_t<Dim::_3D>({ -1, -1, 1 })) {
        return im1_jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, -1 })) {
        return im1__j0_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, 0 })) {
        return im1__j0__k0;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, 1 })) {
        return im1__j0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, -1 })) {
        return im1_jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, 0 })) {
        return im1_jp1__k0;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, 1 })) {
        return im1_jp1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, -1 })) {
        return i0__jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, 0 })) {
        return i0__jm1__k0;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, 1 })) {
        return i0__jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 0, -1 })) {
        return i0___j0_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 0, 1 })) {
        return i0___j0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, -1 })) {
        return i0__jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, 0 })) {
        return i0__jp1__k0;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, 1 })) {
        return i0__jp1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, -1 })) {
        return ip1_jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, 0 })) {
        return ip1_jm1__k0;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, 1 })) {
        return ip1_jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, -1 })) {
        return ip1__j0_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, 0 })) {
        return ip1__j0__k0;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, 1 })) {
        return ip1__j0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, -1 })) {
        return ip1_jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, 0 })) {
        return ip1_jp1__k0;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, 1 })) {
        return ip1_jp1_kp1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  Inline auto SendTag(short tag, bool im1, bool ip1) -> short {
    return ((im1) * (PrtlSendTag<Dim::_1D>::im1 - 1) +
            (ip1) * (PrtlSendTag<Dim::_1D>::ip1 - 1) + 1) *
           tag;
  }

  Inline auto SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1) -> short {
    return ((im1 && jm1) * (PrtlSendTag<Dim::_2D>::im1_jm1 - 1) +
            (im1 && jp1) * (PrtlSendTag<Dim::_2D>::im1_jp1 - 1) +
            (ip1 && jm1) * (PrtlSendTag<Dim::_2D>::ip1_jm1 - 1) +
            (ip1 && jp1) * (PrtlSendTag<Dim::_2D>::ip1_jp1 - 1) +
            (im1 && !jp1 && !jm1) * (PrtlSendTag<Dim::_2D>::im1__j0 - 1) +
            (ip1 && !jp1 && !jm1) * (PrtlSendTag<Dim::_2D>::ip1__j0 - 1) +
            (jm1 && !ip1 && !im1) * (PrtlSendTag<Dim::_2D>::i0__jm1 - 1) +
            (jp1 && !ip1 && !im1) * (PrtlSendTag<Dim::_2D>::i0__jp1 - 1) + 1) *
           tag;
  }

  Inline auto SendTag(short tag,
                      bool  im1,
                      bool  ip1,
                      bool  jm1,
                      bool  jp1,
                      bool  km1,
                      bool  kp1) -> short {
    return ((im1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::im1_jm1_km1 - 1) +
            (im1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::im1_jm1_kp1 - 1) +
            (im1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::im1_jp1_km1 - 1) +
            (im1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::im1_jp1_kp1 - 1) +
            (ip1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::ip1_jm1_km1 - 1) +
            (ip1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1_jm1_kp1 - 1) +
            (ip1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::ip1_jp1_km1 - 1) +
            (ip1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1_jp1_kp1 - 1) +
            (im1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::im1_jm1__k0 - 1) +
            (im1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::im1_jp1__k0 - 1) +
            (ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::ip1_jm1__k0 - 1) +
            (ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::ip1_jp1__k0 - 1) +
            (im1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim::_3D>::im1__j0_km1 - 1) +
            (im1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim::_3D>::im1__j0_kp1 - 1) +
            (ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim::_3D>::ip1__j0_km1 - 1) +
            (ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1__j0_kp1 - 1) +
            (!im1 && !ip1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::i0__jm1_km1 - 1) +
            (!im1 && !ip1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::i0__jm1_kp1 - 1) +
            (!im1 && !ip1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::i0__jp1_km1 - 1) +
            (!im1 && !ip1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::i0__jp1_kp1 - 1) +
            (!im1 && !ip1 && !jm1 && !jp1 && km1) *
              (PrtlSendTag<Dim::_3D>::i0___j0_km1 - 1) +
            (!im1 && !ip1 && !jm1 && !jp1 && kp1) *
              (PrtlSendTag<Dim::_3D>::i0___j0_kp1 - 1) +
            (!im1 && !ip1 && jm1 && !km1 && !kp1) *
              (PrtlSendTag<Dim::_3D>::i0__jm1__k0 - 1) +
            (!im1 && !ip1 && jp1 && !km1 && !kp1) *
              (PrtlSendTag<Dim::_3D>::i0__jp1__k0 - 1) +
            (im1 && !jm1 && !jp1 && !km1 && !kp1) *
              (PrtlSendTag<Dim::_3D>::im1__j0__k0 - 1) +
            (ip1 && !jm1 && !jp1 && !km1 && !kp1) *
              (PrtlSendTag<Dim::_3D>::ip1__j0__k0 - 1) +
            1) *
           tag;
  }
} // namespace mpi

#endif // GLOBAL_ARCH_MPI_TAGS_H
