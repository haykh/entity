/**
 * @file kernels/comm.hpp
 * @brief Kernels used during communications
 * @implements
 *   - kernel::comm::PrepareOutgoingPrtls_kernel<>
 *   - kernel::comm::PopulatePrtlSendBuffer_kernel<>
 *   - kernel::comm::ExtractReceivedPrtls_kernel<>
 * @namespaces:
 *   - kernel::comm::
 */

#ifndef KERNELS_COMM_HPP
#define KERNELS_COMM_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include <Kokkos_Core.hpp>

namespace kernel::comm {
  using namespace ntt;

  template <Dimension D>
  class PrepareOutgoingPrtls_kernel {
    const array_t<int*>   shifts_in_x1, shifts_in_x2, shifts_in_x3;
    array_t<std::size_t*> outgoing_indices;

    const std::size_t npart, npart_alive, npart_dead, ntags;

    array_t<int*>         i1, i1_prev, i2, i2_prev, i3, i3_prev;
    const array_t<short*> tag;

    const array_t<std::size_t*> tag_offsets;

    array_t<std::size_t*> current_offset;

  public:
    PrepareOutgoingPrtls_kernel(const array_t<int*>&         shifts_in_x1,
                                const array_t<int*>&         shifts_in_x2,
                                const array_t<int*>&         shifts_in_x3,
                                array_t<std::size_t*>&       outgoing_indices,
                                std::size_t                  npart,
                                std::size_t                  npart_alive,
                                std::size_t                  npart_dead,
                                std::size_t                  ntags,
                                array_t<int*>&               i1,
                                array_t<int*>&               i1_prev,
                                array_t<int*>&               i2,
                                array_t<int*>&               i2_prev,
                                array_t<int*>&               i3,
                                array_t<int*>&               i3_prev,
                                const array_t<short*>&       tag,
                                const array_t<std::size_t*>& tag_offsets)
      : shifts_in_x1 { shifts_in_x1 }
      , shifts_in_x2 { shifts_in_x2 }
      , shifts_in_x3 { shifts_in_x3 }
      , outgoing_indices { outgoing_indices }
      , npart { npart }
      , npart_alive { npart_alive }
      , npart_dead { npart_dead }
      , ntags { ntags }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , tag { tag }
      , tag_offsets { tag_offsets }
      , current_offset { "current_offset", ntags } {}

    Inline void operator()(index_t p) const {
      if (tag(p) != ParticleTag::alive) {
        // dead or to-be-sent
        auto idx_for_tag = Kokkos::atomic_fetch_add(&current_offset(tag(p)), 1);
        if (tag(p) != ParticleTag::dead) {
          idx_for_tag += npart_dead;
        }
        if (tag(p) > 2) {
          idx_for_tag += tag_offsets(tag(p) - 3);
        }
        if (idx_for_tag >= npart - npart_alive) {
          raise::KernelError(HERE, "Outgoing indices idx exceeds the array size");
        }
        outgoing_indices(idx_for_tag) = p;
        // apply offsets
        if (tag(p) != ParticleTag::dead) {
          if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
            i1(p)      += shifts_in_x1(tag(p) - 2);
            i1_prev(p) += shifts_in_x1(tag(p) - 2);
          }
          if constexpr (D == Dim::_2D or D == Dim::_3D) {
            i2(p)      += shifts_in_x2(tag(p) - 2);
            i2_prev(p) += shifts_in_x2(tag(p) - 2);
          }
          if constexpr (D == Dim::_3D) {
            i3(p)      += shifts_in_x3(tag(p) - 2);
            i3_prev(p) += shifts_in_x3(tag(p) - 2);
          }
        }
      }
    }
  };

  template <Dimension D, Coord::type C>
  class PopulatePrtlSendBuffer_kernel {
    array_t<int*>      send_buff_int;
    array_t<real_t*>   send_buff_real;
    array_t<prtldx_t*> send_buff_prtldx;
    array_t<real_t*>   send_buff_pld;

    const unsigned short NINTS, NREALS, NPRTLDX, NPLDS;
    const std::size_t    idx_offset;

    const array_t<int*>         i1, i1_prev, i2, i2_prev, i3, i3_prev;
    const array_t<prtldx_t*>    dx1, dx1_prev, dx2, dx2_prev, dx3, dx3_prev;
    const array_t<real_t*>      ux1, ux2, ux3, weight, phi;
    const array_t<real_t**>     pld;
    array_t<short*>             tag;
    const array_t<std::size_t*> outgoing_indices;

  public:
    PopulatePrtlSendBuffer_kernel(array_t<int*>&               send_buff_int,
                                  array_t<real_t*>&            send_buff_real,
                                  array_t<prtldx_t*>&          send_buff_prtldx,
                                  array_t<real_t*>&            send_buff_pld,
                                  unsigned short               NINTS,
                                  unsigned short               NREALS,
                                  unsigned short               NPRTLDX,
                                  unsigned short               NPLDS,
                                  std::size_t                  idx_offset,
                                  const array_t<int*>&         i1,
                                  const array_t<int*>&         i1_prev,
                                  const array_t<prtldx_t*>&    dx1,
                                  const array_t<prtldx_t*>&    dx1_prev,
                                  const array_t<int*>&         i2,
                                  const array_t<int*>&         i2_prev,
                                  const array_t<prtldx_t*>&    dx2,
                                  const array_t<prtldx_t*>&    dx2_prev,
                                  const array_t<int*>&         i3,
                                  const array_t<int*>&         i3_prev,
                                  const array_t<prtldx_t*>&    dx3,
                                  const array_t<prtldx_t*>&    dx3_prev,
                                  const array_t<real_t*>&      ux1,
                                  const array_t<real_t*>&      ux2,
                                  const array_t<real_t*>&      ux3,
                                  const array_t<real_t*>&      weight,
                                  const array_t<real_t*>&      phi,
                                  const array_t<real_t**>&     pld,
                                  array_t<short*>&             tag,
                                  const array_t<std::size_t*>& outgoing_indices)
      : send_buff_int { send_buff_int }
      , send_buff_real { send_buff_real }
      , send_buff_prtldx { send_buff_prtldx }
      , send_buff_pld { send_buff_pld }
      , NINTS { NINTS }
      , NREALS { NREALS }
      , NPRTLDX { NPRTLDX }
      , NPLDS { NPLDS }
      , idx_offset { idx_offset }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx1_prev { dx1_prev }
      , dx2 { dx2 }
      , dx2_prev { dx2_prev }
      , dx3 { dx3 }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , weight { weight }
      , phi { phi }
      , pld { pld }
      , tag { tag }
      , outgoing_indices { outgoing_indices } {}

    Inline void operator()(index_t p) const {
      const auto idx = outgoing_indices(idx_offset + p);
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        send_buff_int(NINTS * p + 0)      = i1(idx);
        send_buff_int(NINTS * p + 1)      = i1_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 0) = dx1(idx);
        send_buff_prtldx(NPRTLDX * p + 1) = dx1_prev(idx);
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        send_buff_int(NINTS * p + 2)      = i2(idx);
        send_buff_int(NINTS * p + 3)      = i2_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 2) = dx2(idx);
        send_buff_prtldx(NPRTLDX * p + 3) = dx2_prev(idx);
      }
      if constexpr (D == Dim::_3D) {
        send_buff_int(NINTS * p + 4)      = i3(idx);
        send_buff_int(NINTS * p + 5)      = i3_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 4) = dx3(idx);
        send_buff_prtldx(NPRTLDX * p + 5) = dx3_prev(idx);
      }
      send_buff_real(NREALS * p + 0) = ux1(idx);
      send_buff_real(NREALS * p + 1) = ux2(idx);
      send_buff_real(NREALS * p + 2) = ux3(idx);
      send_buff_real(NREALS * p + 3) = weight(idx);
      if constexpr (D == Dim::_2D and C != Coord::Cart) {
        send_buff_real(NREALS * p + 4) = phi(idx);
      }
      if (NPLDS > 0) {
        for (auto l { 0u }; l < NPLDS; ++l) {
          send_buff_pld(NPLDS * p + l) = pld(idx, l);
        }
      }
      tag(idx) = ParticleTag::dead;
    }
  };

  template <Dimension D, Coord::type C>
  class ExtractReceivedPrtls_kernel {
    const array_t<int*>      recv_buff_int;
    const array_t<real_t*>   recv_buff_real;
    const array_t<prtldx_t*> recv_buff_prtldx;
    const array_t<real_t*>   recv_buff_pld;

    const unsigned short NINTS, NREALS, NPRTLDX, NPLDS;
    const std::size_t    npart, npart_holes;

    array_t<int*>               i1, i1_prev, i2, i2_prev, i3, i3_prev;
    array_t<prtldx_t*>          dx1, dx1_prev, dx2, dx2_prev, dx3, dx3_prev;
    array_t<real_t*>            ux1, ux2, ux3, weight, phi;
    array_t<real_t**>           pld;
    array_t<short*>             tag;
    const array_t<std::size_t*> outgoing_indices;

  public:
    ExtractReceivedPrtls_kernel(const array_t<int*>&         recv_buff_int,
                                const array_t<real_t*>&      recv_buff_real,
                                const array_t<prtldx_t*>&    recv_buff_prtldx,
                                const array_t<real_t*>&      recv_buff_pld,
                                unsigned short               NINTS,
                                unsigned short               NREALS,
                                unsigned short               NPRTLDX,
                                unsigned short               NPLDS,
                                std::size_t                  npart,
                                array_t<int*>&               i1,
                                array_t<int*>&               i1_prev,
                                array_t<prtldx_t*>&          dx1,
                                array_t<prtldx_t*>&          dx1_prev,
                                array_t<int*>&               i2,
                                array_t<int*>&               i2_prev,
                                array_t<prtldx_t*>&          dx2,
                                array_t<prtldx_t*>&          dx2_prev,
                                array_t<int*>&               i3,
                                array_t<int*>&               i3_prev,
                                array_t<prtldx_t*>&          dx3,
                                array_t<prtldx_t*>&          dx3_prev,
                                array_t<real_t*>&            ux1,
                                array_t<real_t*>&            ux2,
                                array_t<real_t*>&            ux3,
                                array_t<real_t*>&            weight,
                                array_t<real_t*>&            phi,
                                array_t<real_t**>&           pld,
                                array_t<short*>&             tag,
                                const array_t<std::size_t*>& outgoing_indices)
      : recv_buff_int { recv_buff_int }
      , recv_buff_real { recv_buff_real }
      , recv_buff_prtldx { recv_buff_prtldx }
      , recv_buff_pld { recv_buff_pld }
      , NINTS { NINTS }
      , NREALS { NREALS }
      , NPRTLDX { NPRTLDX }
      , NPLDS { NPLDS }
      , npart { npart }
      , npart_holes { outgoing_indices.extent(0) }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx1_prev { dx1_prev }
      , dx2 { dx2 }
      , dx2_prev { dx2_prev }
      , dx3 { dx3 }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , weight { weight }
      , phi { phi }
      , pld { pld }
      , tag { tag }
      , outgoing_indices { outgoing_indices } {}

    Inline void operator()(index_t p) const {
      std::size_t idx;
      if (p >= npart_holes) {
        idx = npart + p - npart_holes;
      } else {
        idx = outgoing_indices(p);
      }
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        i1(idx)       = recv_buff_int(NINTS * p + 0);
        i1_prev(idx)  = recv_buff_int(NINTS * p + 1);
        dx1(idx)      = recv_buff_prtldx(NPRTLDX * p + 0);
        dx1_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 1);
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        i2(idx)       = recv_buff_int(NINTS * p + 2);
        i2_prev(idx)  = recv_buff_int(NINTS * p + 3);
        dx2(idx)      = recv_buff_prtldx(NPRTLDX * p + 2);
        dx2_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 3);
      }
      if constexpr (D == Dim::_3D) {
        i3(idx)       = recv_buff_int(NINTS * p + 4);
        i3_prev(idx)  = recv_buff_int(NINTS * p + 5);
        dx3(idx)      = recv_buff_prtldx(NPRTLDX * p + 4);
        dx3_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 5);
      }
      ux1(idx)    = recv_buff_real(NREALS * p + 0);
      ux2(idx)    = recv_buff_real(NREALS * p + 1);
      ux3(idx)    = recv_buff_real(NREALS * p + 2);
      weight(idx) = recv_buff_real(NREALS * p + 3);
      if constexpr (D == Dim::_2D and C != Coord::Cart) {
        phi(idx) = recv_buff_real(NREALS * p + 4);
      }
      if (NPLDS > 0) {
        for (auto l { 0u }; l < NPLDS; ++l) {
          pld(idx, l) = recv_buff_pld(NPLDS * p + l);
        }
      }
      tag(idx) = ParticleTag::alive;
    }
  };

} // namespace kernel::comm

#endif // KERNELS_COMM_HPP
