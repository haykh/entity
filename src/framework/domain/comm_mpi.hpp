/**
 * @file framework/domain/comm_mpi.hpp
 * @brief MPI communication routines
 * @implements
 *   - comm::CommunicateField<> -> void
 * @namespaces:
 *   - comm::
 * @note This should only be included if the MPI_ENABLED flag is set
 */

#ifndef FRAMEWORK_DOMAIN_COMM_MPI_HPP
#define FRAMEWORK_DOMAIN_COMM_MPI_HPP

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "arch/mpi_tags.h"
#include "utils/error.h"

#include "framework/containers/particles.h"

#include "kernels/comm.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <numeric>
#include <vector>

namespace comm {
  using namespace ntt;

  template <Dimension D, int N>
  inline void CommunicateField(unsigned int                      idx,
                               ndfield_t<D, N>&                  fld,
                               ndfield_t<D, N>&                  fld_buff,
                               unsigned int                      send_idx,
                               unsigned int                      recv_idx,
                               int                               send_rank,
                               int                               recv_rank,
                               const std::vector<range_tuple_t>& send_slice,
                               const std::vector<range_tuple_t>& recv_slice,
                               const range_tuple_t&              comps,
                               bool                              additive) {
    raise::ErrorIf(send_rank < 0 && recv_rank < 0,
                   "CommunicateField called with negative ranks",
                   HERE);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    raise::ErrorIf(
      (send_rank == rank && send_idx != idx) ||
        (recv_rank == rank && recv_idx != idx),
      "Multiple-domain single-rank communication not yet implemented",
      HERE);
    if ((send_idx == idx) and (recv_idx == idx)) {
      //  trivial copy if sending to self and receiving from self

      if (not additive) {

        // simply filling the ghost cells
        if constexpr (D == Dim::_1D) {
          Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], comps),
                            Kokkos::subview(fld, send_slice[0], comps));
        } else if constexpr (D == Dim::_2D) {
          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], comps),
            Kokkos::subview(fld, send_slice[0], send_slice[1], comps));
        } else if constexpr (D == Dim::_3D) {

          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], recv_slice[2], comps),
            Kokkos::subview(fld, send_slice[0], send_slice[1], send_slice[2], comps));
        }
      } else {
        // adding received fields to ghosts + active
        if constexpr (D == Dim::_1D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
              { recv_slice[0].first, comps.first },
              { recv_slice[0].second, comps.second }),
            Lambda(index_t i1, index_t ci) {
              fld_buff(i1, ci) += fld(i1 - offset_x1, ci);
            });
        } else if constexpr (D == Dim::_2D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          const auto offset_x2 = (long int)(recv_slice[1].first) -
                                 (long int)(send_slice[1].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
              { recv_slice[0].first, recv_slice[1].first, comps.first },
              { recv_slice[0].second, recv_slice[1].second, comps.second }),
            Lambda(index_t i1, index_t i2, index_t ci) {
              fld_buff(i1, i2, ci) += fld(i1 - offset_x1, i2 - offset_x2, ci);
            });
        } else if constexpr (D == Dim::_3D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          const auto offset_x2 = (long int)(recv_slice[1].first) -
                                 (long int)(send_slice[1].first);
          const auto offset_x3 = (long int)(recv_slice[2].first) -
                                 (long int)(send_slice[2].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, AccelExeSpace>(
              { recv_slice[0].first,
                recv_slice[1].first,
                recv_slice[2].first,
                comps.first },
              { recv_slice[0].second,
                recv_slice[1].second,
                recv_slice[2].second,
                comps.second }),
            Lambda(index_t i1, index_t i2, index_t i3, index_t ci) {
              fld_buff(i1, i2, i3, ci) += fld(i1 - offset_x1,
                                              i2 - offset_x2,
                                              i3 - offset_x3,
                                              ci);
            });
        }
      }
    } else {
      std::size_t nsend { comps.second - comps.first },
        nrecv { comps.second - comps.first };
      ndarray_t<static_cast<unsigned short>(D) + 1> send_fld, recv_fld;

      for (short d { 0 }; d < (short)D; ++d) {
        if (send_rank >= 0) {
          nsend *= (send_slice[d].second - send_slice[d].first);
        }
        if (recv_rank >= 0) {
          nrecv *= (recv_slice[d].second - recv_slice[d].first);
        }
      }
      if (send_rank >= 0) {
        if constexpr (D == Dim::_1D) {
          send_fld = ndarray_t<2>("send_fld",
                                  send_slice[0].second - send_slice[0].first,
                                  comps.second - comps.first);
          Kokkos::deep_copy(send_fld, Kokkos::subview(fld, send_slice[0], comps));
        } else if constexpr (D == Dim::_2D) {
          send_fld = ndarray_t<3>("send_fld",
                                  send_slice[0].second - send_slice[0].first,
                                  send_slice[1].second - send_slice[1].first,
                                  comps.second - comps.first);
          Kokkos::deep_copy(
            send_fld,
            Kokkos::subview(fld, send_slice[0], send_slice[1], comps));
        } else if constexpr (D == Dim::_3D) {
          send_fld = ndarray_t<4>("send_fld",
                                  send_slice[0].second - send_slice[0].first,
                                  send_slice[1].second - send_slice[1].first,
                                  send_slice[2].second - send_slice[2].first,
                                  comps.second - comps.first);
          Kokkos::deep_copy(
            send_fld,
            Kokkos::subview(fld, send_slice[0], send_slice[1], send_slice[2], comps));
        }
      }
      if (recv_rank >= 0) {
        if constexpr (D == Dim::_1D) {
          recv_fld = ndarray_t<2>("recv_fld",
                                  recv_slice[0].second - recv_slice[0].first,
                                  comps.second - comps.first);
        } else if constexpr (D == Dim::_2D) {
          recv_fld = ndarray_t<3>("recv_fld",
                                  recv_slice[0].second - recv_slice[0].first,
                                  recv_slice[1].second - recv_slice[1].first,
                                  comps.second - comps.first);
        } else if constexpr (D == Dim::_3D) {
          recv_fld = ndarray_t<4>("recv_fld",
                                  recv_slice[0].second - recv_slice[0].first,
                                  recv_slice[1].second - recv_slice[1].first,
                                  recv_slice[2].second - recv_slice[2].first,
                                  comps.second - comps.first);
        }
      }

      if (send_rank >= 0 && recv_rank >= 0) {
        MPI_Sendrecv(send_fld.data(),
                     nsend,
                     mpi::get_type<real_t>(),
                     send_rank,
                     0,
                     recv_fld.data(),
                     nrecv,
                     mpi::get_type<real_t>(),
                     recv_rank,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
      } else if (send_rank >= 0) {
        MPI_Send(send_fld.data(),
                 nsend,
                 mpi::get_type<real_t>(),
                 send_rank,
                 0,
                 MPI_COMM_WORLD);

      } else if (recv_rank >= 0) {
        MPI_Recv(recv_fld.data(),
                 nrecv,
                 mpi::get_type<real_t>(),
                 recv_rank,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        raise::Error("CommunicateField called with negative ranks", HERE);
      }

      if (recv_rank >= 0) {

        // !TODO: perhaps directly recv to the fld?
        if (not additive) {
          if constexpr (D == Dim::_1D) {
            Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], comps), recv_fld);
          } else if constexpr (D == Dim::_2D) {
            Kokkos::deep_copy(
              Kokkos::subview(fld, recv_slice[0], recv_slice[1], comps),
              recv_fld);
          } else if constexpr (D == Dim::_3D) {
            Kokkos::deep_copy(
              Kokkos::subview(fld, recv_slice[0], recv_slice[1], recv_slice[2], comps),
              recv_fld);
          }
        } else {
          if constexpr (D == Dim::_1D) {
            const auto offset_x1 = recv_slice[0].first;
            const auto offset_c  = comps.first;
            Kokkos::parallel_for(
              "CommunicateField-extract",
              Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
                { recv_slice[0].first, comps.first },
                { recv_slice[0].second, comps.second }),
              Lambda(index_t i1, index_t ci) {
                fld_buff(i1, ci) += recv_fld(i1 - offset_x1, ci - offset_c);
              });
          } else if constexpr (D == Dim::_2D) {
            const auto offset_x1 = recv_slice[0].first;
            const auto offset_x2 = recv_slice[1].first;
            const auto offset_c  = comps.first;
            Kokkos::parallel_for(
              "CommunicateField-extract",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
                { recv_slice[0].first, recv_slice[1].first, comps.first },
                { recv_slice[0].second, recv_slice[1].second, comps.second }),
              Lambda(index_t i1, index_t i2, index_t ci) {
                fld_buff(i1, i2, ci) += recv_fld(i1 - offset_x1,
                                                 i2 - offset_x2,
                                                 ci - offset_c);
              });
          } else if constexpr (D == Dim::_3D) {
            const auto offset_x1 = recv_slice[0].first;
            const auto offset_x2 = recv_slice[1].first;
            const auto offset_x3 = recv_slice[2].first;
            const auto offset_c  = comps.first;
            Kokkos::parallel_for(
              "CommunicateField-extract",
              Kokkos::MDRangePolicy<Kokkos::Rank<4>, AccelExeSpace>(
                { recv_slice[0].first,
                  recv_slice[1].first,
                  recv_slice[2].first,
                  comps.first },
                { recv_slice[0].second,
                  recv_slice[1].second,
                  recv_slice[2].second,
                  comps.second }),
              Lambda(index_t i1, index_t i2, index_t i3, index_t ci) {
                fld_buff(i1, i2, i3, ci) += recv_fld(i1 - offset_x1,
                                                     i2 - offset_x2,
                                                     i3 - offset_x3,
                                                     ci - offset_c);
              });
          }
        }
      }
    }
  }

  void ParticleSendRecvCount(int          send_rank,
                             int          recv_rank,
                             std::size_t  send_count,
                             std::size_t& recv_count) {
    if ((send_rank >= 0) && (recv_rank >= 0)) {
      MPI_Sendrecv(&send_count,
                   1,
                   mpi::get_type<std::size_t>(),
                   send_rank,
                   0,
                   &recv_count,
                   1,
                   mpi::get_type<std::size_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_rank >= 0) {
      MPI_Send(&send_count, 1, mpi::get_type<std::size_t>(), send_rank, 0, MPI_COMM_WORLD);
    } else if (recv_rank >= 0) {
      MPI_Recv(&recv_count,
               1,
               mpi::get_type<std::size_t>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      raise::Error("ParticleSendRecvCount called with negative ranks", HERE);
    }
  }

  template <Dimension D, Coord::type C>
  void CommunicateParticles(Particles<D, C>&                species,
                            const array_t<std::size_t*>&    outgoing_indices,
                            const array_t<std::size_t*>&    tag_offsets,
                            const std::vector<std::size_t>& npptag_vec,
                            const std::vector<std::size_t>& npptag_recv_vec,
                            const std::vector<int>&         send_ranks,
                            const std::vector<int>&         recv_ranks,
                            const dir::dirs_t<D>&           dirs_to_comm) {
    // number of arrays of each type to send/recv
    const unsigned short NREALS = 4 + static_cast<unsigned short>(
                                        D == Dim::_2D and C != Coord::Cart);
    const unsigned short NINTS   = 2 * static_cast<unsigned short>(D);
    const unsigned short NPRTLDX = 2 * static_cast<unsigned short>(D);
    const unsigned short NPLDS   = species.npld();

    // buffers to store recv data
    const auto       npart_alive = npptag_vec[ParticleTag::alive];
    const auto       npart_dead  = npptag_vec[ParticleTag::dead];
    const auto       npart_send  = outgoing_indices.extent(0) - npart_dead;
    const auto       npart_recv  = std::accumulate(npptag_recv_vec.begin(),
                                            npptag_recv_vec.end(),
                                            static_cast<std::size_t>(0));
    array_t<int*>    recv_buff_int { "recv_buff_int", npart_recv * NINTS };
    array_t<real_t*> recv_buff_real { "recv_buff_real", npart_recv * NREALS };
    array_t<prtldx_t*> recv_buff_prtldx { "recv_buff_prtldx", npart_recv * NPRTLDX };
    array_t<real_t*> recv_buff_pld;

    if (NPLDS > 0) {
      recv_buff_pld = array_t<real_t*> { "recv_buff_pld", npart_recv * NPLDS };
    }

    auto iteration        = 0;
    auto current_received = 0;

    for (const auto& direction : dirs_to_comm) {
      const auto send_rank     = send_ranks[iteration];
      const auto recv_rank     = recv_ranks[iteration];
      const auto tag_send      = mpi::PrtlSendTag<D>::dir2tag(direction);
      const auto tag_recv      = mpi::PrtlSendTag<D>::dir2tag(-direction);
      const auto npart_send_in = npptag_vec[tag_send];
      const auto npart_recv_in = npptag_recv_vec[tag_recv - 2];
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      array_t<int*> send_buff_int { "send_buff_int", npart_send_in * NINTS };
      array_t<real_t*> send_buff_real { "send_buff_real", npart_send_in * NREALS };
      array_t<prtldx_t*> send_buff_prtldx { "send_buff_prtldx",
                                            npart_send_in * NPRTLDX };
      array_t<real_t*>   send_buff_pld;
      if (NPLDS > 0) {
        send_buff_pld = array_t<real_t*> { "send_buff_pld", npart_send_in * NPLDS };
      }

      auto tag_offsets_h = Kokkos::create_mirror_view(tag_offsets);
      Kokkos::deep_copy(tag_offsets_h, tag_offsets);

      std::size_t idx_offset = npart_dead;
      if (tag_send > 2) {
        idx_offset += tag_offsets_h(tag_send - 3);
      }
      // clang-format off
      Kokkos::parallel_for(
        "PopulatePrtlSendBuffer",
        npart_send_in,
        kernel::comm::PopulatePrtlSendBuffer_kernel<D, C>(
          send_buff_int, send_buff_real, send_buff_prtldx, send_buff_pld,
          NINTS, NREALS, NPRTLDX, NPLDS, idx_offset,
          species.i1, species.i1_prev, species.dx1, species.dx1_prev,
          species.i2, species.i2_prev, species.dx2, species.dx2_prev,
          species.i3, species.i3_prev, species.dx3, species.dx3_prev,
          species.ux1, species.ux2, species.ux3, 
          species.weight, species.phi, species.pld, species.tag,
          outgoing_indices)
      );
      // clang-format on

      const auto recv_offset_int    = current_received * NINTS;
      const auto recv_offset_real   = current_received * NREALS;
      const auto recv_offset_prtldx = current_received * NPRTLDX;
      const auto recv_offset_pld    = current_received * NPLDS;

      if ((send_rank >= 0) and (recv_rank >= 0) and (npart_send_in > 0) and
          (npart_recv_in > 0)) {
        raise::ErrorIf(recv_offset_int + npart_recv_in * NINTS >
                         recv_buff_int.extent(0),
                       "incorrect # of recv particles",
                       HERE);
        MPI_Sendrecv(send_buff_int.data(),
                     npart_send_in * NINTS,
                     mpi::get_type<int>(),
                     send_rank,
                     0,
                     recv_buff_int.data() + recv_offset_int,
                     npart_recv_in * NINTS,
                     mpi::get_type<int>(),
                     recv_rank,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buff_real.data(),
                     npart_send_in * NREALS,
                     mpi::get_type<real_t>(),
                     send_rank,
                     0,
                     recv_buff_real.data() + recv_offset_real,
                     npart_recv_in * NREALS,
                     mpi::get_type<real_t>(),
                     recv_rank,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buff_prtldx.data(),
                     npart_send_in * NPRTLDX,
                     mpi::get_type<prtldx_t>(),
                     send_rank,
                     0,
                     recv_buff_prtldx.data() + recv_offset_prtldx,
                     npart_recv_in * NPRTLDX,
                     mpi::get_type<prtldx_t>(),
                     recv_rank,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (NPLDS > 0) {
          MPI_Sendrecv(send_buff_pld.data(),
                       npart_send_in * NPLDS,
                       mpi::get_type<real_t>(),
                       send_rank,
                       0,
                       recv_buff_pld.data() + recv_offset_pld,
                       npart_recv_in * NPLDS,
                       mpi::get_type<real_t>(),
                       recv_rank,
                       0,
                       MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
        }
      } else if ((send_rank >= 0) and (npart_send_in > 0)) {
        MPI_Send(send_buff_int.data(),
                 npart_send_in * NINTS,
                 mpi::get_type<int>(),
                 send_rank,
                 0,
                 MPI_COMM_WORLD);
        MPI_Send(send_buff_real.data(),
                 npart_send_in * NREALS,
                 mpi::get_type<real_t>(),
                 send_rank,
                 0,
                 MPI_COMM_WORLD);
        MPI_Send(send_buff_prtldx.data(),
                 npart_send_in * NPRTLDX,
                 mpi::get_type<prtldx_t>(),
                 send_rank,
                 0,
                 MPI_COMM_WORLD);
        if (NPLDS > 0) {
          MPI_Send(send_buff_pld.data(),
                   npart_send_in * NPLDS,
                   mpi::get_type<real_t>(),
                   send_rank,
                   0,
                   MPI_COMM_WORLD);
        }
      } else if ((recv_rank >= 0) and (npart_recv_in > 0)) {
        raise::ErrorIf(recv_offset_int + npart_recv_in * NINTS >
                         recv_buff_int.extent(0),
                       "incorrect # of recv particles",
                       HERE);
        MPI_Recv(recv_buff_int.data() + recv_offset_int,
                 npart_recv_in * NINTS,
                 mpi::get_type<int>(),
                 recv_rank,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(recv_buff_real.data() + recv_offset_real,
                 npart_recv_in * NREALS,
                 mpi::get_type<real_t>(),
                 recv_rank,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(recv_buff_prtldx.data() + recv_offset_prtldx,
                 npart_recv_in * NPRTLDX,
                 mpi::get_type<prtldx_t>(),
                 recv_rank,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        if (NPLDS > 0) {
          MPI_Recv(recv_buff_pld.data() + recv_offset_pld,
                   npart_recv_in * NPLDS,
                   mpi::get_type<real_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
        }
      }
      current_received += npart_recv_in;
      iteration++;

    } // end direction loop

    // clang-format off
    Kokkos::parallel_for(
      "PopulateFromRecvBuffer",
      npart_recv,
      kernel::comm::ExtractReceivedPrtls_kernel<D, C>(
            recv_buff_int, recv_buff_real, recv_buff_prtldx, recv_buff_pld,
            NINTS, NREALS, NPRTLDX, NPLDS,
            species.npart(),
            species.i1, species.i1_prev, species.dx1, species.dx1_prev,
            species.i2, species.i2_prev, species.dx2, species.dx2_prev,
            species.i3, species.i3_prev, species.dx3, species.dx3_prev,
            species.ux1, species.ux2, species.ux3,
            species.weight, species.phi, species.pld, species.tag,
            outgoing_indices)
    );
    // clang-format on

    const auto npart       = species.npart();
    const auto npart_holes = outgoing_indices.extent(0);
    if (npart_recv > npart_holes) {
      species.set_npart(npart + npart_recv - npart_holes);
    }
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_MPI_HPP
