#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/containers/particles.h"
#include "framework/specialization_registry.h"
#include "output/utils/readers.h"
#include "output/utils/writers.h"

#include "kernels/prtls_to_phys.hpp"

#include <Kokkos_Core.hpp>
#include <adios2.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace ntt {
  /* * * * * * * * *
   * Output
   * * * * * * * * */
  template <Dimension D, Coord::type C>
  void Particles<D, C>::OutputDeclare(adios2::IO& io) const {
    const auto n_addition_coords = ((D == Dim::_2D) and (C != Coord::Cart)) ? 1
                                                                            : 0;
    for (auto d { 0u }; d < D + n_addition_coords; ++d) {
      io.DefineVariable<real_t>(fmt::format("pX%d_%d", d + 1, index()),
                                { adios2::UnknownDim },
                                { adios2::UnknownDim },
                                { adios2::UnknownDim });
    }
    for (auto d { 0u }; d < Dim::_3D; ++d) {
      io.DefineVariable<real_t>(fmt::format("pU%d_%d", d + 1, index()),
                                { adios2::UnknownDim },
                                { adios2::UnknownDim },
                                { adios2::UnknownDim });
    }
    io.DefineVariable<real_t>(fmt::format("pW_%d", index()),
                              { adios2::UnknownDim },
                              { adios2::UnknownDim },
                              { adios2::UnknownDim });
    if (npld_r() > 0) {
      for (auto pr { 0 }; pr < npld_r(); ++pr) {
        io.DefineVariable<real_t>(fmt::format("pPLDR%d_%d", pr, index()),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
      }
    }
    auto num_track_plds = 0;
    if (use_tracking()) {
      io.DefineVariable<npart_t>(fmt::format("pIDX_%d", index()),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
#if !defined(MPI_ENABLED)
      num_track_plds = 1;
#else
      num_track_plds = 2;
      io.DefineVariable<npart_t>(fmt::format("pRNK_%d", index()),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
#endif
    }
    if (npld_i() > num_track_plds) {
      for (auto pr { num_track_plds }; pr < npld_i(); ++pr) {
        io.DefineVariable<npart_t>(
          fmt::format("pPLDI%d_%d", pr - num_track_plds, index()),
          { adios2::UnknownDim },
          { adios2::UnknownDim },
          { adios2::UnknownDim });
      }
    }
  }

  template <Dimension D, Coord::type C>
  template <SimEngine::type S, class M>
  void Particles<D, C>::OutputWrite(adios2::IO&     io,
                                    adios2::Engine& writer,
                                    npart_t         prtl_stride,
                                    std::size_t     domains_total,
                                    std::size_t     domains_offset,
                                    const M&        metric) {
    if (not is_sorted()) {
      RemoveDead();
    }
    npart_t           nout;
    array_t<npart_t*> out_indices;
    if (!use_tracking()) {
      nout = npart() / prtl_stride;
    } else {
      nout               = 0u;
      const auto tag_d   = this->tag;
      const auto pld_i_d = this->pld_i;
      Kokkos::parallel_reduce(
        "CountOutputParticles",
        rangeActiveParticles(),
        Lambda(index_t p, npart_t & l_nout) {
          if ((tag_d(p) == ParticleTag::alive) and
              (pld_i_d(p, pldi::spcCtr) % prtl_stride == 0)) {
            l_nout += 1;
          }
        },
        nout);
      out_indices = array_t<npart_t*> { "out_indices", nout };
      array_t<npart_t> out_counter { "out_counter" };
      Kokkos::parallel_for(
        "RecordOutputIndices",
        rangeActiveParticles(),
        Lambda(index_t p) {
          if ((tag_d(p) == ParticleTag::alive) and
              (pld_i_d(p, pldi::spcCtr) % prtl_stride == 0)) {
            const auto p_out   = Kokkos::atomic_fetch_add(&out_counter(), 1);
            out_indices(p_out) = p;
          }
        });
    }

    npart_t nout_offset = 0;
    npart_t nout_total  = nout;
#if defined(MPI_ENABLED)
    auto nout_total_vec = std::vector<npart_t>(domains_total);
    MPI_Allgather(&nout,
                  1,
                  mpi::get_type<npart_t>(),
                  nout_total_vec.data(),
                  1,
                  mpi::get_type<npart_t>(),
                  MPI_COMM_WORLD);
    nout_total = 0;
    for (auto r = 0; r < domains_total; ++r) {
      if (r < domains_offset) {
        nout_offset += nout_total_vec[r];
      }
      nout_total += nout_total_vec[r];
    }
#endif // MPI_ENABLED

    array_t<real_t*> buff_x1, buff_x2, buff_x3;
    array_t<real_t*> buff_ux1 { "ux1", nout };
    array_t<real_t*> buff_ux2 { "ux2", nout };
    array_t<real_t*> buff_ux3 { "ux3", nout };
    array_t<real_t*> buff_wei { "w", nout };
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      buff_x1 = array_t<real_t*> { "x1", nout };
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      buff_x2 = array_t<real_t*> { "x2", nout };
    }
    if constexpr (D == Dim::_3D or ((D == Dim::_2D) and (C != Coord::Cart))) {
      buff_x3 = array_t<real_t*> { "x3", nout };
    }
    array_t<real_t**>  buff_pldr;
    array_t<npart_t**> buff_pldi;

    if (npld_r() > 0) {
      buff_pldr = array_t<real_t**> { "pldr", nout, npld_r() };
    }
    if (npld_i() > 0) {
      buff_pldi = array_t<npart_t**> { "pldi", nout, npld_i() };
    }

    if (nout > 0) {
      if (!use_tracking()) {
        // clang-format off
        Kokkos::parallel_for(
          "PrtlToPhys",
          nout,
          kernel::PrtlToPhys_kernel<S, M, false>(prtl_stride, out_indices,
                                                 buff_x1, buff_x2, buff_x3,
                                                 buff_ux1, buff_ux2, buff_ux3,
                                                 buff_wei, 
                                                 buff_pldr, buff_pldi,
                                                 i1, i2, i3,
                                                 dx1, dx2, dx3,
                                                 ux1, ux2, ux3,
                                                 phi, weight, 
                                                 pld_r, pld_i,
                                                 metric));
        // clang-format on
      } else {
        // clang-format off
        Kokkos::parallel_for(
          "PrtlToPhys",
          nout,
          kernel::PrtlToPhys_kernel<S, M, true>(prtl_stride, out_indices,
                                                buff_x1, buff_x2, buff_x3,
                                                buff_ux1, buff_ux2, buff_ux3,
                                                buff_wei, 
                                                buff_pldr, buff_pldi,
                                                i1, i2, i3,
                                                dx1, dx2, dx3,
                                                ux1, ux2, ux3,
                                                phi, weight, 
                                                pld_r, pld_i,
                                                metric));
        // clang-format on
      }
    }
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("pW_%d", index()),
                              buff_wei,
                              nout,
                              nout_total,
                              nout_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("pU1_%d", index()),
                              buff_ux1,
                              nout,
                              nout_total,
                              nout_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("pU2_%d", index()),
                              buff_ux2,
                              nout,
                              nout_total,
                              nout_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("pU3_%d", index()),
                              buff_ux3,
                              nout,
                              nout_total,
                              nout_offset);
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      out::Write1DArray<real_t>(io,
                                writer,
                                fmt::format("pX1_%d", index()),
                                buff_x1,
                                nout,
                                nout_total,
                                nout_offset);
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      out::Write1DArray<real_t>(io,
                                writer,
                                fmt::format("pX2_%d", index()),
                                buff_x2,
                                nout,
                                nout_total,
                                nout_offset);
    }
    if constexpr (D == Dim::_3D or ((D == Dim::_2D) and (C != Coord::Cart))) {
      out::Write1DArray<real_t>(io,
                                writer,
                                fmt::format("pX3_%d", index()),
                                buff_x3,
                                nout,
                                nout_total,
                                nout_offset);
    }

    if (npld_r() > 0) {
      for (auto pr { 0 }; pr < npld_r(); ++pr) {
        auto buff_sub = Kokkos::subview(buff_pldr, Kokkos::ALL, pr);
        out::Write1DSubArray<real_t, decltype(buff_sub)>(
          io,
          writer,
          fmt::format("pPLDR%d_%d", pr, index()),
          buff_sub,
          nout,
          nout_total,
          nout_offset);
      }
    }
    auto num_track_plds = 0;
    if (use_tracking()) {
#if !defined(MPI_ENABLED)
      num_track_plds = 1;
      {
        auto buff_sub = Kokkos::subview(buff_pldi,
                                        Kokkos::ALL,
                                        static_cast<std::size_t>(pldi::spcCtr));
        out::Write1DSubArray<npart_t, decltype(buff_sub)>(
          io,
          writer,
          fmt::format("pIDX_%d", index()),
          buff_sub,
          nout,
          nout_total,
          nout_offset);
      }
#else
      num_track_plds = 2;
      {
        auto buff_sub = Kokkos::subview(buff_pldi,
                                        Kokkos::ALL,
                                        static_cast<std::size_t>(pldi::spcCtr));
        out::Write1DSubArray<npart_t, decltype(buff_sub)>(
          io,
          writer,
          fmt::format("pIDX_%d", index()),
          buff_sub,
          nout,
          nout_total,
          nout_offset);
      }
      {
        auto buff_sub = Kokkos::subview(buff_pldi,
                                        Kokkos::ALL,
                                        static_cast<std::size_t>(pldi::domIdx));
        out::Write1DSubArray<npart_t, decltype(buff_sub)>(
          io,
          writer,
          fmt::format("pRNK_%d", index()),
          buff_sub,
          nout,
          nout_total,
          nout_offset);
      }
#endif
    }
    if (npld_i() > num_track_plds) {
      for (auto pr { num_track_plds }; pr < npld_i(); ++pr) {
        auto buff_sub = Kokkos::subview(buff_pldi,
                                        Kokkos::ALL,
                                        static_cast<std::size_t>(pr));
        out::Write1DSubArray<npart_t, decltype(buff_sub)>(
          io,
          writer,
          fmt::format("pPLDI%d_%d", pr - num_track_plds, index()),
          buff_sub,
          nout,
          nout_total,
          nout_offset);
      }
    }
  }

  /* * * * * * * * *
   * Checkpoints
   * * * * * * * * */

  template <Dimension D, Coord::type C>
  void Particles<D, C>::CheckpointDeclare(adios2::IO& io) const {
    logger::Checkpoint(
      fmt::format("Declaring particle checkpoint for species #%d", index()),
      HERE);

    io.DefineVariable<npart_t>(fmt::format("s%d_npart", index()),
                               { adios2::UnknownDim },
                               { adios2::UnknownDim },
                               { adios2::UnknownDim });
    io.DefineVariable<npart_t>(fmt::format("s%d_counter", index()),
                               { adios2::UnknownDim },
                               { adios2::UnknownDim },
                               { adios2::UnknownDim });
    for (auto d { 0u }; d < static_cast<unsigned short>(D); ++d) {
      io.DefineVariable<int>(fmt::format("s%d_i%d", index(), d + 1),
                             { adios2::UnknownDim },
                             { adios2::UnknownDim },
                             { adios2::UnknownDim });
      io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d", index(), d + 1),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
      io.DefineVariable<int>(fmt::format("s%d_i%d_prev", index(), d + 1),
                             { adios2::UnknownDim },
                             { adios2::UnknownDim },
                             { adios2::UnknownDim });
      io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d_prev", index(), d + 1),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
    }

    if constexpr (D == Dim::_2D and C != ntt::Coord::Cart) {
      io.DefineVariable<real_t>(fmt::format("s%d_phi", index()),
                                { adios2::UnknownDim },
                                { adios2::UnknownDim },
                                { adios2::UnknownDim });
    }

    for (auto d { 0u }; d < 3; ++d) {
      io.DefineVariable<real_t>(fmt::format("s%d_ux%d", index(), d + 1),
                                { adios2::UnknownDim },
                                { adios2::UnknownDim },
                                { adios2::UnknownDim });
    }

    io.DefineVariable<short>(fmt::format("s%d_tag", index()),
                             { adios2::UnknownDim },
                             { adios2::UnknownDim },
                             { adios2::UnknownDim });
    io.DefineVariable<real_t>(fmt::format("s%d_weight", index()),
                              { adios2::UnknownDim },
                              { adios2::UnknownDim },
                              { adios2::UnknownDim });
    if (npld_r() > 0) {
      io.DefineVariable<real_t>(fmt::format("s%d_pld_r", index()),
                                { adios2::UnknownDim, npld_r() },
                                { adios2::UnknownDim, 0 },
                                { adios2::UnknownDim, npld_r() });
    }
    if (npld_i() > 0) {
      io.DefineVariable<npart_t>(fmt::format("s%d_pld_i", index()),
                                 { adios2::UnknownDim, npld_i() },
                                 { adios2::UnknownDim, 0 },
                                 { adios2::UnknownDim, npld_i() });
    }
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::CheckpointRead(adios2::IO&     io,
                                       adios2::Engine& reader,
                                       std::size_t     domains_total,
                                       std::size_t     domains_offset) {
    logger::Checkpoint(
      fmt::format("Reading particle checkpoint for species #%d", index()),
      HERE);
    raise::ErrorIf(npart() > 0,
                   "Particles already initialized before reading checkpoint",
                   HERE);
    npart_t npart_offset = 0u;
    npart_t npart_read;

    out::ReadVariable<npart_t>(io,
                               reader,
                               fmt::format("s%d_npart", index()),
                               npart_read,
                               domains_offset);
    set_npart(npart_read);

#if defined(MPI_ENABLED)
    {
      const auto           npart_send = npart();
      std::vector<npart_t> glob_nparts(domains_total);
      MPI_Allgather(&npart_send,
                    1,
                    mpi::get_type<npart_t>(),
                    glob_nparts.data(),
                    1,
                    mpi::get_type<npart_t>(),
                    MPI_COMM_WORLD);
      for (auto d { 0u }; d < domains_offset; ++d) {
        npart_offset += glob_nparts[d];
      }
    }
#endif
    out::ReadVariable<npart_t>(io,
                               reader,
                               fmt::format("s%d_counter", index()),
                               m_counter,
                               domains_offset);

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i1", index()),
                            i1,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx1", index()),
                                 dx1,
                                 npart(),
                                 npart_offset);
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i1_prev", index()),
                            i1_prev,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx1_prev", index()),
                                 dx1_prev,
                                 npart(),
                                 npart_offset);
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i2", index()),
                            i2,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx2", index()),
                                 dx2,
                                 npart(),
                                 npart_offset);
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i2_prev", index()),
                            i2_prev,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx2_prev", index()),
                                 dx2_prev,
                                 npart(),
                                 npart_offset);
    }

    if constexpr (D == Dim::_3D) {
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i3", index()),
                            i3,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx3", index()),
                                 dx3,
                                 npart(),
                                 npart_offset);
      out::Read1DArray<int>(io,
                            reader,
                            fmt::format("s%d_i3_prev", index()),
                            i3_prev,
                            npart(),
                            npart_offset);
      out::Read1DArray<prtldx_t>(io,
                                 reader,
                                 fmt::format("s%d_dx3_prev", index()),
                                 dx3_prev,
                                 npart(),
                                 npart_offset);
    }

    if constexpr (D == Dim::_2D and C != Coord::Cart) {
      out::Read1DArray<real_t>(io,
                               reader,
                               fmt::format("s%d_phi", index()),
                               phi,
                               npart(),
                               npart_offset);
    }

    out::Read1DArray<real_t>(io,
                             reader,
                             fmt::format("s%d_ux1", index()),
                             ux1,
                             npart(),
                             npart_offset);
    out::Read1DArray<real_t>(io,
                             reader,
                             fmt::format("s%d_ux2", index()),
                             ux2,
                             npart(),
                             npart_offset);
    out::Read1DArray<real_t>(io,
                             reader,
                             fmt::format("s%d_ux3", index()),
                             ux3,
                             npart(),
                             npart_offset);
    out::Read1DArray<short>(io,
                            reader,
                            fmt::format("s%d_tag", index()),
                            tag,
                            npart(),
                            npart_offset);
    out::Read1DArray<real_t>(io,
                             reader,
                             fmt::format("s%d_weight", index()),
                             weight,
                             npart(),
                             npart_offset);

    if (npld_r() > 0) {
      out::Read2DArray<real_t>(io,
                               reader,
                               fmt::format("s%d_pld_r", index()),
                               pld_r,
                               npld_r(),
                               npart(),
                               npart_offset);
    }

    if (npld_i() > 0) {
      out::Read2DArray<npart_t>(io,
                                reader,
                                fmt::format("s%d_pld_i", index()),
                                pld_i,
                                npld_i(),
                                npart(),
                                npart_offset);
    }
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::CheckpointWrite(adios2::IO&     io,
                                        adios2::Engine& writer,
                                        std::size_t     domains_total,
                                        std::size_t     domains_offset) const {
    logger::Checkpoint(
      fmt::format("Writing particle checkpoint for species #%d", index()),
      HERE);

    npart_t npart_offset = 0u;
    npart_t npart_total  = npart();

#if defined(MPI_ENABLED)
    {
      std::vector<npart_t> glob_nparts(domains_total);
      MPI_Allgather(&m_npart,
                    1,
                    mpi::get_type<npart_t>(),
                    glob_nparts.data(),
                    1,
                    mpi::get_type<npart_t>(),
                    MPI_COMM_WORLD);
      npart_total = 0u;
      for (auto r = 0; r < domains_total; ++r) {
        if (r < domains_offset) {
          npart_offset += glob_nparts[r];
        }
        npart_total += glob_nparts[r];
      }
    }
#endif

    out::WriteVariable<npart_t>(io,
                                writer,
                                fmt::format("s%d_npart", index()),
                                npart(),
                                domains_total,
                                domains_offset);
    out::WriteVariable<npart_t>(io,
                                writer,
                                fmt::format("s%d_counter", index()),
                                npart(),
                                domains_total,
                                domains_offset);

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i1", index()),
                             i1,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx1", index()),
                                  dx1,
                                  npart(),
                                  npart_total,
                                  npart_offset);
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i1_prev", index()),
                             i1_prev,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx1_prev", index()),
                                  dx1_prev,
                                  npart(),
                                  npart_total,
                                  npart_offset);
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i2", index()),
                             i2,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx2", index()),
                                  dx2,
                                  npart(),
                                  npart_total,
                                  npart_offset);
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i2_prev", index()),
                             i2_prev,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx2_prev", index()),
                                  dx2_prev,
                                  npart(),
                                  npart_total,
                                  npart_offset);
    }

    if constexpr (D == Dim::_3D) {
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i3", index()),
                             i3,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx3", index()),
                                  dx3,
                                  npart(),
                                  npart_total,
                                  npart_offset);
      out::Write1DArray<int>(io,
                             writer,
                             fmt::format("s%d_i3_prev", index()),
                             i3_prev,
                             npart(),
                             npart_total,
                             npart_offset);
      out::Write1DArray<prtldx_t>(io,
                                  writer,
                                  fmt::format("s%d_dx3_prev", index()),
                                  dx3_prev,
                                  npart(),
                                  npart_total,
                                  npart_offset);
    }

    if constexpr (D == Dim::_2D and C != Coord::Cart) {
      out::Write1DArray<real_t>(io,
                                writer,
                                fmt::format("s%d_phi", index()),
                                phi,
                                npart(),
                                npart_total,
                                npart_offset);
    }

    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("s%d_ux1", index()),
                              ux1,
                              npart(),
                              npart_total,
                              npart_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("s%d_ux2", index()),
                              ux2,
                              npart(),
                              npart_total,
                              npart_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("s%d_ux3", index()),
                              ux3,
                              npart(),
                              npart_total,
                              npart_offset);
    out::Write1DArray<short>(io,
                             writer,
                             fmt::format("s%d_tag", index()),
                             tag,
                             npart(),
                             npart_total,
                             npart_offset);
    out::Write1DArray<real_t>(io,
                              writer,
                              fmt::format("s%d_weight", index()),
                              weight,
                              npart(),
                              npart_total,
                              npart_offset);
    if (npld_r() > 0) {
      out::Write2DArray<real_t>(io,
                                writer,
                                fmt::format("s%d_pld_r", index()),
                                pld_r,
                                npld_r(),
                                npart(),
                                npart_total,
                                npart_offset);
    }

    if (npld_i() > 0) {
      out::Write2DArray<npart_t>(io,
                                 writer,
                                 fmt::format("s%d_pld_i", index()),
                                 pld_i,
                                 npld_i(),
                                 npart(),
                                 npart_total,
                                 npart_offset);
    }
  }

#define PARTICLES_OUTPUT_DECLARE(D, C)                                         \
  template void Particles<D, C>::OutputDeclare(adios2::IO&) const;

  PARTICLES_OUTPUT_DECLARE(Dim::_1D, Coord::Cart)
  PARTICLES_OUTPUT_DECLARE(Dim::_2D, Coord::Cart)
  PARTICLES_OUTPUT_DECLARE(Dim::_3D, Coord::Cart)
  PARTICLES_OUTPUT_DECLARE(Dim::_2D, Coord::Sph)
  PARTICLES_OUTPUT_DECLARE(Dim::_2D, Coord::Qsph)
#undef PARTICLES_OUTPUT_DECLARE

#define PARTICLES_OUTPUT_WRITE(E, M, D)                                         \
  template void Particles<M<D>::Dim, M<D>::CoordType>::OutputWrite<E<M<D>>::S, \
                                                                   M<D>>(      \
    adios2::IO&,                                                               \
    adios2::Engine&,                                                           \
    npart_t,                                                                   \
    std::size_t,                                                               \
    std::size_t,                                                               \
    const M<D>&);

  NTT_FOREACH_SPECIALIZATION(PARTICLES_OUTPUT_WRITE)
#undef PARTICLES_OUTPUT_WRITE

#define PARTICLES_CHECKPOINTS(D, C)                                            \
  template void Particles<D, C>::CheckpointDeclare(adios2::IO&) const;         \
  template void Particles<D, C>::CheckpointRead(adios2::IO&,                   \
                                                adios2::Engine&,               \
                                                std::size_t,                   \
                                                std::size_t);                  \
  template void Particles<D, C>::CheckpointWrite(adios2::IO&,                  \
                                                 adios2::Engine&,              \
                                                 std::size_t,                  \
                                                 std::size_t) const;

  PARTICLES_CHECKPOINTS(Dim::_1D, Coord::Cart)
  PARTICLES_CHECKPOINTS(Dim::_2D, Coord::Cart)
  PARTICLES_CHECKPOINTS(Dim::_3D, Coord::Cart)
  PARTICLES_CHECKPOINTS(Dim::_2D, Coord::Sph)
  PARTICLES_CHECKPOINTS(Dim::_2D, Coord::Qsph)
  PARTICLES_CHECKPOINTS(Dim::_3D, Coord::Sph)
  PARTICLES_CHECKPOINTS(Dim::_3D, Coord::Qsph)
#undef PARTICLES_CHECKPOINTS

} // namespace ntt
