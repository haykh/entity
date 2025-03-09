#include "enums.h"
#include "global.h"

#include "utils/comparators.h"

#include "checkpoint/reader.h"
#include "checkpoint/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <mpi.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace ntt;
using namespace checkpoint;

void cleanup() {
  namespace fs = std::filesystem;
  fs::path temp_path { "checkpoints" };
  fs::remove_all(temp_path);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    // assuming 4 ranks
    // |------|------|
    // |  2   |  3   |
    // |------|------|
    // |      |      |
    // |  0   |  1   |
    // |------|------|
    const std::size_t g_nx1    = 20;
    const std::size_t g_nx2    = 15;
    const std::size_t g_nx1_gh = g_nx1 + 4 * N_GHOSTS;
    const std::size_t g_nx2_gh = g_nx2 + 4 * N_GHOSTS;

    const std::size_t l_nx1 = 10;
    const std::size_t l_nx2 = (rank < 2) ? 10 : 5;

    const std::size_t l_nx1_gh = l_nx1 + 2 * N_GHOSTS;
    const std::size_t l_nx2_gh = l_nx2 + 2 * N_GHOSTS;

    const std::size_t l_corner_x1 = (rank % 2 == 0) ? 0 : l_nx1_gh;
    const std::size_t l_corner_x2 = (rank < 2) ? 0 : l_nx2_gh;

    const std::size_t i1min = N_GHOSTS;
    const std::size_t i2min = N_GHOSTS;
    const std::size_t i1max = l_nx1 + N_GHOSTS;
    const std::size_t i2max = l_nx2 + N_GHOSTS;

    const std::size_t npart1 = (rank % 2 + rank) * 23 + 100;
    const std::size_t npart2 = (rank % 2 + rank) * 37 + 100;

    std::size_t npart1_offset = 0;
    std::size_t npart2_offset = 0;

    std::size_t npart1_globtot = 0;
    std::size_t npart2_globtot = 0;

    for (auto r = 0; r < rank - 1; ++r) {
      npart1_offset += (r % 2 + r) * 23 + 100;
      npart2_offset += (r % 2 + r) * 37 + 100;
    }

    for (auto r = 0; r < size; ++r) {
      npart1_globtot += (r % 2 + r) * 23 + 100;
      npart2_globtot += (r % 2 + r) * 37 + 100;
    }

    // init data
    ndfield_t<Dim::_2D, 6> field1 { "fld1", l_nx1_gh, l_nx2_gh };
    ndfield_t<Dim::_2D, 6> field2 { "fld2", l_nx1_gh, l_nx2_gh };

    array_t<int*>     i1 { "i_1", npart1 };
    array_t<real_t*>  u1 { "u_1", npart1 };
    array_t<int*>     i2 { "i_2", npart2 };
    array_t<real_t*>  u2 { "u_2", npart2 };
    array_t<real_t**> plds1 { "plds_1", npart1, 3 };

    {
      // fill data
      Kokkos::parallel_for(
        "fillFlds",
        CreateRangePolicy<Dim::_2D>({ i1min, i2min }, { i1max, i2max }),
        Lambda(index_t i1, index_t i2) {
          field1(i1, i2, 0) = static_cast<real_t>(i1 + i2);
          field1(i1, i2, 1) = static_cast<real_t>(i1 * i2);
          field1(i1, i2, 2) = static_cast<real_t>(i1 / i2);
          field1(i1, i2, 3) = static_cast<real_t>(i1 - i2);
          field1(i1, i2, 4) = static_cast<real_t>(i2 / i1);
          field1(i1, i2, 5) = static_cast<real_t>(i1);
          field2(i1, i2, 0) = static_cast<real_t>(-(i1 + i2));
          field2(i1, i2, 1) = static_cast<real_t>(-(i1 * i2));
          field2(i1, i2, 2) = static_cast<real_t>(-(i1 / i2));
          field2(i1, i2, 3) = static_cast<real_t>(-(i1 - i2));
          field2(i1, i2, 4) = static_cast<real_t>(-(i2 / i1));
          field2(i1, i2, 5) = static_cast<real_t>(-i1);
        });
      Kokkos::parallel_for(
        "fillPrtl1",
        npart1,
        Lambda(index_t p) {
          u1(p)       = static_cast<real_t>(p);
          i1(p)       = static_cast<int>(p);
          plds1(p, 0) = static_cast<real_t>(p);
          plds1(p, 1) = static_cast<real_t>(p * p);
          plds1(p, 2) = static_cast<real_t>(p * p * p);
        });
      Kokkos::parallel_for(
        "fillPrtl2",
        npart2,
        Lambda(index_t p) {
          u2(p) = -static_cast<real_t>(p);
          i2(p) = -static_cast<int>(p);
        });
    }

    adios2::ADIOS adios;

    {
      // write checkpoint
      Writer writer;
      writer.init(&adios, 0, 0.0, 1);

      writer.defineFieldVariables(SimEngine::GRPIC,
                                  { g_nx1_gh, g_nx2_gh },
                                  { l_corner_x1, l_corner_x2 },
                                  { l_nx1_gh, l_nx2_gh });

      writer.defineParticleVariables(Coord::Sph, Dim::_2D, 2, { 3, 0 });

      writer.beginSaving(0, 0.0);

      writer.saveField<Dim::_2D, 6>("em", field1);
      writer.saveField<Dim::_2D, 6>("em0", field2);

      writer.savePerDomainVariable<std::size_t>("s1_npart", 1, 0, npart1);
      writer.savePerDomainVariable<std::size_t>("s2_npart", 1, 0, npart2);

      writer.saveParticleQuantity<int>("s1_i1",
                                       npart1_globtot,
                                       npart1_offset,
                                       npart1,
                                       i1);
      writer.saveParticleQuantity<real_t>("s1_ux1",
                                          npart1_globtot,
                                          npart1_offset,
                                          npart1,
                                          u1);
      writer.saveParticleQuantity<int>("s2_i1",
                                       npart2_globtot,
                                       npart2_offset,
                                       npart2,
                                       i2);
      writer.saveParticleQuantity<real_t>("s2_ux1",
                                          npart2_globtot,
                                          npart2_offset,
                                          npart2,
                                          u2);

      writer.saveParticlePayloads("s1_plds",
                                  3,
                                  npart1_globtot,
                                  npart1_offset,
                                  npart1,
                                  plds1);

      writer.endSaving();
    }

    {
      // read checkpoint
      ndfield_t<Dim::_2D, 6> field1_read { "fld1_read", l_nx1_gh, l_nx2_gh };
      ndfield_t<Dim::_2D, 6> field2_read { "fld2_read", l_nx1_gh, l_nx2_gh };

      array_t<int*>     i1_read { "i_1", npart1 };
      array_t<real_t*>  u1_read { "u_1", npart1 };
      array_t<int*>     i2_read { "i_2", npart2 };
      array_t<real_t*>  u2_read { "u_2", npart2 };
      array_t<real_t**> plds1_read { "plds_1", npart1, 3 };

      adios2::IO     io     = adios.DeclareIO("checkpointRead");
      adios2::Engine reader = io.Open("checkpoints/step-00000000.bp",
                                      adios2::Mode::Read);
      reader.BeginStep();

      auto fieldRange = adios2::Box<adios2::Dims>({ l_corner_x1, l_corner_x2, 0 },
                                                  { l_nx1_gh, l_nx2_gh, 6 });
      ReadFields<Dim::_2D, 6>(io, reader, "em", fieldRange, field1_read);
      ReadFields<Dim::_2D, 6>(io, reader, "em0", fieldRange, field2_read);

      auto [nprtl1, noff1] = ReadParticleCount(io, reader, 0, rank, size);
      auto [nprtl2, noff2] = ReadParticleCount(io, reader, 1, rank, size);

      ReadParticleData<real_t>(io, reader, "ux1", 0, u1_read, nprtl1, noff1);
      ReadParticleData<real_t>(io, reader, "ux1", 1, u2_read, nprtl2, noff2);
      ReadParticleData<int>(io, reader, "i1", 0, i1_read, nprtl1, noff1);
      ReadParticleData<int>(io, reader, "i1", 1, i2_read, nprtl2, noff2);
      ReadParticlePayloads(io, reader, 0, plds1_read, 3, nprtl1, noff1);

      reader.EndStep();
      reader.Close();

      // check the validity
      Kokkos::parallel_for(
        "checkFields",
        CreateRangePolicy<Dim::_2D>({ 0, 0 }, { l_nx1_gh, l_nx2_gh }),
        Lambda(index_t i1, index_t i2) {
          for (int i = 0; i < 6; ++i) {
            if (not cmp::AlmostEqual(field1(i1, i2, i), field1_read(i1, i2, i))) {
              raise::KernelError(HERE, "Field1 read failed");
            }
            if (not cmp::AlmostEqual(field2(i1, i2, i), field2_read(i1, i2, i))) {
              raise::KernelError(HERE, "Field2 read failed");
            }
          }
        });

      raise::ErrorIf(npart1 != nprtl1, "Particle count 1 mismatch", HERE);
      raise::ErrorIf(npart2 != nprtl2, "Particle count 2 mismatch", HERE);
      raise::ErrorIf(noff1 != npart1_offset, "Particle offset 1 mismatch", HERE);
      raise::ErrorIf(noff2 != npart2_offset, "Particle offset 2 mismatch", HERE);

      Kokkos::parallel_for(
        "checkPrtl1",
        nprtl1,
        Lambda(index_t p) {
          if (not cmp::AlmostEqual(u1(p), u1_read(p))) {
            raise::KernelError(HERE, "u1 read failed");
          }
          if (i1(p) != i1_read(p)) {
            raise::KernelError(HERE, "i1 read failed");
          }
          for (auto l = 0; l < 3; ++l) {
            if (not cmp::AlmostEqual(plds1(p, l), plds1_read(p, l))) {
              raise::KernelError(HERE, "plds1 read failed");
            }
          }
        });
      Kokkos::parallel_for(
        "checkPrtl2",
        nprtl2,
        Lambda(index_t p) {
          if (not cmp::AlmostEqual(u2(p), u2_read(p))) {
            raise::KernelError(HERE, "u2 read failed");
          }
          if (i2(p) != i2_read(p)) {
            raise::KernelError(HERE, "i2 read failed");
          }
        });
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    cleanup();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  Kokkos::finalize();
  return 0;
}
