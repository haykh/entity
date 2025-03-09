#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"

#include "output/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <mpi.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

void cleanup() {
  namespace fs = std::filesystem;
  fs::path tempfile_path { "test.h5" };
  fs::remove(tempfile_path);
}

#define CEILDIV(a, b)                                                          \
  (static_cast<int>(math::ceil(static_cast<real_t>(a) / static_cast<real_t>(b))))

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  try {
    using namespace ntt;
    constexpr auto nx1    = 10;
    constexpr auto nx1_gh = nx1 + 2 * N_GHOSTS;
    constexpr auto i1min  = N_GHOSTS;
    constexpr auto i1max  = nx1 + N_GHOSTS;
    constexpr auto dwn1   = 3;

    ndfield_t<Dim::_1D, 3>   field { "fld", nx1_gh };
    std::vector<std::string> field_names;

    {
      // fill data
      Kokkos::parallel_for(
        "fill",
        CreateRangePolicy<Dim::_1D>({ i1min }, { i1max }),
        Lambda(index_t i1) {
          const auto i1_ = static_cast<real_t>(i1);
          field(i1, 0)   = i1_;
          field(i1, 1)   = -i1_;
          field(i1, 2)   = SQR(i1_);
        });
    }

    adios2::ADIOS adios { MPI_COMM_WORLD };

    {
      // write
      auto writer = out::Writer();
      writer.init(&adios, "hdf5", "test");
      writer.defineMeshLayout({ static_cast<unsigned long>(mpi_size) * nx1 },
                              { static_cast<unsigned long>(mpi_rank) * nx1 },
                              { nx1 },
                              { dwn1 },
                              false,
                              Coord::Cart);
      writer.defineFieldOutputs(SimEngine::SRPIC, { "E" });

      std::vector<std::size_t> addresses;
      for (auto i = 0; i < 3; ++i) {
        field_names.push_back(writer.fieldWriters()[0].name(i));
        addresses.push_back(i);
      }
      writer.beginWriting(0, 0.0);
      writer.writeField<Dim::_1D, 3>(field_names, field, addresses);
      writer.endWriting();

      writer.beginWriting(1, 0.1);
      writer.writeField<Dim::_1D, 3>(field_names, field, addresses);
      writer.endWriting();
      adios.ExitComputationBlock();
    }

    adios.FlushAll();

    {
      // read
      adios2::IO io = adios.DeclareIO("read-test");
      io.SetEngine("hdf5");
      adios2::Engine reader = io.Open("test.h5", adios2::Mode::Read, MPI_COMM_SELF);
      raise::ErrorIf(io.InquireAttribute<unsigned int>("NGhosts").Data()[0] != 0,
                     "NGhosts is not correct",
                     HERE);
      raise::ErrorIf(io.InquireAttribute<std::size_t>("Dimension").Data()[0] != 1,
                     "Dimension is not correct",
                     HERE);
      for (std::size_t step = 0; reader.BeginStep() == adios2::StepStatus::OK;
           ++step) {
        std::size_t step_read;
        long double time_read;

        reader.Get(io.InquireVariable<std::size_t>("Step"),
                   &step_read,
                   adios2::Mode::Sync);
        reader.Get(io.InquireVariable<long double>("Time"),
                   &time_read,
                   adios2::Mode::Sync);
        raise::ErrorIf(step_read != step, "Step is not correct", HERE);
        raise::ErrorIf((float)time_read != (float)step * 0.1f,
                       "Time is not correct",
                       HERE);

        const auto l_size   = nx1;
        const auto l_offset = nx1 * mpi_rank;
        const auto g_size   = nx1 * mpi_size;

        const double n = l_size;
        const double d = dwn1;
        const double l = l_offset;
        const double f = math::ceil(l / d) * d - l;

        const auto first_cell = static_cast<std::size_t>(f);
        const auto l_size_dwn = static_cast<std::size_t>(math::ceil((n - f) / d));
        const auto l_corner_dwn = static_cast<std::size_t>(math::ceil(l / d));

        array_t<real_t*> field_read {};
        int              cntr = 0;
        for (const auto& name : field_names) {
          auto fieldVar = io.InquireVariable<real_t>(name);
          if (fieldVar) {
            raise::ErrorIf(fieldVar.Shape().size() != 1,
                           fmt::format("%s is not 1D", name.c_str()),
                           HERE);
            auto        dims  = fieldVar.Shape();
            std::size_t nx1_r = dims[0];
            raise::ErrorIf((nx1_r != CEILDIV(nx1 * mpi_size, dwn1)),
                           fmt::format("%s = %ld is not %d",
                                       name.c_str(),
                                       nx1_r,
                                       CEILDIV(nx1 * mpi_size, dwn1)),
                           HERE);

            fieldVar.SetSelection(
              adios2::Box<adios2::Dims>({ l_corner_dwn }, { l_size_dwn }));
            field_read        = array_t<real_t*>(name, l_size_dwn);
            auto field_read_h = Kokkos::create_mirror_view(field_read);
            reader.Get(fieldVar, field_read_h.data(), adios2::Mode::Sync);
            Kokkos::deep_copy(field_read, field_read_h);

            Kokkos::parallel_for(
              "check",
              CreateRangePolicy<Dim::_1D>({ 0 }, { l_size_dwn }),
              Lambda(index_t i1) {
                if (not cmp::AlmostEqual(
                      field_read(i1),
                      field(i1 * dwn1 + first_cell + i1min, cntr))) {
                  printf("\n:::::::::::::::\nfield_read(%ld) = %f != "
                         "field(%ld, %d) = %f\n:::::::::::::::\n",
                         i1,
                         field_read(i1),
                         i1 * dwn1 + first_cell + i1min,
                         cntr,
                         field(i1 * dwn1 + first_cell + i1min, cntr));
                  raise::KernelError(HERE, "Field is not read correctly");
                }
              });
          } else {
            raise::Error("Field not found", HERE);
          }
          ++cntr;
        }
      }
      reader.Close();
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    CallOnce([]() {
      cleanup();
    });
    MPI_Finalize();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  MPI_Finalize();
  Kokkos::finalize();
  return 0;
}

#undef CEILDIV
