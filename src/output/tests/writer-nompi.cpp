#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "output/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;

void cleanup() {
  namespace fs = std::filesystem;
  fs::path tempfile_path { "test.h5" };
  fs::remove(tempfile_path);
}

#define CEILDIV(a, b)                                                          \
  (static_cast<int>(math::ceil(static_cast<real_t>(a) / static_cast<real_t>(b))))

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    constexpr auto nx1    = 10;
    constexpr auto nx1_gh = nx1 + 2 * N_GHOSTS;
    constexpr auto nx2    = 14;
    constexpr auto nx2_gh = nx2 + 2 * N_GHOSTS;
    constexpr auto nx3    = 17;
    constexpr auto nx3_gh = nx3 + 2 * N_GHOSTS;
    constexpr auto i1min  = N_GHOSTS;
    constexpr auto i2min  = N_GHOSTS;
    constexpr auto i3min  = N_GHOSTS;
    constexpr auto i1max  = nx1 + N_GHOSTS;
    constexpr auto i2max  = nx2 + N_GHOSTS;
    constexpr auto i3max  = nx3 + N_GHOSTS;

    constexpr auto dwn1 = 2;
    constexpr auto dwn2 = 1;
    constexpr auto dwn3 = 5;

    ndfield_t<Dim::_3D, 3>   field { "fld", nx1_gh, nx2_gh, nx3_gh };
    std::vector<std::string> field_names;

    {
      // fill data
      Kokkos::parallel_for(
        "fill",
        CreateRangePolicy<Dim::_3D>({ i1min, i2min, i3min },
                                    { i1max, i2max, i3max }),
        Lambda(index_t i1, index_t i2, index_t i3) {
          const auto i1_       = static_cast<real_t>(i1);
          const auto i2_       = static_cast<real_t>(i2);
          const auto i3_       = static_cast<real_t>(i3);
          field(i1, i2, i3, 0) = i1_;
          field(i1, i2, i3, 1) = i2_;
          field(i1, i2, i3, 2) = i3_;
        });
    }

    adios2::ADIOS adios;

    {
      // write
      auto writer = out::Writer();
      writer.init(&adios, "hdf5", "test", false);
      writer.defineMeshLayout({ nx1, nx2, nx3 },
                              { 0, 0, 0 },
                              { nx1, nx2, nx3 },
                              { dwn1, dwn2, dwn3 },
                              false,
                              Coord::Cart);
      writer.defineFieldOutputs(SimEngine::SRPIC, { "E", "B", "Rho_1_3", "N_2" });

      std::vector<std::size_t> addresses;
      for (auto i = 0; i < 3; ++i) {
        field_names.push_back(writer.fieldWriters()[0].name(i));
        addresses.push_back(i);
      }
      writer.beginWriting(WriteMode::Fields, 10, 123.0);
      writer.writeField<Dim::_3D, 3>(field_names, field, addresses);
      writer.endWriting(WriteMode::Fields);

      writer.beginWriting(WriteMode::Fields, 20, 123.4);
      writer.writeField<Dim::_3D, 3>(field_names, field, addresses);
      writer.endWriting(WriteMode::Fields);
    }

    adios.FlushAll();

    {
      // read
      adios2::IO io = adios.DeclareIO("read-test");
      io.SetEngine("hdf5");
      adios2::Engine reader = io.Open("test.h5", adios2::Mode::Read);
      const auto layoutRight = io.InquireAttribute<int>("LayoutRight").Data()[0] ==
                               1;

      raise::ErrorIf(io.InquireAttribute<unsigned int>("NGhosts").Data()[0] != 0,
                     "NGhosts is not correct",
                     HERE);
      raise::ErrorIf(io.InquireAttribute<std::size_t>("Dimension").Data()[0] != 3,
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
        raise::ErrorIf(step_read != (step + 1) * 10, "Step is not correct", HERE);
        raise::ErrorIf((float)time_read != 123 + (float)step * 0.4f,
                       "Time is not correct",
                       HERE);

        array_t<real_t***> field_read {};

        int cntr = 0;
        for (const auto& name : field_names) {
          auto fieldVar = io.InquireVariable<real_t>(name);
          if (fieldVar) {
            raise::ErrorIf(fieldVar.Shape().size() != 3,
                           fmt::format("%s is not 3D", name.c_str()),
                           HERE);

            auto        dims  = fieldVar.Shape();
            std::size_t nx1_r = dims[0];
            std::size_t nx2_r = dims[1];
            std::size_t nx3_r = dims[2];
            if (!layoutRight) {
              std::swap(nx1_r, nx3_r);
            }
            raise::ErrorIf((nx1_r != CEILDIV(nx1, dwn1)) ||
                             (nx2_r != CEILDIV(nx2, dwn2)) ||
                             (nx3_r != CEILDIV(nx3, dwn3)),
                           fmt::format("%s = %ldx%ldx%ld is not %dx%dx%d",
                                       name.c_str(),
                                       nx1_r,
                                       nx2_r,
                                       nx3_r,
                                       CEILDIV(nx1, dwn1),
                                       CEILDIV(nx2, dwn2),
                                       CEILDIV(nx3, dwn3)),
                           HERE);

            if (!layoutRight) {
              std::swap(nx1_r, nx3_r);
            }
            fieldVar.SetSelection(
              adios2::Box<adios2::Dims>({ 0, 0, 0 }, { nx1_r, nx2_r, nx3_r }));
            if (!layoutRight) {
              std::swap(nx1_r, nx3_r);
            }
            field_read        = array_t<real_t***>(name, nx1_r, nx2_r, nx3_r);
            auto field_read_h = Kokkos::create_mirror_view(field_read);
            reader.Get(fieldVar, field_read_h.data(), adios2::Mode::Sync);
            Kokkos::deep_copy(field_read, field_read_h);

            Kokkos::parallel_for(
              "check",
              CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { nx1_r, nx2_r, nx3_r }),
              Lambda(index_t i1, index_t i2, index_t i3) {
                if (not cmp::AlmostEqual(field_read(i1, i2, i3),
                                         field(i1 * dwn1 + i1min,
                                               i2 * dwn2 + i2min,
                                               i3 * dwn3 + i3min,
                                               cntr))) {
                  printf("\n:::::::::::::::\nfield_read(%ld, %ld, %ld) = %f != "
                         "field(%ld, %ld, %ld, %d) = %f\n:::::::::::::::\n",
                         i1,
                         i2,
                         i3,
                         field_read(i1, i2, i3),
                         i1 * dwn1 + i1min,
                         i2 * dwn2 + i2min,
                         i3 * dwn3 + i3min,
                         cntr,
                         field(i1 * dwn1 + i1min,
                               i2 * dwn2 + i2min,
                               i3 * dwn3 + i3min,
                               cntr));
                  raise::KernelError(HERE, "Field is not read correctly");
                }
              });
          } else {
            raise::Error("Field not found", HERE);
          }
          ++cntr;
        }
        reader.EndStep();
      }
      reader.Close();
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

#undef CEILDIV
