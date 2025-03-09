#include "output/writer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/param_container.h"
#include "utils/tools.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

#include <filesystem>
#include <string>
#include <vector>

namespace out {

  void Writer::init(adios2::ADIOS*     ptr_adios,
                    const std::string& engine,
                    const std::string& title,
                    bool               use_separate_files) {
    m_separate_files = use_separate_files;
    m_engine         = engine;
    p_adios          = ptr_adios;

    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);

    m_io = p_adios->DeclareIO("Entity::Output");
    m_io.SetEngine(engine);

    m_io.DefineVariable<std::size_t>("Step");
    m_io.DefineVariable<long double>("Time");
    m_fname = title;
  }

  void Writer::addTracker(const std::string& type,
                          std::size_t        interval,
                          long double        interval_time) {
    m_trackers.insert({ type, tools::Tracker(type, interval, interval_time) });
  }

  auto Writer::shouldWrite(const std::string& type,
                           std::size_t        step,
                           long double        time) -> bool {
    if (m_trackers.find(type) != m_trackers.end()) {
      return m_trackers.at(type).shouldWrite(step, time);
    } else {
      raise::Error(fmt::format("Tracker type %s not found", type.c_str()), HERE);
      return false;
    }
  }

  void Writer::setMode(adios2::Mode mode) {
    m_mode = mode;
  }

  void Writer::defineMeshLayout(const std::vector<std::size_t>&  glob_shape,
                                const std::vector<std::size_t>&  loc_corner,
                                const std::vector<std::size_t>&  loc_shape,
                                const std::vector<unsigned int>& dwn,
                                bool                             incl_ghosts,
                                Coord                            coords) {
    m_flds_ghosts = incl_ghosts;
    m_dwn         = dwn;

    m_flds_g_shape  = glob_shape;
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;

    for (std::size_t i { 0 }; i < glob_shape.size(); ++i) {
      raise::ErrorIf(dwn[i] != 1 && incl_ghosts,
                     "Downsampling with ghosts not supported",
                     HERE);

      const double g = glob_shape[i];
      const double d = m_dwn[i];
      const double l = loc_corner[i];
      const double n = loc_shape[i];
      const double f = math::ceil(l / d) * d - l;
      m_flds_g_shape_dwn.push_back(static_cast<std::size_t>(math::ceil(g / d)));
      m_flds_l_corner_dwn.push_back(static_cast<std::size_t>(math::ceil(l / d)));
      m_flds_l_first.push_back(static_cast<std::size_t>(f));
      m_flds_l_shape_dwn.push_back(
        static_cast<std::size_t>(math::ceil((n - f) / d)));
    }

    m_io.DefineAttribute("NGhosts", incl_ghosts ? N_GHOSTS : 0);
    m_io.DefineAttribute("Dimension", m_flds_g_shape.size());
    m_io.DefineAttribute("Coordinates", std::string(coords.to_string()));

    for (std::size_t i { 0 }; i < m_flds_g_shape.size(); ++i) {
      // cell-centers
      adios2::Dims g_shape  = { m_flds_g_shape_dwn[i] };
      adios2::Dims l_corner = { m_flds_l_corner_dwn[i] };
      adios2::Dims l_shape  = { m_flds_l_shape_dwn[i] };
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1),
                                  g_shape,
                                  l_corner,
                                  l_shape,
                                  adios2::ConstantDims);
      // cell-edges
      const auto   is_last  = (m_flds_l_corner[i] + m_flds_l_shape[i] ==
                            m_flds_g_shape[i]);
      adios2::Dims g_shape1 = { m_flds_g_shape_dwn[i] + 1 };
      adios2::Dims l_shape1 = { m_flds_l_shape_dwn[i] + (is_last ? 1 : 0) };
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1) + "e",
                                  g_shape1,
                                  l_corner,
                                  l_shape1,
                                  adios2::ConstantDims);
    }

    if constexpr (std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                               Kokkos::LayoutRight>::value) {
      m_io.DefineAttribute("LayoutRight", 1);
    } else {
      std::reverse(m_flds_g_shape_dwn.begin(), m_flds_g_shape_dwn.end());
      std::reverse(m_flds_l_corner_dwn.begin(), m_flds_l_corner_dwn.end());
      std::reverse(m_flds_l_shape_dwn.begin(), m_flds_l_shape_dwn.end());
      m_io.DefineAttribute("LayoutRight", 0);
    }
  }

  void Writer::defineFieldOutputs(const SimEngine&                S,
                                  const std::vector<std::string>& flds_out) {
    m_flds_writers.clear();
    raise::ErrorIf((m_flds_g_shape_dwn.size() == 0) ||
                     (m_flds_l_corner_dwn.size() == 0) ||
                     (m_flds_l_shape_dwn.size() == 0),
                   "Mesh layout must be defined before field output",
                   HERE);
    for (const auto& fld : flds_out) {
      m_flds_writers.emplace_back(S, fld);
    }
    for (const auto& fld : m_flds_writers) {
      if (fld.comp.size() == 0) {
        // scalar
        m_io.DefineVariable<real_t>(fld.name(),
                                    m_flds_g_shape_dwn,
                                    m_flds_l_corner_dwn,
                                    m_flds_l_shape_dwn,
                                    adios2::ConstantDims);
      } else {
        // vector or tensor
        for (std::size_t i { 0 }; i < fld.comp.size(); ++i) {
          m_io.DefineVariable<real_t>(fld.name(i),
                                      m_flds_g_shape_dwn,
                                      m_flds_l_corner_dwn,
                                      m_flds_l_shape_dwn,
                                      adios2::ConstantDims);
        }
      }
    }
  }

  void Writer::defineParticleOutputs(Dimension                          dim,
                                     const std::vector<unsigned short>& specs) {
    m_prtl_writers.clear();
    for (const auto& s : specs) {
      m_prtl_writers.emplace_back(s);
    }
    for (const auto& prtl : m_prtl_writers) {
      for (auto d { 0u }; d < dim; ++d) {
        m_io.DefineVariable<real_t>(prtl.name("X", d + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }
      for (auto d { 0u }; d < Dim::_3D; ++d) {
        m_io.DefineVariable<real_t>(prtl.name("U", d + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }
      m_io.DefineVariable<real_t>(prtl.name("W", 0),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
    }
  }

  void Writer::defineSpectraOutputs(const std::vector<unsigned short>& specs) {
    m_spectra_writers.clear();
    for (const auto& s : specs) {
      m_spectra_writers.emplace_back(s);
    }
    m_io.DefineVariable<real_t>("sEbn", {}, {}, { adios2::UnknownDim });
    for (const auto& sp : m_spectra_writers) {
      m_io.DefineVariable<real_t>(sp.name(), {}, {}, { adios2::UnknownDim });
    }
  }

  void Writer::writeAttrs(const prm::Parameters& params) {
    params.write(m_io);
  }

  template <Dimension D, int N>
  void WriteField(adios2::IO&               io,
                  adios2::Engine&           writer,
                  const std::string&        varname,
                  const ndfield_t<D, N>&    field,
                  std::size_t               comp,
                  std::vector<unsigned int> dwn,
                  std::vector<std::size_t>  first_cell,
                  bool                      ghosts) {
    // when dwn != 1 in any direction, it is assumed that ghosts == false
    auto         var      = io.InquireVariable<real_t>(varname);
    const auto   gh_zones = ghosts ? 0 : N_GHOSTS;
    ndarray_t<D> output_field {};

    if constexpr (D == Dim::_1D) {
      if (ghosts || dwn[0] == 1) {
        auto slice_i1 = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
        auto slice    = Kokkos::subview(field, slice_i1, comp);
        output_field  = array_t<real_t*> { "output_field", slice.extent(0) };
        Kokkos::deep_copy(output_field, slice);
      } else {

        const auto   dwn1          = dwn[0];
        const double first_cell1_d = first_cell[0];
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];

        const auto nx1_dwn = static_cast<std::size_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));

        output_field = array_t<real_t*> { "output_field", nx1_dwn };
        Kokkos::parallel_for(
          "outputField",
          nx1_dwn,
          Lambda(index_t i1) {
            output_field(i1) = field(first_cell1 + i1 * dwn1 + N_GHOSTS, comp);
          });
      }
    } else if constexpr (D == Dim::_2D) {
      if (ghosts || (dwn[0] == 1 && dwn[1] == 1)) {
        auto slice_i1 = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
        auto slice_i2 = range_tuple_t(gh_zones, field.extent(1) - gh_zones);
        auto slice    = Kokkos::subview(field, slice_i1, slice_i2, comp);
        output_field  = array_t<real_t**> { "output_field",
                                            slice.extent(0),
                                            slice.extent(1) };
        Kokkos::deep_copy(output_field, slice);
      } else {
        const auto   dwn1          = dwn[0];
        const auto   dwn2          = dwn[1];
        const double first_cell1_d = first_cell[0];
        const double first_cell2_d = first_cell[1];
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const double nx2_full      = field.extent(1) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];
        const auto   first_cell2   = first_cell[1];

        const auto nx1_dwn = static_cast<std::size_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));
        const auto nx2_dwn = static_cast<std::size_t>(
          math::ceil((nx2_full - first_cell2_d) / dwn2));
        output_field = array_t<real_t**> { "output_field", nx1_dwn, nx2_dwn };
        Kokkos::parallel_for(
          "outputField",
          CreateRangePolicy<Dim::_2D>({ 0, 0 }, { nx1_dwn, nx2_dwn }),
          Lambda(index_t i1, index_t i2) {
            output_field(i1, i2) = field(first_cell1 + i1 * dwn1 + N_GHOSTS,
                                         first_cell2 + i2 * dwn2 + N_GHOSTS,
                                         comp);
          });
      }
    } else if constexpr (D == Dim::_3D) {
      if (ghosts || (dwn[0] == 1 && dwn[1] == 1 && dwn[2] == 1)) {
        auto slice_i1 = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
        auto slice_i2 = range_tuple_t(gh_zones, field.extent(1) - gh_zones);
        auto slice_i3 = range_tuple_t(gh_zones, field.extent(2) - gh_zones);
        auto slice = Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp);
        output_field = array_t<real_t***> { "output_field",
                                            slice.extent(0),
                                            slice.extent(1),
                                            slice.extent(2) };
        Kokkos::deep_copy(output_field, slice);
      } else {
        const auto   dwn1          = dwn[0];
        const auto   dwn2          = dwn[1];
        const auto   dwn3          = dwn[2];
        const double first_cell1_d = first_cell[0];
        const double first_cell2_d = first_cell[1];
        const double first_cell3_d = first_cell[2];
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const double nx2_full      = field.extent(1) - 2 * N_GHOSTS;
        const double nx3_full      = field.extent(2) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];
        const auto   first_cell2   = first_cell[1];
        const auto   first_cell3   = first_cell[2];

        const auto nx1_dwn = static_cast<std::size_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));
        const auto nx2_dwn = static_cast<std::size_t>(
          math::ceil((nx2_full - first_cell2_d) / dwn2));
        const auto nx3_dwn = static_cast<std::size_t>(
          math::ceil((nx3_full - first_cell3_d) / dwn3));

        output_field = array_t<real_t***> { "output_field", nx1_dwn, nx2_dwn, nx3_dwn };
        Kokkos::parallel_for(
          "outputField",
          CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { nx1_dwn, nx2_dwn, nx3_dwn }),
          Lambda(index_t i1, index_t i2, index_t i3) {
            output_field(i1, i2, i3) = field(first_cell1 + i1 * dwn1 + N_GHOSTS,
                                             first_cell2 + i2 * dwn2 + N_GHOSTS,
                                             first_cell3 + i3 * dwn3 + N_GHOSTS,
                                             comp);
          });
      }
    }
    auto output_field_h = Kokkos::create_mirror_view(output_field);
    Kokkos::deep_copy(output_field_h, output_field);
    writer.Put(var, output_field_h);
  }

  template <Dimension D, int N>
  void Writer::writeField(const std::vector<std::string>& names,
                          const ndfield_t<D, N>&          fld,
                          const std::vector<std::size_t>& addresses) {
    raise::ErrorIf(addresses.size() > N,
                   "addresses vector size must be less than N",
                   HERE);
    raise::ErrorIf(names.size() != addresses.size(),
                   "# of names != # of addresses ",
                   HERE);
    for (std::size_t i { 0 }; i < addresses.size(); ++i) {
      WriteField<D, N>(m_io,
                       m_writer,
                       names[i],
                       fld,
                       addresses[i],
                       m_dwn,
                       m_flds_l_first,
                       m_flds_ghosts);
    }
  }

  void Writer::writeParticleQuantity(const array_t<real_t*>& array,
                                     std::size_t             glob_total,
                                     std::size_t             loc_offset,
                                     const std::string&      varname) {
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetShape({ glob_total });
    var.SetSelection(
      adios2::Box<adios2::Dims>({ loc_offset }, { array.extent(0) }));
    auto array_h = Kokkos::create_mirror_view(array);
    Kokkos::deep_copy(array_h, array);
    m_writer.Put<real_t>(var, array_h);
  }

  void Writer::writeSpectrum(const array_t<real_t*>& counts,
                             const std::string&      varname) {
    auto counts_h = Kokkos::create_mirror_view(counts);
    Kokkos::deep_copy(counts_h, counts);
#if defined(MPI_ENABLED)
    array_t<real_t*> counts_all { "counts_all", counts.extent(0) };
    auto             counts_h_all = Kokkos::create_mirror_view(counts_all);
    int              rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(counts_h.data(),
               counts_h_all.data(),
               counts_h.extent(0),
               mpi::get_type<real_t>(),
               MPI_SUM,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank == MPI_ROOT_RANK) {
      auto var = m_io.InquireVariable<real_t>(varname);
      var.SetSelection(adios2::Box<adios2::Dims>({}, { counts.extent(0) }));
      m_writer.Put<real_t>(var, counts_h_all);
    }
#else
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetSelection(adios2::Box<adios2::Dims>({}, { counts.extent(0) }));
    m_writer.Put<real_t>(var, counts_h);
#endif
  }

  void Writer::writeSpectrumBins(const array_t<real_t*>& e_bins,
                                 const std::string&      varname) {
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != MPI_ROOT_RANK) {
      return;
    }
#endif
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetSelection(adios2::Box<adios2::Dims>({}, { e_bins.extent(0) }));
    auto e_bins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(e_bins_h, e_bins);
    m_writer.Put<real_t>(var, e_bins_h);
  }

  void Writer::writeMesh(unsigned short          dim,
                         const array_t<real_t*>& xc,
                         const array_t<real_t*>& xe) {
    auto varc = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1));
    auto vare = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1) + "e");
    auto xc_h = Kokkos::create_mirror_view(xc);
    auto xe_h = Kokkos::create_mirror_view(xe);
    Kokkos::deep_copy(xc_h, xc);
    Kokkos::deep_copy(xe_h, xe);
    m_writer.Put(varc, xc_h);
    m_writer.Put(vare, xe_h);
  }

  void Writer::beginWriting(WriteModeTags write_mode,
                            std::size_t   tstep,
                            long double   time) {
    raise::ErrorIf(write_mode == WriteMode::None, "None is not a valid mode", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_active_mode != WriteMode::None) {
      raise::Fatal("Already writing", HERE);
    }
    m_active_mode = write_mode;
    try {
      std::string       filename;
      const std::string ext = m_engine == "hdf5" ? "h5" : "bp";
      if (m_separate_files) {
        std::string mode_str;
        if (m_active_mode == WriteMode::Fields) {
          mode_str = "fields";
        } else if (m_active_mode == WriteMode::Particles) {
          mode_str = "particles";
        } else if (m_active_mode == WriteMode::Spectra) {
          mode_str = "spectra";
        } else {
          raise::Fatal("Unknown write mode", HERE);
        }
        CallOnce(
          [](auto& main_path, auto& mode_path) {
            const std::filesystem::path main { main_path };
            const std::filesystem::path mode { mode_path };
            if (!std::filesystem::exists(main_path)) {
              std::filesystem::create_directory(main_path);
            }
            if (!std::filesystem::exists(main / mode)) {
              std::filesystem::create_directory(main / mode);
            }
          },
          m_fname,
          mode_str);
        filename = fmt::format("%s/%s/%s.%08lu.%s",
                               m_fname.c_str(),
                               mode_str.c_str(),
                               mode_str.c_str(),
                               tstep,
                               ext.c_str());
        m_mode   = adios2::Mode::Write;
      } else {
        filename = fmt::format("%s.%s", m_fname.c_str(), ext.c_str());
        m_mode   = std::filesystem::exists(filename) ? adios2::Mode::Append
                                                     : adios2::Mode::Write;
      }
      m_writer = m_io.Open(filename, m_mode);
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }
    m_writer.BeginStep();
    m_writer.Put(m_io.InquireVariable<std::size_t>("Step"), &tstep);
    m_writer.Put(m_io.InquireVariable<long double>("Time"), &time);
  }

  void Writer::endWriting(WriteModeTags write_mode) {
    raise::ErrorIf(write_mode == WriteMode::None, "None is not a valid mode", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_active_mode == WriteMode::None) {
      raise::Fatal("Not writing", HERE);
    }
    if (m_active_mode != write_mode) {
      raise::Fatal("Writing mode mismatch", HERE);
    }
    m_active_mode = WriteMode::None;
    m_writer.EndStep();
    m_writer.Close();
  }

  template void Writer::writeField<Dim::_1D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_1D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 6>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_2D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_2D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 6>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_3D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_3D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 6>&,
                                                const std::vector<std::size_t>&);

  template void WriteField<Dim::_1D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_1D, 3>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);
  template void WriteField<Dim::_1D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_1D, 6>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);
  template void WriteField<Dim::_2D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_2D, 3>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);
  template void WriteField<Dim::_2D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_2D, 6>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);
  template void WriteField<Dim::_3D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_3D, 3>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);
  template void WriteField<Dim::_3D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_3D, 6>&,
                                        std::size_t,
                                        std::vector<unsigned int>,
                                        std::vector<std::size_t>,
                                        bool);

} // namespace out
