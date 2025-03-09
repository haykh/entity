#include "checkpoint/writer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/parameters.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace checkpoint {

  void Writer::init(adios2::ADIOS* ptr_adios,
                    std::size_t    interval,
                    long double    interval_time,
                    int            keep) {
    m_keep    = keep;
    m_enabled = keep != 0;
    if (not m_enabled) {
      return;
    }
    m_tracker.init("checkpoint", interval, interval_time);
    p_adios = ptr_adios;
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);

    m_io = p_adios->DeclareIO("Entity::Checkpoint");
    m_io.SetEngine("BPFile");

    m_io.DefineVariable<std::size_t>("Step");
    m_io.DefineVariable<long double>("Time");
    m_io.DefineAttribute("NGhosts", ntt::N_GHOSTS);

    CallOnce([]() {
      const std::filesystem::path save_path { "checkpoints" };
      if (!std::filesystem::exists(save_path)) {
        std::filesystem::create_directory(save_path);
      }
    });
  }

  void Writer::defineFieldVariables(const ntt::SimEngine&           S,
                                    const std::vector<std::size_t>& glob_shape,
                                    const std::vector<std::size_t>& loc_corner,
                                    const std::vector<std::size_t>& loc_shape) {
    auto gs6 = std::vector<std::size_t>(glob_shape.begin(), glob_shape.end());
    auto lc6 = std::vector<std::size_t>(loc_corner.begin(), loc_corner.end());
    auto ls6 = std::vector<std::size_t>(loc_shape.begin(), loc_shape.end());
    gs6.push_back(6);
    lc6.push_back(0);
    ls6.push_back(6);

    m_io.DefineVariable<real_t>("em", gs6, lc6, ls6);
    if (S == ntt::SimEngine::GRPIC) {
      m_io.DefineVariable<real_t>("em0", gs6, lc6, ls6);
      auto gs3 = std::vector<std::size_t>(glob_shape.begin(), glob_shape.end());
      auto lc3 = std::vector<std::size_t>(loc_corner.begin(), loc_corner.end());
      auto ls3 = std::vector<std::size_t>(loc_shape.begin(), loc_shape.end());
      gs3.push_back(3);
      lc3.push_back(0);
      ls3.push_back(3);
      m_io.DefineVariable<real_t>("cur0", gs3, lc3, ls3);
    }
  }

  void Writer::defineParticleVariables(const ntt::Coord& C,
                                       Dimension         dim,
                                       std::size_t       nspec,
                                       const std::vector<unsigned short>& nplds) {
    raise::ErrorIf(nplds.size() != nspec,
                   "Number of payloads does not match the number of species",
                   HERE);
    for (auto s { 0u }; s < nspec; ++s) {
      m_io.DefineVariable<std::size_t>(fmt::format("s%d_npart", s + 1),
                                       { adios2::UnknownDim },
                                       { adios2::UnknownDim },
                                       { adios2::UnknownDim });

      for (auto d { 0u }; d < dim; ++d) {
        m_io.DefineVariable<int>(fmt::format("s%d_i%d", s + 1, d + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
        m_io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d", s + 1, d + 1),
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim });
        m_io.DefineVariable<int>(fmt::format("s%d_i%d_prev", s + 1, d + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
        m_io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d_prev", s + 1, d + 1),
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim });
      }

      if (dim == Dim::_2D and C != ntt::Coord::Cart) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_phi", s + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }

      for (auto d { 0u }; d < 3; ++d) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_ux%d", s + 1, d + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }

      m_io.DefineVariable<short>(fmt::format("s%d_tag", s + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
      m_io.DefineVariable<real_t>(fmt::format("s%d_weight", s + 1),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
      if (nplds[s] > 0) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_plds", s + 1),
                                    { adios2::UnknownDim, nplds[s] },
                                    { adios2::UnknownDim, 0 },
                                    { adios2::UnknownDim, nplds[s] });
      }
    }
  }

  auto Writer::shouldSave(std::size_t step, long double time) -> bool {
    return m_enabled and m_tracker.shouldWrite(step, time);
  }

  void Writer::beginSaving(std::size_t step, long double time) {
    raise::ErrorIf(!m_enabled, "Checkpoint is not enabled", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_writing_mode) {
      raise::Fatal("Already writing", HERE);
    }
    m_writing_mode = true;
    try {
      auto fname      = fmt::format("checkpoints/step-%08lu.bp", step);
      m_writer        = m_io.Open(fname, adios2::Mode::Write);
      auto meta_fname = fmt::format("checkpoints/meta-%08lu.toml", step);
      m_written.push_back({ fname, meta_fname });
      logger::Checkpoint(fmt::format("Writing checkpoint to %s and %s",
                                     fname.c_str(),
                                     meta_fname.c_str()),
                         HERE);
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }

    m_writer.BeginStep();
    m_writer.Put(m_io.InquireVariable<std::size_t>("Step"), &step);
    m_writer.Put(m_io.InquireVariable<long double>("Time"), &time);
  }

  void Writer::endSaving() {
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (!m_writing_mode) {
      raise::Fatal("Not writing", HERE);
    }
    m_writing_mode = false;
    m_writer.EndStep();
    m_writer.Close();

    // optionally remove the oldest checkpoint
    CallOnce([&]() {
      if (m_keep > 0 and m_written.size() > (std::size_t)m_keep) {
        const auto oldest = m_written.front();
        if (std::filesystem::exists(oldest.first) and
            std::filesystem::exists(oldest.second)) {
          std::filesystem::remove_all(oldest.first);
          std::filesystem::remove(oldest.second);
          m_written.erase(m_written.begin());
        } else {
          raise::Warning("Checkpoint file does not exist for some reason", HERE);
        }
      }
    });
  }

  template <typename T>
  void Writer::savePerDomainVariable(const std::string& varname,
                                     std::size_t        total,
                                     std::size_t        offset,
                                     T                  data) {
    auto var = m_io.InquireVariable<T>(varname);
    var.SetShape({ total });
    var.SetSelection(adios2::Box<adios2::Dims>({ offset }, { 1 }));
    m_writer.Put(var, &data);
  }

  void Writer::saveAttrs(const ntt::SimulationParams& params, long double time) {
    CallOnce([&]() {
      std::ofstream metadata;
      if (m_written.empty()) {
        raise::Fatal("No checkpoint file to save metadata", HERE);
      }
      metadata.open(m_written.back().second.c_str());
      metadata << "[metadata]\n"
               << "  time = " << time << "\n\n"
               << params.data() << std::endl;
      metadata.close();
    });
  }

  template <Dimension D, int N>
  void Writer::saveField(const std::string&     fieldname,
                         const ndfield_t<D, N>& field) {
    auto field_h = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(field_h, field);
    m_writer.Put(m_io.InquireVariable<real_t>(fieldname),
                 field_h.data(),
                 adios2::Mode::Sync);
  }

  template <typename T>
  void Writer::saveParticleQuantity(const std::string& quantity,
                                    std::size_t        glob_total,
                                    std::size_t        loc_offset,
                                    std::size_t        loc_size,
                                    const array_t<T*>& data) {
    const auto slice = range_tuple_t(0, loc_size);
    auto       var   = m_io.InquireVariable<T>(quantity);

    var.SetShape({ glob_total });
    var.SetSelection(adios2::Box<adios2::Dims>({ loc_offset }, { loc_size }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice);
    m_writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

  void Writer::saveParticlePayloads(const std::string&       quantity,
                                    std::size_t              nplds,
                                    std::size_t              glob_total,
                                    std::size_t              loc_offset,
                                    std::size_t              loc_size,
                                    const array_t<real_t**>& data) {
    const auto slice = range_tuple_t(0, loc_size);
    auto       var   = m_io.InquireVariable<real_t>(quantity);

    var.SetShape({ glob_total, nplds });
    var.SetSelection(
      adios2::Box<adios2::Dims>({ loc_offset, 0 }, { loc_size, nplds }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice, range_tuple_t(0, nplds));
    m_writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

  template void Writer::savePerDomainVariable<int>(const std::string&,
                                                   std::size_t,
                                                   std::size_t,
                                                   int);
  template void Writer::savePerDomainVariable<float>(const std::string&,
                                                     std::size_t,
                                                     std::size_t,
                                                     float);
  template void Writer::savePerDomainVariable<double>(const std::string&,
                                                      std::size_t,
                                                      std::size_t,
                                                      double);
  template void Writer::savePerDomainVariable<std::size_t>(const std::string&,
                                                           std::size_t,
                                                           std::size_t,
                                                           std::size_t);

  template void Writer::saveField<Dim::_1D, 3>(const std::string&,
                                               const ndfield_t<Dim::_1D, 3>&);
  template void Writer::saveField<Dim::_1D, 6>(const std::string&,
                                               const ndfield_t<Dim::_1D, 6>&);
  template void Writer::saveField<Dim::_2D, 3>(const std::string&,
                                               const ndfield_t<Dim::_2D, 3>&);
  template void Writer::saveField<Dim::_2D, 6>(const std::string&,
                                               const ndfield_t<Dim::_2D, 6>&);
  template void Writer::saveField<Dim::_3D, 3>(const std::string&,
                                               const ndfield_t<Dim::_3D, 3>&);
  template void Writer::saveField<Dim::_3D, 6>(const std::string&,
                                               const ndfield_t<Dim::_3D, 6>&);

  template void Writer::saveParticleQuantity<int>(const std::string&,
                                                  std::size_t,
                                                  std::size_t,
                                                  std::size_t,
                                                  const array_t<int*>&);
  template void Writer::saveParticleQuantity<float>(const std::string&,
                                                    std::size_t,
                                                    std::size_t,
                                                    std::size_t,
                                                    const array_t<float*>&);
  template void Writer::saveParticleQuantity<double>(const std::string&,
                                                     std::size_t,
                                                     std::size_t,
                                                     std::size_t,
                                                     const array_t<double*>&);
  template void Writer::saveParticleQuantity<short>(const std::string&,
                                                    std::size_t,
                                                    std::size_t,
                                                    std::size_t,
                                                    const array_t<short*>&);
} // namespace checkpoint
