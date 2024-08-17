#include "checkpoint/reader.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  template <Dimension D, int N>
  void ReadFields(adios2::IO&                      io,
                  adios2::Engine&                  reader,
                  const std::string&               field,
                  const adios2::Box<adios2::Dims>& range,
                  ndfield_t<D, N>&                 array) {
    logger::Checkpoint(fmt::format("Reading field: %s", field.c_str()), HERE);
    auto field_var = io.InquireVariable<real_t>(field);
    field_var.SetStepSelection(adios2::Box<std::size_t>({ 0 }, { 1 }));
    field_var.SetSelection(range);
    auto array_h = Kokkos::create_mirror_view(array);
    reader.Get(field_var, array_h.data());
    Kokkos::deep_copy(array, array_h);
  }

  auto ReadParticleCount(adios2::IO&     io,
                         adios2::Engine& reader,
                         unsigned short  s,
                         std::size_t     local_dom,
                         std::size_t     ndomains)
    -> std::pair<std::size_t, std::size_t> {
    logger::Checkpoint(fmt::format("Reading particle count for: %d", s + 1), HERE);
    auto npart_var = io.InquireVariable<std::size_t>(
      fmt::format("s%d_npart", s + 1));
    raise::ErrorIf(
      npart_var.Shape()[0] != ndomains or npart_var.Shape().size() != 1,
      "npart_var.Shape()[0] != ndomains or npart_var.Shape().size() != 1",
      HERE);
    npart_var.SetStepSelection(adios2::Box<std::size_t>({ 0 }, { 1 }));
    npart_var.SetSelection(
      adios2::Box<adios2::Dims>({ 0 }, { npart_var.Shape()[0] }));
    std::vector<std::size_t> nparts(ndomains);
    reader.Get(npart_var, nparts.data(), adios2::Mode::Sync);

    const auto  loc_npart    = nparts[local_dom];
    std::size_t offset_npart = 0;
    for (auto d { 0u }; d < local_dom; ++d) {
      offset_npart += nparts[d];
    }
    return { loc_npart, offset_npart };
  }

  template <typename T>
  void ReadParticleData(adios2::IO&        io,
                        adios2::Engine&    reader,
                        const std::string& quantity,
                        unsigned short     s,
                        array_t<T*>&       array,
                        std::size_t        count,
                        std::size_t        offset) {
    logger::Checkpoint(
      fmt::format("Reading quantity: s%d_%s", s + 1, quantity.c_str()),
      HERE);
    auto var = io.InquireVariable<T>(
      fmt::format("s%d_%s", s + 1, quantity.c_str()));
    var.SetStepSelection(adios2::Box<std::size_t>({ 0 }, { 1 }));
    var.SetSelection(adios2::Box<adios2::Dims>({ offset }, { count }));
    auto array_h = Kokkos::create_mirror_view(array);
    reader.Get(var, array_h.data());
    Kokkos::deep_copy(array, array_h);
  }

  template void ReadFields<Dim::_1D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_1D, 3>&);
  template void ReadFields<Dim::_2D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_2D, 3>&);
  template void ReadFields<Dim::_3D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_3D, 3>&);
  template void ReadFields<Dim::_1D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_1D, 6>&);
  template void ReadFields<Dim::_2D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_2D, 6>&);
  template void ReadFields<Dim::_3D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const adios2::Box<adios2::Dims>&,
                                        ndfield_t<Dim::_3D, 6>&);

  template void ReadParticleData<int>(adios2::IO&,
                                      adios2::Engine&,
                                      const std::string&,
                                      unsigned short,
                                      array_t<int*>&,
                                      std::size_t,
                                      std::size_t);
  template void ReadParticleData<float>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        unsigned short,
                                        array_t<float*>&,
                                        std::size_t,
                                        std::size_t);
  template void ReadParticleData<double>(adios2::IO&,
                                         adios2::Engine&,
                                         const std::string&,
                                         unsigned short,
                                         array_t<double*>&,
                                         std::size_t,
                                         std::size_t);
  template void ReadParticleData<short>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        unsigned short,
                                        array_t<short*>&,
                                        std::size_t,
                                        std::size_t);

} // namespace checkpoint
