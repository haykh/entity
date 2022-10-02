#include "global.h"
#include "output_csv.h"
#include "meshblock.h"

#include <string>
#include <cassert>

namespace ntt {
  namespace csv {
    void ensureFileExists(const std::string&) {}

    template <Dimension D, SimulationType S>
    void writeField(const std::string&, const Meshblock<D, S>&, const em&) {}

    template <Dimension D, SimulationType S>
    void writeField(const std::string&, const Meshblock<D, S>&, const cur&) {}

    template <Dimension D, SimulationType S>
    void writeParticle(std::string,
                       const Meshblock<D, S>&,
                       const std::size_t&,
                       const std::size_t&,
                       const OutputMode&) {}
  } // namespace csv
} // namespace ntt

#ifdef PIC_SIMTYPE

using Meshblock1D = ntt::Meshblock<ntt::Dim1, ntt::SimulationType::PIC>;
using Meshblock2D = ntt::Meshblock<ntt::Dim2, ntt::SimulationType::PIC>;
using Meshblock3D = ntt::Meshblock<ntt::Dim3, ntt::SimulationType::PIC>;

template void ntt::csv::writeField<ntt::Dim1, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock1D&,
                                                                        const em&);
template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock2D&,
                                                                        const em&);
template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock3D&,
                                                                        const em&);

template void ntt::csv::writeField<ntt::Dim1, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock1D&,
                                                                        const cur&);
template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock2D&,
                                                                        const cur&);
template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::PIC>(const std::string&,
                                                                        const Meshblock3D&,
                                                                        const cur&);

template void ntt::csv::writeParticle<ntt::Dim1, ntt::SimulationType::PIC>(
  std::string, const Meshblock1D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dim2, ntt::SimulationType::PIC>(
  std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dim3, ntt::SimulationType::PIC>(
  std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const OutputMode&);

#elif defined(GRPIC_SIMTYPE)

using Meshblock2D = ntt::Meshblock<ntt::Dim2, ntt::SimulationType::GRPIC>;
using Meshblock3D = ntt::Meshblock<ntt::Dim3, ntt::SimulationType::GRPIC>;

template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::GRPIC>(const std::string&,
                                                                          const Meshblock2D&,
                                                                          const em&);
template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::GRPIC>(const std::string&,
                                                                          const Meshblock3D&,
                                                                          const em&);

template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::GRPIC>(const std::string&,
                                                                          const Meshblock2D&,
                                                                          const cur&);
template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::GRPIC>(const std::string&,
                                                                          const Meshblock3D&,
                                                                          const cur&);

template void ntt::csv::writeParticle<ntt::Dim2, ntt::SimulationType::GRPIC>(
  std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dim3, ntt::SimulationType::GRPIC>(
  std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const OutputMode&);

#endif