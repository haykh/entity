/**
 * @file checkpoint/writer.h
 * @brief Class that dumps checkpoints
 * @implements
 *   - checkpoint::Writer
 * @cpp:
 *   - writer.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_WRITER_H
#define CHECKPOINT_WRITER_H

#include "enums.h"
#include "global.h"

#include "utils/tools.h"

#include "framework/parameters.h"

#include <adios2.h>

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  class Writer {
    adios2::ADIOS* p_adios { nullptr };

    adios2::IO     m_io;
    adios2::Engine m_writer;

    tools::Tracker m_tracker {};

    bool m_writing_mode { false };

    std::vector<std::pair<std::string, std::string>> m_written;

    int  m_keep;
    bool m_enabled;

  public:
    Writer() {}

    ~Writer() = default;

    void init(adios2::ADIOS*, std::size_t, long double, int);

    auto shouldSave(std::size_t, long double) -> bool;

    void beginSaving(std::size_t, long double);
    void endSaving();

    void saveAttrs(const ntt::SimulationParams&, long double);

    template <typename T>
    void savePerDomainVariable(const std::string&, std::size_t, std::size_t, T);

    template <Dimension D, int N>
    void saveField(const std::string&, const ndfield_t<D, N>&);

    template <typename T>
    void saveParticleQuantity(const std::string&,
                              std::size_t,
                              std::size_t,
                              std::size_t,
                              const array_t<T*>&);

    void saveParticlePayloads(const std::string&,
                              std::size_t,
                              std::size_t,
                              std::size_t,
                              std::size_t,
                              const array_t<real_t**>&);

    void defineFieldVariables(const ntt::SimEngine&,
                              const std::vector<std::size_t>&,
                              const std::vector<std::size_t>&,
                              const std::vector<std::size_t>&);

    void defineParticleVariables(const ntt::Coord&,
                                 Dimension,
                                 std::size_t,
                                 const std::vector<unsigned short>&);

    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }
  };

} // namespace checkpoint

#endif // CHECKPOINT_WRITER_H
