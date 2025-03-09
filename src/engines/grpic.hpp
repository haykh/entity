/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"

#include "utils/timer.h"
#include "utils/toml.h"

#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "engines/engine.hpp"

namespace ntt {

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using base_t = Engine<SimEngine::GRPIC, M>;

    using Engine<SimEngine::GRPIC, M>::m_params;
    using Engine<SimEngine::GRPIC, M>::m_metadomain;

  public:
    static constexpr auto S { SimEngine::SRPIC };

    GRPICEngine(const SimulationParams& params) : base_t { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers&, Domain<SimEngine::GRPIC, M>&) override {}
  };

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H
