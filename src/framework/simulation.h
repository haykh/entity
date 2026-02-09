/**
 * @file framework/simulation.h
 * @brief Simulation class which creates and calles the engines
 * @implements
 *   - ntt::Simulation
 * @cpp:
 *   - simulation.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - DEBUG
 */

#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "enums.h"

#include "utils/error.h"
#include "utils/toml.h"

#include "engines/traits.h"
#include "framework/parameters/parameters.h"

namespace ntt {

  class Simulation {
    SimulationParams m_params;

    Dimension m_requested_dimension;
    SimEngine m_requested_engine { SimEngine::INVALID };
    Metric    m_requested_metric { Metric::INVALID };

  public:
    Simulation(int argc, char* argv[]);
    ~Simulation();

    template <template <class> class E, template <Dimension> class M, Dimension D>
      requires traits::engine::HasRun<E<M<D>>>
    inline void run() {
      using engine_t = E<M<D>>;
      try {
        engine_t engine { m_params };
        engine.run();
      } catch (const std::exception& e) {
        raise::Fatal(e.what(), HERE);
      }
    }

    [[nodiscard]]
    inline auto requested_dimension() const -> Dimension {
      return m_requested_dimension;
    }

    [[nodiscard]]
    inline auto requested_engine() const -> SimEngine {
      return m_requested_engine;
    }

    [[nodiscard]]
    inline auto requested_metric() const -> Metric {
      return m_requested_metric;
    }
  };

} // namespace ntt

#endif // FRAMEWORK_SIMULATION_H
