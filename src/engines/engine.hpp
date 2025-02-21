/**
 * @file engines/engine.hpp
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
 * @cpp:
 *   - engine_init.cpp
 *   - engine_printer.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI
 *   - DEBUG
 *   - OUTPUT_ENABLED
 *   - CUDA_ENABLED
 */

#ifndef ENGINES_ENGINE_H
#define ENGINES_ENGINE_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/progressbar.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "framework/containers/species.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include "pgen.hpp"

#include <Kokkos_Core.hpp>

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif // OUTPUT_ENABLED

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <map>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  class Engine {
    static_assert(M::is_metric, "template arg for Engine class has to be a metric");
    static_assert(user::PGen<S, M>::is_pgen, "unrecognized problem generator");

  protected:
#if defined(OUTPUT_ENABLED)
  #if defined(MPI_ENABLED)
    adios2::ADIOS m_adios { MPI_COMM_WORLD };
  #else
    adios2::ADIOS m_adios;
  #endif
#endif

    SimulationParams m_params;
    Metadomain<S, M> m_metadomain;
    user::PGen<S, M> m_pgen;

    const bool        is_resuming;
    const long double runtime;
    const real_t      dt;
    const std::size_t max_steps;
    const std::size_t start_step;
    const long double start_time;
    long double       time;
    std::size_t       step;

  public:
    static constexpr bool pgen_is_ok {
      traits::check_compatibility<S>::value(user::PGen<S, M>::engines) and
      traits::check_compatibility<M::MetricType>::value(user::PGen<S, M>::metrics) and
      traits::check_compatibility<M::Dim>::value(user::PGen<S, M>::dimensions)
    };

    static constexpr Dimension D { M::Dim };
    static constexpr bool      is_engine { true };

    Engine(const SimulationParams& params)
      : m_params { params }
      , m_metadomain { m_params.get<unsigned int>("simulation.domain.number"),
                       m_params.get<std::vector<int>>(
                         "simulation.domain.decomposition"),
                       m_params.get<std::vector<std::size_t>>(
                         "grid.resolution"),
                       m_params.get<boundaries_t<real_t>>("grid.extent"),
                       m_params.get<boundaries_t<FldsBC>>(
                         "grid.boundaries.fields"),
                       m_params.get<boundaries_t<PrtlBC>>(
                         "grid.boundaries.particles"),
                       m_params.get<std::map<std::string, real_t>>(
                         "grid.metric.params"),
                       m_params.get<std::vector<ParticleSpecies>>(
                         "particles.species") }
      , m_pgen { m_params, m_metadomain }
      , is_resuming { m_params.get<bool>("checkpoint.is_resuming") }
      , runtime { m_params.get<long double>("simulation.runtime") }
      , dt { m_params.get<real_t>("algorithms.timestep.dt") }
      , max_steps { static_cast<std::size_t>(runtime / dt) }
      , start_step { m_params.get<std::size_t>("checkpoint.start_step") }
      , start_time { m_params.get<long double>("checkpoint.start_time") }
      , time { start_time }
      , step { start_step } {
      raise::ErrorIf(not pgen_is_ok, "Problem generator is not compatible with the picked engine/metric/dimension", HERE);
      print_report();
    }

    ~Engine() = default;

    void init();
    void print_report() const;

    virtual void step_forward(timer::Timers&, Domain<S, M>&) = 0;

    void run();
  };

} // namespace ntt

#endif // ENGINES_ENGINE_H
