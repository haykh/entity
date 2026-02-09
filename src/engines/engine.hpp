/**
 * @file engines/engine.hpp
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
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

#include "arch/mpi_aliases.h"
#include "utils/diag.h"
#include "utils/reporter.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "archetypes/field_setter.h"
#include "archetypes/traits.h"
#include "engines/reporter.h"
#include "engines/traits.h"
#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include "pgen.hpp"

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
#endif

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
  #include <adios2/cxx/KokkosView.h>
#endif // OUTPUT_ENABLED

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <map>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
    requires traits::engine::IsCompatibleWithEngine<S, M, user::PGen>
  class Engine {

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

    const bool       is_resuming;
    const simtime_t  runtime;
    const real_t     dt;
    const timestep_t max_steps;
    const timestep_t start_step;
    const simtime_t  start_time;
    simtime_t        time;
    timestep_t       step;

  public:
    static constexpr Dimension D { M::Dim };

    Engine(const SimulationParams& params)
      : m_params { params }
      , m_metadomain { m_params.get<unsigned int>("simulation.domain.number"),
                       m_params.get<std::vector<int>>(
                         "simulation.domain.decomposition"),
                       m_params.get<std::vector<ncells_t>>("grid.resolution"),
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
      , runtime { m_params.get<simtime_t>("simulation.runtime") }
      , dt { m_params.get<real_t>("algorithms.timestep.dt") }
      , max_steps { static_cast<timestep_t>(runtime / dt) }
      , start_step { m_params.get<timestep_t>("checkpoint.start_step") }
      , start_time { m_params.get<simtime_t>("checkpoint.start_time") }
      , time { start_time }
      , step { start_step } {}

    ~Engine() = default;

    void init();
    void print_report() const;

    virtual void step_forward(timer::Timers&, Domain<S, M>&) = 0;

    void run();

    auto engineParams() const -> prm::Parameters {
      auto parameters = prm::Parameters {};
      parameters.set("dt", static_cast<real_t>(dt));
      parameters.set("time", static_cast<simtime_t>(time));
      return parameters;
    }
  };

  template <SimEngine::type S, class M>
    requires traits::engine::IsCompatibleWithEngine<S, M, user::PGen>
  void Engine<S, M>::init() {
    m_metadomain.InitStatsWriter(m_params, is_resuming);
#if defined(OUTPUT_ENABLED)
    m_metadomain.InitWriter(&m_adios, m_params);
    m_metadomain.InitCheckpointWriter(&m_adios, m_params);
#endif
    logger::Checkpoint("Initializing Engine", HERE);
    if (not is_resuming) {
      // start a new simulation with initial conditions
      logger::Checkpoint("Loading initial conditions", HERE);
      if constexpr (arch::traits::pgen::HasInitFlds<user::PGen<S, M>>) {
        logger::Checkpoint("Initializing fields from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          Kokkos::parallel_for(
            "InitFields",
            loc_dom.mesh.rangeActiveCells(),
            arch::SetEMFields_kernel<decltype(m_pgen.init_flds), S, M> {
              loc_dom.fields.em,
              m_pgen.init_flds,
              loc_dom.mesh.metric });
        });
      }
      if constexpr (
        arch::traits::pgen::HasInitPrtls<user::PGen<S, M>, Domain<S, M>>) {
        logger::Checkpoint("Initializing particles from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          m_pgen.InitPrtls(loc_dom);
        });
      }
    } else {
#if defined(OUTPUT_ENABLED)
      // read simulation data from the checkpoint
      raise::ErrorIf(
        m_params.template get<timestep_t>("checkpoint.start_step") == 0,
        "Resuming simulation from a checkpoint requires a valid start_step",
        HERE);
      logger::Checkpoint("Resuming simulation from a checkpoint", HERE);
      m_metadomain.ContinueFromCheckpoint(&m_adios, m_params);
#else
      raise::Error(
        "Resuming simulation from a checkpoint requires -D output=ON",
        HERE);
#endif
    }
    print_report();
  }

  template <SimEngine::type S, class M>
    requires traits::engine::IsCompatibleWithEngine<S, M, user::PGen>
  void Engine<S, M>::print_report() const {
    const auto colored_stdout = m_params.template get<bool>(
      "diagnostics.colored_stdout");
    std::string report = "";
    CallOnce(
      [&](auto& metadomain, auto& params) {
        const auto pgen  = std::string(PGEN);
        report          += reporter::Backend();

        report += ReportSimulationConfig(params,
                                         pgen,
                                         S,
                                         Metric(M::MetricType),
                                         dt,
                                         runtime,
                                         max_steps,
                                         metadomain.ndomains_per_dim(),
                                         metadomain.ndomains());
        if (metadomain.species_params().size() > 0) {
          report += "\n";
          reporter::AddCategory(report, 4, "Particles");
        }
        for (const auto& species : metadomain.species_params()) {
          report += species.Report();
        }
        report.pop_back();
      },
      m_metadomain,
      m_params);
    info::Print(report, colored_stdout);

    report = "\n";
    CallOnce([&]() {
      reporter::AddCategory(report, 4, "Domains");
      report.pop_back();
    });
    info::Print(report, colored_stdout);

    for (unsigned int idx { 0 }; idx < m_metadomain.ndomains(); ++idx) {
      auto is_local = false;
      for (const auto& lidx : m_metadomain.l_subdomain_indices()) {
        is_local |= (idx == lidx);
      }
      if (is_local) {
        const auto& domain = m_metadomain.subdomain(idx);
        report             = domain.Report();
        if (idx == m_metadomain.ndomains() - 1) {
          report += "\n\n";
        }
        info::Print(report, colored_stdout, true, false);
      }
#if defined(MPI_ENABLED)
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
  }

  template <SimEngine::type S, class M>
    requires traits::engine::IsCompatibleWithEngine<S, M, user::PGen>
  void Engine<S, M>::run() {
    init();

    auto timers = timer::Timers {
      { "FieldSolver",
       "CurrentFiltering", "CurrentDeposit",
       "ParticlePusher", "FieldBoundaries",
       "ParticleBoundaries", "Communications",
       "Injector", "Custom",
       "PrtlClear", "Output",
       "Checkpoint" },
      []() {
        Kokkos::fence();
       },
      m_params.get<bool>("diagnostics.blocking_timers")
    };
    const auto diag_interval = m_params.template get<timestep_t>(
      "diagnostics.interval");

    auto       time_history   = pbar::DurationHistory { 1000 };
    const auto clear_interval = m_params.template get<timestep_t>(
      "particles.clear_interval");

    // main algorithm loop
    while (step < max_steps) {
      // run the engine-dependent algorithm step
      m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
        step_forward(timers, dom);
      });
      // poststep (if defined)
      if constexpr (
        arch::traits::pgen::HasCustomPostStep<decltype(m_pgen), Domain<S, M>>) {
        timers.start("Custom");
        m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
          m_pgen.CustomPostStep(step, time, dom);
        });
        timers.stop("Custom");
      }
      auto print_prtl_clear = (clear_interval > 0 and
                               step % clear_interval == 0 and step > 0);

      // advance time & step
      time += dt;
      ++step;

      auto print_output     = false;
      auto print_checkpoint = false;
#if defined(OUTPUT_ENABLED)
      timers.start("Output");
      if constexpr (
        arch::traits::pgen::HasCustomFieldOutput<decltype(m_pgen), Domain<S, M>>) {
        auto lambda_custom_field_output = [&](const std::string&    name,
                                              ndfield_t<M::Dim, 6>& buff,
                                              index_t               idx,
                                              timestep_t            step,
                                              simtime_t             time,
                                              const Domain<S, M>&   dom) {
          m_pgen.CustomFieldOutput(name, buff, idx, step, time, dom);
        };
        print_output &= m_metadomain.Write(m_params,
                                           step,
                                           step - 1,
                                           time,
                                           time - dt,
                                           lambda_custom_field_output);
      } else {
        print_output &= m_metadomain.Write(m_params, step, step - 1, time, time - dt);
      }
      if constexpr (
        arch::traits::pgen::HasCustomStatOutput<decltype(m_pgen), Domain<S, M>>) {
        auto lambda_custom_stat = [&](const std::string&  name,
                                      timestep_t          step,
                                      simtime_t           time,
                                      const Domain<S, M>& dom) -> real_t {
          return m_pgen.CustomStat(name, step, time, dom);
        };
        print_output &= m_metadomain.WriteStats(m_params,
                                                step,
                                                step - 1,
                                                time,
                                                time - dt,
                                                lambda_custom_stat);
      } else {
        print_output &= m_metadomain.WriteStats(m_params,
                                                step,
                                                step - 1,
                                                time,
                                                time - dt);
      }
      timers.stop("Output");

      timers.start("Checkpoint");
      print_checkpoint = m_metadomain.WriteCheckpoint(m_params,
                                                      step,
                                                      step - 1,
                                                      time,
                                                      time - dt);
      timers.stop("Checkpoint");
#endif

      // advance time_history
      time_history.tick();
      // print timestep report
      if (diag_interval > 0 and step % diag_interval == 0) {
        diag::printDiagnostics(
          step - 1,
          max_steps,
          time - dt,
          dt,
          timers,
          time_history,
          m_metadomain.l_ncells(),
          m_metadomain.species_labels(),
          m_metadomain.l_npart_perspec(),
          m_metadomain.l_maxnpart_perspec(),
          print_prtl_clear,
          print_output,
          print_checkpoint,
          m_params.get<bool>("diagnostics.colored_stdout"));
      }
      timers.resetAll();
    }
  }

} // namespace ntt

#endif // ENGINES_ENGINE_H
