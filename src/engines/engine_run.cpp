#include "enums.h"

#include "arch/traits.h"
#include "utils/diag.h"

#include "engine_registry.h"

#include "framework/domain/domain.h"

#include "engines/engine.hpp"

namespace ntt {

  template <SimEngine::type S, class M>
  void Engine<S, M>::run() {
    if constexpr (pgen_is_ok) {
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
      const auto diag_interval = m_params.get<timestep_t>(
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
          traits::has_method<traits::pgen::custom_poststep_t, decltype(m_pgen)>::value) {
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
          traits::has_method<traits::pgen::custom_field_output_t, decltype(m_pgen)>::value) {
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
          traits::has_method<traits::pgen::custom_stat_t, decltype(m_pgen)>::value) {
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
  }

#define NTT_INSTANTIATE_ENGINE_RUN(E, M, D) template void Engine<E<M<D>>::S, M<D>>::run();
  NTT_ENGINE_METRIC_DIMENSION_REGISTRY(NTT_INSTANTIATE_ENGINE_RUN)
#undef NTT_INSTANTIATE_ENGINE_RUN

} // namespace ntt
