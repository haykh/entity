#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "engine_registry.h"

#include "archetypes/field_setter.h"

#include "engines/engine.hpp"

#include <Kokkos_Core.hpp>

#include <string>

namespace ntt {

  template <SimEngine::type S, class M>
  void Engine<S, M>::init() {
    if constexpr (pgen_is_ok) {
      m_metadomain.InitStatsWriter(m_params, is_resuming);
#if defined(OUTPUT_ENABLED)
      m_metadomain.InitWriter(&m_adios, m_params);
      m_metadomain.InitCheckpointWriter(&m_adios, m_params);
#endif
      logger::Checkpoint("Initializing Engine", HERE);
      if (not is_resuming) {
        // start a new simulation with initial conditions
        logger::Checkpoint("Loading initial conditions", HERE);
        if constexpr (
          traits::has_member<traits::pgen::init_flds_t, user::PGen<S, M>>::value) {
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
          traits::has_member<traits::pgen::init_prtls_t, user::PGen<S, M>>::value) {
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
    }
    print_report();
  }

#define NTT_INSTANTIATE_ENGINE(E, M, D) template class Engine<E<M<D>>::S, M<D>>;
  NTT_ENGINE_METRIC_DIMENSION_REGISTRY(NTT_INSTANTIATE_ENGINE)
#undef NTT_INSTANTIATE_ENGINE

} // namespace ntt
