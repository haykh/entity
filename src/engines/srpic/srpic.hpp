/**
 * @file engines/srpic.hpp
 * @brief Simulation engien class which specialized on SRPIC
 * @implements
 *   - ntt::SRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 */

#ifndef ENGINES_SRPIC_SRPIC_H
#define ENGINES_SRPIC_SRPIC_H

#include "enums.h"
#include "global.h"

#include "utils/numeric.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "engines/srpic/currents.h"
#include "engines/srpic/fields_bcs.h"
#include "engines/srpic/fieldsolvers.h"
#include "engines/srpic/particle_pusher.h"
#include "engines/srpic/particles_bcs.h"
#include "engines/traits.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

#include "engines/engine.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace ntt {

  template <class M>
    requires traits::engine::IsCompatibleWithSRPICEngine<M, user::PGen>
  class SRPICEngine : public Engine<SimEngine::SRPIC, M> {

    using base_t   = Engine<SimEngine::SRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::SRPIC, M>;
    using domain_t = Domain<SimEngine::SRPIC, M>;
    // contents
    using base_t::m_metadomain;
    using base_t::m_params;
    using base_t::m_pgen;
    // methods
    using base_t::init;
    // variables
    using base_t::dt;
    using base_t::max_steps;
    using base_t::runtime;
    using base_t::step;
    using base_t::time;

  public:
    static constexpr auto S { SimEngine::SRPIC };

    SRPICEngine(const SimulationParams& params) : base_t { params } {}

    ~SRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.fieldsolver.enable");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.deposit.enable");
      const auto clear_interval = m_params.template get<std::size_t>(
        "particles.clear_interval");

      if (step == 0) {
        // communicate fields and apply BCs on the first timestep
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::E);
        srpic::FieldBoundaries(dom,
                               m_metadomain.mesh().metric,
                               m_metadomain.mesh(),
                               m_pgen,
                               this->engineParams(),
                               m_params,
                               BC::B | BC::E);
        srpic::ParticleInjector(m_metadomain, dom, m_params);
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        srpic::Faraday(dom, this->engineParams(), m_params, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        srpic::FieldBoundaries(dom,
                               m_metadomain.mesh().metric,
                               m_metadomain.mesh(),
                               m_pgen,
                               this->engineParams(),
                               m_params,
                               BC::B);
        timers.stop("FieldBoundaries");
        Kokkos::fence();
      }

      {
        timers.start("ParticlePusher");
        srpic::ParticlePush(dom,
                            m_metadomain.mesh(),
                            m_metadomain.mesh().metric,
                            this->engineParams(),
                            m_params,
                            m_pgen);
        timers.stop("ParticlePusher");

        if (deposit_enabled) {
          timers.start("CurrentDeposit");
          srpic::CurrentsDeposit(dom, this->engineParams());
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.SynchronizeFields(dom, Comm::J);
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

          timers.start("CurrentFiltering");
          srpic::CurrentsFilter(m_metadomain, dom, m_params);
          timers.stop("CurrentFiltering");
        }

        timers.start("Communications");
        m_metadomain.CommunicateParticles(dom);
        timers.stop("Communications");
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        srpic::Faraday(dom, this->engineParams(), m_params, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        srpic::FieldBoundaries(dom,
                               m_metadomain.mesh().metric,
                               m_metadomain.mesh(),
                               m_pgen,
                               this->engineParams(),
                               m_params,
                               BC::B);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        srpic::Ampere(dom, this->engineParams(), m_params, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          srpic::CurrentsAmpere(dom, this->engineParams(), m_params, m_pgen);
          timers.stop("FieldSolver");
        }

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::E | Comm::J);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        srpic::FieldBoundaries(dom,
                               m_metadomain.mesh().metric,
                               m_metadomain.mesh(),
                               m_pgen,
                               this->engineParams(),
                               m_params,
                               BC::E);
        timers.stop("FieldBoundaries");
      }

      {
        timers.start("Injector");
        srpic::ParticleInjector(m_metadomain, dom, m_params);
        timers.stop("Injector");
      }

      if (clear_interval > 0 and step % clear_interval == 0 and step > 0) {
        timers.start("PrtlClear");
        m_metadomain.RemoveDeadParticles(dom);
        timers.stop("PrtlClear");
      }
    }
  };

} // namespace ntt

#endif // ENGINES_SRPIC_SRPIC_H
