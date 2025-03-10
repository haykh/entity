/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - grpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "engines/engine.hpp"
#include "kernels/ampere_gr.hpp"
#include "kernels/aux_fields_gr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/faraday_gr.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_pusher_gr.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>
#include <utility>

namespace ntt {

  enum class gr_getE {
    D0_B,
    D_B0
  };
  enum class gr_getH {
    D_B0,
    D0_B0
  };
  enum class gr_faraday {
    aux,
    main
  };
  enum class gr_ampere {
    init,
    aux,
    main
  };
  enum class gr_bc {
    main,
    aux
  };

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using base_t   = Engine<SimEngine::GRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::GRPIC, M>;
    using domain_t = Domain<SimEngine::GRPIC, M>;
    // constexprs
    using base_t::pgen_is_ok;
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
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine(SimulationParams& params) : base_t { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.toggles.fieldsolver");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.toggles.deposit");
      const auto clear_interval = m_params.template get<std::size_t>(
        "particles.clear_interval");

      if (step == 0) {
        if (fieldsolver_enabled) {
          // communicate fields and apply BCs on the first timestep
          /**
           * Initially: em0::B   --
           *            em0::D   --
           *            em::B    at -1/2
           *            em::D    at -1/2
           *
           *            cur0::J  --
           *            cur::J   --
           *
           *            aux::E   --
           *            aux::H   --
           *
           *            x_prtl   at -1/2
           *            u_prtl   at -1/2
           */

          /**
           * em0::D, em::D, em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom,
                                         Comm::B | Comm::B0 | Comm::D | Comm::D0);
          FieldBoundaries(dom, BC::B | BC::D, gr_bc::main);

          /**
           * em0::B <- em::B
           * em0::D <- em::D
           *
           * Now: em0::B & em0::D at -1/2
           */
          CopyFields(dom);

          /**
           * aux::E <- alpha * em::D + beta x em0::B
           * aux::H <- alpha * em::B0 - beta x em::D
           *
           * Now: aux::E & aux::H at -1/2
           */
          ComputeAuxE(dom, gr_getE::D_B0);
          ComputeAuxH(dom, gr_getH::D_B0);

          /**
           * aux::E, aux::H <- boundary conditions
           */
          FieldBoundaries(dom, BC::H | BC::E, gr_bc::aux);

          /**
           * em0::B <- (em0::B) <- -curl aux::E
           *
           * Now: em0::B at 0
           */
          Faraday(dom, gr_faraday::aux, HALF);

          /**
           * em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
          FieldBoundaries(dom, BC::B, gr_bc::main);

          /**
           * em::D <- (em0::D) <- curl aux::H
           *
           * Now: em::D at 0
           */
          Ampere(dom, gr_ampere::init, HALF);

          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          FieldBoundaries(dom, BC::D, gr_bc::main);

          /**
           * aux::E <- alpha * em::D + beta x em0::B
           * aux::H <- alpha * em0::B - beta x em::D
           *
           * Now: aux::E & aux::H at 0
           */
          ComputeAuxE(dom, gr_getE::D_B0);
          ComputeAuxH(dom, gr_getH::D_B0);

          /**
           * aux::E, aux::H <- boundary conditions
           */
          FieldBoundaries(dom, BC::H | BC::E, gr_bc::aux);

          // !ADD: GR -- particles?

          /**
           * em0::B <- (em::B) <- -curl aux::E
           *
           * Now: em0::B at 1/2
           */
          Faraday(dom, gr_faraday::main, ONE);
          /**
           * em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
          FieldBoundaries(dom, BC::B, gr_bc::main);

          /**
           * em0::D <- (em0::D) <- curl aux::H
           *
           * Now: em0::D at 1/2
           */
          Ampere(dom, gr_ampere::aux, ONE);
          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          FieldBoundaries(dom, BC::D, gr_bc::main);

          /**
           * aux::H <- alpha * em0::B - beta x em0::D
           *
           * Now: aux::H at 1/2
           */
          ComputeAuxH(dom, gr_getH::D0_B0);
          /**
           * aux::H <- boundary conditions
           */
          FieldBoundaries(dom, BC::H, gr_bc::aux);

          /**
           * em0::D <- (em::D) <- curl aux::H
           *
           * Now: em0::D at 1
           *      em::D at 0
           */
          Ampere(dom, gr_ampere::main, ONE);
          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          FieldBoundaries(dom, BC::D, gr_bc::main);

          /**
           * em::D <-> em0::D
           * em::B <-> em0::B
           * em::J <-> em0::J
           */
          SwapFields(dom);
          /**
           * Finally: em0::B   at -1/2
           *          em0::D   at 0
           *          em::B    at 1/2
           *          em::D    at 1
           *
           *          cur0::J  --
           *          cur::J   --
           *
           *          aux::E   --
           *          aux::H   --
           *
           *          x_prtl   at 1
           *          u_prtl   at 1/2
           */
        } else {
          /**
           * em0::B <- em::B
           * em0::D <- em::D
           *
           * Now: em0::B & em0::D at -1/2
           */
          CopyFields(dom);
        }
      }

      /**
       * Initially: em0::B   at n-3/2
       *            em0::D   at n-1
       *            em::B    at n-1/2
       *            em::D    at n
       *
       *            cur0::J  --
       *            cur::J   at n-1/2
       *
       *            aux::E   --
       *            aux::H   --
       *
       *            x_prtl   at n
       *            u_prtl   at n-1/2
       */

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        /**
         * em0::D <- (em0::D + em::D) / 2
         * em0::B <- (em0::B + em::B) / 2
         *
         * Now: em0::D at n-1/2
         *      em0::B at n-1
         */
        TimeAverageDB(dom);
        /**
         * aux::E <- alpha * em0::D + beta x em::B
         *
         * Now: aux::E at n-1/2
         */
        ComputeAuxE(dom, gr_getE::D0_B);
        timers.stop("FieldSolver");

        timers.start("FieldBoundaries");
        /**
         * aux::E <- boundary conditions
         */
        FieldBoundaries(dom, BC::E, gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::B <- (em0::B) <- -curl aux::E
         *
         * Now: em0::B at n
         */
        Faraday(dom, gr_faraday::aux, ONE);
        timers.stop("FieldSolver");

        /**
         * em0::B, em::B <- boundary conditions
         */
        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::B, gr_bc::main);
        timers.stop("FieldBoundaries");
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
        timers.stop("Communications");

        timers.start("FieldSolver");
        /**
         * aux::H <- alpha * em0::B - beta x em::D
         *
         * Now: aux::H at n
         */
        ComputeAuxH(dom, gr_getH::D_B0);
        timers.stop("FieldSolver");

        timers.start("FieldBoundaries");
        /**
         * aux::H <- boundary conditions
         */
        FieldBoundaries(dom, BC::H, gr_bc::aux);
        timers.stop("FieldBoundaries");
      }

      {
        /**
         * x_prtl, u_prtl <- em::D, em0::B
         *
         * Now: x_prtl at n + 1, u_prtl at n + 1/2
         */
        timers.start("ParticlePusher");
        ParticlePush(dom);
        timers.stop("ParticlePusher");

        /**
         * cur0::J <- current deposition
         *
         * Now: cur0::J at n+1/2
         */
        if (deposit_enabled) {
          timers.start("CurrentDeposit");
          Kokkos::deep_copy(dom.fields.cur0, ZERO);
          CurrentsDeposit(dom);
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.SynchronizeFields(dom, Comm::J);
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

          timers.start("FieldBoundaries");
          CurrentsBoundaryConditions(dom);
          timers.stop("FieldBoundaries");

          timers.start("CurrentFiltering");
          CurrentsFilter(dom);
          timers.stop("CurrentFiltering");
        }

        timers.start("Communications");
        m_metadomain.CommunicateParticles(dom);
        timers.stop("Communications");
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        if (deposit_enabled) {
          /**
           * cur::J <- (cur0::J + cur::J) / 2
           *
           * Now: cur::J at n
           */
          TimeAverageJ(dom);
        }
        /**
         * aux::Е <- alpha * em::D + beta x em0::B
         *
         * Now: aux::Е at n
         */
        ComputeAuxE(dom, gr_getE::D_B0);
        timers.stop("FieldSolver");

        timers.start("FieldBoundaries");
        /**
         * aux::Е <- boundary conditions
         */
        FieldBoundaries(dom, BC::E, gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::B <- (em::B) <- -curl aux::E
         *
         * Now: em0::B at n+1/2
         *      em::B at n-1/2
         */
        Faraday(dom, gr_faraday::main, ONE);
        timers.stop("FieldSolver");

        /**
         * em0::B, em::B <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::B, gr_bc::main);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::D <- (em0::D) <- curl aux::H
         *
         * Now: em0::D at n+1/2
         */
        Ampere(dom, gr_ampere::aux, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          /**
           * em0::D <- (em0::D) <- cur::J
           *
           * Now: em0::D at n+1/2
           */
          AmpereCurrents(dom, gr_ampere::aux);
          timers.stop("FieldSolver");
        }

        /**
         * em0::D, em::D <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::D, gr_bc::main);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * aux::H <- alpha * em0::B - beta x em0::D
         *
         * Now: aux::H at n+1/2
         */
        ComputeAuxH(dom, gr_getH::D0_B0);
        timers.stop("FieldSolver");

        timers.start("FieldBoundaries");
        /**
         * aux::H <- boundary conditions
         */
        FieldBoundaries(dom, BC::B, gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::D <- (em::D) <- curl aux::H
         *
         * Now: em0::D at n+1
         *      em::D at n
         */
        Ampere(dom, gr_ampere::main, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          /**
           * em0::D <- (em0::D) <- cur0::J
           *
           * Now: em0::D at n+1
           */
          AmpereCurrents(dom, gr_ampere::main);
          timers.stop("FieldSolver");
        }
        timers.start("FieldSolver");
        /**
         * em::D <-> em0::D
         * em::B <-> em0::B
         * cur::J <-> cur0::J
         */
        SwapFields(dom);
        timers.stop("FieldSolver");

        /**
         * em0::D, em::D <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::D, gr_bc::main);
        timers.stop("FieldBoundaries");
      }

      if (clear_interval > 0 and step % clear_interval == 0 and step > 0) {
        timers.start("PrtlClear");
        m_metadomain.RemoveDeadParticles(dom);
        timers.stop("PrtlClear");
      }

      /**
       * Finally: em0::B   at n-1/2
       *          em0::D   at n
       *          em::B    at n+1/2
       *          em::D    at n+1
       *
       *          cur0::J  (at n)
       *          cur::J   at n+1/2
       *
       *          aux::E   (at n+1/2)
       *          aux::H   (at n)
       *
       *          x_prtl   at n+1
       *          u_prtl   at n+1/2
       */
    }

    /* algorithm substeps --------------------------------------------------- */
    void FieldBoundaries(domain_t& domain, BCTags tags, const gr_bc& g) {
      if (g == gr_bc::main) {
        for (auto& direction : dir::Directions<M::Dim>::orth) {
          if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::MATCH) {
            MatchFieldsIn(direction, domain, tags);
          } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::AXIS) {
            if (domain.mesh.flds_bc_in(direction) == FldsBC::AXIS) {
              AxisFieldsIn(direction, domain, tags);
            }
          } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::CUSTOM) {
            if (domain.mesh.flds_bc_in(direction) == FldsBC::CUSTOM) {
              CustomFieldsIn(direction, domain, tags, g);
            }
          } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::HORIZON) {
            if (domain.mesh.flds_bc_in(direction) == FldsBC::HORIZON) {
              HorizonFieldsIn(direction, domain, tags, g);
            }
          }
        } // loop over directions
      } else if (g == gr_bc::aux) {
        for (auto& direction : dir::Directions<M::Dim>::orth) {
          if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::HORIZON) {
            if (domain.mesh.flds_bc_in(direction) == FldsBC::HORIZON) {
              HorizonFieldsIn(direction, domain, tags, g);
            }
          }
        }
      }
    }

    void MatchFieldsIn(dir::direction_t<M::Dim> direction,
                       domain_t&                domain,
                       BCTags                   tags) {
      /**
       * match boundaries
       */
      const auto ds = m_params.template get<real_t>("grid.boundaries.match.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      xg_max  = m_metadomain.mesh().extent(dim).second;
      xg_min  = xg_max - ds;
      xg_edge = xg_max;

      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<unsigned short>(dim)) {
          box.push_back({ xg_min, xg_max });
          incl_ghosts.push_back({ false, true });
        } else {
          box.push_back(Range::All);
          incl_ghosts.push_back({ true, true });
        }
      }
      if (not domain.mesh.Intersects(box)) {
        return;
      }
      const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> range_min { 0 };
      tuple_t<std::size_t, M::Dim> range_max { 0 };

      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }
      if (dim == in::x1) {
        Kokkos::parallel_for(
          "MatchBoundaries",
          CreateRangePolicy<M::Dim>(range_min, range_max),
          kernel::bc::MatchBoundaries_kernel<S, decltype(m_pgen.init_flds), M, in::x1>(
            domain.fields.em,
            m_pgen.init_flds,
            domain.mesh.metric,
            xg_edge,
            ds,
            tags));
        Kokkos::parallel_for(
          "MatchBoundaries",
          CreateRangePolicy<M::Dim>(range_min, range_max),
          kernel::bc::MatchBoundaries_kernel<S, decltype(m_pgen.init_flds), M, in::x1>(
            domain.fields.em0,
            m_pgen.init_flds,
            domain.mesh.metric,
            xg_edge,
            ds,
            tags));
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    }

    void CurrentsBoundaryConditions(domain_t& domain) {
      /**
       * match boundaries
       */
      const auto ds = m_params.template get<real_t>("grid.boundaries.match.ds");
      const auto dim = in::x1;
      real_t     xg_min, xg_max, xg_edge;
      xg_max  = m_metadomain.mesh().extent(dim).second;
      xg_min  = xg_max - ds;
      xg_edge = xg_max;

      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<unsigned short>(dim)) {
          box.push_back({ xg_min, xg_max });
          incl_ghosts.push_back({ false, true });
        } else {
          box.push_back(Range::All);
          incl_ghosts.push_back({ true, true });
        }
      }
      if (not domain.mesh.Intersects(box)) {
        return;
      }
      const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> range_min { 0 };
      tuple_t<std::size_t, M::Dim> range_max { 0 };

      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }
      Kokkos::parallel_for(
        "AbsorbCurrentsGR",
        CreateRangePolicy<M::Dim>(range_min, range_max),
        kernel::bc::AbsorbCurrentsGR_kernel<M, 1>(domain.fields.cur0,
                                                  domain.mesh.metric,
                                                  xg_edge,
                                                  ds));
    }

    void HorizonFieldsIn(dir::direction_t<M::Dim> direction,
                         domain_t&                domain,
                         BCTags                   tags,
                         const gr_bc&             g) {
      /**
       * open boundaries
       */
      raise::ErrorIf(M::CoordType == Coord::Cart,
                     "Invalid coordinate type for horizon BCs",
                     HERE);
      raise::ErrorIf(direction.get_dim() != in::x1,
                     "Invalid horizon direction, should be x1",
                     HERE);
      const auto i1_min = domain.mesh.i_min(in::x1);
      auto range = CreateRangePolicy<Dim::_1D>({ domain.mesh.i_min(in::x2) },
                                               { domain.mesh.i_max(in::x2) + 1 });
      if (g == gr_bc::main) {
        Kokkos::parallel_for(
          "OpenBCFields",
          range,
          kernel::bc::HorizonBoundaries_kernel<M, false>(domain.fields.em,
                                                         i1_min,
                                                         tags));
        Kokkos::parallel_for(
          "OpenBCFields",
          range,
          kernel::bc::HorizonBoundaries_kernel<M, false>(domain.fields.em0,
                                                         i1_min,
                                                         tags));
      } else if (g == gr_bc::aux) {
        Kokkos::parallel_for(
          "OpenBCFields",
          range,
          kernel::bc::HorizonBoundaries_kernel<M, true>(domain.fields.aux,
                                                        i1_min,
                                                        tags));
      }
    }

    void AxisFieldsIn(dir::direction_t<M::Dim> direction,
                      domain_t&                domain,
                      BCTags                   tags) {
      /**
       * axis boundaries
       */
      raise::ErrorIf(M::CoordType == Coord::Cart,
                     "Invalid coordinate type for axis BCs",
                     HERE);
      raise::ErrorIf(direction.get_dim() != in::x2,
                     "Invalid axis direction, should be x2",
                     HERE);
      const auto i2_min = domain.mesh.i_min(in::x2);
      const auto i2_max = domain.mesh.i_max(in::x2);
      auto range = CreateRangePolicy<Dim::_1D>({ domain.mesh.i_min(in::x1) - 1 },
                                               { domain.mesh.i_max(in::x1) });
      if (direction.get_sign() < 0) {
        Kokkos::parallel_for(
          "AxisBCFields",
          range,
          kernel::bc::AxisBoundariesGR_kernel<M::Dim, false>(domain.fields.em,
                                                             i2_min,
                                                             tags));
        Kokkos::parallel_for(
          "AxisBCFields",
          range,
          kernel::bc::AxisBoundariesGR_kernel<M::Dim, false>(domain.fields.em0,
                                                             i2_min,
                                                             tags));
      } else {
        Kokkos::parallel_for(
          "AxisBCFields",
          range,
          kernel::bc::AxisBoundariesGR_kernel<M::Dim, true>(domain.fields.em,
                                                            i2_max,
                                                            tags));
        Kokkos::parallel_for(
          "AxisBCFields",
          range,
          kernel::bc::AxisBoundariesGR_kernel<M::Dim, true>(domain.fields.em0,
                                                            i2_max,
                                                            tags));
      }
    }

    void CustomFieldsIn(dir::direction_t<M::Dim> direction,
                        domain_t&                domain,
                        BCTags                   tags,
                        const gr_bc&             g) {
      (void)direction;
      (void)domain;
      (void)tags;
      (void)g;
      raise::Error("Custom boundaries not implemented", HERE);
      // if constexpr (
      //   traits::has_member<traits::pgen::custom_fields_t, pgen_t>::value) {
      //   const auto [box, custom_fields] = m_pgen.CustomFields(time);
      //   if (domain.mesh.Intersects(box)) {
      //   }
      //
      // } else {
      //   raise::Error("Custom boundaries not implemented", HERE);
      // }
    }

    /**
     * @brief Swaps em and em0 fields, cur and cur0 currents.
     */
    void SwapFields(domain_t& domain) {
      std::swap(domain.fields.em, domain.fields.em0);
      std::swap(domain.fields.cur, domain.fields.cur0);
    }

    /**
     * @brief Copies em fields into em0
     */
    void CopyFields(domain_t& domain) {
      Kokkos::deep_copy(domain.fields.em0, domain.fields.em);
    }

    void ComputeAuxE(domain_t& domain, const gr_getE& g) {
      auto range = range_with_axis_BCs(domain);
      if (g == gr_getE::D0_B) {
        Kokkos::parallel_for(
          "ComputeAuxE",
          range,
          kernel::gr::ComputeAuxE_kernel<M>(domain.fields.em0, // D
                                            domain.fields.em,  // B
                                            domain.fields.aux, // E
                                            domain.mesh.metric));
      } else if (g == gr_getE::D_B0) {
        Kokkos::parallel_for("ComputeAuxE",
                             range,
                             kernel::gr::ComputeAuxE_kernel<M>(domain.fields.em,
                                                               domain.fields.em0,
                                                               domain.fields.aux,
                                                               domain.mesh.metric));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    void ComputeAuxH(domain_t& domain, const gr_getH& g) {
      auto range = range_with_axis_BCs(domain);
      if (g == gr_getH::D_B0) {
        Kokkos::parallel_for(
          "ComputeAuxH",
          range,
          kernel::gr::ComputeAuxH_kernel<M>(domain.fields.em,  // D
                                            domain.fields.em0, // B
                                            domain.fields.aux, // H
                                            domain.mesh.metric));
      } else if (g == gr_getH::D0_B0) {
        Kokkos::parallel_for("ComputeAuxH",
                             range,
                             kernel::gr::ComputeAuxH_kernel<M>(domain.fields.em0,
                                                               domain.fields.em0,
                                                               domain.fields.aux,
                                                               domain.mesh.metric));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    auto range_with_axis_BCs(const domain_t& domain) -> range_t<M::Dim> {
      auto range = domain.mesh.rangeActiveCells();
      /**
       * @brief taking one extra cell in the x1 and x2 directions if AXIS BCs
       */
      if constexpr (M::Dim == Dim::_2D) {
        if (domain.mesh.flds_bc_in({ 0, +1 }) == FldsBC::AXIS) {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1) - 1, domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        }
      } else if constexpr (M::Dim == Dim::_3D) {
        raise::Error("Invalid dimension", HERE);
      }
      return range;
    }

    void Faraday(domain_t& domain, const gr_faraday& g, real_t fraction = ONE) {
      logger::Checkpoint("Launching Faraday kernel", HERE);
      const auto dT = fraction *
                      m_params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      if (g == gr_faraday::aux) {
        Kokkos::parallel_for(
          "Faraday",
          domain.mesh.rangeActiveCells(),
          kernel::gr::Faraday_kernel<M>(domain.fields.em0, // Bin
                                        domain.fields.em0, // Bout
                                        domain.fields.aux, // E
                                        domain.mesh.metric,
                                        dT,
                                        domain.mesh.n_active(in::x2),
                                        domain.mesh.flds_bc()));
      } else if (g == gr_faraday::main) {
        Kokkos::parallel_for(
          "Faraday",
          domain.mesh.rangeActiveCells(),
          kernel::gr::Faraday_kernel<M>(domain.fields.em,
                                        domain.fields.em0,
                                        domain.fields.aux,
                                        domain.mesh.metric,
                                        dT,
                                        domain.mesh.n_active(in::x2),
                                        domain.mesh.flds_bc()));

      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    void Ampere(domain_t& domain, const gr_ampere& g, real_t fraction = ONE) {
      logger::Checkpoint("Launching Ampere kernel", HERE);
      const auto dT = fraction *
                      m_params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      auto range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto ni2 = domain.mesh.n_active(in::x2);

      if (g == gr_ampere::aux) {
        // First push, updates D0 with J.
        Kokkos::parallel_for("Ampere-1",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em0, // Din
                                                          domain.fields.em0, // Dout
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else if (g == gr_ampere::main) {
        // Second push, updates D with J0 but assigns it to D0.
        Kokkos::parallel_for("Ampere-2",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.fields.em0,
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else if (g == gr_ampere::init) {
        // Second push, updates D with J0 and assigns it to D.
        Kokkos::parallel_for("Ampere-3",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.fields.em,
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    void AmpereCurrents(domain_t& domain, const gr_ampere& g) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
      const auto q0    = m_params.template get<real_t>("scales.q0");
      const auto B0    = m_params.template get<real_t>("scales.B0");
      const auto coeff = -dt * q0 / B0;
      auto       range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto ni2 = domain.mesh.n_active(in::x2);

      if (g == gr_ampere::aux) {
        // Updates D0 with J: D0(n-1/2) -> (J(n)) -> D0(n+1/2)
        Kokkos::parallel_for(
          "AmpereCurrentsAux",
          range,
          kernel::gr::CurrentsAmpere_kernel<M>(domain.fields.em0,
                                               domain.fields.cur,
                                               domain.mesh.metric,
                                               coeff,
                                               ni2,
                                               domain.mesh.flds_bc()));
      } else if (g == gr_ampere::main) {
        // Updates D0 with J0: D0(n) -> (J0(n+1/2)) -> D0(n+1)
        Kokkos::parallel_for(
          "AmpereCurrentsMain",
          range,
          kernel::gr::CurrentsAmpere_kernel<M>(domain.fields.em0,
                                               domain.fields.cur0,
                                               domain.mesh.metric,
                                               coeff,
                                               ni2,
                                               domain.mesh.flds_bc()));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    void TimeAverageDB(domain_t& domain) {
      Kokkos::parallel_for("TimeAverageDB",
                           domain.mesh.rangeActiveCells(),
                           kernel::gr::TimeAverageDB_kernel<M>(domain.fields.em,
                                                               domain.fields.em0,
                                                               domain.mesh.metric));
    }

    void TimeAverageJ(domain_t& domain) {
      Kokkos::parallel_for("TimeAverageJ",
                           domain.mesh.rangeActiveCells(),
                           kernel::gr::TimeAverageJ_kernel<M>(domain.fields.cur,
                                                              domain.fields.cur0,
                                                              domain.mesh.metric));
    }

    void CurrentsDeposit(domain_t& domain) {
      auto scatter_cur0 = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur0);
      for (auto& species : domain.species) {
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);
        if (species.npart() == 0 || cmp::AlmostZero(species.charge())) {
          continue;
        }
        Kokkos::parallel_for("CurrentsDeposit",
                             species.rangeActiveParticles(),
                             kernel::DepositCurrents_kernel<SimEngine::GRPIC, M>(
                               scatter_cur0,
                               species.i1,
                               species.i2,
                               species.i3,
                               species.i1_prev,
                               species.i2_prev,
                               species.i3_prev,
                               species.dx1,
                               species.dx2,
                               species.dx3,
                               species.dx1_prev,
                               species.dx2_prev,
                               species.dx3_prev,
                               species.ux1,
                               species.ux2,
                               species.ux3,
                               species.phi,
                               species.weight,
                               species.tag,
                               domain.mesh.metric,
                               (real_t)(species.charge()),
                               dt));
      }
      Kokkos::Experimental::contribute(domain.fields.cur0, scatter_cur0);
    }

    void CurrentsFilter(domain_t& domain) {
      logger::Checkpoint("Launching currents filtering kernels", HERE);
      auto range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto nfilter = m_params.template get<unsigned short>(
        "algorithms.current_filters");
      tuple_t<std::size_t, M::Dim> size;
      size[0] = domain.mesh.n_active(in::x1);
      size[1] = domain.mesh.n_active(in::x2);

      // !TODO: this needs to be done more efficiently
      for (unsigned short i = 0; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur0);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur0,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        m_metadomain.CommunicateFields(domain, Comm::J); // J0
      }
    }

    void ParticlePush(domain_t& domain) {
      for (auto& species : domain.species) {
        species.set_unsorted();
        logger::Checkpoint(
          fmt::format("Launching particle pusher kernel for %d [%s] : %lu",
                      species.index(),
                      species.label().c_str(),
                      species.npart()),
          HERE);
        if (species.npart() == 0) {
          continue;
        }
        const auto q_ovr_m = species.mass() > ZERO
                               ? species.charge() / species.mass()
                               : ZERO;
        //  coeff = q / m (dt / 2) omegaB0
        const auto coeff   = q_ovr_m * HALF * dt *
                           m_params.template get<real_t>(
                             "algorithms.timestep.correction") *
                           m_params.template get<real_t>("scales.omegaB0");
        const auto eps = m_params.template get<real_t>(
          "algorithms.gr.pusher_eps");
        const auto niter = m_params.template get<unsigned short>(
          "algorithms.gr.pusher_niter");
        // clang-format off
        if (species.pusher() == PrtlPusher::PHOTON) {
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, kernel::gr::Massless_t>(
          0,
          species.npart());

        Kokkos::parallel_for(
          "ParticlePusher",
          range_policy,
          kernel::gr::Pusher_kernel<M>(
              domain.fields.em,
              domain.fields.em0,
              species.i1,        species.i2,       species.i3,
              species.i1_prev,   species.i2_prev,  species.i3_prev,
              species.dx1,       species.dx2,      species.dx3,
              species.dx1_prev,  species.dx2_prev, species.dx3_prev,
              species.ux1,       species.ux2,      species.ux3,
              species.phi,       species.tag,
              domain.mesh.metric,
              coeff, dt,
              domain.mesh.n_active(in::x1),
              domain.mesh.n_active(in::x2),
              domain.mesh.n_active(in::x3),
              eps, niter,
              domain.mesh.prtl_bc()
          ));
        } else if (species.pusher() == PrtlPusher::BORIS) {
          auto range_policy = Kokkos::RangePolicy<AccelExeSpace, kernel::gr::Massive_t>(
          0,
          species.npart());
          Kokkos::parallel_for(
            "ParticlePusher",
            range_policy,
            kernel::gr::Pusher_kernel<M>(
                domain.fields.em,
                domain.fields.em0,
                species.i1,        species.i2,       species.i3,
                species.i1_prev,   species.i2_prev,  species.i3_prev,
                species.dx1,       species.dx2,      species.dx3,
                species.dx1_prev,  species.dx2_prev, species.dx3_prev,
                species.ux1,       species.ux2,      species.ux3,
                species.phi,       species.tag,
                domain.mesh.metric,
                coeff, dt,
                domain.mesh.n_active(in::x1),
                domain.mesh.n_active(in::x2),
                domain.mesh.n_active(in::x3),
                eps, niter,
                domain.mesh.prtl_bc()
          ));
        } else if (species.pusher() == PrtlPusher::NONE) {
          // do nothing
        } else {
          raise::Error("not implemented", HERE);
        }
        // clang-format on
      }
    }
  };
} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H
