/**
 * @file engines/srpic.h
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

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"

#include "engines/engine.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "kernels/ampere_mink.hpp"
#include "kernels/ampere_sr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/faraday_mink.hpp"
#include "kernels/faraday_sr.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_pusher_sr.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>

namespace ntt {

  template <class M>
  class SRPICEngine : public Engine<SimEngine::SRPIC, M> {

    using base_t   = Engine<SimEngine::SRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::SRPIC, M>;
    using domain_t = Domain<SimEngine::SRPIC, M>;
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
    static constexpr auto S { SimEngine::SRPIC };

    SRPICEngine(SimulationParams& params) : base_t { params } {}

    ~SRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.toggles.fieldsolver");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.toggles.deposit");

      if (step == 0) {
        // communicate fields and apply BCs on the first timestep
        m_metadomain.Communicate(dom, Comm::B | Comm::E);
        FieldBoundaries(dom, BC::B | BC::E);
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        Faraday(dom, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.Communicate(dom, Comm::B);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::B);
        timers.stop("FieldBoundaries");
      }

      {
        timers.start("ParticlePusher");
        ParticlePush(dom);
        timers.stop("ParticlePusher");

        if (deposit_enabled) {
          timers.start("CurrentDeposit");
          Kokkos::deep_copy(dom.fields.cur, ZERO);
          CurrentsDeposit(dom);
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.Communicate(dom, Comm::J_sync);
          m_metadomain.Communicate(dom, Comm::J);
          timers.stop("Communications");

          timers.start("CurrentFiltering");
          CurrentsFilter(dom);
          timers.stop("CurrentFiltering");
        }

        timers.start("Communications");
        m_metadomain.Communicate(dom, Comm::Prtl);
        timers.stop("Communications");
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        Faraday(dom, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.Communicate(dom, Comm::B);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::B);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        Ampere(dom, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          CurrentsAmpere(dom);
          timers.stop("FieldSolver");
        }

        timers.start("Communications");
        m_metadomain.Communicate(dom, Comm::E | Comm::J);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::E);
        timers.stop("FieldBoundaries");
      }
    }

    /* algorithm substeps --------------------------------------------------- */
    void Faraday(domain_t& domain, real_t fraction = ONE) {
      logger::Checkpoint("Launching Faraday kernel", HERE);
      const auto dT = fraction *
                      m_params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto dx = math::sqrt(domain.mesh.metric.template h_<1, 1>({}));
        real_t     coeff1, coeff2;
        if constexpr (M::Dim == Dim::_2D) {
          coeff1 = dT / SQR(dx);
          coeff2 = dT;
        } else {
          coeff1 = dT / dx;
          coeff2 = ZERO;
        }
        Kokkos::parallel_for(
          "Faraday",
          domain.mesh.rangeActiveCells(),
          kernel::mink::Faraday_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
      } else {
        Kokkos::parallel_for("Faraday",
                             domain.mesh.rangeActiveCells(),
                             kernel::sr::Faraday_kernel<M>(domain.fields.em,
                                                           domain.mesh.metric,
                                                           dT,
                                                           domain.mesh.flds_bc()));
      }
    }

    void Ampere(domain_t& domain, real_t fraction = ONE) {
      logger::Checkpoint("Launching Ampere kernel", HERE);
      const auto dT = fraction *
                      m_params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto dx = math::sqrt(domain.mesh.metric.template h_<1, 1>({}));
        real_t     coeff1, coeff2;
        if constexpr (M::Dim == Dim::_2D) {
          coeff1 = dT / SQR(dx);
          coeff2 = dT;
        } else {
          coeff1 = dT / dx;
          coeff2 = ZERO;
        }

        Kokkos::parallel_for(
          "Ampere",
          domain.mesh.rangeActiveCells(),
          kernel::mink::Ampere_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
      } else {
        range_t<M::Dim> range {};
        if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        }
        const auto ni2 = domain.mesh.n_active(in::x2);
        Kokkos::parallel_for("Ampere",
                             range,
                             kernel::sr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      }
    }

    void ParticlePush(domain_t& domain) {
      for (auto& species : domain.species) {
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
                           m_params.template get<real_t>("scales.omegaB0");
        PrtlPusher::type pusher;
        if (species.pusher() == PrtlPusher::PHOTON) {
          pusher = PrtlPusher::PHOTON;
        } else if (species.pusher() == PrtlPusher::BORIS) {
          pusher = PrtlPusher::BORIS;
        } else if (species.pusher() == PrtlPusher::VAY) {
          pusher = PrtlPusher::VAY;
        } else {
          raise::Fatal("Invalid particle pusher", HERE);
        }
        const auto cooling = species.cooling();

        // coefficients to be forwarded to the dispatcher
        // gca
        const auto has_gca         = species.use_gca();
        const auto gca_larmor_max  = has_gca ? m_params.template get<real_t>(
                                                "algorithms.gca.larmor_max")
                                             : ZERO;
        const auto gca_eovrb_max   = has_gca ? m_params.template get<real_t>(
                                               "algorithms.gca.e_ovr_b_max")
                                             : ZERO;
        // cooling
        const auto has_synchrotron = (cooling == Cooling::SYNCHROTRON);
        const auto sync_grad       = has_synchrotron
                                       ? m_params.template get<real_t>(
                                     "algorithms.synchrotron.gamma_rad")
                                       : ZERO;
        const auto sync_coeff      = has_synchrotron
                                       ? (real_t)(0.1) * dt *
                                      m_params.template get<real_t>(
                                        "scales.omegaB0") /
                                      (SQR(sync_grad) * species.mass())
                                       : ZERO;

        // toggle to indicate whether pgen defines the external force
        bool has_extforce = false;
        if constexpr (traits::has_member<traits::pgen::ext_force_t, pgen_t>::value) {
          has_extforce = true;
          // toggle to indicate whether the ext force applies to current species
          if (traits::has_member<traits::species_t, decltype(pgen_t::ext_force)>::value) {
            has_extforce &= std::find(m_pgen.ext_force.species.begin(),
                                      m_pgen.ext_force.species.end(),
                                      species.index()) !=
                            m_pgen.ext_force.species.end();
          }
        }

        kernel::sr::CoolingTags cooling_tags = 0;
        if (cooling == Cooling::SYNCHROTRON) {
          cooling_tags = kernel::sr::Cooling::Synchrotron;
        }
        Kokkos::parallel_for(
          "ParticlePusher",
          species.rangeActiveParticles(),
          kernel::sr::Pusher_kernel<M, pgen_t>(pusher,
                                               has_gca,
                                               false,
                                               cooling_tags,
                                               domain.fields.em,
                                               species.index(),
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
                                               species.tag,
                                               domain.mesh.metric,
                                               m_pgen,
                                               time,
                                               coeff,
                                               dt,
                                               domain.mesh.n_active(in::x1),
                                               domain.mesh.n_active(in::x2),
                                               domain.mesh.n_active(in::x3),
                                               domain.mesh.prtl_bc(),
                                               gca_larmor_max,
                                               gca_eovrb_max,
                                               sync_coeff));
      }
    }

    void CurrentsDeposit(domain_t& domain) {
      auto scatter_cur = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur);
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
                             kernel::DepositCurrents_kernel<SimEngine::SRPIC, M>(
                               scatter_cur,
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
      Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
    }

    void CurrentsAmpere(domain_t& domain) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
      const auto q0    = m_params.template get<real_t>("scales.q0");
      const auto n0    = m_params.template get<real_t>("scales.n0");
      const auto B0    = m_params.template get<real_t>("scales.B0");
      const auto coeff = -dt * q0 * n0 / B0;
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto V0 = m_params.template get<real_t>("scales.V0");

        Kokkos::parallel_for(
          "Ampere",
          domain.mesh.rangeActiveCells(),
          kernel::mink::CurrentsAmpere_kernel<M::Dim>(domain.fields.em,
                                                      domain.fields.cur,
                                                      coeff / V0,
                                                      ONE / n0));
      } else {
        range_t<M::Dim> range {};
        if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        }
        const auto ni2 = domain.mesh.n_active(in::x2);
        Kokkos::parallel_for(
          "Ampere",
          range,
          kernel::sr::CurrentsAmpere_kernel<M>(domain.fields.em,
                                               domain.fields.cur,
                                               domain.mesh.metric,
                                               coeff,
                                               ONE / n0,
                                               ni2,
                                               domain.mesh.flds_bc()));
      }
    }

    void CurrentsFilter(domain_t& domain) {
      logger::Checkpoint("Launching currents filtering kernels", HERE);
      range_t<M::Dim> range = domain.mesh.rangeActiveCells();
      if constexpr (M::CoordType != Coord::Cart) {
        /**
         * @brief taking one extra cell in the x2 direction
         *    . . . . .
         *    . ^= =^ .
         *    . |* *\*.
         *    . |* *\*.
         *    . ^- -^ .
         *    . . . . .
         */
        if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        } else if constexpr (M::Dim == Dim::_3D) {
          range = CreateRangePolicy<Dim::_3D>({ domain.mesh.i_min(in::x1),
                                                domain.mesh.i_min(in::x2),
                                                domain.mesh.i_min(in::x3) },
                                              { domain.mesh.i_max(in::x1),
                                                domain.mesh.i_max(in::x2) + 1,
                                                domain.mesh.i_max(in::x3) });
        }
      }
      const auto nfilter = m_params.template get<unsigned short>(
        "algorithms.current_filters");
      tuple_t<std::size_t, M::Dim> size;
      if constexpr (M::Dim == Dim::_1D || M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[0] = domain.mesh.n_active(in::x1);
      }
      if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[1] = domain.mesh.n_active(in::x2);
      }
      if constexpr (M::Dim == Dim::_3D) {
        size[2] = domain.mesh.n_active(in::x3);
      }
      auto sync = 0;
      for (unsigned short i = 0; i < nfilter; ++i) {
        ++sync;
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        if (sync == N_GHOSTS) {
          sync = 0;
          m_metadomain.Communicate(domain, Comm::J);
        }
      }
    }

    void FieldBoundaries(domain_t& domain, BCTags tags) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::ABSORB) {
          AbsorbFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::AXIS) {
          AxisFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::ATMOSPHERE) {
          AtmosphereFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::CUSTOM) {
          raise::Error("Custom boundaries not implemented", HERE);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::HORIZON) {
          raise::Error("HORIZON BCs only applicable for GR", HERE);
        }
      } // loop over directions
    }

    void AbsorbFieldsIn(dir::direction_t<M::Dim> direction,
                        domain_t&                domain,
                        BCTags                   tags) {
      /**
       * absorbing boundaries
       */
      const auto ds = m_params.template get<real_t>(
        "grid.boundaries.absorb.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      auto       sign = direction.get_sign();
      if (sign > 0) { // + direction
        xg_max  = m_metadomain.mesh().extent(dim).second;
        xg_min  = xg_max - ds;
        xg_edge = xg_max;
      } else { // - direction
        xg_min  = m_metadomain.mesh().extent(dim).first;
        xg_max  = xg_min + ds;
        xg_edge = xg_min;
      }
      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<unsigned short>(dim)) {
          box.push_back({ xg_min, xg_max });
          if (sign > 0) {
            incl_ghosts.push_back({ false, true });
          } else {
            incl_ghosts.push_back({ true, false });
          }
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
          "AbsorbFields",
          CreateRangePolicy<M::Dim>(range_min, range_max),
          kernel::AbsorbBoundaries_kernel<M, 1>(domain.fields.em,
                                                domain.mesh.metric,
                                                xg_edge,
                                                ds,
                                                tags));
      } else if (dim == in::x2) {
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          Kokkos::parallel_for(
            "AbsorbFields",
            CreateRangePolicy<M::Dim>(range_min, range_max),
            kernel::AbsorbBoundaries_kernel<M, 2>(domain.fields.em,
                                                  domain.mesh.metric,
                                                  xg_edge,
                                                  ds,
                                                  tags));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else if (dim == in::x3) {
        if constexpr (M::Dim == Dim::_3D) {
          Kokkos::parallel_for(
            "AbsorbFields",
            CreateRangePolicy<M::Dim>(range_min, range_max),
            kernel::AbsorbBoundaries_kernel<M, 3>(domain.fields.em,
                                                  domain.mesh.metric,
                                                  xg_edge,
                                                  ds,
                                                  tags));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
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
      if (direction.get_sign() < 0) {
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em, i2_min, tags));
      } else {
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em, i2_max, tags));
      }
    }

    void AtmosphereFieldsIn(dir::direction_t<M::Dim> direction,
                            domain_t&                domain,
                            BCTags                   tags) {
      /**
       * atmosphere boundaries
       */
      if constexpr (traits::has_member<traits::pgen::field_driver_t, pgen_t>::value) {
        const auto min_buff = m_params.template get<unsigned short>(
                                "algorithms.current_filters") +
                              2;
        const auto  buffer_ncells = min_buff > 5 ? min_buff : 5;
        const auto  sign          = direction.get_sign();
        const auto  dim           = direction.get_dim();
        const auto  dd            = static_cast<unsigned short>(dim);
        real_t      xg_min { ZERO }, xg_max { ZERO };
        std::size_t ig_min, ig_max, il_edge;
        if (sign > 0) { // + direction
          ig_min = m_metadomain.mesh().n_active(dim) - buffer_ncells;
          ig_max = m_metadomain.mesh().n_active(dim);
        } else { // - direction
          ig_min = 0;
          ig_max = buffer_ncells;
        }
        if (dim == in::x1) {
          xg_min = m_metadomain.mesh().metric.template convert<1, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_min));
          xg_max = m_metadomain.mesh().metric.template convert<1, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(ig_max));
        } else if (dim == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            xg_min = m_metadomain.mesh().metric.template convert<2, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(ig_min));
            xg_max = m_metadomain.mesh().metric.template convert<2, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(ig_max));
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else if (dim == in::x3) {
          if constexpr (M::Dim == Dim::_3D) {
            xg_min = m_metadomain.mesh().metric.template convert<3, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(ig_min));
            xg_max = m_metadomain.mesh().metric.template convert<3, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(ig_max));
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          raise::Error("Invalid dimension", HERE);
        }
        boundaries_t<real_t> box;
        boundaries_t<bool>   incl_ghosts;
        for (unsigned short d { 0 }; d < M::Dim; ++d) {
          if (d == dd) {
            box.push_back({ xg_min, xg_max });
            if (sign > 0) {
              incl_ghosts.push_back({ false, true });
            } else {
              incl_ghosts.push_back({ true, false });
            }
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
        auto field_driver = m_pgen.FieldDriver(time);
        if (sign > 0) {
          il_edge = range_min[dd] - N_GHOSTS;
        } else {
          il_edge = range_max[dd] - 1 - N_GHOSTS;
        }
        const auto range = CreateRangePolicy<M::Dim>(range_min, range_max);
        if (dim == in::x1) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "AtmosphereBCFields",
              range,
              kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, true, in::x1>(
                domain.fields.em,
                field_driver,
                domain.mesh.metric,
                il_edge,
                tags));
          } else {
            Kokkos::parallel_for(
              "AtmosphereBCFields",
              range,
              kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, false, in::x1>(
                domain.fields.em,
                field_driver,
                domain.mesh.metric,
                il_edge,
                tags));
          }
        } else if (dim == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, true, in::x2>(
                  domain.fields.em,
                  field_driver,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, false, in::x2>(
                  domain.fields.em,
                  field_driver,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else if (dim == in::x3) {
          if constexpr (M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, true, in::x3>(
                  domain.fields.em,
                  field_driver,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::AtmosphereBoundaries_kernel<decltype(field_driver), M, false, in::x3>(
                  domain.fields.em,
                  field_driver,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else {
        raise::Error("Field driver not implemented in PGEN for atmosphere BCs",
                     HERE);
      }
    }
  };

} // namespace ntt

#endif // ENGINES_SRPIC_SRPIC_H
