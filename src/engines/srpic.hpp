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

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "metrics/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

#include "engines/engine.hpp"
#include "kernels/ampere_mink.hpp"
#include "kernels/ampere_sr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/faraday_mink.hpp"
#include "kernels/faraday_sr.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_pusher_sr.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <utility>

namespace ntt {

  template <class M>
    requires IsCompatibleWithEngine<SimEngine::SRPIC, M> &&
             metric::traits::HasH_ij<M> && metric::traits::HasConvert_i<M> &&
             metric::traits::HasSqrtH_ij<M>
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
        FieldBoundaries(dom, BC::B | BC::E);
        ParticleInjector(dom);
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        Faraday(dom, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::B);
        timers.stop("FieldBoundaries");
        Kokkos::fence();
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
          m_metadomain.SynchronizeFields(dom, Comm::J);
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

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
        Faraday(dom, HALF);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B);
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
        m_metadomain.CommunicateFields(dom, Comm::E | Comm::J);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::E);
        timers.stop("FieldBoundaries");
      }

      {
        timers.start("Injector");
        ParticleInjector(dom);
        timers.stop("Injector");
      }

      if (clear_interval > 0 and step % clear_interval == 0 and step > 0) {
        timers.start("PrtlClear");
        m_metadomain.RemoveDeadParticles(dom);
        timers.stop("PrtlClear");
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
        const auto deltax = m_params.template get<real_t>(
          "algorithms.fieldsolver.delta_x");
        const auto deltay = m_params.template get<real_t>(
          "algorithms.fieldsolver.delta_y");
        const auto betaxy = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_xy");
        const auto betayx = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_yx");
        const auto deltaz = m_params.template get<real_t>(
          "algorithms.fieldsolver.delta_z");
        const auto betaxz = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_xz");
        const auto betazx = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_zx");
        const auto betayz = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_yz");
        const auto betazy = m_params.template get<real_t>(
          "algorithms.fieldsolver.beta_zy");
        real_t coeff1, coeff2;
        if constexpr (M::Dim == Dim::_2D) {
          coeff1 = dT / SQR(dx);
          coeff2 = dT;
        } else {
          coeff1 = dT / dx;
          coeff2 = ZERO;
        }
        Kokkos::parallel_for("Faraday",
                             domain.mesh.rangeActiveCells(),
                             kernel::mink::Faraday_kernel<M::Dim>(domain.fields.em,
                                                                  coeff1,
                                                                  coeff2,
                                                                  deltax,
                                                                  deltay,
                                                                  betaxy,
                                                                  betayx,
                                                                  deltaz,
                                                                  betaxz,
                                                                  betazx,
                                                                  betayz,
                                                                  betazy));
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
      auto range = range_with_axis_BCs(domain);
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
          range,
          kernel::mink::Ampere_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
      } else {
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
      real_t gx1 { ZERO }, gx2 { ZERO }, gx3 { ZERO }, ds { ZERO };
      real_t x_surf { ZERO };
      bool   has_atmosphere = false;
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (m_metadomain.mesh().prtl_bc_in(direction) == PrtlBC::ATMOSPHERE) {
          raise::ErrorIf(has_atmosphere,
                         "Only one direction is allowed to have atm boundaries",
                         HERE);
          has_atmosphere = true;
          const auto g   = m_params.template get<real_t>(
            "grid.boundaries.atmosphere.g");
          ds = m_params.template get<real_t>("grid.boundaries.atmosphere.ds");
          const auto [sign, dim, xg_min, xg_max] = get_atm_extent(direction);
          if (dim == in::x1) {
            gx1 = sign > 0 ? g : -g;
            gx2 = ZERO;
            gx3 = ZERO;
          } else if (dim == in::x2) {
            gx1 = ZERO;
            gx2 = sign > 0 ? g : -g;
            gx3 = ZERO;
          } else if (dim == in::x3) {
            gx1 = ZERO;
            gx2 = ZERO;
            gx3 = sign > 0 ? g : -g;
          } else {
            raise::Error("Invalid dimension", HERE);
          }
          if (sign > 0) {
            x_surf = xg_min;
          } else {
            x_surf = xg_max;
          }
        }
      }
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or (species.npart() == 0)) {
          continue;
        }
        species.set_unsorted();
        logger::Checkpoint(
          fmt::format("Launching particle pusher kernel for %d [%s] : %lu",
                      species.index(),
                      species.label().c_str(),
                      species.npart()),
          HERE);

        kernel::sr::PusherParams pusher_params {};
        pusher_params.pusher_flags         = species.pusher();
        pusher_params.radiative_drag_flags = species.radiative_drag_flags();
        pusher_params.mass                 = species.mass();
        pusher_params.charge               = species.charge();
        pusher_params.time                 = time;
        pusher_params.dt                   = dt;
        pusher_params.omegaB0 = m_params.template get<real_t>("scales.omegaB0");
        pusher_params.ni1     = domain.mesh.n_active(in::x1);
        pusher_params.ni2     = domain.mesh.n_active(in::x2);
        pusher_params.ni3     = domain.mesh.n_active(in::x3);
        pusher_params.boundaries = domain.mesh.prtl_bc();

        if (species.pusher() & ParticlePusher::GCA) {
          pusher_params.gca_params.set(
            "larmor_max",
            m_params.template get<real_t>("algorithms.gca.larmor_max"));
          pusher_params.gca_params.set(
            "e_ovr_b_max",
            m_params.template get<real_t>("algorithms.gca.e_ovr_b_max"));
        }

        if (species.radiative_drag_flags() & RadiativeDrag::SYNCHROTRON) {
          pusher_params.radiative_drag_params.set(
            "synchrotron_gamma_rad",
            m_params.template get<real_t>(
              "radiation.drag.synchrotron.gamma_rad"));
        }

        if (species.radiative_drag_flags() & RadiativeDrag::COMPTON) {
          pusher_params.radiative_drag_params.set(
            "compton_gamma_rad",
            m_params.template get<real_t>("radiation.drag.compton.gamma_rad"));
        }

        kernel::sr::PusherArrays pusher_arrays {};
        pusher_arrays.sp       = species.index();
        pusher_arrays.i1       = species.i1;
        pusher_arrays.i2       = species.i2;
        pusher_arrays.i3       = species.i3;
        pusher_arrays.i1_prev  = species.i1_prev;
        pusher_arrays.i2_prev  = species.i2_prev;
        pusher_arrays.i3_prev  = species.i3_prev;
        pusher_arrays.dx1      = species.dx1;
        pusher_arrays.dx2      = species.dx2;
        pusher_arrays.dx3      = species.dx3;
        pusher_arrays.dx1_prev = species.dx1_prev;
        pusher_arrays.dx2_prev = species.dx2_prev;
        pusher_arrays.dx3_prev = species.dx3_prev;
        pusher_arrays.ux1      = species.ux1;
        pusher_arrays.ux2      = species.ux2;
        pusher_arrays.ux3      = species.ux3;
        pusher_arrays.phi      = species.phi;
        pusher_arrays.tag      = species.tag;

        // toggle to indicate whether pgen defines the external force
        bool has_extforce = false;
        if constexpr (arch::traits::pgen::HasExtForce<pgen_t>) {
          has_extforce = true;
          // toggle to indicate whether the ext force applies to current species
          if (
            ::traits::has_member<::traits::species_t, decltype(pgen_t::ext_force)>::value) {
            has_extforce &= std::find(m_pgen.ext_force.species.begin(),
                                      m_pgen.ext_force.species.end(),
                                      species.index()) !=
                            m_pgen.ext_force.species.end();
          }
        }

        pusher_params.ext_force = has_extforce;

        if (not has_atmosphere and not has_extforce) {
          Kokkos::parallel_for("ParticlePusher",
                               species.rangeActiveParticles(),
                               kernel::sr::Pusher_kernel<M>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric));
        } else if (has_atmosphere and not has_extforce) {
          const auto force =
            kernel::sr::Force<M::PrtlDim, M::CoordType, kernel::sr::NoForce_t, true> {
              { gx1, gx2, gx3 },
              x_surf,
              ds
          };
          Kokkos::parallel_for(
            "ParticlePusher",
            species.rangeActiveParticles(),
            kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                          pusher_arrays,
                                                          domain.fields.em,
                                                          domain.mesh.metric,
                                                          force));
        } else if (not has_atmosphere and has_extforce) {
          if constexpr (arch::traits::pgen::HasExtForce<pgen_t>) {
            const auto force =
              kernel::sr::Force<M::PrtlDim, M::CoordType, decltype(m_pgen.ext_force), false> {
                m_pgen.ext_force
              };
            Kokkos::parallel_for(
              "ParticlePusher",
              species.rangeActiveParticles(),
              kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric,
                                                            force));
          } else {
            raise::Error("External force not implemented", HERE);
          }
        } else { // has_atmosphere and has_extforce
          if constexpr (arch::traits::pgen::HasExtForce<pgen_t>) {
            const auto force =
              kernel::sr::Force<M::PrtlDim, M::CoordType, decltype(m_pgen.ext_force), true> {
                m_pgen.ext_force,
                { gx1, gx2, gx3 },
                x_surf,
                ds
            };
            Kokkos::parallel_for(
              "ParticlePusher",
              species.rangeActiveParticles(),
              kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric,
                                                            force));
          } else {
            raise::Error("External force not implemented", HERE);
          }
        }
      }
    }

    void ParticleInjector(domain_t& domain, InjTags tags = Inj::None) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (m_metadomain.mesh().prtl_bc_in(direction) == PrtlBC::ATMOSPHERE) {
          AtmosphereParticlesIn(direction, domain, tags);
        }
      }
    }

    template <unsigned short O>
    void deposit_with(const Particles<M::Dim, M::CoordType>& species,
                      const M&                               metric,
                      const scatter_ndfield_t<M::Dim, 3>&    scatter_cur,
                      real_t                                 dt) {
      // clang-format off
      Kokkos::parallel_for("CurrentsDeposit",
                          species.rangeActiveParticles(),
                          kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, O>(
                            scatter_cur,
                            species.i1, species.i2, species.i3,
                            species.i1_prev, species.i2_prev, species.i3_prev,
                            species.dx1, species.dx2, species.dx3,
                            species.dx1_prev, species.dx2_prev, species.dx3_prev,
                            species.ux1, species.ux2, species.ux3,
                            species.phi, species.weight, species.tag,
                            metric, (real_t)(species.charge()), dt));
      // clang-format on
    }

    void CurrentsDeposit(domain_t& domain) {
      auto scatter_cur = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur);
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or
            (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
          continue;
        }
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);

        deposit_with<SHAPE_ORDER>(species, domain.mesh.metric, scatter_cur, dt);
      }
      Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
    }

    void CurrentsAmpere(domain_t& domain) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
      const auto q0 = m_params.template get<real_t>("scales.q0");
      const auto n0 = m_params.template get<real_t>("scales.n0");
      const auto B0 = m_params.template get<real_t>("scales.B0");
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto V0    = m_params.template get<real_t>("scales.V0");
        const auto ppc0  = m_params.template get<real_t>("particles.ppc0");
        const auto coeff = -dt * q0 / (B0 * V0);
        if constexpr (arch::traits::pgen::HasExtCurrent<pgen_t>) {
          const std::vector<real_t> xmin { domain.mesh.extent(in::x1).first,
                                           domain.mesh.extent(in::x2).first,
                                           domain.mesh.extent(in::x3).first };
          const auto                ext_current = m_pgen.ext_current;
          const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});
          // clang-format off
          Kokkos::parallel_for(
            "Ampere",
            domain.mesh.rangeActiveCells(),
            kernel::mink::CurrentsAmpere_kernel<M::Dim, decltype(ext_current)>(
              domain.fields.em, domain.fields.cur,
              coeff, ppc0, ext_current, xmin, dx));
          // clang-format on
        } else {
          Kokkos::parallel_for(
            "Ampere",
            domain.mesh.rangeActiveCells(),
            kernel::mink::CurrentsAmpere_kernel<M::Dim>(domain.fields.em,
                                                        domain.fields.cur,
                                                        coeff,
                                                        ppc0));
        }
      } else {
        // non-minkowski
        const auto coeff = -dt * q0 * n0 / B0;
        auto       range = range_with_axis_BCs(domain);
        const auto ni2   = domain.mesh.n_active(in::x2);
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
      auto       range   = range_with_axis_BCs(domain);
      const auto nfilter = m_params.template get<unsigned short>(
        "algorithms.current_filters");
      tuple_t<ncells_t, M::Dim> size;
      if constexpr (M::Dim == Dim::_1D || M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[0] = domain.mesh.n_active(in::x1);
      }
      if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[1] = domain.mesh.n_active(in::x2);
      }
      if constexpr (M::Dim == Dim::_3D) {
        size[2] = domain.mesh.n_active(in::x3);
      }
      // !TODO: this needs to be done more efficiently
      for (auto i { 0u }; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        m_metadomain.CommunicateFields(domain, Comm::J);
      }
    }

    void FieldBoundaries(domain_t& domain, BCTags tags) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::MATCH) {
          MatchFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::AXIS) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::AXIS) {
            AxisFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::ATMOSPHERE) {
          AtmosphereFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::FIXED) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::FIXED) {
            FixedFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::CONDUCTOR) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CONDUCTOR) {
            PerfectConductorFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::CUSTOM) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CUSTOM) {
            CustomFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::HORIZON) {
          raise::Error("HORIZON BCs only applicable for GR", HERE);
        }
      } // loop over directions
    }

    void MatchFieldsIn(dir::direction_t<M::Dim> direction,
                       domain_t&                domain,
                       BCTags                   tags) {
      /**
       * matching boundaries
       */
      const auto ds_array = m_params.template get<boundaries_t<real_t>>(
        "grid.boundaries.match.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      auto       sign = direction.get_sign();
      real_t     ds;
      if (sign > 0) { // + direction
        ds      = ds_array[(short)dim].second;
        xg_max  = m_metadomain.mesh().extent(dim).second;
        xg_min  = xg_max - ds;
        xg_edge = xg_max;
      } else { // - direction
        ds      = ds_array[(short)dim].first;
        xg_min  = m_metadomain.mesh().extent(dim).first;
        xg_max  = xg_min + ds;
        xg_edge = xg_min;
      }
      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (dim_t d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<dim_t>(dim)) {
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
      tuple_t<ncells_t, M::Dim> range_min { 0 };
      tuple_t<ncells_t, M::Dim> range_max { 0 };

      for (auto d { 0u }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }

      if (dim == in::x1) {
        if constexpr (arch::traits::pgen::HasMatchFields<pgen_t>) {
          auto match_fields = m_pgen.MatchFields(time);
          call_match_fields<decltype(match_fields), in::x1>(domain.fields.em,
                                                            domain.mesh.flds_bc(),
                                                            match_fields,
                                                            domain.mesh.metric,
                                                            xg_edge,
                                                            ds,
                                                            tags,
                                                            range_min,
                                                            range_max);
        } else if constexpr (arch::traits::pgen::HasMatchFieldsInX1<pgen_t>) {
          auto match_fields = m_pgen.MatchFieldsInX1(time);
          call_match_fields<decltype(match_fields), in::x1>(domain.fields.em,
                                                            domain.mesh.flds_bc(),
                                                            match_fields,
                                                            domain.mesh.metric,
                                                            xg_edge,
                                                            ds,
                                                            tags,
                                                            range_min,
                                                            range_max);
        }
      } else if (dim == in::x2) {
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          if constexpr (arch::traits::pgen::HasMatchFields<pgen_t>) {
            auto match_fields = m_pgen.MatchFields(time);
            call_match_fields<decltype(match_fields), in::x2>(domain.fields.em,
                                                              domain.mesh.flds_bc(),
                                                              match_fields,
                                                              domain.mesh.metric,
                                                              xg_edge,
                                                              ds,
                                                              tags,
                                                              range_min,
                                                              range_max);
          } else if constexpr (arch::traits::pgen::HasMatchFieldsInX2<pgen_t>) {
            auto match_fields = m_pgen.MatchFieldsInX2(time);
            call_match_fields<decltype(match_fields), in::x2>(domain.fields.em,
                                                              domain.mesh.flds_bc(),
                                                              match_fields,
                                                              domain.mesh.metric,
                                                              xg_edge,
                                                              ds,
                                                              tags,
                                                              range_min,
                                                              range_max);
          }
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else if (dim == in::x3) {
        if constexpr (M::Dim == Dim::_3D) {
          if constexpr (arch::traits::pgen::HasMatchFields<pgen_t>) {
            auto match_fields = m_pgen.MatchFields(time);
            call_match_fields<decltype(match_fields), in::x3>(domain.fields.em,
                                                              domain.mesh.flds_bc(),
                                                              match_fields,
                                                              domain.mesh.metric,
                                                              xg_edge,
                                                              ds,
                                                              tags,
                                                              range_min,
                                                              range_max);
          } else if constexpr (arch::traits::pgen::HasMatchFieldsInX3<pgen_t>) {
            auto match_fields = m_pgen.MatchFieldsInX3(time);
            call_match_fields<decltype(match_fields), in::x3>(domain.fields.em,
                                                              domain.mesh.flds_bc(),
                                                              match_fields,
                                                              domain.mesh.metric,
                                                              xg_edge,
                                                              ds,
                                                              tags,
                                                              range_min,
                                                              range_max);
          }
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    }

    void AxisFieldsIn(dir::direction_t<M::Dim> direction,
                      domain_t&                domain,
                      BCTags                   tags) {
      /**
       * axis boundaries
       */
      if constexpr (M::CoordType != Coord::Cart) {
        raise::ErrorIf(direction.get_dim() != in::x2,
                       "Invalid axis direction, should be x2",
                       HERE);
        const auto i2_min = domain.mesh.i_min(in::x2);
        const auto i2_max = domain.mesh.i_max(in::x2);
        if (direction.get_sign() < 0) {
          Kokkos::parallel_for(
            "AxisBCFields",
            domain.mesh.n_all(in::x1),
            kernel::bc::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em,
                                                             i2_min,
                                                             tags));
        } else {
          Kokkos::parallel_for(
            "AxisBCFields",
            domain.mesh.n_all(in::x1),
            kernel::bc::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em,
                                                            i2_max,
                                                            tags));
        }
      } else {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Invalid coordinate type for axis BCs", HERE);
      }
    }

    void FixedFieldsIn(dir::direction_t<M::Dim> direction,
                       domain_t&                domain,
                       BCTags                   tags) {
      /**
       * fixed field boundaries
       */
      const auto sign = direction.get_sign();
      const auto dim  = direction.get_dim();
      raise::ErrorIf(dim != in::x1 and M::CoordType != Coord::Cart,
                     "Fixed BCs only implemented for x1 in "
                     "non-cartesian coordinates",
                     HERE);
      em normal_b_comp, tang_e_comp1, tang_e_comp2;
      if (dim == in::x1) {
        normal_b_comp = em::bx1;
        tang_e_comp1  = em::ex2;
        tang_e_comp2  = em::ex3;
      } else if (dim == in::x2) {
        normal_b_comp = em::bx2;
        tang_e_comp1  = em::ex1;
        tang_e_comp2  = em::ex3;
      } else if (dim == in::x3) {
        normal_b_comp = em::bx3;
        tang_e_comp1  = em::ex1;
        tang_e_comp2  = em::ex2;
      } else {
        raise::Error("Invalid dimension", HERE);
      }
      std::vector<ncells_t> xi_min, xi_max;
      const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
      for (dim_t d { 0u }; d < M::Dim; ++d) {
        const auto dd = all_dirs[d];
        if (dim == dd) {
          if (sign > 0) { // + direction
            xi_min.push_back(domain.mesh.n_all(dd) - N_GHOSTS);
            xi_max.push_back(domain.mesh.n_all(dd));
          } else { // - direction
            xi_min.push_back(0);
            xi_max.push_back(N_GHOSTS);
          }
        } else {
          xi_min.push_back(0);
          xi_max.push_back(domain.mesh.n_all(dd));
        }
      }
      raise::ErrorIf(xi_min.size() != xi_max.size() or
                       xi_min.size() != static_cast<std::size_t>(M::Dim),
                     "Invalid range size",
                     HERE);
      std::vector<unsigned short> comps;
      if (tags & BC::E) {
        comps.push_back(tang_e_comp1);
        comps.push_back(tang_e_comp2);
      }
      if (tags & BC::B) {
        comps.push_back(normal_b_comp);
      }
      if constexpr (arch::traits::pgen::HasFixFieldsConst<pgen_t>) {
        for (const auto& comp : comps) {
          auto       value     = ZERO;
          bool       shouldset = false;
          // if fix field function present, read from it
          const auto newset    = m_pgen.FixFieldsConst(
            (bc_in)(sign * ((short)dim + 1)),
            (em)comp);
          value     = newset.first;
          shouldset = newset.second;
          if (shouldset) {
            if constexpr (M::Dim == Dim::_1D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                comp),
                value);
            } else if constexpr (M::Dim == Dim::_2D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                std::make_pair(xi_min[1], xi_max[1]),
                                comp),
                value);
            } else if constexpr (M::Dim == Dim::_3D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                std::make_pair(xi_min[1], xi_max[1]),
                                std::make_pair(xi_min[2], xi_max[2]),
                                comp),
                value);
            } else {
              raise::Error("Invalid dimension", HERE);
            }
          }
        }
      } else {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Fixed fields not present (both const and non-const)", HERE);
      }
    }

    void PerfectConductorFieldsIn(dir::direction_t<M::Dim> direction,
                                  domain_t&                domain,
                                  BCTags                   tags) {
      /**
       * perfect conductor field boundaries
       */
      if constexpr (M::CoordType != Coord::Cart) {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error(
          "Perfect conductor BCs only applicable to cartesian coordinates",
          HERE);
      } else {
        const auto sign = direction.get_sign();
        const auto dim  = direction.get_dim();

        std::vector<std::size_t> xi_min, xi_max;

        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };

        for (auto d { 0u }; d < M::Dim; ++d) {
          const auto dd = all_dirs[d];
          if (dim == dd) {
            xi_min.push_back(0);
            xi_max.push_back((sign < 0) ? (N_GHOSTS + 1) : N_GHOSTS);
          } else {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd));
          }
        }
        raise::ErrorIf(xi_min.size() != xi_max.size() or
                         xi_min.size() != static_cast<std::size_t>(M::Dim),
                       "Invalid range size",
                       HERE);

        range_t<M::Dim> range;
        if constexpr (M::Dim == Dim::_1D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1] },
                                            { xi_max[0], xi_max[1] });
        } else if constexpr (M::Dim == Dim::_3D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1], xi_min[2] },
                                            { xi_max[0], xi_max[1], xi_max[2] });
        } else {
          raise::Error("Invalid dimension", HERE);
        }
        std::size_t i_edge;
        if (sign > 0) {
          i_edge = domain.mesh.i_max(dim);
        } else {
          i_edge = domain.mesh.i_min(dim);
        }

        if (dim == in::x1) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x1, true>(
                domain.fields.em,
                i_edge,
                tags));
          } else {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x1, false>(
                domain.fields.em,
                i_edge,
                tags));
          }
        } else if (dim == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x2, true>(
                  domain.fields.em,
                  i_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x2, false>(
                  domain.fields.em,
                  i_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          if constexpr (M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x3, true>(
                  domain.fields.em,
                  i_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x3, false>(
                  domain.fields.em,
                  i_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        }
      }
    }

    void AtmosphereFieldsIn(dir::direction_t<M::Dim> direction,
                            domain_t&                domain,
                            BCTags                   tags) {
      /**
       * atmosphere field boundaries
       */
      if constexpr (arch::traits::pgen::HasAtmFields<pgen_t>) {
        const auto [sign, dim, xg_min, xg_max] = get_atm_extent(direction);
        const auto           dd                = static_cast<dim_t>(dim);
        boundaries_t<real_t> box;
        boundaries_t<bool>   incl_ghosts;
        for (auto d { 0u }; d < M::Dim; ++d) {
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

        for (auto d { 0u }; d < M::Dim; ++d) {
          range_min[d] = intersect_range[d].first;
          range_max[d] = intersect_range[d].second;
        }
        auto        atm_fields = m_pgen.AtmFields(time);
        std::size_t il_edge;
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
              kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x1>(
                domain.fields.em,
                atm_fields,
                domain.mesh.metric,
                il_edge,
                tags));
          } else {
            Kokkos::parallel_for(
              "AtmosphereBCFields",
              range,
              kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x1>(
                domain.fields.em,
                atm_fields,
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
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x2>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x2>(
                  domain.fields.em,
                  atm_fields,
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
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x3>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x3>(
                  domain.fields.em,
                  atm_fields,
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
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Atm fields not implemented in PGEN for atmosphere BCs", HERE);
      }
    }

    void CustomFieldsIn(dir::direction_t<M::Dim> direction,
                        domain_t&                domain,
                        BCTags                   tags) {
      (void)direction;
      (void)domain;
      (void)tags;
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

    void AtmosphereParticlesIn(const dir::direction_t<M::Dim>& direction,
                               domain_t&                       domain,
                               InjTags                         tags) {
      const auto [sign, dim, xg_min, xg_max] = get_atm_extent(direction);

      const auto x_surf = sign > 0 ? xg_min : xg_max;
      const auto ds     = m_params.template get<real_t>(
        "grid.boundaries.atmosphere.ds");
      const auto temp = m_params.template get<real_t>(
        "grid.boundaries.atmosphere.temperature");
      const auto height = m_params.template get<real_t>(
        "grid.boundaries.atmosphere.height");
      const auto species = m_params.template get<std::pair<spidx_t, spidx_t>>(
        "grid.boundaries.atmosphere.species");
      const auto nmax = m_params.template get<real_t>(
        "grid.boundaries.atmosphere.density");

      Kokkos::deep_copy(domain.fields.bckp, ZERO);
      auto scatter_bckp = Kokkos::Experimental::create_scatter_view(
        domain.fields.bckp);
      const auto use_weights = M::CoordType != Coord::Cart;
      const auto ni2         = domain.mesh.n_active(in::x2);
      const auto inv_n0      = ONE / m_params.template get<real_t>("scales.n0");

      // compute the density of the two species
      if (tags & Inj::AssumeEmpty) {
        if constexpr (M::Dim == Dim::_1D) {
          Kokkos::deep_copy(
            Kokkos::subview(domain.fields.bckp, Kokkos::ALL, std::make_pair(0, 1)),
            ZERO);
        } else if constexpr (M::Dim == Dim::_2D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.bckp,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            std::make_pair(0, 1)),
                            ZERO);
        } else if constexpr (M::Dim == Dim::_3D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.bckp,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            std::make_pair(0, 1)),
                            ZERO);
        }
      } else {
        for (const auto& sp :
             std::vector<spidx_t> { species.first, species.second }) {
          auto& prtl_spec = domain.species[sp - 1];
          if (prtl_spec.npart() == 0) {
            continue;
          }
          // clang-format off
          Kokkos::parallel_for(
            "ComputeMoments",
            prtl_spec.rangeActiveParticles(),
            kernel::ParticleMoments_kernel<SimEngine::SRPIC, M, FldsID::Rho, 6>(
              {}, scatter_bckp, 0,
              prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
              prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
              prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
              prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
              prtl_spec.mass(), prtl_spec.charge(),
              use_weights,
              domain.mesh.metric, domain.mesh.flds_bc(),
              ni2, inv_n0, 0));
          // clang-format on
          prtl_spec.set_unsorted();
        }
        Kokkos::Experimental::contribute(domain.fields.bckp, scatter_bckp);
        m_metadomain.SynchronizeFields(domain, Comm::Bckp, { 0, 1 });
      }

      const auto maxwellian = arch::Maxwellian<S, M> { domain.mesh.metric,
                                                       domain.random_pool,
                                                       temp };

      if (dim == in::x1) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x1> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x1> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else if (dim == in::x2) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x2> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x2> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else if (dim == in::x3) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x3> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x3> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist = arch::Replenish<S, M, 6, decltype(target_density)> {
            domain.mesh.metric,
            domain.fields.bckp,
            0,
            target_density,
            nmax
          };
          arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(spatial_dist)>(
            m_params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
      return;
    }

  private:
    /**
     * @brief Get the buffer region of the atmosphere and the direction
     * @param direction direction in which the atmosphere is applied
     * @return tuple: [sign of the direction, the direction (as in::), the min and max extent
     * @note xg_min and xg_max are the extents where the fields are set, not the atmosphere itself
     * @note i.e.
     *
     *         fields set      particles injected
     * ghost zone  |               |
     *    v        v               v
     * |....|...........|*******************.....  -> x1
     * ^                ^
     * xg_min         xg_max
     * |                |                  |
     * |<--  buffer  -->|<-- atmosphere -->|
     *
     * in this case the function returns { -1, in::x1, xg_min, xg_max }
     */
    auto get_atm_extent(dir::direction_t<M::Dim> direction) const
      -> std::tuple<short, in, real_t, real_t> {
      const auto sign     = direction.get_sign();
      const auto dim      = direction.get_dim();
      const auto min_buff = m_params.template get<unsigned short>(
                              "algorithms.current_filters") +
                            2;
      const auto buffer_ncells = min_buff > 5 ? min_buff : 5;
      if (M::CoordType != Coord::Cart and (dim != in::x1 or sign > 0)) {
        raise::Error("For non-cartesian coordinates atmosphere BCs is "
                     "possible only in -x1 (@ rmin)",
                     HERE);
      }
      real_t   xg_min { ZERO }, xg_max { ZERO };
      ncells_t ig_min, ig_max;
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
      return { sign, dim, xg_min, xg_max };
    }

    auto range_with_axis_BCs(const domain_t& domain) -> range_t<M::Dim> {
      auto range = domain.mesh.rangeActiveCells();
      if constexpr (M::CoordType != Coord::Cart) {
        /**
         * @brief taking one extra cell in the x2 direction if AXIS BCs
         */
        if constexpr (M::Dim == Dim::_2D) {
          if (domain.mesh.flds_bc_in({ 0, +1 }) == FldsBC::AXIS) {
            range = CreateRangePolicy<Dim::_2D>(
              { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
              { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
          }
        } else if constexpr (M::Dim == Dim::_3D) {
          if (domain.mesh.flds_bc_in({ 0, +1, 0 }) == FldsBC::AXIS) {
            range = CreateRangePolicy<Dim::_3D>({ domain.mesh.i_min(in::x1),
                                                  domain.mesh.i_min(in::x2),
                                                  domain.mesh.i_min(in::x3) },
                                                { domain.mesh.i_max(in::x1),
                                                  domain.mesh.i_max(in::x2) + 1,
                                                  domain.mesh.i_max(in::x3) });
          }
        }
      }
      return range;
    }

    template <class T, in O>
    void call_match_fields(ndfield_t<M::Dim, 6>&       fields,
                           const boundaries_t<FldsBC>& boundaries,
                           const T&                    match_fields,
                           const M&                    metric,
                           real_t                      xg_edge,
                           real_t                      ds,
                           BCTags                      tags,
                           tuple_t<ncells_t, M::Dim>&  range_min,
                           tuple_t<ncells_t, M::Dim>&  range_max) {
      Kokkos::parallel_for(
        "MatchFields",
        CreateRangePolicy<M::Dim>(range_min, range_max),
        kernel::bc::MatchBoundaries_kernel<SimEngine::SRPIC, T, M, O>(fields,
                                                                      match_fields,
                                                                      metric,
                                                                      xg_edge,
                                                                      ds,
                                                                      tags,
                                                                      boundaries));
    }
  };

} // namespace ntt

#endif // ENGINES_SRPIC_SRPIC_H
