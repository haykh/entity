#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
#include <utility>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : Bmag { bmag }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { drift_ux } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return Bmag * math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  drift_ux, temperature, temperature_ratio, filling_fraction;
    // injector properties
    const real_t  injector_velocity, injection_start, dt;
    const int     injection_frequency;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , filling_fraction { p.template get<real_t>("setup.filling_fraction", 1.0) }
      , injector_velocity { p.template get<real_t>("setup.injector_velocity", 1.0) }
      , injection_start { p.template get<real_t>("setup.injection_start", 0.0) }
      , injection_frequency { p.template get<int>("setup.injection_frequency", 100) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") } {}

    inline PGen() {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&, const em& comp) const
      -> std::pair<real_t, bool> {
      if (comp == em::ex1) {
        return { init_flds.ex1({ ZERO }), true };
      } else if (comp == em::ex2) {
        return { ZERO, true };
      } else if (comp == em::ex3) {
        return { ZERO, true };
      } else if (comp == em::bx1) {
        return { init_flds.bx1({ ZERO }), true };
      } else if (comp == em::bx2) {
        return { init_flds.bx2({ ZERO }), true };
      } else if (comp == em::bx3) {
        return { init_flds.bx3({ ZERO }), true };
      } else {
        raise::Error("Invalid component", HERE);
        return { ZERO, false };
      }
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      /*
       *  Plasma setup as partially filled box
       *
       *  Plasma setup:
       *
       * global_xmin                            global_xmax
       * |                                      |
       * V                                      V
       * |:::::::::::|..........................|
       *             ^
       *             |
       *        filling_fraction
       */

      // minimum and maximum position of particles
      real_t xg_min = global_xmin;
      real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.push_back({ xg_min, xg_max });
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           box);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /*
       *  Replenish plasma in a moving injector
       *
       *  Injector setup:
       *
       * global_xmin           purge/replenish  global_xmax
       * |         x_init            |          |
       * V           v               V          V
       * |:::::::::::;::::::::::|\\\\\\\\|......|
       *                       xmin    xmax
       *                                 ^
       *                                 |
       *                           moving injector
       */

      // check if the injector should be active
      if (step % injection_frequency != 0) {
        return;
      }

      // initial position of injector
      const auto x_init = global_xmin +
                          filling_fraction * (global_xmax - global_xmin);

      // compute the position of the injector after the current timestep
      auto xmax = x_init + injector_velocity *
                             (std::max<real_t>(time - injection_start, ZERO) + dt);
      if (xmax >= global_xmax) {
        xmax = global_xmax;
      }

      // compute the beginning of the injected region
      auto xmin = xmax - injection_frequency * dt;
      if (xmin <= global_xmin) {
        xmin = global_xmin;
      }

      // define indice range to reset fields
      boundaries_t<bool> incl_ghosts;
      for (auto d = 0; d < M::Dim; ++d) {
        incl_ghosts.push_back({ false, false });
      }

      // define box to reset fields
      boundaries_t<real_t> purge_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          purge_box.push_back({ xmin, global_xmax });
        } else {
          purge_box.push_back(Range::All);
        }
      }

      const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
      for (auto d = 0; d < M::Dim; ++d) {
        x_min[d] = extent[d].first;
        x_max[d] = extent[d].second;
      }

      Kokkos::parallel_for("ResetFields",
                           CreateRangePolicy<M::Dim>(x_min, x_max),
                           arch::SetEMFields_kernel<decltype(init_flds), S, M> {
                             domain.fields.em,
                             init_flds,
                             domain.mesh.metric });
      global_domain.CommunicateFields(domain, Comm::E | Comm::B);

      /*
        tag particles inside the injection zone as dead
      */
      const auto& mesh = domain.mesh;

      // loop over particle species
      for (auto s { 0u }; s < 2; ++s) {
        // get particle properties
        auto& species = domain.species[s];
        auto  i1      = species.i1;
        auto  dx1     = species.dx1;
        auto  tag     = species.tag;

        Kokkos::parallel_for(
          "RemoveParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            // check if the particle is already dead
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x_Cd = static_cast<real_t>(i1(p)) +
                              static_cast<real_t>(dx1(p));
            const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
              x_Cd);

            if (x_Ph > xmin) {
              tag(p) = ParticleTag::dead;
            }
          });
      }

      /*
          Inject slab of fresh plasma
      */

      // define box to inject into
      boundaries_t<real_t> inj_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          inj_box.push_back({ xmin, xmax });
        } else {
          inj_box.push_back(Range::All);
        }
      }

      // same maxwell distribution as above
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           inj_box);
    }
  };
} // namespace user
#endif
