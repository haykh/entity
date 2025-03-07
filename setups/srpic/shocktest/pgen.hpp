#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <utility>
#include <vector>

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
    Inline auto bx1(const coord_t<D>& x_ph) const -> real_t {
      // return Bmag * math::cos(Btheta);
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>& x_ph) const -> real_t {
      // return Bmag * math::sin(Btheta) * math::sin(Bphi);
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_ph) const -> real_t {
      // return Bmag * math::sin(Btheta) * math::cos(Bphi);
      // return ONE + 0.01 * (ONE - math::tanh(20.*(-1.5 + x_ph[0]))*math::tanh(20.*(-0.5
      // + x_ph[0])))/2.0 * math::sin(4.0 * constant::PI * x_ph[0]);
      return ZERO;
    }

    // electric field components
    Inline auto ex1(const coord_t<D>& x_ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_ph) const -> real_t {
      // return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
      // return ZERO - 0.01 * (ONE - math::tanh(20.*(-1.5 + x_ph[0]))*math::tanh(20.*(-0.5
      // + x_ph[0])))/2.0 * math::sin(4.0 * constant::PI * x_ph[0]);
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_ph) const -> real_t {
      // return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
      return ZERO;
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t drift_ux, temperature;

    const std::vector<real_t> x1arr_e, x2arr_e, ux1arr_e, ux2arr_e, ux3arr_e;
    const std::vector<real_t> x1arr_i, x2arr_i, ux1arr_i, ux2arr_i, ux3arr_i;

    const real_t            Btheta, Bphi, Bmag;
    InitFields<D>           init_flds;
    const Metadomain<S, M>* metadomain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , x1arr_e { p.template get<std::vector<real_t>>("setup.x_e") }
      , x2arr_e { p.template get<std::vector<real_t>>("setup.y_e") }
      , ux1arr_e { p.template get<std::vector<real_t>>("setup.ux_e") }
      , ux2arr_e { p.template get<std::vector<real_t>>("setup.uy_e") }
      , ux3arr_e { p.template get<std::vector<real_t>>("setup.uz_e") }
      , x1arr_i { p.template get<std::vector<real_t>>("setup.x_i") }
      , x2arr_i { p.template get<std::vector<real_t>>("setup.y_i") }
      , ux1arr_i { p.template get<std::vector<real_t>>("setup.ux_i") }
      , ux2arr_i { p.template get<std::vector<real_t>>("setup.uy_i") }
      , ux3arr_i { p.template get<std::vector<real_t>>("setup.uz_i") }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , metadomain { &m } {}

    inline PGen() {}

    auto FixFieldsConst(const bc_in&, const em& comp) const
      -> std::pair<real_t, bool> {
      if (comp == em::ex2) {
        return { init_flds.ex2({ ZERO }), true };
      } else if (comp == em::ex3) {
        return { init_flds.ex3({ ZERO }), true };
      } else {
        return { ZERO, false };
      }
    }

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {
      arch::InjectGlobally(*metadomain,
                           domain,
                           1,
                           {
                             {  "x1",  x1arr_e },
                             {  "x2",  x2arr_e },
                             { "ux1", ux1arr_e },
                             { "ux2", ux1arr_e },
                             { "ux3", ux3arr_e }
      });
      arch::InjectGlobally(*metadomain,
                           domain,
                           2,
                           {
                             {  "x1",  x1arr_i },
                             {  "x2",  x2arr_i },
                             { "ux1", ux1arr_i },
                             { "ux2", ux1arr_i },
                             { "ux3", ux3arr_i }
      });
    }
  };

} // namespace user

#endif
