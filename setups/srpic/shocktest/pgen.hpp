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
      // return ONE + 0.01 * (ONE - math::tanh(20.*(-1.5 + x_ph[0]))*math::tanh(20.*(-0.5 + x_ph[0])))/2.0 * math::sin(4.0 * constant::PI * x_ph[0]);
      return ZERO;
    }

    // electric field components
    Inline auto ex1(const coord_t<D>& x_ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_ph) const -> real_t {
      // return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
      // return ZERO - 0.01 * (ONE - math::tanh(20.*(-1.5 + x_ph[0]))*math::tanh(20.*(-0.5 + x_ph[0])))/2.0 * math::sin(4.0 * constant::PI * x_ph[0]);
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

    const real_t  Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux } {}

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

        auto& species_e = domain.species[0];
        auto& species_p = domain.species[1];

        array_t<std::size_t> elec_ind("elec_ind");
        array_t<std::size_t> pos_ind("pos_ind");
          
        auto offset_e = species_e.npart();
        auto offset_p = species_p.npart();

        auto ux1_e    = species_e.ux1;
        auto ux2_e    = species_e.ux2;
        auto ux3_e    = species_e.ux3;
        auto i1_e     = species_e.i1;
        auto i2_e     = species_e.i2;
        auto dx1_e    = species_e.dx1;
        auto dx2_e    = species_e.dx2;
        auto phi_e    = species_e.phi;
        auto weight_e = species_e.weight;
        auto tag_e    = species_e.tag;

        auto ux1_p    = species_p.ux1;
        auto ux2_p    = species_p.ux2;
        auto ux3_p    = species_p.ux3;
        auto i1_p     = species_p.i1;
        auto i2_p     = species_p.i2;
        auto dx1_p    = species_p.dx1;
        auto dx2_p    = species_p.dx2;
        auto phi_p    = species_p.phi;
        auto weight_p = species_p.weight;
        auto tag_p    = species_p.tag;

        int nseed = 1;

        Kokkos::parallel_for("init_particles", nseed, KOKKOS_LAMBDA(const int& s) {

          // ToDo: fix this
          auto i1_ = math::floor(10);
          auto i2_ = math::floor(64);
          auto dx1_ = HALF;
          auto dx2_ = HALF;


              auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
              auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);

              i1_e(elec_p + offset_e) = i1_;
              dx1_e(elec_p + offset_e) = dx1_;
              i2_e(elec_p + offset_e) = i2_;
              dx2_e(elec_p + offset_e) = dx2_;
              ux1_e(elec_p + offset_e) = -0.5;
              ux2_e(elec_p + offset_e) = 0.5;
              ux3_e(elec_p + offset_e) = ZERO;
              weight_e(elec_p + offset_e) = ONE;
              tag_e(elec_p + offset_e) = ParticleTag::alive;

              i1_p(pos_p + offset_p) = i1_;
              dx1_p(pos_p + offset_p) = dx1_;
              i2_p(pos_p + offset_p) = i2_;
              dx2_p(pos_p + offset_p) = dx2_;
              ux1_p(pos_p + offset_p) = 0.5;
              ux2_p(pos_p + offset_p) = -0.5;
              ux3_p(pos_p + offset_p) = ZERO;
              weight_p(pos_p + offset_p) = ONE;
              tag_p(pos_p + offset_p) = ParticleTag::alive;


          });


            auto elec_ind_h = Kokkos::create_mirror(elec_ind);
            Kokkos::deep_copy(elec_ind_h, elec_ind);
            species_e.set_npart(offset_e + elec_ind_h());

            auto pos_ind_h = Kokkos::create_mirror(pos_ind);
            Kokkos::deep_copy(pos_ind_h, pos_ind);
            species_p.set_npart(offset_p + pos_ind_h());


    }


    // inline void InitPrtls(Domain<S, M>& local_domain) {
    //   const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
    //                                                   local_domain.random_pool,
    //                                                   temperature,
    //                                                   -drift_ux,
    //                                                   in::x1);

    //   const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
    //     energy_dist,
    //     { 1, 2 });
    //   arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
    //     params,
    //     local_domain,
    //     injector,
    //     1.0);
    // }

  };

} // namespace user

#endif
