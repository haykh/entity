#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <fstream>
#include <iostream>

enum {
  REAL = 0,
  IMAG = 1
};

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct ExtForce {
    ExtForce(array_t<real_t* [2]> amplitudes, real_t SX1, real_t SX2, real_t SX3)
      : amps { amplitudes }
      , sx1 { SX1 }
      , sx2 { SX2 }
      , sx3 { SX3 } {}

    const std::vector<unsigned short> species { 1, 2 };

    ExtForce() = default;

    Inline auto fx1(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t k01 = ONE * constant::TWO_PI / sx1;
      real_t k02 = ZERO * constant::TWO_PI / sx2;
      real_t k03 = ZERO * constant::TWO_PI / sx3;
      real_t k04 = ONE;
      real_t k11 = ZERO * constant::TWO_PI / sx1;
      real_t k12 = ONE * constant::TWO_PI / sx2;
      real_t k13 = ZERO * constant::TWO_PI / sx3;
      real_t k14 = ONE;
      real_t k21 = ZERO * constant::TWO_PI / sx1;
      real_t k22 = ZERO * constant::TWO_PI / sx2;
      real_t k23 = ONE * constant::TWO_PI / sx3;
      real_t k24 = ONE;

      // return 0.1 * cos(2.0 * constant::TWO_PI * x_Ph[1]);

      return (k14 * amps(0, REAL) *
                math::cos(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2]) +
              k14 * amps(0, IMAG) *
                math::sin(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2])) +
             (k24 * amps(1, REAL) *
                math::cos(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]) +
              k24 * amps(1, IMAG) *
                math::sin(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]));
    }

    Inline auto fx2(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t k01 = ONE * constant::TWO_PI / sx1;
      real_t k02 = ZERO * constant::TWO_PI / sx2;
      real_t k03 = ZERO * constant::TWO_PI / sx3;
      real_t k04 = ONE;
      real_t k11 = ZERO * constant::TWO_PI / sx1;
      real_t k12 = ONE * constant::TWO_PI / sx2;
      real_t k13 = ZERO * constant::TWO_PI / sx3;
      real_t k14 = ONE;
      real_t k21 = ZERO * constant::TWO_PI / sx1;
      real_t k22 = ZERO * constant::TWO_PI / sx2;
      real_t k23 = ONE * constant::TWO_PI / sx3;
      real_t k24 = ONE;
      return (k04 * amps(2, REAL) *
                math::cos(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * x_Ph[2]) +
              k04 * amps(2, IMAG) *
                math::sin(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * x_Ph[2])) +
             (k24 * amps(3, REAL) *
                math::cos(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]) +
              k24 * amps(3, IMAG) *
                math::sin(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]));
      // return ZERO;
    }

    Inline auto fx3(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t k01 = ONE * constant::TWO_PI / sx1;
      real_t k02 = ZERO * constant::TWO_PI / sx2;
      real_t k03 = ZERO * constant::TWO_PI / sx3;
      real_t k04 = ONE;
      real_t k11 = ZERO * constant::TWO_PI / sx1;
      real_t k12 = ONE * constant::TWO_PI / sx2;
      real_t k13 = ZERO * constant::TWO_PI / sx3;
      real_t k14 = ONE;
      real_t k21 = ZERO * constant::TWO_PI / sx1;
      real_t k22 = ZERO * constant::TWO_PI / sx2;
      real_t k23 = ONE * constant::TWO_PI / sx3;
      real_t k24 = ONE;
      return (k04 * amps(4, REAL) *
                math::cos(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * x_Ph[2]) +
              k04 * amps(4, IMAG) *
                math::sin(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * x_Ph[2])) +
             (k14 * amps(5, REAL) *
                math::cos(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2]) +
              k14 * amps(5, IMAG) *
                math::sin(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2]));
      // return ZERO;
    }

  private:
    array_t<real_t* [2]> amps;
    const real_t         sx1, sx2, sx3;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t         SX1, SX2, SX3;
    const real_t         temperature, machno;
    const unsigned int   nmodes;
    const real_t         amp0, phi0;
    array_t<real_t* [2]> amplitudes;
    ExtForce<M::PrtlDim> ext_force;
    const real_t         dt;

    inline PGen(const SimulationParams& params, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { params }
      , SX1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , SX2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , SX3 { global_domain.mesh().extent(in::x3).second -
              global_domain.mesh().extent(in::x3).first }
      // , SX1 { 2.0 }
      // , SX2 { 2.0 }
      // , SX3 { 2.0 }
      , temperature { params.template get<real_t>("problem.temperature", 0.1) }
      , machno { params.template get<real_t>("problem.machno", 0.1) }
      , nmodes { params.template get<unsigned int>("setup.nmodes", 6) }
      , amp0 { machno * temperature / static_cast<real_t>(nmodes) }
      , phi0 { INV_4 } // !TODO: randomize
      , amplitudes { "DrivingModes", nmodes }
      , ext_force { amplitudes, SX1, SX2, SX3 }
      , dt { params.template get<real_t>("algorithms.timestep.dt") } {
      Init();
    }

    void Init() {
      // initializing amplitudes
      auto       amplitudes_ = amplitudes;
      const auto amp0_       = amp0;
      const auto phi0_       = phi0;
      Kokkos::parallel_for(
        "RandomAmplitudes",
        amplitudes.extent(0),
        Lambda(index_t i) {
          amplitudes_(i, REAL) = amp0_ * math::cos(phi0_);
          amplitudes_(i, IMAG) = amp0_ * math::sin(phi0_);
        });
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      {
        const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temperature);
        const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
          energy_dist,
          { 1, 2 });
        const real_t ndens = 0.9;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }

      {
        const auto energy_dist = arch::PowerlawDist<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        0.1, 100.0, -3.0);
        const auto injector = arch::UniformInjector<S, M, arch::PowerlawDist>(
          energy_dist,
          { 1, 2 });
        const real_t ndens = 0.1;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }
    }

    void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {
      auto omega0 = 0.6 * math::sqrt(temperature * machno * constant::TWO_PI / SX1);
      auto gamma0 = 0.5 * math::sqrt(temperature * machno * constant::TWO_PI / SX2);
      auto sigma0 = amp0 * math::sqrt(static_cast<real_t>(nmodes) * gamma0);
      auto pool   = domain.random_pool;
      Kokkos::parallel_for(
        "RandomAmplitudes",
        amplitudes.extent(0),
        ClassLambda(index_t i) {
          auto       rand_gen = pool.get_state();
          const auto unr      = Random<real_t>(rand_gen) - HALF;
          const auto uni      = Random<real_t>(rand_gen) - HALF;
          pool.free_state(rand_gen);
          const auto ampr_prev = amplitudes(i, REAL);
          const auto ampi_prev = amplitudes(i, IMAG);
          amplitudes(i, REAL)  = (ampr_prev * math::cos(omega0 * dt) +
                                 ampi_prev * math::sin(omega0 * dt)) *
                                  math::exp(-gamma0 * dt) +
                                unr * sigma0;
          amplitudes(i, IMAG) = (-ampr_prev * math::sin(omega0 * dt) +
                                 ampi_prev * math::cos(omega0 * dt)) *
                                  math::exp(-gamma0 * dt) +
                                uni * sigma0;
        });

      // auto fext_en_total = ZERO;
      // for (auto& species : domain.species) {
      //   auto pld    = species.pld[0];
      //   auto weight = species.weight;
      //   Kokkos::parallel_reduce(
      //     "ExtForceEnrg",
      //     species.rangeActiveParticles(),
      //     ClassLambda(index_t p, real_t & fext_en) {
      //       fext_en += pld(p) * weight(p);
      //     },
      //     fext_en_total);
      // }

      // auto pkin_en_total = ZERO;
      // for (auto& species : domain.species) {
      //   auto ux1    = species.ux1;
      //   auto ux2    = species.ux2;
      //   auto ux3    = species.ux3;
      //   auto weight = species.weight;
      //   Kokkos::parallel_reduce(
      //     "KinEnrg",
      //     species.rangeActiveParticles(),
      //     ClassLambda(index_t p, real_t & pkin_en) {
      //       pkin_en += (math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p))) -
      //                   ONE) *
      //                  weight(p);
      //     },
      //     pkin_en_total);
      // }
      // // Weight the macroparticle integral by sim parameters
      // pkin_en_total /= params.template get<real_t>("scales.n0");

      // std::ofstream myfile;
      // if (time == 0) {
      //   myfile.open("fextenrg.txt");
      // } else {
      //   myfile.open("fextenrg.txt", std::ios_base::app);
      // }
      // myfile << fext_en_total << std::endl;
      // myfile.close();

      // if (time == 0) {
      //   myfile.open("kenrg.txt");
      // } else {
      //   myfile.open("kenrg.txt", std::ios_base::app);
      // }
      // myfile << pkin_en_total << std::endl;
      // myfile.close();

      // if constexpr (D == Dim::_3D) {
        
      //   auto metric = domain.mesh.metric;
        
      //   auto benrg_total = ZERO;
      //   auto EB          = domain.fields.em;
      //   Kokkos::parallel_reduce(
      //     "BEnrg",
      //     domain.mesh.rangeActiveCells(),
      //     Lambda(index_t i1, index_t i2, index_t i3, real_t & benrg) {
      //       coord_t<Dim::_3D> x_Cd { ZERO };
      //       vec_t<Dim::_3D>   b_Cntrv { EB(i1, i2, i3, em::bx1),
      //                                 EB(i1, i2, i3, em::bx2),
      //                                 EB(i1, i2, i3, em::bx3) };
      //       vec_t<Dim::_3D>   b_XYZ;
      //       metric.template transform<Idx::U, Idx::T>(x_Cd,
      //                                                             b_Cntrv,
      //                                                             b_XYZ);
      //       benrg += (SQR(b_XYZ[0]) + SQR(b_XYZ[1]) + SQR(b_XYZ[2]));
      //     },
      //     benrg_total);
      //   benrg_total *= params.template get<real_t>("scales.sigma0") * HALF;

      //   if (time == 0) {
      //     myfile.open("bsqenrg.txt");
      //   } else {
      //     myfile.open("bsqenrg.txt", std::ios_base::app);
      //   }
      //   myfile << benrg_total << std::endl;
      //   myfile.close();
      //   auto eenrg_total = ZERO;
      //   Kokkos::parallel_reduce(
      //     "BEnrg",
      //     domain.mesh.rangeActiveCells(),
      //     Lambda(index_t i1, index_t i2, index_t i3, real_t & eenrg) {
      //       coord_t<Dim::_3D> x_Cd { ZERO };
      //       vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, i3, em::ex1),
      //                                 EB(i1, i2, i3, em::ex2),
      //                                 EB(i1, i2, i3, em::ex3) };
      //       vec_t<Dim::_3D>   e_XYZ;
      //       metric.template transform<Idx::U, Idx::T>(x_Cd,
      //                                                             e_Cntrv,
      //                                                             e_XYZ);            
      //       eenrg += (SQR(e_XYZ[0]) + SQR(e_XYZ[1]) + SQR(e_XYZ[2]));
      //     },
      //     eenrg_total);
      //   eenrg_total *= params.template get<real_t>("scales.sigma0") * HALF;
  

      //   if (time == 0) {
      //     myfile.open("esqenrg.txt");
      //   } else {
      //     myfile.open("esqenrg.txt", std::ios_base::app);
      //   }
      //   myfile << eenrg_total << std::endl;
      //   myfile.close();
      // }
    }
  };

} // namespace user

#endif