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
  struct InitFields {
    InitFields(real_t Bnorm)
      : Bnorm { Bnorm } {
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return Bnorm;
    }

  private:
    const real_t Bnorm;
  };

  template <SimEngine::type S, class M>
  struct PowerlawDist : public arch::EnergyDistribution<S, M> {
    PowerlawDist(const M&               metric,
             random_number_pool_t&      pool,
             real_t                     g_min,
             real_t                     g_max,
             real_t                     pl_ind)
      : arch::EnergyDistribution<S, M> { metric }
      , g_min { g_min }
      , g_max { g_max }
      , random_pool { pool }
      , pl_ind { pl_ind } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      // if (sp == 1) {
         auto   rand_gen = random_pool.get_state();
         auto   rand_X1 = Random<real_t>(rand_gen);
         auto   rand_gam = ONE;
         if (pl_ind != -1.0) {
            rand_gam += math::pow(math::pow(g_min,ONE + pl_ind) + (-math::pow(g_min,ONE + pl_ind) + math::pow(g_max,ONE + pl_ind))*rand_X1,ONE/(ONE + pl_ind));
         } else {
            rand_gam += math::pow(g_min,ONE - rand_X1)*math::pow(g_max,rand_X1);
         }
         auto   rand_u = math::sqrt( SQR(rand_gam) - ONE );

        if constexpr (M::Dim == Dim::_1D) {
          v[0] = ZERO;
        } else if constexpr (M::Dim == Dim::_2D) {
          v[0] = ZERO;
          v[1] = ZERO;
        } else {
          auto rand_X2 = Random<real_t>(rand_gen);
          auto rand_X3 = Random<real_t>(rand_gen);
          v[0]   = rand_u * (TWO * rand_X2 - ONE);
          v[2]   = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
          v[1]   = v[2] * math::cos(constant::TWO_PI * rand_X3);
          v[2]   = v[2] * math::sin(constant::TWO_PI * rand_X3);
        }
        random_pool.free_state(rand_gen);
      // } else {
      //   v[0] = ZERO;
      //   v[1] = ZERO;
      //   v[2] = ZERO;
      // }
    }

  private:
    const real_t g_min, g_max, pl_ind;
    random_number_pool_t random_pool;
  };

  template <Dimension D>
  struct ExtForce {
    ExtForce(array_t<real_t* [2]> amplitudes, real_t SX1, real_t SX2, real_t SX3)
      : amps { amplitudes }
      , sx1 { SX1 }
      , sx2 { SX2 }
      , sx3 { SX3 } 
      , k01 {ONE * constant::TWO_PI / sx1}
      , k02 {ZERO * constant::TWO_PI / sx2}
      , k03 {ZERO * constant::TWO_PI / sx3}
      , k04 {ONE}
      , k11 {ZERO * constant::TWO_PI / sx1}
      , k12 {ONE * constant::TWO_PI / sx2}
      , k13 {ZERO * constant::TWO_PI / sx3}
      , k14 {ONE}
      , k21 {ZERO * constant::TWO_PI / sx1}
      , k22 {ZERO * constant::TWO_PI / sx2}
      , k23 {ONE * constant::TWO_PI / sx3}
      , k24 {ONE} {}

    const std::vector<unsigned short> species { 1, 2 };

    ExtForce() = default;

    Inline auto fx1(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {

      return ZERO;
      // return (k14 * amps(0, REAL) *
      //           math::cos(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * 0.0) +
      //         k14 * amps(0, IMAG) *
      //           math::sin(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * 0.0)) ;

      // return ONE * math::cos(ONE * constant::TWO_PI * x_Ph[1]);

    }

    Inline auto fx2(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {

      // return (k04 * amps(1, REAL) *
      //           math::cos(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * 0.0) +
      //         k04 * amps(1, IMAG) *
      //           math::sin(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * 0.0)) ;
      return ZERO;
    }

    Inline auto fx3(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {

      // return (k04 * amps(4, REAL) *
      //           math::cos(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * 0.0) +
      //         k04 * amps(4, IMAG) *
      //           math::sin(k01 * x_Ph[0] + k02 * x_Ph[1] + k03 * 0.0)) +
      //        (k14 * amps(5, REAL) *
      //           math::cos(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * 0.0) +
      //         k14 * amps(5, IMAG) *
      //           math::sin(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * 0.0));
      return ZERO;
    }

  private:
    array_t<real_t* [2]> amps;
    const real_t         sx1, sx2, sx3;
    const real_t         k01, k02, k03, k04, k11, k12, k13, k14, k21, k22, k23, k24;
  };

    template <Dimension D>
  struct ExtCurrent {
    ExtCurrent(array_t<real_t* [2]> amplitudes, real_t SX1, real_t SX2, real_t SX3, array_t<real_t*> damp0)
      : amps { amplitudes }
      , sx1 { SX1 }
      , sx2 { SX2 }
      , sx3 { SX3 }
      , damp { damp0 }
      , k11 {ZERO * constant::TWO_PI / sx1}
      , k12 { ONE * constant::TWO_PI / sx2}
      , k21 {ZERO * constant::TWO_PI / sx1}
      , k22 {-ONE * constant::TWO_PI / sx2}
      , k31 { ONE * constant::TWO_PI / sx1}
      , k32 {ZERO * constant::TWO_PI / sx2}
      , k41 {-ONE * constant::TWO_PI / sx1}
      , k42 {ZERO * constant::TWO_PI / sx2} 
      , k51 {ZERO * constant::TWO_PI / sx1}
      , k52 { ONE * constant::TWO_PI / sx2}
      , k61 {ZERO * constant::TWO_PI / sx1}
      , k62 {-ONE * constant::TWO_PI / sx2}
      , k71 { ONE * constant::TWO_PI / sx1}
      , k72 {ZERO * constant::TWO_PI / sx2}
      , k81 {-ONE * constant::TWO_PI / sx1}
      , k82 {ZERO * constant::TWO_PI / sx2} {}

    ExtCurrent() = default;

    Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {

      return ZERO;
      // return damp(4) * amps(4, REAL) *
      //          math::cos(k51 * x_Ph[0] + k52 * x_Ph[1]) -
      //        damp(4) * amps(4, IMAG) *
      //          math::sin(k51 * x_Ph[0] + k52 * x_Ph[1]) +
      //        damp(5) * amps(5, REAL) *
      //          math::cos(k61 * x_Ph[0] + k62 * x_Ph[1]) -
      //        damp(5) * amps(5, IMAG) *
      //          math::sin(k61 * x_Ph[0] + k62 * x_Ph[1]);
    }

    Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {

      return ZERO;
      // return damp(6) * amps(6, REAL) *
      //          math::cos(k71 * x_Ph[0] + k72 * x_Ph[1]) -
      //        damp(6) * amps(6, IMAG) *
      //          math::sin(k71 * x_Ph[0] + k72 * x_Ph[1]) +
      //        damp(7) * amps(7, REAL) *
      //          math::cos(k81 * x_Ph[0] + k82 * x_Ph[1]) -
      //        damp(7) * amps(7, IMAG) *
      //          math::sin(k81 * x_Ph[0] + k82 * x_Ph[1]);
    }

    Inline auto jx3(const coord_t<D>& x_Ph) const -> real_t {

      // return ZERO;
      return damp(0) * amps(0, REAL) *
               math::cos(k11 * x_Ph[0] + k12 * x_Ph[1]) -
             damp(0) * amps(0, IMAG) *
               math::sin(k11 * x_Ph[0] + k12 * x_Ph[1]) +
             damp(1) * amps(1, REAL) *
               math::cos(k21 * x_Ph[0] + k22 * x_Ph[1]) -
             damp(1) * amps(1, IMAG) *
               math::sin(k21 * x_Ph[0] + k22 * x_Ph[1]) +
             damp(2) * amps(2, REAL) *
                math::cos(k31 * x_Ph[0] + k32 * x_Ph[1]) -
              damp(2) * amps(2, IMAG) *
                math::sin(k31 * x_Ph[0] + k32 * x_Ph[1]) +
              damp(3) * amps(3, REAL) *
                math::cos(k41 * x_Ph[0] + k42 * x_Ph[1]) -
              damp(3) * amps(3, IMAG) *
                math::sin(k41 * x_Ph[0] + k42 * x_Ph[1]);
    }

  private:
    array_t<real_t* [2]> amps;
    array_t<real_t*> damp;
    const real_t         sx1, sx2, sx3;
    const real_t         k11, k12, k21, k22, k31, k32, k41, k42;
    const real_t         k51, k52, k61, k62, k71, k72, k81, k82;
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
    const real_t         temperature, machno, Bnorm;
    const unsigned int   nmodes;
    const real_t         amp0;
    const real_t        pl_gamma_min, pl_gamma_max, pl_index;
    array_t<real_t* [2]> amplitudes;
    array_t<real_t*> phi0, rands, damp0;
    ExtForce<M::PrtlDim> ext_force;
    ExtCurrent<M::PrtlDim>   ext_current;
    const real_t         dt;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& params, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { params }
      , SX1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , SX2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      // , SX3 { global_domain.mesh().extent(in::x3).second -
      //         global_domain.mesh().extent(in::x3).first }
      , SX3 { TWO }
      , temperature { params.template get<real_t>("setup.temperature", 0.16) }
      , machno { params.template get<real_t>("setup.machno", 1.0) }
      , nmodes { params.template get<unsigned int>("setup.nmodes", 8) }
      , Bnorm { params.template get<real_t>("setup.Bnorm", 0.0) }
      , pl_gamma_min { params.template get<real_t>("setup.pl_gamma_min", 0.1) }
      , pl_gamma_max { params.template get<real_t>("setup.pl_gamma_max", 100.0) }
      , pl_index { params.template get<real_t>("setup.pl_index", -2.0) } 
      , dt { params.template get<real_t>("algorithms.timestep.dt") } 
      , amp0 { machno * temperature / static_cast<real_t>(nmodes) * 0.1 }
      , damp0 { "Damping", nmodes }  
      , phi0 { "DrivingPhases", nmodes }
      , amplitudes { "DrivingModes", nmodes }
      , rands { "RandomNumbers", 2*nmodes }
      , ext_force { amplitudes, SX1, SX2, SX3 }
      , ext_current { amplitudes, SX1, SX2, SX3, damp0}
      , init_flds { Bnorm } {
      // Initializing random phases
      auto phi0_ = Kokkos::create_mirror_view(phi0);
      auto damp0_ = Kokkos::create_mirror_view(damp0);
      // srand (static_cast <unsigned> (12345));
      for (int i = 0; i < nmodes; ++i) {
        phi0_(i) = constant::TWO_PI * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
        damp0_(i) = ZERO;
      }
      Kokkos::deep_copy(phi0, phi0_);
      Kokkos::deep_copy(damp0, damp0_);

      #if defined(MPI_ENABLED)
        int              rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast(phi0.data(), phi0.extent(0), mpi::get_type<real_t>(), 0, MPI_COMM_WORLD);
      #endif

      // Initializing amplitudes
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
          amplitudes_(i, REAL) = amp0_ * math::cos(phi0_(i));
          amplitudes_(i, IMAG) = amp0_ * math::sin(phi0_(i));
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
        const real_t ndens = 1.0;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }

      {
        // const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
        //                                                 local_domain.random_pool,
        //                                                 temperature*100);        
        // const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
        //                                                 local_domain.random_pool,
        //                                                 temperature * 2,
        //                                                 10.0,
        //                                                 1);
        // const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
        //   energy_dist,
        //   { 1, 2 });

        const auto energy_dist = PowerlawDist<S, M>(local_domain.mesh.metric,
                                                     local_domain.random_pool,
                                                     pl_gamma_min,
                                                     pl_gamma_max,
                                                     pl_index);  

        const auto injector = arch::UniformInjector<S, M, PowerlawDist>(
          energy_dist,
          { 1, 2 });  


        const real_t ndens = 0.0;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }
    }

    void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {
      // auto omega0 = 0.5*0.6 * math::sqrt(temperature * machno) * constant::TWO_PI / SX1;
      // auto gamma0 = 0.5*0.5 * math::sqrt(temperature * machno) * constant::TWO_PI / SX2;
      const auto mag0 = params.template get<real_t>("scales.sigma0");
      const auto vA0 = math::sqrt(mag0/(mag0 + 1.3333333333333333));
      const auto omega0 = 0.5 * 0.6 * vA0 * constant::TWO_PI / this->SX1;
      const auto gamma0 = 0.5 * 0.5 * vA0 * constant::TWO_PI / this->SX1;
      const auto sigma0 = this->amp0 * math::sqrt(static_cast<real_t>(this->nmodes) * gamma0 / this->dt);
      const auto pool   = domain.random_pool;
      const auto dt_    = this->dt;

      #if defined(MPI_ENABLED)
        int              rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      #endif

      Kokkos::parallel_for(
        "RandomNumbers",
        rands.extent(0),
        ClassLambda(index_t i) {
          auto       rand_gen = pool.get_state();
          rands(i) = Random<real_t>(rand_gen);
          pool.free_state(rand_gen);
        });

      #if defined(MPI_ENABLED)
        MPI_Bcast(rands.data(), rands.extent(0), mpi::get_type<real_t>(), 0, MPI_COMM_WORLD);
      #endif

      // auto rand_m = Kokkos::create_mirror_view(rands);
      // Kokkos::deep_copy(rand_m, rands);
      // printf("rands(0) = %f\n", rand_m(0));

      Kokkos::parallel_for(
        "RandomAmplitudes",
        amplitudes.extent(0),
        ClassLambda(index_t i) {
          const auto unr      = rands(i) - HALF;
          const auto uni      = rands(amplitudes.extent(0) + i) - HALF;
          const auto ampr_prev = amplitudes(i, REAL);
          const auto ampi_prev = amplitudes(i, IMAG);
          auto omega0in = omega0;

          amplitudes(i, REAL)  = (ampr_prev * math::cos(omega0in * dt_) +
                                 ampi_prev * math::sin(omega0in * dt_)) *
                                  math::exp(-gamma0 * dt_) +
                                unr * sigma0 * dt_;
          amplitudes(i, IMAG) = (-ampr_prev * math::sin(omega0in * dt_) +
                                 ampi_prev * math::cos(omega0in * dt_)) *
                                  math::exp(-gamma0 * dt_) +
                                uni * sigma0 * dt_;

          if(damp0(i) < ONE) {
            damp0(i) += dt_;
          }
        });

      auto amplitudes_ = Kokkos::create_mirror_view(amplitudes);
      Kokkos::deep_copy(amplitudes_, amplitudes);
      // for (int i = 0; i < nmodes; ++i) {
      //   printf("amplitudes_(%d, REAL) = %f\n, rank = %d", i, amplitudes_(i, REAL), rank);
      // }

      auto fext_en_total = ZERO;
      for (auto& species : domain.species) {
        auto fext_en_s = ZERO;
        auto pld    = species.pld[0];
        auto weight = species.weight;
        Kokkos::parallel_reduce(
          "ExtForceEnrg",
          species.rangeActiveParticles(),
          ClassLambda(index_t p, real_t & fext_en) {
            fext_en += pld(p) * weight(p);
          },
          fext_en_s);
      #if defined(MPI_ENABLED)
        auto fext_en_sg = ZERO;
        MPI_Allreduce(&fext_en_s, &fext_en_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
        fext_en_total += fext_en_sg;
      #else
        fext_en_total += fext_en_s; 
      #endif
      }

      // Weight the macroparticle integral by sim parameters
      fext_en_total /= params.template get<real_t>("scales.n0");

      auto pkin_en_total = ZERO;
      for (auto& species : domain.species) {
        auto pkin_en_s = ZERO;
        auto ux1    = species.ux1;
        auto ux2    = species.ux2;
        auto ux3    = species.ux3;
        auto weight = species.weight;
        Kokkos::parallel_reduce(
          "KinEnrg",
          species.rangeActiveParticles(),
          ClassLambda(index_t p, real_t & pkin_en) {
            pkin_en += (math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p))) -
                        ONE) *
                       weight(p);
          },
          pkin_en_s);
      #if defined(MPI_ENABLED)
        auto pkin_en_sg = ZERO;
        MPI_Allreduce(&pkin_en_s, &pkin_en_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
        pkin_en_total += pkin_en_sg;
      #else
        pkin_en_total += pkin_en_s;
      #endif
      }

      // Weight the macroparticle integral by sim parameters
      pkin_en_total /= params.template get<real_t>("scales.n0");
        
      auto benrg_total = ZERO;
      auto eenrg_total = ZERO;
      auto ej_total = ZERO;

      if constexpr (D == Dim::_3D) {
        
        auto metric = domain.mesh.metric;
        
        auto benrg_s = ZERO;
        auto EB          = domain.fields.em;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, index_t i3, real_t & benrg) {
            coord_t<Dim::_3D> x_Cd { ZERO };
            vec_t<Dim::_3D>   b_Cntrv { EB(i1, i2, i3, em::bx1),
                                      EB(i1, i2, i3, em::bx2),
                                      EB(i1, i2, i3, em::bx3) };
            vec_t<Dim::_3D>   b_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd,
                                                                  b_Cntrv,
                                                                  b_XYZ);
            benrg += (SQR(b_XYZ[0]) + SQR(b_XYZ[1]) + SQR(b_XYZ[2]));
          },
          benrg_s);
        #if defined(MPI_ENABLED)
          auto benrg_sg = ZERO;
          MPI_Allreduce(&benrg_s, &benrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          benrg_total += benrg_sg;
        #else
          benrg_total += benrg_s;
        #endif

      // Weight the field integral by sim parameters
        benrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

        auto eenrg_s = ZERO;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, index_t i3, real_t & eenrg) {
            coord_t<Dim::_3D> x_Cd { ZERO };
            vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, i3, em::ex1),
                                      EB(i1, i2, i3, em::ex2),
                                      EB(i1, i2, i3, em::ex3) };
            vec_t<Dim::_3D>   e_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd, e_Cntrv, e_XYZ);            
            eenrg += (SQR(e_XYZ[0]) + SQR(e_XYZ[1]) + SQR(e_XYZ[2]));
          },
          eenrg_s);

        #if defined(MPI_ENABLED)
          auto eenrg_sg = ZERO;
          MPI_Allreduce(&eenrg_s, &eenrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          eenrg_total += eenrg_sg;  
        #else
          eenrg_total += eenrg_s;
        #endif

      // Weight the field integral by sim parameters
        eenrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

      }

      if constexpr (D == Dim::_2D) {
        
        auto metric = domain.mesh.metric;
        
        auto benrg_s = ZERO;
        auto EB          = domain.fields.em;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, real_t & benrg) {
            coord_t<Dim::_2D> x_Cd { ZERO };
            vec_t<Dim::_3D>   b_Cntrv { EB(i1, i2, em::bx1),
                                      EB(i1, i2, em::bx2),
                                      EB(i1, i2, em::bx3) };
            vec_t<Dim::_3D>   b_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd,
                                                                  b_Cntrv,
                                                                  b_XYZ);
            benrg += (SQR(b_XYZ[0]) + SQR(b_XYZ[1]) + SQR(b_XYZ[2]));
          },
          benrg_s);
        #if defined(MPI_ENABLED)
          auto benrg_sg = ZERO;
          MPI_Allreduce(&benrg_s, &benrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          benrg_total += benrg_sg;
        #else
          benrg_total += benrg_s;
        #endif

      // Weight the field integral by sim parameters
        benrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

        auto eenrg_s = ZERO;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, real_t & eenrg) {
            coord_t<Dim::_2D> x_Cd { ZERO };
            vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, em::ex1),
                                      EB(i1, i2, em::ex2),
                                      EB(i1, i2, em::ex3) };
            vec_t<Dim::_3D>   e_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd, e_Cntrv, e_XYZ);            
            eenrg += (SQR(e_XYZ[0]) + SQR(e_XYZ[1]) + SQR(e_XYZ[2]));
          },
          eenrg_s);

        #if defined(MPI_ENABLED)
          auto eenrg_sg = ZERO;
          MPI_Allreduce(&eenrg_s, &eenrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          eenrg_total += eenrg_sg;  
        #else
          eenrg_total += eenrg_s;
        #endif

      // Weight the field integral by sim parameters
        eenrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

        auto ej_s = ZERO;
        auto ext_current_ = this->ext_current;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, real_t & ej) {
            coord_t<Dim::_2D> x_Cd { i1, i2 };
            vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, em::ex1),
                                      EB(i1, i2, em::ex2),
                                      EB(i1, i2, em::ex3) };
            vec_t<Dim::_3D>   e_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd, e_Cntrv, e_XYZ);  

            coord_t<Dim::_2D> xp_Ph { ZERO };
            xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(x_Cd[0]);
            xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(x_Cd[1]);

            ej -= (e_XYZ[0] * ext_current_.jx1(xp_Ph) +
                   e_XYZ[1] * ext_current_.jx2(xp_Ph) +
                   e_XYZ[2] * ext_current_.jx3(xp_Ph));

          },
          ej_s);

        #if defined(MPI_ENABLED)
          auto ej_sg = ZERO;
          MPI_Allreduce(&ej_s, &ej_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          ej_total += ej_sg;
        #else
          ej_total += ej_s;
        #endif

      // Weight the field integral by sim parameters
        ej_total *= params.template get<real_t>("scales.V0") / params.template get<real_t>("scales.larmor0");

      }

      std::ofstream myfile1;
      std::ofstream myfile2;
      std::ofstream myfile3;
      std::ofstream myfile4;
      std::ofstream myfile5;
      std::ofstream myfile6;
      

      #if defined(MPI_ENABLED)

        if(rank == MPI_ROOT_RANK) {

          printf("fext_en_total: %f, pkin_en_total: %f, benrg_total: %f, eenrg_total: %f, MPI rank %d\n", fext_en_total, pkin_en_total, benrg_total, eenrg_total, MPI_ROOT_RANK);
          
          if (time == 0) {
            myfile1.open("fextenrg.txt");
          } else {
            myfile1.open("fextenrg.txt", std::ios_base::app);
          }
          myfile1 << fext_en_total << std::endl;

          if (time == 0) {
            myfile2.open("kenrg.txt");
          } else {
            myfile2.open("kenrg.txt", std::ios_base::app);
          }
          myfile2 << pkin_en_total << std::endl;

          if (time == 0) {
            myfile3.open("bsqenrg.txt");
          } else {
            myfile3.open("bsqenrg.txt", std::ios_base::app);
          }
          myfile3 << benrg_total << std::endl;

          if (time == 0) {
            myfile4.open("esqenrg.txt");
          } else {
            myfile4.open("esqenrg.txt", std::ios_base::app);
          }
          myfile4 << eenrg_total << std::endl;

          if (time == 0) {
            myfile5.open("amps.txt");
          } else {
            myfile5.open("amps.txt", std::ios_base::app);
          }

          for (int i = 0; i < nmodes; ++i) {
            myfile5 << amplitudes_(i, REAL) << " " << amplitudes_(i, IMAG) << " ";
          }
          myfile5 << std::endl;

          if (time == 0) {
            myfile6.open("ejenrg.txt");
          } else {
            myfile6.open("ejenrg.txt", std::ios_base::app);
          }
          myfile6 << ej_total << std::endl;

        }

      #else

          if (time == 0) {
            myfile1.open("fextenrg.txt");
          } else {
            myfile1.open("fextenrg.txt", std::ios_base::app);
          }
          myfile1 << fext_en_total << std::endl;

          if (time == 0) {
            myfile2.open("kenrg.txt");
          } else {
            myfile2.open("kenrg.txt", std::ios_base::app);
          }
          myfile2 << pkin_en_total << std::endl;

          if (time == 0) {
            myfile3.open("bsqenrg.txt");
          } else {
            myfile3.open("bsqenrg.txt", std::ios_base::app);
          }
          myfile3 << benrg_total << std::endl;

          if (time == 0) {
            myfile4.open("esqenrg.txt");
          } else {
            myfile4.open("esqenrg.txt", std::ios_base::app);
          }
          myfile4 << eenrg_total << std::endl;

          if (time == 0) {
            myfile5.open("amps.txt");
          } else {
            myfile5.open("amps.txt", std::ios_base::app);
          }
          for (int i = 0; i < nmodes; ++i) {
            myfile5 << amplitudes_(i, REAL) << " " << amplitudes_(i, IMAG) << " ";
          }
          myfile5 << std::endl;

          if (time == 0) {
            myfile6.open("ejenrg.txt");
          } else {
            myfile6.open("ejenrg.txt", std::ios_base::app);
          }
          myfile6 << ej_total << std::endl;
          
      #endif
    }
  };

} // namespace user

#endif