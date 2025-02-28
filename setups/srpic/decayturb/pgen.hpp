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
    InitFields(real_t Bnorm, array_t<real_t* [8]> amplitudes, array_t<real_t* [8]> phi )
      : Bnorm { Bnorm } 
      , B0x1 { ZERO }
      , B0x2 { ZERO }
      , B0x3 { Bnorm } 
      , amp { amplitudes}
      , phi { phi } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {

      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k-1, l-1) * kb1 / kbnorm * math::sin(kdotx + phi(k-1, l-1));

        }
      }

      return B0x1 + dBvec;       

    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      
      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k-1, l-1) * kb2 / kbnorm * math::sin(kdotx + phi(k-1, l-1));

        }
      }

      return B0x2 + dBvec;        
      }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      
      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k-1, l-1) * kb3 / kbnorm * math::sin(kdotx + phi(k-1, l-1));

        }
      }

      return B0x3 + dBvec;        
    }

  private:
    const real_t Bnorm;
    const real_t B0x1, B0x2, B0x3;
    array_t<real_t* [8]> amp;
    array_t<real_t* [8]> phi;
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
    array_t<real_t* [8]> amplitudes, phi0;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& params, Metadomain<S, M>& global_domain)
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
      , amp0 { machno * temperature / static_cast<real_t>(8) }
      , phi0 { "DrivingPhases", 8 }
      , amplitudes { "DrivingModes", 8 }
      , init_flds { Bnorm, amplitudes, phi0 } {
      // Initializing random phases
      auto phi0_ = Kokkos::create_mirror_view(phi0);
      auto amplitudes_ = Kokkos::create_mirror_view(amplitudes);
      srand (static_cast <unsigned> (12345));
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          phi0_(i, j) = constant::TWO_PI * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
          amplitudes_(i, j) = amp0 * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
        }
      }
      Kokkos::deep_copy(phi0, phi0_);
      Kokkos::deep_copy(amplitudes, amplitudes_);
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
    }
  };

} // namespace user

#endif