#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include <stdlib.h>
#endif // MPI_ENABLED

namespace user {
  using namespace ntt;

  // initializing guide field and curl(B) = J_ext at the initial time step
  template <Dimension D>
  struct InitFields {
    InitFields(array_t<real_t**>& k,
               array_t<real_t*>&  a_real,
               array_t<real_t*>&  a_imag,
               array_t<real_t*>&  a_real_inv,
               array_t<real_t*>&  a_imag_inv)
      : k { k }
      , a_real { a_real }
      , a_imag { a_imag }
      , a_real_inv { a_real_inv }
      , a_imag_inv { a_imag_inv }
      , n_modes { a_real.size() } {};

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      auto bx1_0 = ZERO;
      if constexpr (D == Dim::_2D) {
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          bx1_0        -= TWO * k(1, i) *
                   (a_real(i) * math::sin(k_dot_r) +
                    a_imag(i) * math::cos(k_dot_r));
          bx1_0 -= TWO * k(1, i) *
                   (a_real_inv(i) * math::sin(k_dot_r) +
                    a_imag_inv(i) * math::cos(k_dot_r));
        }
        return bx1_0;
      }
      if constexpr (D == Dim::_3D) {
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          bx1_0 -= TWO * k(1, i) *
                   (a_real(i) * math::sin(k_dot_r) +
                    a_imag(i) * math::cos(k_dot_r));
        }
        return bx1_0;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      auto bx2_0 = ZERO;
      if constexpr (D == Dim::_2D) {
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          bx2_0        += TWO * k(0, i) *
                   (a_real(i) * math::sin(k_dot_r) +
                    a_imag(i) * math::cos(k_dot_r));
          bx2_0 += TWO * k(0, i) *
                   (a_real_inv(i) * math::sin(k_dot_r) +
                    a_imag_inv(i) * math::cos(k_dot_r));
        }
        return bx2_0;
      }
      if constexpr (D == Dim::_3D) {
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          bx2_0 += TWO * k(0, i) *
                   (a_real(i) * math::sin(k_dot_r) +
                    a_imag(i) * math::cos(k_dot_r));
        }
        return bx2_0;
      }
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }

    array_t<real_t**> k;
    array_t<real_t*>  a_real;
    array_t<real_t*>  a_imag;
    array_t<real_t*>  a_real_inv;
    array_t<real_t*>  a_imag_inv;
    std::size_t       n_modes;
  };

  inline auto init_pool(unsigned int seed) -> unsigned int {
    if (seed == 0) {
      unsigned int new_seed = static_cast<unsigned int>(rand());
#if defined(MPI_ENABLED)
      MPI_Bcast(&new_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif // MPI_ENABLED
      return new_seed;
    } else {
      return static_cast<unsigned int>(seed);
    }
  }

  template <Dimension D>
  inline auto init_wavenumbers() -> std::vector<std::vector<real_t>> {
    if constexpr (D == Dim::_2D) {
      return {
        {  1, 0 },
        {  0, 1 },
        {  1, 1 },
        { -1, 1 }
      };
    } else if constexpr (D == Dim::_3D) {
      return {
        {  1,  0,  1 },
        {  0,  1,  1 },
        { -1,  0,  1 },
        {  0, -1,  1 },
        {  1,  0, -1 },
        {  0,  1, -1 },
        { -1,  0, -1 },
        {  0, -1, -1 }
      };
    } else {
      raise::Error("Invalid dimension", HERE);
      return {};
    }
  }

  // external current definition
  template <Dimension D>
  struct ExternalCurrent {
    ExternalCurrent(real_t                            dB,
                    real_t                            om0,
                    real_t                            g0,
                    std::vector<std::vector<real_t>>& wavenumbers,
                    unsigned int                      seed,
                    real_t                            Lx,
                    real_t                            Ly,
                    real_t                            Lz)
      : wavenumbers { wavenumbers }
      , n_modes { wavenumbers.size() }
      , dB { dB }
      , Lx { Lx }
      , Ly { Ly }
      , Lz { Lz }
      , seed { seed }
      , omega_0 { om0 }
      , gamma_0 { g0 }
      , k { "wavevector", D, n_modes }
      , a_real { "a_real", n_modes }
      , a_imag { "a_imag", n_modes }
      , a_real_inv { "a_real_inv", n_modes }
      , a_imag_inv { "a_imag_inv", n_modes }
      , A0 { "A0", n_modes } {
      // initializing random generator
      srand(seed);
      // initializing wavevectors
      auto k_host = Kokkos::create_mirror_view(k);
      if constexpr (D == Dim::_2D) {
        for (auto i = 0u; i < n_modes; i++) {
          k_host(0, i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
          k_host(1, i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
        }
      }
      if constexpr (D == Dim::_3D) {
        for (auto i = 0u; i < n_modes; i++) {
          k_host(0, i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
          k_host(1, i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
          k_host(2, i) = constant::TWO_PI * wavenumbers[i][2] / Lz;
        }
      }
      // initializing initial complex amplitudes
      auto a_real_host     = Kokkos::create_mirror_view(a_real);
      auto a_imag_host     = Kokkos::create_mirror_view(a_imag);
      auto a_real_inv_host = Kokkos::create_mirror_view(a_real_inv);
      auto a_imag_inv_host = Kokkos::create_mirror_view(a_imag_inv);
      auto A0_host         = Kokkos::create_mirror_view(A0);

      real_t prefac { ZERO };
      if constexpr (D == Dim::_2D) {
        prefac = HALF; // HALF = 1/sqrt(twice modes due to reality condition * twice the frequencies due to sign change)
      } else if constexpr (D == Dim::_3D) {
        prefac = ONE;
      }
      for (auto i = 0u; i < n_modes; i++) {
        auto k_perp = math::sqrt(
          k_host(0, i) * k_host(0, i) + k_host(1, i) * k_host(1, i));
        real_t phase = static_cast<real_t>(rand()) /
                       static_cast<real_t>(RAND_MAX) * constant::TWO_PI;
        A0_host(i)     = dB / math::sqrt((real_t)n_modes) / k_perp * prefac;
        a_real_host(i) = A0_host(i) * math::cos(phase);
        a_imag_host(i) = A0_host(i) * math::sin(phase);
        phase = static_cast<real_t>(rand()) / static_cast<real_t>(RAND_MAX) *
                constant::TWO_PI;
        a_imag_inv_host(i) = A0_host(i) * math::cos(phase);
        a_real_inv_host(i) = A0_host(i) * math::sin(phase);
      }

      Kokkos::deep_copy(a_real, a_real_host);
      Kokkos::deep_copy(a_imag, a_imag_host);
      Kokkos::deep_copy(a_real_inv, a_real_inv_host);
      Kokkos::deep_copy(a_imag_inv, a_imag_inv_host);
      Kokkos::deep_copy(A0, A0_host);
      Kokkos::deep_copy(k, k_host);
    };

    Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return ZERO;
      }
      if constexpr (D == Dim::_3D) {
        real_t jx1_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx1_ant -= TWO * k(0, i) * k(2, i) *
                     (a_real(i) * math::cos(k_dot_r) -
                      a_imag(i) * math::sin(k_dot_r));
        }
        return jx1_ant;
      }
    }

    Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return ZERO;
      } else if constexpr (D == Dim::_3D) {
        real_t jx2_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx2_ant -= TWO * k(1, i) * k(2, i) *
                     (a_real(i) * math::cos(k_dot_r) -
                      a_imag(i) * math::sin(k_dot_r));
        }
        return jx2_ant;
      }
    }

    Inline auto jx3(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        real_t jx3_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_perp_sq  = k(0, i) * k(0, i) + k(1, i) * k(1, i);
          auto k_dot_r    = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          jx3_ant        += TWO * k_perp_sq *
                     (a_real(i) * math::cos(k_dot_r) -
                      a_imag(i) * math::sin(k_dot_r));
          jx3_ant += TWO * k_perp_sq *
                     (a_real_inv(i) * math::cos(k_dot_r) -
                      a_imag_inv(i) * math::sin(k_dot_r));
        }
        return jx3_ant;
      } else if constexpr (D == Dim::_3D) {
        real_t jx3_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_perp_sq = k(0, i) * k(0, i) + k(1, i) * k(1, i);
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx3_ant += TWO * k_perp_sq *
                     (a_real(i) * math::cos(k_dot_r) -
                      a_imag(i) * math::sin(k_dot_r));
        }
        return jx3_ant;
      }
    }

  private:
    const std::vector<std::vector<real_t>> wavenumbers;
    const std::size_t                      n_modes;
    const real_t                           dB, Lx, Ly, Lz;
    const unsigned int                     seed;

  public:
    const real_t      omega_0, gamma_0;
    array_t<real_t**> k;
    array_t<real_t*>  a_real;
    array_t<real_t*>  a_imag;
    array_t<real_t*>  a_real_inv;
    array_t<real_t*>  a_imag_inv;
    array_t<real_t*>  A0;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines =
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics =
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      arch::traits::pgen::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t                     temperature, dB, omega_0, gamma_0;
    const real_t                     Lx, Ly, Lz, escape_dist;
    const unsigned int               random_seed;
    std::vector<std::vector<real_t>> wavenumbers;
    random_number_pool_t             random_pool;

    // debugging, will delete later
    real_t total_sum           = ZERO;
    real_t total_sum_inv       = ZERO;
    real_t number_of_timesteps = ZERO;

    ExternalCurrent<D> ext_current;
    InitFields<D>      init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , dB { p.template get<real_t>("setup.dB", ONE) }
      , omega_0 { p.template get<real_t>("setup.omega_0") }
      , gamma_0 { p.template get<real_t>("setup.gamma_0") }
      , wavenumbers { init_wavenumbers<D>() }
      , random_seed { p.template get<unsigned int>("setup.seed", 0) }
      , random_pool { init_pool(random_seed) }
      , Lx { global_domain.mesh().extent(in::x1).second -
             global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second -
             global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second -
             global_domain.mesh().extent(in::x3).first }
      , escape_dist { p.template get<real_t>("setup.escape_dist", HALF * Lx) }
      , ext_current { dB, omega_0, gamma_0, wavenumbers, init_pool(random_seed),
                      Lx, Ly,      Lz }
      , init_flds { ext_current.k,
                    ext_current.a_real,
                    ext_current.a_imag,
                    ext_current.a_real_inv,
                    ext_current.a_imag_inv } {};

    inline void InitPrtls(Domain<S, M>& domain) {
      arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, temperature, { 1, 2 });
    }

    void CustomPostStep(timestep_t, simtime_t, Domain<S, M>& domain) {
#if defined(MPI_ENABLED)
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
      // update amplitudes of antenna
      const auto  dt = params.template get<real_t>("algorithms.timestep.dt");
      const auto& ext_curr = ext_current;
      Kokkos::parallel_for(
        "Antenna amplitudes",
        wavenumbers.size(),
        ClassLambda(index_t i) {
          auto       generator  = random_pool.get_state();
          const auto u_imag     = Random<real_t>(generator) - HALF;
          const auto u_real     = Random<real_t>(generator) - HALF;
          const auto u_real_inv = Random<real_t>(generator) - HALF;
          const auto u_imag_inv = Random<real_t>(generator) - HALF;
          random_pool.free_state(generator);

          auto a_real_prev     = ext_curr.a_real(i);
          auto a_imag_prev     = ext_curr.a_imag(i);
          auto a_real_inv_prev = ext_curr.a_real_inv(i);
          auto a_imag_inv_prev = ext_curr.a_imag_inv(i);
          ext_curr.a_real(i) = (a_real_prev * math::cos(ext_curr.omega_0 * dt) +
                                a_imag_prev * math::sin(ext_curr.omega_0 * dt)) *
                                 math::exp(-ext_curr.gamma_0 * dt) +
                               ext_curr.A0(i) *
                                 math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                                 u_real * dt;

          ext_curr.a_imag(i) = (a_imag_prev * math::cos(ext_curr.omega_0 * dt) -
                                a_real_prev * math::sin(ext_curr.omega_0 * dt)) *
                                 math::exp(-ext_curr.gamma_0 * dt) +
                               ext_curr.A0(i) *
                                 math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                                 u_imag * dt;

          ext_curr.a_real_inv(
            i) = (a_real_inv_prev * math::cos(-ext_curr.omega_0 * dt) +
                  a_imag_inv_prev * math::sin(-ext_curr.omega_0 * dt)) *
                   math::exp(-ext_curr.gamma_0 * dt) +
                 ext_curr.A0(i) * math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                   u_real_inv * dt;

          ext_curr.a_imag_inv(
            i) = (a_imag_inv_prev * math::cos(-ext_curr.omega_0 * dt) -
                  a_real_inv_prev * math::sin(-ext_curr.omega_0 * dt)) *
                   math::exp(-ext_curr.gamma_0 * dt) +
                 ext_curr.A0(i) * math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                   u_imag_inv * dt;
        });

      // particle escape (resample velocities)
      const auto energy_dist = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      domain.random_pool,
                                                      temperature);
      for (const auto& sp : { 0, 1 }) {
        if (domain.species[sp].npld_r() > 1) {
          const auto& ux1   = domain.species[sp].ux1;
          const auto& ux2   = domain.species[sp].ux2;
          const auto& ux3   = domain.species[sp].ux3;
          const auto& pld_r = domain.species[sp].pld_r;
          const auto& tag   = domain.species[sp].tag;
          const auto  L     = escape_dist;
          Kokkos::parallel_for(
            "UpdatePld",
            domain.species[sp].npart(),
            Lambda(index_t p) {
              if (tag(p) == ParticleTag::dead) {
                return;
              }
              const auto gamma = math::sqrt(
                ONE + ux1(p) * ux1(p) + ux2(p) * ux2(p) + ux3(p) * ux3(p));
              pld_r(p, 0) += ux1(p) * dt / gamma;
              pld_r(p, 1) += ux2(p) * dt / gamma;

              if ((math::abs(pld_r(p, 0)) > L) or (math::abs(pld_r(p, 1)) > L)) {
                coord_t<D>      x_Ph { ZERO };
                vec_t<Dim::_3D> u_Mxw { ZERO };
                energy_dist(x_Ph, u_Mxw);
                ux1(p)      = u_Mxw[0];
                ux2(p)      = u_Mxw[1];
                ux3(p)      = u_Mxw[2];
                pld_r(p, 0) = ZERO;
                pld_r(p, 1) = ZERO;
              }
            });
        }
      }
    }
  };
} // namespace user

#endif
