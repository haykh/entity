#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <vector>

/* -------------------------------------------------------------------------- */
/* Local macros    (same as in particle_pusher_sr.hpp)                        */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    I = static_cast<int>((XI + 1)) - 1;                                        \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

namespace user {
  using namespace ntt;

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

    const Metadomain<S, M>& global_domain;

    const real_t temperature, temperature_gradient;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , temperature { params.template get<real_t>("setup.temperature", 0.0) }
      , temperature_gradient {
        params.template get<real_t>("setup.temperature_gradient", 0.0)
      } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto empty = std::vector<real_t> {};
      const auto x1_e  = params.template get<std::vector<real_t>>("setup.x1_e",
                                                                 empty);
      const auto x2_e  = params.template get<std::vector<real_t>>("setup.x2_e",
                                                                 empty);
      const auto x3_e  = params.template get<std::vector<real_t>>("setup.x3_e",
                                                                 empty);
      const auto phi_e = params.template get<std::vector<real_t>>("setup.phi_e",
                                                                  empty);
      const auto ux1_e = params.template get<std::vector<real_t>>("setup.ux1_e",
                                                                  empty);
      const auto ux2_e = params.template get<std::vector<real_t>>("setup.ux2_e",
                                                                  empty);
      const auto ux3_e = params.template get<std::vector<real_t>>("setup.ux3_e",
                                                                  empty);

      const auto x1_i  = params.template get<std::vector<real_t>>("setup.x1_i",
                                                                 empty);
      const auto x2_i  = params.template get<std::vector<real_t>>("setup.x2_i",
                                                                 empty);
      const auto x3_i  = params.template get<std::vector<real_t>>("setup.x3_i",
                                                                 empty);
      const auto phi_i = params.template get<std::vector<real_t>>("setup.phi_i",
                                                                  empty);
      const auto ux1_i = params.template get<std::vector<real_t>>("setup.ux1_i",
                                                                  empty);
      const auto ux2_i = params.template get<std::vector<real_t>>("setup.ux2_i",
                                                                  empty);
      const auto ux3_i = params.template get<std::vector<real_t>>("setup.ux3_i",
                                                                  empty);
      std::map<std::string, std::vector<real_t>> data_e {
        {  "x1",  x1_e },
        {  "x2",  x2_e },
        { "ux1", ux1_e },
        { "ux2", ux2_e },
        { "ux3", ux3_e }
      };
      std::map<std::string, std::vector<real_t>> data_i {
        {  "x1",  x1_i },
        {  "x2",  x2_i },
        { "ux1", ux1_i },
        { "ux2", ux2_i },
        { "ux3", ux3_i }
      };
      if constexpr (M::CoordType == Coord::Cart or D == Dim::_3D) {
        data_e["x3"] = x3_e;
        data_i["x3"] = x3_i;
      } else if constexpr (D == Dim::_2D) {
        data_e["phi"] = phi_e;
        data_i["phi"] = phi_i;
      }

      arch::InjectGlobally<S, M>(global_domain, local_domain, (spidx_t)1, data_e);
      arch::InjectGlobally<S, M>(global_domain, local_domain, (spidx_t)2, data_i);
    }

    auto FixFieldsConst(const bc_in&, const em&) const -> std::pair<real_t, bool> {
      return { ZERO, false };
    }

    struct CustomPrtlUpdate {
      random_number_pool_t pool;
      real_t               temp_cold, temp_hot;
      real_t               xmin, xmax;

      template <class Coord, class PusherKernel>
      Inline void operator()(index_t p, Coord& xp, PusherKernel& pusher) const {

        const auto x_Cd = static_cast<real_t>(pusher.i1(p)) +
                          static_cast<real_t>(pusher.dx1(p));
        const auto x_Ph = pusher.metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
        vec_t<Dim::_3D>       v { ZERO };
        const coord_t<M::Dim> x_dummy { ZERO };

        // step 1: calculate the particle 3 velocity
        const real_t gamma_p = math::sqrt(
          ONE + SQR(pusher.ux1(p)) + SQR(pusher.ux2(p)) + SQR(pusher.ux3(p)));

        const real_t beta_x_p = math::abs(pusher.ux1(p)) / gamma_p;
        const real_t xp_prev = i_di_to_Xi(pusher.i1_prev(p), pusher.dx1_prev(p));

        // Reflecting boundary that resamples velocity
        if (x_Ph < xmin) {
          arch::JuttnerSinge(v, temp_cold, pool);

          // calculate the time for the particle to reach the wall
          const int      delta_i1_to_wall  = pusher.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = pusher.dx1_prev(p);
          const real_t dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t dt_to_wall =
            dx_to_wall /
            pusher.metric.template transform<1, Idx::XYZ, Idx::U>(x_dummy,
                                                                  beta_x_p);

          // update the velocity to the post-collision value (reflection with new speed sampled from Juttner distribution)
          pusher.ux1(p) = math::abs(v[0]);
          pusher.ux2(p) = v[1];
          pusher.ux3(p) = v[2];

          // calculate remaining time after the collision
          const real_t remaining_dt            = pusher.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(ONE +
                                                            SQR(pusher.ux1(p)) +
                                                            SQR(pusher.ux2(p)) +
                                                            SQR(pusher.ux3(p)));

          // update the position to the post-collision value (reflection with new speed sampled from Juttner distribution)
          pusher.i1(p)  = 0;
          pusher.dx1(p) = pusher.metric.template transform<1, Idx::XYZ, Idx::U>(
                            x_dummy,
                            pusher.ux1(p)) *
                          remaining_dt_inv_energy;

          // Re-sync coordinate variable xp for subsequent boundary procedures
          xp[0] = static_cast<real_t>(pusher.i1(p)) +
                  static_cast<real_t>(pusher.dx1(p));

        } else if (x_Ph > xmax) {
          arch::JuttnerSinge(v, temp_hot, pool);

          // step 2: calculate the time for the particle to reach the piston
          const int      delta_i1_to_wall  = pusher.ni1 - 1 - pusher.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = ONE - pusher.dx1_prev(p);
          const real_t dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t dt_to_wall =
            dx_to_wall /
            pusher.metric.template transform<1, Idx::XYZ, Idx::U>(x_dummy,
                                                                  beta_x_p);

          // update the velocity to the post-collision value (reflection with new speed sampled from Juttner distribution)
          pusher.ux1(p) = -math::abs(v[0]);
          pusher.ux2(p) = v[1];
          pusher.ux3(p) = v[2];

          // step 3: calculate remaining time after the collision
          const real_t remaining_dt            = pusher.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(ONE +
                                                            SQR(pusher.ux1(p)) +
                                                            SQR(pusher.ux2(p)) +
                                                            SQR(pusher.ux3(p)));

          // update the position to the post-collision value (reflection with new speed sampled from Juttner distribution)
          pusher.i1(p)  = pusher.ni1 - 2;
          pusher.dx1(p) = ONE -
                          pusher.metric.template transform<1, Idx::XYZ, Idx::U>(
                            x_dummy,
                            math::abs(pusher.ux1(p))) *
                            remaining_dt_inv_energy;

          // Re-sync coordinate variable xp for subsequent boundary procedures
          xp[0] = static_cast<real_t>(pusher.i1(p)) +
                  static_cast<real_t>(pusher.dx1(p));
        }
      }
    };

    template <class D>
    auto CustomParticleUpdate(simtime_t time, spidx_t sp, D& domain) const {

      return CustomPrtlUpdate {
        domain.random_pool(),
        temperature / domain.species[sp - 1].mass(), // sp is 1-indexed
        temperature_gradient * temperature / domain.species[sp - 1].mass(),
        global_domain.mesh().extent(in::x1).first, // xmin
        global_domain.mesh().extent(in::x1).second // xmax
      };
    }
  };

} // namespace user

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // PROBLEM_GENERATOR_H
