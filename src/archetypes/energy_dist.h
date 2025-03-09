/**
 * @file archetypes/energy_dist.hpp
 * @brief Defines an archetype for energy distributions
 * @implements
 *   - arch::EnergyDistribution<>
 *   - arch::Cold<> : arch::EnergyDistribution<>
 *   - arch::Powerlaw<> : arch::EnergyDistribution<>
 *   - arch::Maxwellian<> : arch::EnergyDistribution<>
 * @namespaces:
 *   - arch::
 * @note
 * The class returns a random velocity according to a coded distribution
 * For Cartesian: the returned velocity is in the global Cartesian basis
 * For non-Cartesian SR: the returned velocity is in the tetrad basis
 * For GR: the returned velocity is in the covariant basis
 */

#ifndef ARCHETYPES_ENERGY_DIST_HPP
#define ARCHETYPES_ENERGY_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct EnergyDistribution {
    static constexpr auto D = M::Dim;
    static constexpr bool is_energy_dist { true };
    static_assert(M::is_metric, "M must be a metric class");

    EnergyDistribution(const M& metric) : metric { metric } {}

    // Takes the physical coordinate of the particle and returns
    //   the velocity in tetrad basis
    // last argument -- is the species index (1, ..., nspec)
    Inline virtual void operator()(const coord_t<D>&,
                                   vec_t<Dim::_3D>& v,
                                   unsigned short = 0) const {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct Cold : public EnergyDistribution<S, M> {
    Cold(const M& metric) : EnergyDistribution<S, M> { metric } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v,
                           unsigned short = 0) const override {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct Powerlaw : public EnergyDistribution<S, M> {
    using EnergyDistribution<S, M>::metric;

    Powerlaw(const M&              metric,
             random_number_pool_t& pool,
             real_t                g_min,
             real_t                g_max,
             real_t                pl_ind)
      : EnergyDistribution<S, M> { metric }
      , pool { pool }
      , g_min { g_min }
      , g_max { g_max }
      , pl_ind { pl_ind } {}

    Inline void operator()(const coord_t<M::Dim>& x_Code,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp = 0) const override {
      auto rand_gen = pool.get_state();
      auto rand_X1  = Random<real_t>(rand_gen);
      auto rand_gam = ONE;

      // Power-law distribution from uniform (see https://mathworld.wolfram.com/RandomNumber.html)
      if (pl_ind != -ONE) {
        rand_gam += math::pow(
          math::pow(g_min, ONE + pl_ind) +
            (-math::pow(g_min, ONE + pl_ind) + math::pow(g_max, ONE + pl_ind)) *
              rand_X1,
          ONE / (ONE + pl_ind));
      } else {
        rand_gam += math::pow(g_min, ONE - rand_X1) * math::pow(g_max, rand_X1);
      }
      auto rand_u  = math::sqrt(SQR(rand_gam) - ONE);
      auto rand_X2 = Random<real_t>(rand_gen);
      auto rand_X3 = Random<real_t>(rand_gen);
      v[0]         = rand_u * (TWO * rand_X2 - ONE);
      v[2]         = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
      v[1]         = v[2] * math::cos(constant::TWO_PI * rand_X3);
      v[2]         = v[2] * math::sin(constant::TWO_PI * rand_X3);

      if constexpr (S == SimEngine::GRPIC) {
        // convert from the tetrad basis to covariant
        vec_t<Dim::_3D> v_Hat;
        v_Hat[0] = v[0];
        v_Hat[1] = v[1];
        v_Hat[2] = v[2];
        metric.template transform<Idx::T, Idx::D>(x_Code, v_Hat, v);
      }

      pool.free_state(rand_gen);
    }

  private:
    const real_t         g_min, g_max, pl_ind;
    random_number_pool_t pool;
  };

  template <SimEngine::type S, class M>
  struct Maxwellian : public EnergyDistribution<S, M> {
    using EnergyDistribution<S, M>::metric;

    Maxwellian(const M&              metric,
               random_number_pool_t& pool,
               real_t                temperature,
               real_t                boost_vel       = ZERO,
               in                    boost_direction = in::x1,
               bool                  zero_current    = true)
      : EnergyDistribution<S, M> { metric }
      , pool { pool }
      , temperature { temperature }
      , boost_velocity { boost_vel }
      , boost_direction { boost_direction }
      , zero_current { zero_current } {
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative",
                     HERE);
      raise::ErrorIf(
        (not cmp::AlmostZero(boost_vel, ZERO)) && (M::CoordType != Coord::Cart),
        "Maxwellian: Boosting is only supported in Cartesian coordinates",
        HERE);
    }

    // Juttner-Synge distribution
    Inline void JS(vec_t<Dim::_3D>& v, const real_t& temp) const {
      auto   rand_gen = pool.get_state();
      real_t randX1, randX2;
      if (temp < static_cast<real_t>(0.5)) {
        // Juttner-Synge distribution using the Box-Muller method - non-relativistic
        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[0]   = randX1 * math::cos(randX2) * math::sqrt(temp);

        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);
      } else {
        // Juttner-Synge distribution using the Sobol method - relativistic
        auto randu   = ONE;
        auto randeta = Random<real_t>(rand_gen);
        while (SQR(randeta) <= SQR(randu) + ONE) {
          randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                   Random<real_t>(rand_gen);
          while (cmp::AlmostZero(randX1)) {
            randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                     Random<real_t>(rand_gen);
          }
          randu  = -temp * math::log(randX1);
          randX2 = Random<real_t>(rand_gen);
          while (cmp::AlmostZero(randX2)) {
            randX2 = Random<real_t>(rand_gen);
          }
          randeta = -temp * math::log(randX1 * randX2);
        }
        randX1 = Random<real_t>(rand_gen);
        randX2 = Random<real_t>(rand_gen);
        v[0]   = randu * (TWO * randX1 - ONE);
        v[2]   = TWO * randu * math::sqrt(randX1 * (ONE - randX1));
        v[1]   = v[2] * math::cos(constant::TWO_PI * randX2);
        v[2]   = v[2] * math::sin(constant::TWO_PI * randX2);
      }
      pool.free_state(rand_gen);
    }

    // Boost a symmetric distribution to a relativistic speed using flipping
    // method https://arxiv.org/pdf/1504.03910.pdf
    Inline void boost(vec_t<Dim::_3D>& v) const {
      const auto boost_dir = static_cast<unsigned short>(boost_direction);
      const auto boost_beta { boost_velocity /
                              math::sqrt(ONE + SQR(boost_velocity)) };
      const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
      auto       rand_gen = pool.get_state();
      if (-boost_beta * v[boost_dir] > gamma * Random<real_t>(rand_gen)) {
        v[boost_dir] = -v[boost_dir];
      }
      pool.free_state(rand_gen);
      v[boost_dir] = math::sqrt(ONE + SQR(boost_velocity)) *
                     (v[boost_dir] + boost_beta * gamma);
    }

    Inline void operator()(const coord_t<M::Dim>& x_Code,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         s = 0) const override {
      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        JS(v, temperature);
      }
      if constexpr (S == SimEngine::GRPIC) {
        // convert from the tetrad basis to covariant
        vec_t<Dim::_3D> v_Hat;
        v_Hat[0] = v[0];
        v_Hat[1] = v[1];
        v_Hat[2] = v[2];
        metric.template transform<Idx::T, Idx::D>(x_Code, v_Hat, v);
      }
      if constexpr (M::CoordType == Coord::Cart) {
        // boost only when using cartesian coordinates
        if (not cmp::AlmostZero(boost_velocity)) {
          boost(v);
          if (not zero_current and s % 2 == 0) {
            v[0] = -v[0];
            v[1] = -v[1];
            v[2] = -v[2];
          }
        }
      }
    }

  private:
    random_number_pool_t pool;

    const real_t temperature;
    const real_t boost_velocity;
    const in     boost_direction;
    const bool   zero_current;
  };

} // namespace arch

#endif // ARCHETYPES_ENERGY_DIST_HPP
