#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"
#include "kernels/injectors.hpp"
#include "kernels/pushers/traits.h"

#include <Kokkos_Pair.hpp>

#include <map>
#include <string>
#include <vector>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  };

  template <class M>
  struct RandomEmission {
    struct Payload {
      real_t photon_energy { ZERO };
    };

    random_number_pool_t random_pool;
    const real_t         probability;

    array_t<npart_t> inj_idx { "inj_idx" };
    const npart_t    inj_offset;

    array_t<int*>      inj_i1, inj_i2, inj_i3;
    array_t<prtldx_t*> inj_dx1, inj_dx2, inj_dx3;
    array_t<real_t*>   inj_ux1, inj_ux2, inj_ux3;
    array_t<real_t*>   inj_phi;
    array_t<real_t*>   inj_weight;
    array_t<short*>    inj_tag;
    array_t<npart_t**> inj_pld_i;

    RandomEmission(random_number_pool_t& random_pool,
                   real_t                probability,
                   npart_t               inj_offset,
                   array_t<int*>&        inj_i1,
                   array_t<int*>&        inj_i2,
                   array_t<int*>&        inj_i3,
                   array_t<prtldx_t*>&   inj_dx1,
                   array_t<prtldx_t*>&   inj_dx2,
                   array_t<prtldx_t*>&   inj_dx3,
                   array_t<real_t*>&     inj_ux1,
                   array_t<real_t*>&     inj_ux2,
                   array_t<real_t*>&     inj_ux3,
                   array_t<real_t*>&     inj_phi,
                   array_t<real_t*>&     inj_weight,
                   array_t<short*>&      inj_tag,
                   array_t<npart_t**>&   inj_pld_i)
      : random_pool { random_pool }
      , probability { probability }
      , inj_offset { inj_offset }
      , inj_i1 { inj_i1 }
      , inj_i2 { inj_i2 }
      , inj_i3 { inj_i3 }
      , inj_dx1 { inj_dx1 }
      , inj_dx2 { inj_dx2 }
      , inj_dx3 { inj_dx3 }
      , inj_ux1 { inj_ux1 }
      , inj_ux2 { inj_ux2 }
      , inj_ux3 { inj_ux3 }
      , inj_phi { inj_phi }
      , inj_weight { inj_weight }
      , inj_tag { inj_tag }
      , inj_pld_i { inj_pld_i } {}

    Inline auto shouldEmit(const coord_t<M::PrtlDim>&,
                           const coord_t<M::PrtlDim>&,
                           const vec_t<Dim::_3D>& u_Ph,
                           const vec_t<Dim::_3D>&,
                           const vec_t<Dim::_3D>&,
                           vec_t<Dim::_3D>& delta_u_Ph,
                           Payload& payload) const -> Kokkos::pair<bool, bool> {
      auto       generator = random_pool.get_state();
      const auto rnd       = Random<real_t>(generator);
      random_pool.free_state(generator);
      if (rnd < probability) {
        delta_u_Ph[0] = -0.1 * u_Ph[0];
        delta_u_Ph[1] = -0.1 * u_Ph[1];
        delta_u_Ph[2] = -0.1 * u_Ph[2];

        const auto uSqr          = NORM_SQR(u_Ph[0], u_Ph[1], u_Ph[2]);
        const auto gammaSqr      = ONE + uSqr;
        const auto delta_uSqr    = NORM_SQR(delta_u_Ph[0],
                                         delta_u_Ph[1],
                                         delta_u_Ph[2]);
        const auto u_dot_delta_u = DOT(u_Ph[0],
                                       u_Ph[1],
                                       u_Ph[2],
                                       delta_u_Ph[0],
                                       delta_u_Ph[1],
                                       delta_u_Ph[2]);
        payload.photon_energy    = math::sqrt(gammaSqr) *
                                (math::sqrt(ONE + delta_uSqr / gammaSqr +
                                            TWO * u_dot_delta_u / gammaSqr) -
                                 ONE);
        return { true, true };
      }
      return { false, false };
    }

    Inline auto emit(const tuple_t<int, M::Dim>&      xi_Cd,
                     const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                     const vec_t<Dim::_3D>&           direction,
                     real_t,
                     real_t,
                     const Payload& payload) const -> void {
      const auto inj_index = Kokkos::atomic_fetch_add(&inj_idx(), 1);
      kernel::InjectParticle<M::Dim, M::CoordType, false>(
        inj_offset + inj_index,
        inj_i1,
        inj_i2,
        inj_i3,
        inj_dx1,
        inj_dx2,
        inj_dx3,
        inj_ux1,
        inj_ux2,
        inj_ux3,
        inj_phi,
        inj_weight,
        inj_tag,
        inj_pld_i,
        xi_Cd,
        dxi_Cd,
        { payload.photon_energy * direction[0],
          payload.photon_energy * direction[1],
          payload.photon_energy * direction[2] });
    }

    auto emitted_species_indices() const -> std::vector<spidx_t> {
      return { 2u };
    }

    auto numbers_injected() const -> std::vector<npart_t> {
      auto inj_idx_h = Kokkos::create_mirror_view(inj_idx);
      Kokkos::deep_copy(inj_idx_h, inj_idx);
      return { inj_idx_h() };
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const Metadomain<S, M>& metadomain;

    const real_t emission_probability;

    InitFields<D> init_flds {};

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , metadomain { metadomain }
      , emission_probability { params.template get<real_t>(
          "setup.emission_probability") } {
      static_assert(kernel::traits::pusher::emission::IsValid<RandomEmission<M>, M>,
                    "RandomEmission does not satisfy the requirements of an "
                    "emission policy");
    }

    inline auto EmissionPolicy(simtime_t, spidx_t, Domain<S, M>& domain) const
      -> RandomEmission<M> {
      return RandomEmission<M> {
        domain.random_pool(),      emission_probability,
        domain.species[1].npart(), domain.species[1].i1,
        domain.species[1].i2,      domain.species[1].i3,
        domain.species[1].dx1,     domain.species[1].dx2,
        domain.species[1].dx3,     domain.species[1].ux1,
        domain.species[1].ux2,     domain.species[1].ux3,
        domain.species[1].phi,     domain.species[1].weight,
        domain.species[1].tag,     domain.species[1].pld_i
      };
    }

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto empty  = std::vector<real_t> {};
      const auto x1_arr = params.template get<std::vector<real_t>>(
        "setup.x1_arr",
        empty);
      const auto x2_arr = params.template get<std::vector<real_t>>(
        "setup.x2_arr",
        empty);
      const auto x3_arr = params.template get<std::vector<real_t>>(
        "setup.x3_arr",
        empty);
      const auto ux1_arr = params.template get<std::vector<real_t>>(
        "setup.ux1_arr",
        empty);
      const auto ux2_arr = params.template get<std::vector<real_t>>(
        "setup.ux2_arr",
        empty);
      const auto ux3_arr = params.template get<std::vector<real_t>>(
        "setup.ux3_arr",
        empty);

      std::map<std::string, std::vector<real_t>> data_arr {
        {  "x1",  x1_arr },
        {  "x2",  x2_arr },
        {  "x3",  x3_arr },
        { "ux1", ux1_arr },
        { "ux2", ux2_arr },
        { "ux3", ux3_arr }
      };
      arch::InjectGlobally<S, M>(metadomain, domain, 1u, data_arr);
    }
  };

} // namespace user

#endif
