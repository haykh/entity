#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/injectors.hpp"
#include "kernels/pushers/sr.hpp"
#include "kernels/pushers/traits.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include <iostream>
#include <vector>

template <class M>
struct EmissionPolicy {
  struct Payload {};

  array_t<npart_t> sp1_idx { "sp1_idx" };
  array_t<npart_t> sp2_idx { "sp2_idx" };

  const unsigned int step;

  const npart_t sp1_offset, sp2_offset;

  array_t<int*>      sp1_i1, sp1_i2, sp1_i3;
  array_t<prtldx_t*> sp1_dx1, sp1_dx2, sp1_dx3;
  array_t<real_t*>   sp1_ux1, sp1_ux2, sp1_ux3;
  array_t<real_t*>   sp1_phi;
  array_t<real_t*>   sp1_weight;
  array_t<short*>    sp1_tag;
  array_t<npart_t**> sp1_pld_i;

  array_t<int*>      sp2_i1, sp2_i2, sp2_i3;
  array_t<prtldx_t*> sp2_dx1, sp2_dx2, sp2_dx3;
  array_t<real_t*>   sp2_ux1, sp2_ux2, sp2_ux3;
  array_t<real_t*>   sp2_phi;
  array_t<real_t*>   sp2_weight;
  array_t<short*>    sp2_tag;
  array_t<npart_t**> sp2_pld_i;

  EmissionPolicy(unsigned int        step,
                 npart_t             sp1_offset,
                 array_t<int*>&      sp1_i1,
                 array_t<int*>&      sp1_i2,
                 array_t<int*>&      sp1_i3,
                 array_t<prtldx_t*>& sp1_dx1,
                 array_t<prtldx_t*>& sp1_dx2,
                 array_t<prtldx_t*>& sp1_dx3,
                 array_t<real_t*>&   sp1_ux1,
                 array_t<real_t*>&   sp1_ux2,
                 array_t<real_t*>&   sp1_ux3,
                 array_t<real_t*>&   sp1_phi,
                 array_t<real_t*>&   sp1_weight,
                 array_t<short*>&    sp1_tag,
                 array_t<npart_t**>& sp1_pld_i,
                 npart_t             sp2_offset,
                 array_t<int*>&      sp2_i1,
                 array_t<int*>&      sp2_i2,
                 array_t<int*>&      sp2_i3,
                 array_t<prtldx_t*>& sp2_dx1,
                 array_t<prtldx_t*>& sp2_dx2,
                 array_t<prtldx_t*>& sp2_dx3,
                 array_t<real_t*>&   sp2_ux1,
                 array_t<real_t*>&   sp2_ux2,
                 array_t<real_t*>&   sp2_ux3,
                 array_t<real_t*>&   sp2_phi,
                 array_t<real_t*>&   sp2_weight,
                 array_t<short*>&    sp2_tag,
                 array_t<npart_t**>& sp2_pld_i)
    : step { step }
    , sp1_offset { sp1_offset }
    , sp1_i1 { sp1_i1 }
    , sp1_i2 { sp1_i2 }
    , sp1_i3 { sp1_i3 }
    , sp1_dx1 { sp1_dx1 }
    , sp1_dx2 { sp1_dx2 }
    , sp1_dx3 { sp1_dx3 }
    , sp1_ux1 { sp1_ux1 }
    , sp1_ux2 { sp1_ux2 }
    , sp1_ux3 { sp1_ux3 }
    , sp1_phi { sp1_phi }
    , sp1_weight { sp1_weight }
    , sp1_tag { sp1_tag }
    , sp1_pld_i { sp1_pld_i }
    , sp2_offset { sp2_offset }
    , sp2_i1 { sp2_i1 }
    , sp2_i2 { sp2_i2 }
    , sp2_i3 { sp2_i3 }
    , sp2_dx1 { sp2_dx1 }
    , sp2_dx2 { sp2_dx2 }
    , sp2_dx3 { sp2_dx3 }
    , sp2_ux1 { sp2_ux1 }
    , sp2_ux2 { sp2_ux2 }
    , sp2_ux3 { sp2_ux3 }
    , sp2_phi { sp2_phi }
    , sp2_weight { sp2_weight }
    , sp2_tag { sp2_tag }
    , sp2_pld_i { sp2_pld_i } {}

  Inline auto shouldEmit(const coord_t<M::PrtlDim>& x_Cd,
                         const coord_t<M::PrtlDim>& x_Ph,
                         const vec_t<Dim::_3D>&     u_Ph,
                         const vec_t<Dim::_3D>&,
                         const vec_t<Dim::_3D>&,
                         vec_t<Dim::_3D>& delta_u_Ph,
                         Payload& payload) const -> Kokkos::pair<bool, bool> {
    delta_u_Ph[0] = -TWO;
    delta_u_Ph[1] = ZERO;
    delta_u_Ph[2] = ZERO;
    return { true, false };
  }

  Inline auto emit(const tuple_t<int, M::Dim>&      xi_Cd,
                   const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                   const vec_t<Dim::_3D>&           direction,
                   real_t,
                   real_t,
                   const Payload&) const -> void {
    if (step % 2u == 0u) {
      const auto sp1_inj_index = Kokkos::atomic_fetch_add(&sp1_idx(), 1);
      kernel::InjectParticle<M::Dim, M::CoordType, false>(
        sp1_offset + sp1_inj_index,
        sp1_i1,
        sp1_i2,
        sp1_i3,
        sp1_dx1,
        sp1_dx2,
        sp1_dx3,
        sp1_ux1,
        sp1_ux2,
        sp1_ux3,
        sp1_phi,
        sp1_weight,
        sp1_tag,
        sp1_pld_i,
        xi_Cd,
        dxi_Cd,
        { direction[0], direction[1], direction[2] });
    } else if (step != 3u) {
      const auto sp2_inj_index = Kokkos::atomic_fetch_add(&sp2_idx(), 1);
      kernel::InjectParticle<M::Dim, M::CoordType, false>(
        sp2_offset + sp2_inj_index,
        sp2_i1,
        sp2_i2,
        sp2_i3,
        sp2_dx1,
        sp2_dx2,
        sp2_dx3,
        sp2_ux1,
        sp2_ux2,
        sp2_ux3,
        sp2_phi,
        sp2_weight,
        sp2_tag,
        sp2_pld_i,
        xi_Cd,
        dxi_Cd,
        { HALF * direction[0], HALF * direction[1], HALF * direction[2] });
    }
  }

  auto emitted_species_indices() const -> std::vector<spidx_t> {
    return { 2u, 3u };
  }

  auto numbers_injected() const -> std::vector<npart_t> {
    auto sp1_idx_h = Kokkos::create_mirror_view(sp1_idx);
    Kokkos::deep_copy(sp1_idx_h, sp1_idx);
    auto sp2_idx_h = Kokkos::create_mirror_view(sp2_idx);
    Kokkos::deep_copy(sp2_idx_h, sp2_idx);
    return { sp1_idx_h(), sp2_idx_h() };
  }
};

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);

  try {
    using namespace ntt;
    metric::Minkowski<Dim::_1D> metric { { 128u }, { { 0.0, 1.0 } }, {} };

    const auto delta_t = metric.dxMin() * 0.5;
    auto emitting_species = Particles<decltype(metric)::Dim, decltype(metric)::CoordType>(
      1,
      "emitter",
      0.0f,
      0.0f,
      100u,
      0u,
      0u,
      ParticlePusher::PHOTON,
      false,
      RadiativeDrag::NONE,
      EmissionType::CUSTOM,
      0u,
      0u);
    auto emitting_tag_h = Kokkos::create_mirror_view(emitting_species.tag);
    emitting_tag_h(0)   = ParticleTag::alive;
    emitting_tag_h(1)   = ParticleTag::alive;
    Kokkos::deep_copy(emitting_species.tag, emitting_tag_h);

    auto emitted_species_1 = Particles<decltype(metric)::Dim, decltype(metric)::CoordType>(
      2,
      "emitted_1",
      1.0f,
      1.0f,
      100u,
      0u,
      0u,
      ParticlePusher::BORIS,
      false,
      RadiativeDrag::NONE,
      EmissionType::NONE,
      0u,
      0u);
    auto emitted_species_2 = Particles<decltype(metric)::Dim, decltype(metric)::CoordType>(
      3,
      "emitted_2",
      1.0f,
      -1.0f,
      100u,
      0u,
      0u,
      ParticlePusher::BORIS,
      false,
      RadiativeDrag::NONE,
      EmissionType::NONE,
      0u,
      0u);

    auto pusher_arrays = emitting_species.PusherKernelArrays();

    kernel::sr::PusherParams pusher_params {};
    pusher_params.pusher_flags = emitting_species.pusher();
    pusher_params.radiative_drag_flags = emitting_species.radiative_drag_flags();
    pusher_params.mass       = emitting_species.mass();
    pusher_params.charge     = emitting_species.charge();
    pusher_params.dt         = delta_t;
    pusher_params.omegaB0    = 1.0f;
    pusher_params.ni1        = 128u;
    pusher_params.ni2        = 1u;
    pusher_params.ni3        = 1u;
    pusher_params.boundaries = {
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
    };

    ndfield_t<Dim::_1D, 6> EB { "EB", 128u + 2u * N_GHOSTS };
    const auto             no_custom_update =
      kernel::sr::NoCustomPrtlUpdate_t<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>> {};

    for (auto step = 0u; step < 7u; ++step) {
      pusher_params.time   = static_cast<simtime_t>(step) * delta_t;
      auto emission_policy = EmissionPolicy<decltype(metric)> {
        step,
        emitted_species_1.npart(),
        emitted_species_1.i1,
        emitted_species_1.i2,
        emitted_species_1.i3,
        emitted_species_1.dx1,
        emitted_species_1.dx2,
        emitted_species_1.dx3,
        emitted_species_1.ux1,
        emitted_species_1.ux2,
        emitted_species_1.ux3,
        emitted_species_1.phi,
        emitted_species_1.weight,
        emitted_species_1.tag,
        emitted_species_1.pld_i,
        emitted_species_2.npart(),
        emitted_species_2.i1,
        emitted_species_2.i2,
        emitted_species_2.i3,
        emitted_species_2.dx1,
        emitted_species_2.dx2,
        emitted_species_2.dx3,
        emitted_species_2.ux1,
        emitted_species_2.ux2,
        emitted_species_2.ux3,
        emitted_species_2.phi,
        emitted_species_2.weight,
        emitted_species_2.tag,
        emitted_species_2.pld_i
      };

      Kokkos::parallel_for(
        "ParticlePusher",
        2u,
        kernel::sr::Pusher_kernel<decltype(metric),
                                  kernel::traits::external::NoPolicy_t,
                                  false,
                                  decltype(emission_policy),
                                  decltype(no_custom_update)>(
          pusher_params,
          pusher_arrays,
          EB,
          metric,
          kernel::traits::external::NoPolicy_t {},
          emission_policy,
          no_custom_update));
      const auto n_injected = emission_policy.numbers_injected();
      emitted_species_1.set_counter(emitted_species_1.counter() + n_injected[0]);
      emitted_species_1.set_npart(emitted_species_1.npart() + n_injected[0]);
      emitted_species_2.set_counter(emitted_species_2.counter() + n_injected[1]);
      emitted_species_2.set_npart(emitted_species_2.npart() + n_injected[1]);
    }
    raise::ErrorIf(emitted_species_1.npart() != 8u,
                   "Unexpected number of particles for emitted species 1",
                   HERE);
    raise::ErrorIf(emitted_species_2.npart() != 4u,
                   "Unexpected number of particles for emitted species 2",
                   HERE);

    auto sp1_tag_h = Kokkos::create_mirror_view(emitted_species_1.tag);
    Kokkos::deep_copy(sp1_tag_h, emitted_species_1.tag);
    auto sp1_ux1_h = Kokkos::create_mirror_view(emitted_species_1.ux1);
    Kokkos::deep_copy(sp1_ux1_h, emitted_species_1.ux1);

    auto sp2_tag_h = Kokkos::create_mirror_view(emitted_species_2.tag);
    Kokkos::deep_copy(sp2_tag_h, emitted_species_2.tag);
    auto sp2_ux1_h = Kokkos::create_mirror_view(emitted_species_2.ux1);
    Kokkos::deep_copy(sp2_ux1_h, emitted_species_2.ux1);

    for (auto i = 0u; i < emitted_species_1.npart(); ++i) {
      raise::ErrorIf(sp1_tag_h(i) != ParticleTag::alive,
                     "Unexpected tag value for emitted species 1",
                     HERE);
      raise::ErrorIf(sp1_ux1_h(i) != ONE,
                     "Unexpected ux1 value for emitted species 1",
                     HERE);
    }
    for (auto i = 0u; i < emitted_species_2.npart(); ++i) {
      raise::ErrorIf(sp2_tag_h(i) != ParticleTag::alive,
                     "Unexpected tag value for emitted species 1",
                     HERE);
      raise::ErrorIf(sp2_ux1_h(i) != HALF,
                     "Unexpected ux1 value for emitted species 1",
                     HERE);
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
