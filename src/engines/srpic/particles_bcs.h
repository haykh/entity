#ifndef ENGINES_SRPIC_PARTICLES_BCS_H
#define ENGINES_SRPIC_PARTICLES_BCS_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include "kernels/particle_moments.hpp"

namespace ntt {
  namespace srpic {

    template <class M>
      requires metric::traits::HasD<M> && metric::traits::HasCoordType<M>
    void AtmosphereParticlesIn(dir::direction_t<M::Dim>         direction,
                               Metadomain<SimEngine::SRPIC, M>& metadomain,
                               Domain<SimEngine::SRPIC, M>&     domain,
                               const SimulationParams&          params,
                               InjTags                          tags) {
      const auto [sign, dim, xg_min, xg_max] = srpic::GetAtmosphereExtent(
        direction,
        metadomain.mesh().metric,
        metadomain.mesh(),
        params);

      const auto x_surf = sign > 0 ? xg_min : xg_max;
      const auto ds     = params.template get<real_t>(
        "grid.boundaries.atmosphere.ds");
      const auto temp = params.template get<real_t>(
        "grid.boundaries.atmosphere.temperature");
      const auto height = params.template get<real_t>(
        "grid.boundaries.atmosphere.height");
      const auto species = params.template get<std::pair<spidx_t, spidx_t>>(
        "grid.boundaries.atmosphere.species");
      const auto nmax = params.template get<real_t>(
        "grid.boundaries.atmosphere.density");

      Kokkos::deep_copy(domain.fields.bckp, ZERO);
      auto scatter_bckp = Kokkos::Experimental::create_scatter_view(
        domain.fields.bckp);
      const auto use_weights = M::CoordType != Coord::Cart;
      const auto ni2         = domain.mesh.n_active(in::x2);
      const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");

      // compute the density of the two species
      if (tags & Inj::AssumeEmpty) {
        if constexpr (M::Dim == Dim::_1D) {
          Kokkos::deep_copy(
            Kokkos::subview(domain.fields.bckp, Kokkos::ALL, std::make_pair(0, 1)),
            ZERO);
        } else if constexpr (M::Dim == Dim::_2D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.bckp,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            std::make_pair(0, 1)),
                            ZERO);
        } else if constexpr (M::Dim == Dim::_3D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.bckp,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            Kokkos::ALL,
                                            std::make_pair(0, 1)),
                            ZERO);
        }
      } else {
        for (const auto& sp :
             std::vector<spidx_t> { species.first, species.second }) {
          auto& prtl_spec = domain.species[sp - 1];
          if (prtl_spec.npart() == 0) {
            continue;
          }
          // clang-format off
          Kokkos::parallel_for(
            "ComputeMoments",
            prtl_spec.rangeActiveParticles(),
            kernel::ParticleMoments_kernel<SimEngine::SRPIC, M, FldsID::Rho, 6>(
              {}, scatter_bckp, 0,
              prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
              prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
              prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
              prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
              prtl_spec.mass(), prtl_spec.charge(),
              use_weights,
              domain.mesh.metric, domain.mesh.flds_bc(),
              ni2, inv_n0, 0));
          // clang-format on
          prtl_spec.set_unsorted();
        }
        Kokkos::Experimental::contribute(domain.fields.bckp, scatter_bckp);
        metadomain.SynchronizeFields(domain, Comm::Bckp, { 0, 1 });
      }

      const auto maxwellian = arch::Maxwellian<SimEngine::SRPIC, M> {
        domain.mesh.metric,
        domain.random_pool(),
        temp
      };

      if (dim == in::x1) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x1> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x1> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else if (dim == in::x2) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x2> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x2> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else if (dim == in::x3) {
        if (sign > 0) {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, true, in::x3> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        } else {
          auto target_density =
            arch::AtmosphereDensityProfile<M::Dim, M::CoordType, false, in::x3> {
              nmax,
              height,
              x_surf,
              ds
            };
          const auto spatial_dist =
            arch::Replenish<SimEngine::SRPIC, M, 6, decltype(target_density)> {
              domain.mesh.metric,
              domain.fields.bckp,
              0,
              target_density,
              nmax
            };
          arch::InjectNonUniform<SimEngine::SRPIC,
                                 M,
                                 decltype(maxwellian),
                                 decltype(maxwellian),
                                 decltype(spatial_dist)>(
            params,
            domain,
            { species.first, species.second },
            { maxwellian, maxwellian },
            spatial_dist,
            nmax,
            use_weights);
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
      return;
    }

    template <class M>
      requires metric::traits::HasD<M>
    void ParticleInjector(Metadomain<SimEngine::SRPIC, M>& metadomain,
                          Domain<SimEngine::SRPIC, M>&     domain,
                          const SimulationParams&          params,
                          InjTags                          tags = Inj::None) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (metadomain.mesh().prtl_bc_in(direction) == PrtlBC::ATMOSPHERE) {
          AtmosphereParticlesIn(direction, metadomain, domain, params, tags);
        }
      }
    }

  } // namespace srpic
} // namespace ntt

#endif
