#ifndef ENGINES_SRPIC_CURRENTS_H
#define ENGINES_SRPIC_CURRENTS_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "metrics/traits.h"

#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"

namespace ntt {
  namespace srpic {

    template <class M, unsigned short O>
      requires metric::traits::HasD<M> && metric::traits::HasCoordType<M>
    void CallDepositKernel(const Particles<M::Dim, M::CoordType>& species,
                           const M&                               local_metric,
                           const scatter_ndfield_t<M::Dim, 3>&    scatter_cur,
                           real_t                                 dt) {
      Kokkos::parallel_for("CurrentsDeposit",
                           species.rangeActiveParticles(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, O>(
                             scatter_cur,
                             species.i1,
                             species.i2,
                             species.i3,
                             species.i1_prev,
                             species.i2_prev,
                             species.i3_prev,
                             species.dx1,
                             species.dx2,
                             species.dx3,
                             species.dx1_prev,
                             species.dx2_prev,
                             species.dx3_prev,
                             species.ux1,
                             species.ux2,
                             species.ux3,
                             species.phi,
                             species.weight,
                             species.tag,
                             local_metric,
                             (real_t)(species.charge()),
                             dt));
    }

    template <class M>
    void CurrentsDeposit(Domain<SimEngine::SRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      Kokkos::deep_copy(domain.fields.cur, ZERO);
      auto scatter_cur = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur);
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or
            (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
          continue;
        }
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);

        CallDepositKernel<M, SHAPE_ORDER>(species, domain.mesh.metric, scatter_cur, dt);
      }
      Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
    }

    template <class M>
      requires metric::traits::HasD<M> && metric::traits::HasCoordType<M>
    void CurrentsFilter(Metadomain<SimEngine::SRPIC, M>& metadomain,
                        Domain<SimEngine::SRPIC, M>&     domain,
                        const SimulationParams&          params) {
      logger::Checkpoint("Launching currents filtering kernels", HERE);
      auto       range   = srpic::RangeWithAxisBCs(domain);
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      tuple_t<ncells_t, M::Dim> size;
      if constexpr (M::Dim == Dim::_1D || M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[0] = domain.mesh.n_active(in::x1);
      }
      if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[1] = domain.mesh.n_active(in::x2);
      }
      if constexpr (M::Dim == Dim::_3D) {
        size[2] = domain.mesh.n_active(in::x3);
      }
      // !TODO: this needs to be done more efficiently
      for (auto i { 0u }; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        metadomain.CommunicateFields(domain, Comm::J);
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_CURRENTS_H
