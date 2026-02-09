#ifndef ENGINES_SRPIC_FIELDSOLVERS_H
#define ENGINES_SRPIC_FIELDSOLVERS_H

#include "enums.h"
#include "global.h"

#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "archetypes/traits.h"
#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

#include "kernels/ampere_mink.hpp"
#include "kernels/ampere_sr.hpp"
#include "kernels/faraday_mink.hpp"
#include "kernels/faraday_sr.hpp"

namespace ntt {
  namespace srpic {

    template <class M>
    void Faraday(Domain<SimEngine::SRPIC, M>& domain,
                 const prm::Parameters&       engine_params,
                 const SimulationParams&      params,
                 real_t                       fraction = ONE) {
      logger::Checkpoint("Launching Faraday kernel", HERE);
      const auto dt = engine_params.get<real_t>("dt");

      const auto dT = fraction *
                      params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto dx = math::sqrt(domain.mesh.metric.template h_<1, 1>({}));
        const auto deltax = params.template get<real_t>(
          "algorithms.fieldsolver.delta_x");
        const auto deltay = params.template get<real_t>(
          "algorithms.fieldsolver.delta_y");
        const auto betaxy = params.template get<real_t>(
          "algorithms.fieldsolver.beta_xy");
        const auto betayx = params.template get<real_t>(
          "algorithms.fieldsolver.beta_yx");
        const auto deltaz = params.template get<real_t>(
          "algorithms.fieldsolver.delta_z");
        const auto betaxz = params.template get<real_t>(
          "algorithms.fieldsolver.beta_xz");
        const auto betazx = params.template get<real_t>(
          "algorithms.fieldsolver.beta_zx");
        const auto betayz = params.template get<real_t>(
          "algorithms.fieldsolver.beta_yz");
        const auto betazy = params.template get<real_t>(
          "algorithms.fieldsolver.beta_zy");
        real_t coeff1, coeff2;
        if constexpr (M::Dim == Dim::_2D) {
          coeff1 = dT / SQR(dx);
          coeff2 = dT;
        } else {
          coeff1 = dT / dx;
          coeff2 = ZERO;
        }
        Kokkos::parallel_for("Faraday",
                             domain.mesh.rangeActiveCells(),
                             kernel::mink::Faraday_kernel<M::Dim>(domain.fields.em,
                                                                  coeff1,
                                                                  coeff2,
                                                                  deltax,
                                                                  deltay,
                                                                  betaxy,
                                                                  betayx,
                                                                  deltaz,
                                                                  betaxz,
                                                                  betazx,
                                                                  betayz,
                                                                  betazy));
      } else {
        Kokkos::parallel_for("Faraday",
                             domain.mesh.rangeActiveCells(),
                             kernel::sr::Faraday_kernel<M>(domain.fields.em,
                                                           domain.mesh.metric,
                                                           dT,
                                                           domain.mesh.flds_bc()));
      }
    }

    template <class M>
    void Ampere(Domain<SimEngine::SRPIC, M>& domain,
                const prm::Parameters&       engine_params,
                const SimulationParams&      params,
                real_t                       fraction = ONE) {
      logger::Checkpoint("Launching Ampere kernel", HERE);
      const auto dt = engine_params.get<real_t>("dt");

      const auto dT = fraction *
                      params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      auto range = RangeWithAxisBCs(domain);
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto dx = math::sqrt(domain.mesh.metric.template h_<1, 1>({}));
        real_t     coeff1, coeff2;
        if constexpr (M::Dim == Dim::_2D) {
          coeff1 = dT / SQR(dx);
          coeff2 = dT;
        } else {
          coeff1 = dT / dx;
          coeff2 = ZERO;
        }

        Kokkos::parallel_for(
          "Ampere",
          range,
          kernel::mink::Ampere_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
      } else {
        const auto ni2 = domain.mesh.n_active(in::x2);
        Kokkos::parallel_for("Ampere",
                             range,
                             kernel::sr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      }
    }

    template <class M, class PG>
    void CurrentsAmpere(Domain<SimEngine::SRPIC, M>& domain,
                        const prm::Parameters&       engine_params,
                        const SimulationParams&      params,
                        const PG&                    pgen) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
      const auto dt = engine_params.get<real_t>("dt");

      const auto q0 = params.template get<real_t>("scales.q0");
      const auto n0 = params.template get<real_t>("scales.n0");
      const auto B0 = params.template get<real_t>("scales.B0");
      if constexpr (M::CoordType == Coord::Cart) {
        // minkowski case
        const auto V0    = params.template get<real_t>("scales.V0");
        const auto ppc0  = params.template get<real_t>("particles.ppc0");
        const auto coeff = -dt * q0 / (B0 * V0);
        if constexpr (arch::traits::pgen::HasExtCurrent<PG>) {
          const std::vector<real_t> xmin { domain.mesh.extent(in::x1).first,
                                           domain.mesh.extent(in::x2).first,
                                           domain.mesh.extent(in::x3).first };
          const auto                ext_current = pgen.ext_current;
          const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});
          Kokkos::parallel_for(
            "Ampere",
            domain.mesh.rangeActiveCells(),
            kernel::mink::CurrentsAmpere_kernel<M::Dim, decltype(ext_current)>(
              domain.fields.em,
              domain.fields.cur,
              coeff,
              ppc0,
              ext_current,
              xmin,
              dx));
        } else {
          Kokkos::parallel_for(
            "Ampere",
            domain.mesh.rangeActiveCells(),
            kernel::mink::CurrentsAmpere_kernel<M::Dim>(domain.fields.em,
                                                        domain.fields.cur,
                                                        coeff,
                                                        ppc0));
        }
      } else {
        // non-minkowski
        const auto coeff = -dt * q0 * n0 / B0;
        auto       range = RangeWithAxisBCs(domain);
        const auto ni2   = domain.mesh.n_active(in::x2);
        Kokkos::parallel_for(
          "Ampere",
          range,
          kernel::sr::CurrentsAmpere_kernel<M>(domain.fields.em,
                                               domain.fields.cur,
                                               domain.mesh.metric,
                                               coeff,
                                               ONE / n0,
                                               ni2,
                                               domain.mesh.flds_bc()));
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_FIELDSOLVERS_H
