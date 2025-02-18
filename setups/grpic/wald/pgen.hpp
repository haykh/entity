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
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_, real_t m_eps) : metric { metric_ }, m_eps { m_eps } {}

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1])) {
        return ONE;
      } else {
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1])) {
        return ZERO;
      } else {
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const M      metric;
    const real_t m_eps;
  };

  template <SimEngine::type S, class M>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    PointDistribution(const std::vector<real_t>& xi_min,
                      const std::vector<real_t>& xi_max,
                      const real_t               sigma_thr,
                      const real_t               dens_thr,
                      const SimulationParams&    params,
                      Domain<S, M>*              domain_ptr)
      : arch::SpatialDistribution<S, M> { domain_ptr->mesh.metric }
      , metric { domain_ptr->mesh.metric }
      , EM { domain_ptr->fields.em }
      , density { domain_ptr->fields.buff }
      , sigma_thr { sigma_thr }
      , dens_thr { dens_thr } {
      std::copy(xi_min.begin(), xi_min.end(), x_min);
      std::copy(xi_max.begin(), xi_max.end(), x_max);

      std::vector<unsigned short> specs {};
      for (auto& sp : domain_ptr->species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }

      Kokkos::deep_copy(density, ZERO);
      auto  scatter_buff = Kokkos::Experimental::create_scatter_view(density);
      // some parameters
      auto& mesh         = domain_ptr->mesh;
      const auto use_weights = params.template get<bool>(
        "particles.use_weights");
      const auto ni2    = mesh.n_active(in::x2);
      const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");

      for (const auto& sp : specs) {
        auto& prtl_spec = domain_ptr->species[sp - 1];
        // clang-format off
        Kokkos::parallel_for(
          "ComputeMoments",
          prtl_spec.rangeActiveParticles(),
          kernel::ParticleMoments_kernel<S, M, FldsID::Rho, 3>({}, scatter_buff, 0u,
                                                               prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
                                                               prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
                                                               prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
                                                               prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
                                                               prtl_spec.mass(), prtl_spec.charge(),
                                                               use_weights,
                                                               metric, mesh.flds_bc(),
                                                               ni2, inv_n0, ZERO));
        // clang-format on
      }
      Kokkos::Experimental::contribute(density, scatter_buff);
    }

    Inline auto sigma_crit(const coord_t<M::Dim>& x_Ph) const -> bool {
      coord_t<M::Dim> xi { ZERO };
      if constexpr (M::Dim == Dim::_2D) {
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        const auto i1 = static_cast<int>(xi[0]) + static_cast<int>(N_GHOSTS);
        const auto i2 = static_cast<int>(xi[1]) + static_cast<int>(N_GHOSTS);
        const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                        EM(i1, i2, em::bx2),
                                        EM(i1, i2, em::bx3) };
        vec_t<Dim::_3D>       B_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
        const auto bsqr =
          DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto dens = density(i1, i2, 0);
        return (bsqr > sigma_thr * dens) || (dens < dens_thr);
      }
      return false;
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d] and sigma_crit(x_Ph);
      }
      return fill ? ONE : ZERO;
    }

  private:
    tuple_t<real_t, M::Dim> x_min;
    tuple_t<real_t, M::Dim> x_max;
    const real_t            sigma_thr;
    const real_t            dens_thr;
    Domain<S, M>*           domain_ptr;
    ndfield_t<M::Dim, 3>    density;
    ndfield_t<M::Dim, 6>    EM;
    const M                 metric;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::GRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const real_t sigma0, sigma_max, multiplicity, nGJ, temperature, m_eps;

    InitFields<M, D> init_flds;

    inline PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , xi_min { p.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { p.template get<std::vector<real_t>>("setup.xi_max") }
      , sigma_max { p.template get<real_t>("setup.sigma_max") }
      , sigma0 { p.template get<real_t>("scales.sigma0") }
      , multiplicity { p.template get<real_t>("setup.multiplicity") }
      , nGJ { p.template get<real_t>("scales.B0") *
              SQR(p.template get<real_t>("scales.skindepth0")) }
      , temperature { p.template get<real_t>("setup.temperature") }
      , m_eps { p.template get<real_t>("setup.m_eps") }
      , init_flds { m.mesh().metric, m_eps } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature);
      const auto spatial_dist = PointDistribution<S, M>(xi_min,
                                                        xi_max,
                                                        sigma_max / sigma0,
                                                        multiplicity * nGJ,
                                                        params,
                                                        &local_domain);

      const auto injector =
        arch::NonUniformInjector<S, M, arch::Maxwellian, PointDistribution>(
          energy_dist,
          spatial_dist,
          { 1, 2 });
      arch::InjectNonUniform<S, M, decltype(injector)>(params,
                                                       local_domain,
                                                       injector,
                                                       1.0,
                                                       true);
    }

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& local_domain) {
      const auto energy_dist  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature);
      const auto spatial_dist = PointDistribution<S, M>(xi_min,
                                                        xi_max,
                                                        sigma_max / sigma0,
                                                        multiplicity * nGJ,
                                                        params,
                                                        &local_domain);

      const auto injector =
        arch::NonUniformInjector<S, M, arch::Maxwellian, PointDistribution>(
          energy_dist,
          spatial_dist,
          { 1, 2 });
      arch::InjectNonUniform<S, M, decltype(injector)>(params,
                                                       local_domain,
                                                       injector,
                                                       1.0,
                                                       true);
    }
  };

} // namespace user

#endif
