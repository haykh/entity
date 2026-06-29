/**
 * @file kernels/particle_moments.hpp
 * @brief Algorithm for computing different moments from particle distribution
 * @implements
 *   - kernel::ParticleMoments_kernel<>
 *   - kernel::ParticleMomentsNew_kernel<>
 *   - kernel::NormalizeVectorByRho_kernel<>
 *   - kernel::Normalize4VelocityByNorm_kernel<>
 *   - kernel::Transform4VelocitySpatialToPhysical_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_PARTICLE_MOMENTS_HPP
#define KERNELS_PARTICLE_MOMENTS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"

#include <cstdint>
#include <vector>

namespace kernel {
  using namespace ntt;

  namespace particle_moment_functor {

    template <class MF>
    concept IsValid = requires(const MF&              fm,
                               const ParticleArrays&  prtls,
                               float                  mass,
                               float                  charge,
                               prtlidx_t              p,
                               list_t<real_t, MF::N>& buffer) {
      { fm(prtls, mass, charge, p, buffer) } -> std::same_as<void>;
    };

    template <class MF>
    concept HasN = requires {
      { MF::N } -> std::convertible_to<uint8_t>;
    };

  } // namespace particle_moment_functor

  template <class MF>
  concept ParticleMomentsFunctor = particle_moment_functor::IsValid<MF> &&
                                   particle_moment_functor::HasN<MF>;

  template <FldsID::type F>
  auto get_contrib(float mass, float charge) -> real_t {
    if constexpr (F == FldsID::Rho) {
      return mass;
    } else if constexpr (F == FldsID::Charge) {
      return charge;
    } else {
      return ONE;
    }
  }

  template <SimEngine::type S, MetricClass M, FldsID::type F, uint8_t N>
  class ParticleMoments_kernel {
    static constexpr auto D = M::Dim;

    static_assert((F == FldsID::Rho) || (F == FldsID::Charge) || (F == FldsID::N) ||
                    (F == FldsID::Nppc) || (F == FldsID::T) || (F == FldsID::V),
                  "Invalid field ID");

    const uint8_t           c1, c2;
    scatter_ndfield_t<D, N> Buff;
    const idx_t             buff_idx;
    const ParticleArrays    particles;
    const float             mass;
    const float             charge;
    const bool              use_weights;
    const M                 metric;
    const int               ni2;
    const real_t            inv_n0;

    const uint8_t                 order;
    const uint8_t                 window;
    const OutputSmoothingTypeFlag smoothing;

    const real_t contrib;
    bool         is_axis_i2min { false }, is_axis_i2max { false };

  public:
    ParticleMoments_kernel(
      const std::vector<uint8_t>&            components,
      const scatter_ndfield_t<D, N>&         scatter_buff,
      idx_t                                  buff_idx,
      const Particles<M::Dim, M::CoordType>& particles,
      bool                                   use_weights,
      const M&                               metric,
      const boundaries_t<FldsBC>&            boundaries,
      ncells_t                               ni2,
      real_t                                 inv_n0,
      uint8_t                                order = 0u,
      OutputSmoothingTypeFlag smoothing = OutputSmoothingType::SPLINE)
      : c1 { not components.empty() ? components[0] : static_cast<uint8_t>(0) }
      , c2 { (components.size() == 2) ? components[1] : static_cast<uint8_t>(0) }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , particles { static_cast<const ParticleArrays&>(particles) }
      , mass { particles.mass() }
      , charge { particles.charge() }
      , use_weights { use_weights }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , inv_n0 { inv_n0 }
      , order { order }
      , smoothing { smoothing }
      , window { static_cast<uint8_t>(math::ceil(static_cast<float>(order) / 2.0f)) }
      , contrib { get_contrib<F>(mass, charge) } {
      raise::ErrorIf(buff_idx >= N, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      raise::ErrorIf(((F == FldsID::Rho) || (F == FldsID::Charge)) && (mass == ZERO),
                     "Rho & Charge for massless particles not defined",
                     HERE);
      if constexpr ((M::CoordType != Coord::Cartesian) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline auto shapeFunction(real_t delta_x) const -> real_t {
      if (smoothing == OutputSmoothingType::SPLINE) {
        if (order == 0) {
          return ONE;
        } else if (order == 1) {
          return prtl_shape::S1(delta_x);
        } else if (order == 2) {
          return prtl_shape::S2(delta_x);
        } else if (order == 3) {
          return prtl_shape::S3(delta_x);
        } else if (order == 4) {
          return prtl_shape::S4(delta_x);
        } else if (order == 5) {
          return prtl_shape::S5(delta_x);
        } else if (order == 6) {
          return prtl_shape::S6(delta_x);
        } else if (order == 7) {
          return prtl_shape::S7(delta_x);
        } else if (order == 8) {
          return prtl_shape::S8(delta_x);
        } else if (order == 9) {
          return prtl_shape::S9(delta_x);
        } else if (order == 10) {
          return prtl_shape::S10(delta_x);
        } else if (order == 11) {
          return prtl_shape::S11(delta_x);
        } else {
          raise::KernelError(HERE, "Unsupported shape function order");
          return ZERO;
        }
      } else if (smoothing == OutputSmoothingType::CONST) {
        return ONE / (TWO * static_cast<real_t>(window) + ONE);
      } else {
        raise::KernelError(HERE, "Unsupported smoothing method");
        return ZERO;
      }
    }

    Inline auto computeStressEnergyComponent(prtlidx_t p) const -> real_t {
      real_t          u0 { ZERO };
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (S == SimEngine::SRPIC) {
        // stress-energy tensor for SR is computed in the tetrad (hatted) basis
        if constexpr (M::CoordType == Coord::Cartesian) {
          u_Phys[0] = particles.ux1(p);
          u_Phys[1] = particles.ux2(p);
          u_Phys[2] = particles.ux3(p);
        } else {
          static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
          coord_t<M::PrtlDim> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                      static_cast<real_t>(particles.dx1(p));
          x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                      static_cast<real_t>(particles.dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(particles.i3(p)) +
                        static_cast<real_t>(particles.dx3(p));
          } else {
            x_Code[2] = particles.phi(p);
          }
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            x_Code,
            { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
            u_Phys);
        }
        u0 = (mass == ZERO)
               ? (NORM(u_Phys[0], u_Phys[1], u_Phys[2]))
               : (math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2])));
      } else if constexpr (S == SimEngine::GRPIC) {
        // stress-energy tensor for GR is computed in contravariant basis
        static_assert(D != Dim::_1D, "GRPIC 1D");
        coord_t<D> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                    static_cast<real_t>(particles.dx1(p));
        x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                    static_cast<real_t>(particles.dx2(p));
        if constexpr (D == Dim::_3D) {
          x_Code[2] = static_cast<real_t>(particles.i3(p)) +
                      static_cast<real_t>(particles.dx3(p));
        }
        // raise full covariant 4-vector to get correct contravariant u^i
        // u^i != h^{ij} u_j
        const real_t    u_0_cov { metric.u_0(
          x_Code,
          { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          (mass == ZERO) ? ZERO : ONE) };
        vec_t<Dim::_4D> u_cntrv_4d { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(
          x_Code,
          { u_0_cov, particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          u_cntrv_4d);
        // in GR: u^0 = Gamma/alpha
        u0 = u_cntrv_4d[0];
        metric.template transform<Idx::U, Idx::PU>(
          x_Code,
          { u_cntrv_4d[1], u_cntrv_4d[2], u_cntrv_4d[3] },
          u_Phys);
      } else {
        raise::KernelError(
          HERE,
          "computeStressEnergyComponent called for non-SRPIC/GRPIC");
      }
      auto T_component = (mass == ZERO ? ONE : mass) / u0;
      for (const auto& c : { c1, c2 }) {
        if (c > 0) {
          T_component *= u_Phys[c - 1];
        } else {
          T_component *= u0;
        }
      }
      return T_component;
    }

    Inline auto computeBulk3VelocityTimesMass(prtlidx_t p) const -> real_t {
      real_t          u0 { ZERO };
      // for bulk 3vel (tetrad basis)
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (M::CoordType == Coord::Cartesian) {
        u_Phys[0] = particles.ux1(p);
        u_Phys[1] = particles.ux2(p);
        u_Phys[2] = particles.ux3(p);
      } else {
        coord_t<M::PrtlDim> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                    static_cast<real_t>(particles.dx1(p));
        x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                    static_cast<real_t>(particles.dx2(p));
        if constexpr (D == Dim::_3D) {
          x_Code[2] = static_cast<real_t>(particles.i3(p)) +
                      static_cast<real_t>(particles.dx3(p));
        } else {
          x_Code[2] = particles.phi(p);
        }
        metric.template transform_xyz<Idx::XYZ, Idx::T>(
          x_Code,
          { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          u_Phys);
      }
      if (mass == ZERO) {
        u0 = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
      } else {
        u0 = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
      }
      if (c1 > 0u) {
        return (mass == ZERO ? ONE : mass) * u_Phys[c1 - 1] / u0;
      } else {
        return (mass == ZERO ? ONE : (mass * math::sqrt(ONE - SQR(ONE / u0))));
      }
    }

    Inline auto computeEckartVelocityFluxComponent(prtlidx_t p) const -> real_t {
      // GR: Eckart frame flux N^μ = m * u^μ / u^0
      static_assert(D != Dim::_1D, "GRPIC 1D");
      coord_t<D> x_Code { ZERO };
      x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                  static_cast<real_t>(particles.dx1(p));
      x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                  static_cast<real_t>(particles.dx2(p));
      if constexpr (D == Dim::_3D) {
        x_Code[2] = static_cast<real_t>(particles.i3(p)) +
                    static_cast<real_t>(particles.dx3(p));
      }
      // raise full covariant 4-vector to get correct contravariant u^μ
      // u^i != h^{ij} u_j
      const real_t    u_0_cov { metric.u_0(
        x_Code,
        { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
        (mass == ZERO) ? ZERO : ONE) };
      vec_t<Dim::_4D> u_cntrv_4d { ZERO };
      metric.template transform_4d<Idx::D, Idx::U>(
        x_Code,
        { u_0_cov, particles.ux1(p), particles.ux2(p), particles.ux3(p) },
        u_cntrv_4d);
      const real_t u0 { u_cntrv_4d[0] };
      // Deposit flux N^μ = mass * u^μ / u^0
      if (c1 == 0) {
        return (mass == ZERO ? ONE : mass);
      } else {
        return (mass == ZERO ? ONE : mass) * u_cntrv_4d[c1] / u0;
      }
    }

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == FldsID::T) {
        coeff = computeStressEnergyComponent(p);
      } else if constexpr (F == FldsID::V) {
        if constexpr (S == SimEngine::GRPIC) {
          coeff = computeEckartVelocityFluxComponent(p);
        } else {
          coeff = computeBulk3VelocityTimesMass(p);
        }
      } else {
        // for other cases, use the `contrib` defined above
        coeff = contrib;
      }
      if constexpr (F != FldsID::Nppc) {
        // for nppc calculation ...
        // ... do not take volume, weights or smoothing into account
        if constexpr (D == Dim::_1D) {
          coeff *= inv_n0 / metric.sqrt_det_h(
                              { static_cast<real_t>(particles.i1(p)) + HALF });
        } else if constexpr (D == Dim::_2D) {
          coeff *= inv_n0 / metric.sqrt_det_h(
                              { static_cast<real_t>(particles.i1(p)) + HALF,
                                static_cast<real_t>(particles.i2(p)) + HALF });
        } else if constexpr (D == Dim::_3D) {
          coeff *= inv_n0 / metric.sqrt_det_h(
                              { static_cast<real_t>(particles.i1(p)) + HALF,
                                static_cast<real_t>(particles.i2(p)) + HALF,
                                static_cast<real_t>(particles.i3(p)) + HALF });
        }
        if (use_weights) {
          coeff *= particles.weight(p);
        }
      }
      auto buff_access = Buff.access();
      if constexpr (D == Dim::_1D) {
        for (auto di1 { -window }; di1 <= window; ++di1) {
          const real_t delta_x1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                            (static_cast<real_t>(di1) + HALF));
          buff_access(particles.i1(p) + di1 + N_GHOSTS,
                      buff_idx) += coeff * shapeFunction(delta_x1);
        }
      } else if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            const real_t delta_x1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                              (static_cast<real_t>(di1) + HALF));
            const real_t delta_x2 = math::abs(static_cast<real_t>(particles.dx2(p)) -
                                              (static_cast<real_t>(di2) + HALF));
            const auto shape_coeff = shapeFunction(delta_x1) *
                                     shapeFunction(delta_x2);
            if constexpr (M::CoordType == Coord::Cartesian) {
              buff_access(particles.i1(p) + di1 + N_GHOSTS,
                          particles.i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff * shape_coeff;
            } else {
              // reflect contribution at axes
              if (is_axis_i2min && (particles.i2(p) + di2 < 0)) {
                buff_access(particles.i1(p) + di1 + N_GHOSTS,
                            N_GHOSTS - (particles.i2(p) + di2),
                            buff_idx) += coeff * shape_coeff;
              } else if (is_axis_i2max && (particles.i2(p) + di2 >= ni2)) {
                buff_access(particles.i1(p) + di1 + N_GHOSTS,
                            2 * ni2 - (particles.i2(p) + di2) + N_GHOSTS,
                            buff_idx) += coeff * shape_coeff;
              } else {
                buff_access(particles.i1(p) + di1 + N_GHOSTS,
                            particles.i2(p) + di2 + N_GHOSTS,
                            buff_idx) += coeff * shape_coeff;
              }
            }
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (auto di3 { -window }; di3 <= window; ++di3) {
          for (auto di2 { -window }; di2 <= window; ++di2) {
            for (auto di1 { -window }; di1 <= window; ++di1) {
              const auto delta_x1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                              (static_cast<real_t>(di1) + HALF));
              const auto delta_x2 = math::abs(static_cast<real_t>(particles.dx2(p)) -
                                              (static_cast<real_t>(di2) + HALF));
              const auto delta_x3 = math::abs(static_cast<real_t>(particles.dx3(p)) -
                                              (static_cast<real_t>(di3) + HALF));
              const auto shape_coeff = shapeFunction(delta_x1) *
                                       shapeFunction(delta_x2) *
                                       shapeFunction(delta_x3);
              if constexpr (M::CoordType == Coord::Cartesian) {
                buff_access(particles.i1(p) + di1 + N_GHOSTS,
                            particles.i2(p) + di2 + N_GHOSTS,
                            particles.i3(p) + di3 + N_GHOSTS,
                            buff_idx) += coeff * shape_coeff;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (particles.i2(p) + di2 < 0)) {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              N_GHOSTS - (particles.i2(p) + di2),
                              particles.i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                } else if (is_axis_i2max && (particles.i2(p) + di2 >= ni2)) {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              2 * ni2 - (particles.i2(p) + di2) + N_GHOSTS,
                              particles.i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                } else {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              particles.i2(p) + di2 + N_GHOSTS,
                              particles.i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                }
              }
            }
          }
        }
      }
    }
  };

  /**
   * @brief Generic moment-deposition kernel.
   *
   * Computes a per-particle scalar via the supplied @p Func and deposits it onto
   * a scatter buffer, automatically applying volume normalization, weighting,
   * smoothing (shape function) and axis reflection. @p Func is any device-
   * callable object with signature
   *   `Inline auto operator()(const ParticleArrays&, prtlidx_t, const M&) const
   *      -> real_t`
   * i.e. it receives the particle container, the particle index, and the metric,
   * and returns the raw (un-normalized, un-weighted, un-smoothed) contribution of
   * that particle to the moment.
   *
   * @tparam M    Metric type
   * @tparam N    Last dimension of the buffer
   * @tparam Func Per-particle contribution functor (deduced)
   */
  template <MetricClass M, uint8_t N, ParticleMomentsFunctor MF>
  class ParticleMomentsNew_kernel {
    static_assert(
      MF::N <= N,
      "Buffer size N must be >= number of components N deposited by Func");
    static constexpr auto D = M::Dim;

    const MF                          func;
    scatter_ndfield_t<D, N>           Buff;
    const Kokkos::Array<idx_t, MF::N> buff_indices;
    const ParticleArrays       particles;
    const float                mass, charge;
    const bool                 use_weights;
    const bool                 apply_norm;
    const M                    metric;
    const int                  ni2;
    const real_t               inv_n0;

    const uint8_t                 order;
    const uint8_t                 window;
    const OutputSmoothingTypeFlag smoothing;

    bool is_axis_i2min { false }, is_axis_i2max { false };

  public:
    ParticleMomentsNew_kernel(
      const MF&                              func,
      const scatter_ndfield_t<D, N>&         scatter_buff,
      const Kokkos::Array<idx_t, MF::N>&     buff_indices,
      const Particles<M::Dim, M::CoordType>& particles,
      bool                                   use_weights,
      const M&                               metric,
      const boundaries_t<FldsBC>&            boundaries,
      ncells_t                               ni2,
      real_t                                 inv_n0,
      uint8_t                                order = 0u,
      OutputSmoothingTypeFlag smoothing  = OutputSmoothingType::SPLINE,
      bool                    apply_norm = true)
      : func { func }
      , Buff { scatter_buff }
      , buff_indices { buff_indices }
      , particles { static_cast<const ParticleArrays&>(particles) }
      , mass { particles.mass() }
      , charge { particles.charge() }
      , use_weights { use_weights }
      , apply_norm { apply_norm }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , inv_n0 { inv_n0 }
      , order { order }
      , smoothing { smoothing }
      , window { static_cast<uint8_t>(
          math::ceil(static_cast<float>(order) / 2.0f)) } {
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      if constexpr ((M::CoordType != Coord::Cartesian) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline auto shapeFunction(real_t delta_x) const -> real_t {
      if (smoothing == OutputSmoothingType::SPLINE) {
        if (order == 0) {
          return ONE;
        } else if (order == 1) {
          return prtl_shape::S1(delta_x);
        } else if (order == 2) {
          return prtl_shape::S2(delta_x);
        } else if (order == 3) {
          return prtl_shape::S3(delta_x);
        } else if (order == 4) {
          return prtl_shape::S4(delta_x);
        } else if (order == 5) {
          return prtl_shape::S5(delta_x);
        } else if (order == 6) {
          return prtl_shape::S6(delta_x);
        } else if (order == 7) {
          return prtl_shape::S7(delta_x);
        } else if (order == 8) {
          return prtl_shape::S8(delta_x);
        } else if (order == 9) {
          return prtl_shape::S9(delta_x);
        } else if (order == 10) {
          return prtl_shape::S10(delta_x);
        } else if (order == 11) {
          return prtl_shape::S11(delta_x);
        } else {
          raise::KernelError(HERE, "Unsupported shape function order");
          return ZERO;
        }
      } else if (smoothing == OutputSmoothingType::CONST) {
        return ONE / (TWO * static_cast<real_t>(window) + ONE);
      } else {
        raise::KernelError(HERE, "Unsupported smoothing method");
        return ZERO;
      }
    }

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) == ParticleTag::dead) {
        return;
      }
      list_t<real_t, MF::N> contributions { ZERO };
      func(particles, mass, charge, p, contributions);
      for (uint8_t i = 0; i < MF::N; ++i) {
        // apply volume normalization and (optionally) particle weights;
        // skipped e.g. for nppc, which counts raw particles per cell
        if constexpr (D == Dim::_1D) {
          contributions[i] *= inv_n0 /
                              metric.sqrt_det_h(
                                { static_cast<real_t>(particles.i1(p)) + HALF });
        } else if constexpr (D == Dim::_2D) {
          contributions[i] *= inv_n0 /
                              metric.sqrt_det_h(
                                { static_cast<real_t>(particles.i1(p)) + HALF,
                                  static_cast<real_t>(particles.i2(p)) + HALF });
        } else if constexpr (D == Dim::_3D) {
          contributions[i] *= inv_n0 /
                              metric.sqrt_det_h(
                                { static_cast<real_t>(particles.i1(p)) + HALF,
                                  static_cast<real_t>(particles.i2(p)) + HALF,
                                  static_cast<real_t>(particles.i3(p)) + HALF });
        }
        if (use_weights) {
          contributions[i] *= particles.weight(p);
        }
      }
      auto buff_access = Buff.access();
      for (uint8_t i = 0; i < MF::N; ++i) {
        const auto coeff    = contributions[i];
        const auto buff_idx = buff_indices[i];
        if constexpr (D == Dim::_1D) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            const real_t delta_x1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                              (static_cast<real_t>(di1) + HALF));
            buff_access(particles.i1(p) + di1 + N_GHOSTS,
                        buff_idx) += coeff * shapeFunction(delta_x1);
          }
        } else if constexpr (D == Dim::_2D) {
          for (auto di2 { -window }; di2 <= window; ++di2) {
            for (auto di1 { -window }; di1 <= window; ++di1) {
              const real_t delta_x1 = math::abs(
                static_cast<real_t>(particles.dx1(p)) -
                (static_cast<real_t>(di1) + HALF));
              const real_t delta_x2 = math::abs(
                static_cast<real_t>(particles.dx2(p)) -
                (static_cast<real_t>(di2) + HALF));
              const auto shape_coeff = shapeFunction(delta_x1) *
                                       shapeFunction(delta_x2);
              if constexpr (M::CoordType == Coord::Cartesian) {
                buff_access(particles.i1(p) + di1 + N_GHOSTS,
                            particles.i2(p) + di2 + N_GHOSTS,
                            buff_idx) += coeff * shape_coeff;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (particles.i2(p) + di2 < 0)) {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              N_GHOSTS - (particles.i2(p) + di2),
                              buff_idx) += coeff * shape_coeff;
                } else if (is_axis_i2max && (particles.i2(p) + di2 >= ni2)) {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              2 * ni2 - (particles.i2(p) + di2) + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                } else {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              particles.i2(p) + di2 + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                }
              }
            }
          }
        } else if constexpr (D == Dim::_3D) {
          for (auto di3 { -window }; di3 <= window; ++di3) {
            for (auto di2 { -window }; di2 <= window; ++di2) {
              for (auto di1 { -window }; di1 <= window; ++di1) {
                const auto delta_x1 = math::abs(
                  static_cast<real_t>(particles.dx1(p)) -
                  (static_cast<real_t>(di1) + HALF));
                const auto delta_x2 = math::abs(
                  static_cast<real_t>(particles.dx2(p)) -
                  (static_cast<real_t>(di2) + HALF));
                const auto delta_x3 = math::abs(
                  static_cast<real_t>(particles.dx3(p)) -
                  (static_cast<real_t>(di3) + HALF));
                const auto shape_coeff = shapeFunction(delta_x1) *
                                         shapeFunction(delta_x2) *
                                         shapeFunction(delta_x3);
                if constexpr (M::CoordType == Coord::Cartesian) {
                  buff_access(particles.i1(p) + di1 + N_GHOSTS,
                              particles.i2(p) + di2 + N_GHOSTS,
                              particles.i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff * shape_coeff;
                } else {
                  // reflect contribution at axes
                  if (is_axis_i2min && (particles.i2(p) + di2 < 0)) {
                    buff_access(particles.i1(p) + di1 + N_GHOSTS,
                                N_GHOSTS - (particles.i2(p) + di2),
                                particles.i3(p) + di3 + N_GHOSTS,
                                buff_idx) += coeff * shape_coeff;
                  } else if (is_axis_i2max && (particles.i2(p) + di2 >= ni2)) {
                    buff_access(particles.i1(p) + di1 + N_GHOSTS,
                                2 * ni2 - (particles.i2(p) + di2) + N_GHOSTS,
                                particles.i3(p) + di3 + N_GHOSTS,
                                buff_idx) += coeff * shape_coeff;
                  } else {
                    buff_access(particles.i1(p) + di1 + N_GHOSTS,
                                particles.i2(p) + di2 + N_GHOSTS,
                                particles.i3(p) + di3 + N_GHOSTS,
                                buff_idx) += coeff * shape_coeff;
                  }
                }
              }
            }
          }
        }
      }
    }
  };

  template <Dimension D, uint8_t N>
  class NormalizeVectorByRho_kernel {
    const ndfield_t<D, N> Rho;
    ndfield_t<D, N>       Vector;
    const uint8_t         c_rho, c_v1, c_v2, c_v3;

  public:
    NormalizeVectorByRho_kernel(const ndfield_t<D, N>& rho,
                                const ndfield_t<D, N>& vector,
                                uint8_t                crho,
                                uint8_t                cv1,
                                uint8_t                cv2,
                                uint8_t                cv3)
      : Rho { rho }
      , Vector { vector }
      , c_rho { crho }
      , c_v1 { cv1 }
      , c_v2 { cv2 }
      , c_v3 { cv3 } {
      raise::ErrorIf(c_rho >= N or c_v1 >= N or c_v2 >= N or c_v3 >= N,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_rho == c_v1 or c_rho == c_v2 or c_rho == c_v3,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_v1 == c_v2 or c_v1 == c_v3 or c_v2 == c_v3,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (not cmp::AlmostZero(Rho(i1, c_rho))) {
          Vector(i1, c_v1) /= Rho(i1, c_rho);
          Vector(i1, c_v2) /= Rho(i1, c_rho);
          Vector(i1, c_v3) /= Rho(i1, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of NormalizeVectorByRho_kernel called for non-1D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (not cmp::AlmostZero(Rho(i1, i2, c_rho))) {
          Vector(i1, i2, c_v1) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v2) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v3) /= Rho(i1, i2, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of NormalizeVectorByRho_kernel called for non-2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (not cmp::AlmostZero(Rho(i1, i2, i3, c_rho))) {
          Vector(i1, i2, i3, c_v1) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v2) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v3) /= Rho(i1, i2, i3, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of NormalizeVectorByRho_kernel called for non-3D");
      }
    }
  };

  template <Dimension D, GRMetricClass M, uint8_t N>
  class Normalize4VelocityByNorm_kernel {
    // Normalizes 4-momentum flux to Eckart frame velocity
    // V^μ = N^μ / sqrt(-N_ν N^ν)
    const ndfield_t<D, N> Flux;           // momentum flux N^μ
    ndfield_t<D, N>       Vector;         // Eckart 4-velocity
    const uint8_t c_u0, c_u1, c_u2, c_u3; // 4-velocity component indices
    const M       metric;

  public:
    Normalize4VelocityByNorm_kernel(const ndfield_t<D, N>& flux,
                                    const ndfield_t<D, N>& vector,
                                    uint8_t                cu0,
                                    uint8_t                cu1,
                                    uint8_t                cu2,
                                    uint8_t                cu3,
                                    const M&               metric)
      : Flux { flux }
      , Vector { vector }
      , c_u0 { cu0 }
      , c_u1 { cu1 }
      , c_u2 { cu2 }
      , c_u3 { cu3 }
      , metric { metric } {
      raise::ErrorIf(c_u0 >= N or c_u1 >= N or c_u2 >= N or c_u3 >= N,
                     "Invalid component index",
                     HERE);
    }

    // ZAMO fallback for empty or pathological cells

    Inline void zamo_fallback_2d(cellidx_t         i1,
                                 cellidx_t         i2,
                                 const coord_t<D>& x_Code) const {
      if constexpr (D == Dim::_2D) {
        const real_t al      = metric.alpha(x_Code);
        Vector(i1, i2, c_u0) = ONE / al;
        Vector(i1, i2, c_u1) = -metric.beta1(x_Code) / al;
        Vector(i1, i2, c_u2) = ZERO;
        Vector(i1, i2, c_u3) = ZERO;
      } else {
        raise::KernelError(
          HERE,
          "2D fallback of Normalize4VelocityByNorm_kernel called for non-2D");
      }
    }

    Inline void zamo_fallback_3d(cellidx_t         i1,
                                 cellidx_t         i2,
                                 cellidx_t         i3,
                                 const coord_t<D>& x_Code) const {
      if constexpr (D == Dim::_3D) {
        const real_t al          = metric.alpha(x_Code);
        Vector(i1, i2, i3, c_u0) = ONE / al;
        Vector(i1, i2, i3, c_u1) = -metric.beta1(x_Code) / al;
        Vector(i1, i2, i3, c_u2) = ZERO;
        Vector(i1, i2, i3, c_u3) = ZERO;
      } else {
        raise::KernelError(
          HERE,
          "3D fallback of Normalize4VelocityByNorm_kernel called for non-3D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;

        vec_t<Dim::_4D> N_cntrv { ZERO, ZERO, ZERO, ZERO };
        N_cntrv[0] = Flux(i1, i2, c_u0);
        N_cntrv[1] = Flux(i1, i2, c_u1);
        N_cntrv[2] = Flux(i1, i2, c_u2);
        N_cntrv[3] = Flux(i1, i2, c_u3);

        // ZAMO fallback for empty cells or overflow (sqrt_det_h -> 0 near axis)
        if (cmp::AlmostZero(N_cntrv[0]) || not math::isfinite(N_cntrv[0])) {
          zamo_fallback_2d(i1, i2, x_Code);
          return;
        }

        vec_t<Dim::_4D> N_cov { ZERO, ZERO, ZERO, ZERO };
        // Compute N_i = g_ij N^j
        metric.template transform_4d<Idx::U, Idx::D>(x_Code, N_cntrv, N_cov);

        // Compute N_ν N^ν = g_μν N^μ N^ν (should be negative for timelike)
        real_t N_norm_sq { N_cov[0] * N_cntrv[0] + N_cov[1] * N_cntrv[1] +
                           N_cov[2] * N_cntrv[2] + N_cov[3] * N_cntrv[3] };

        // ZAMO fallback for spacelike, null, or NaN
        if (not(N_norm_sq < ZERO)) {
          zamo_fallback_2d(i1, i2, x_Code);
          return;
        }

        real_t norm = math::sqrt(-N_norm_sq);

        if (not cmp::AlmostZero(norm)) {
          Vector(i1, i2, c_u0) = N_cntrv[0] / norm;
          Vector(i1, i2, c_u1) = N_cntrv[1] / norm;
          Vector(i1, i2, c_u2) = N_cntrv[2] / norm;
          Vector(i1, i2, c_u3) = N_cntrv[3] / norm;
        } else {
          zamo_fallback_2d(i1, i2, x_Code);
        }
      } else {
        raise::KernelError(HERE, "2D implementation of Normalize4VelocityByNorm_kernel called for non-2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;
        x_Code[2] = COORD(i3) + HALF;

        vec_t<Dim::_4D> N_cntrv { ZERO, ZERO, ZERO, ZERO };
        N_cntrv[0] = Flux(i1, i2, i3, c_u0); // N^0
        N_cntrv[1] = Flux(i1, i2, i3, c_u1); // N^1
        N_cntrv[2] = Flux(i1, i2, i3, c_u2); // N^2
        N_cntrv[3] = Flux(i1, i2, i3, c_u3); // N^3

        // ZAMO fallback for empty cells or overflow (sqrt_det_h -> 0 near axis)
        if (cmp::AlmostZero(N_cntrv[0]) || not math::isfinite(N_cntrv[0])) {
          zamo_fallback_3d(i1, i2, i3, x_Code);
          return;
        }

        vec_t<Dim::_4D> N_cov { ZERO };
        // Compute N_i = g_ij N^j
        metric.template transform_4d<Idx::U, Idx::D>(x_Code, N_cntrv, N_cov);

        // Compute N_ν N^ν = g_μν N^μ N^ν (should be negative for timelike)
        real_t N_norm_sq { N_cov[0] * N_cntrv[0] + N_cov[1] * N_cntrv[1] +
                           N_cov[2] * N_cntrv[2] + N_cov[3] * N_cntrv[3] };

        // ZAMO fallback for spacelike, null, or NaN
        if (not(N_norm_sq < ZERO)) {
          zamo_fallback_3d(i1, i2, i3, x_Code);
          return;
        }

        real_t norm = math::sqrt(-N_norm_sq);

        if (not cmp::AlmostZero(norm)) {
          Vector(i1, i2, i3, c_u0) = N_cntrv[0] / norm;
          Vector(i1, i2, i3, c_u1) = N_cntrv[1] / norm;
          Vector(i1, i2, i3, c_u2) = N_cntrv[2] / norm;
          Vector(i1, i2, i3, c_u3) = N_cntrv[3] / norm;
        } else {
          zamo_fallback_3d(i1, i2, i3, x_Code);
        }
      } else {
        raise::KernelError(HERE, "3D implementation of Normalize4VelocityByNorm_kernel called for non-3D");
      }
    }
  };

  template <Dimension D, GRMetricClass M, uint8_t N>
  class Transform4VelocitySpatialToPhysical_kernel {
    // Transforms spatial components of 4-velocity from coordinate to physical
    // basis u^0 (Gamma/alpha) remains unchanged as it's unitless
    ndfield_t<D, N> Vector;
    const uint8_t   c_u1, c_u2, c_u3;
    const M         metric;

  public:
    Transform4VelocitySpatialToPhysical_kernel(ndfield_t<D, N>& vector,
                                               uint8_t          cu1,
                                               uint8_t          cu2,
                                               uint8_t          cu3,
                                               const M&         metric)
      : Vector { vector }
      , c_u1 { cu1 }
      , c_u2 { cu2 }
      , c_u3 { cu3 }
      , metric { metric } {
      raise::ErrorIf(c_u1 >= N or c_u2 >= N or c_u3 >= N,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;

        vec_t<Dim::_3D> u_cntrv { Vector(i1, i2, c_u1),
                                  Vector(i1, i2, c_u2),
                                  Vector(i1, i2, c_u3) };

        vec_t<Dim::_3D> u_phys { ZERO };
        // Transform spatial components from coordinate contravariant to physical contravariant
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_cntrv, u_phys);

        Vector(i1, i2, c_u1) = u_phys[0];
        Vector(i1, i2, c_u2) = u_phys[1];
        Vector(i1, i2, c_u3) = u_phys[2];
      } else {
        raise::KernelError(HERE, "2D implementation of Transform4VelocitySpatialToPhysical_kernel called for non-2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;
        x_Code[2] = COORD(i3) + HALF;

        vec_t<Dim::_3D> u_cntrv { Vector(i1, i2, i3, c_u1),
                                  Vector(i1, i2, i3, c_u2),
                                  Vector(i1, i2, i3, c_u3) };

        vec_t<Dim::_3D> u_phys { ZERO };
        // Transform spatial components from coordinate contravariant to physical contravariant
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_cntrv, u_phys);

        Vector(i1, i2, i3, c_u1) = u_phys[0];
        Vector(i1, i2, i3, c_u2) = u_phys[1];
        Vector(i1, i2, i3, c_u3) = u_phys[2];
      } else {
        raise::KernelError(HERE, "3D implementation of Transform4VelocitySpatialToPhysical_kernel called for non-3D");
      }
    }
  };

  template <SimEngine::type S, MetricClass M>
  class ParticleDistribution_kernel {
    const ParticleArrays particles;
    const bool           is_massive;
    const M              metric;

    const real_t e_min, e_max;
    const bool   log_bins;
    const size_t n_bins;

    scatter_array_t<real_t*> dn_scatter;

  public:
    ParticleDistribution_kernel(const Particles<M::Dim, M::CoordType>& particles,
                                const scatter_array_t<real_t*>& dn_scatter,
                                real_t                          e_min,
                                real_t                          e_max,
                                bool                            log_bins,
                                size_t                          n_bins,
                                const M&                        metric)
      : particles { static_cast<const ParticleArrays&>(particles) }
      , is_massive { (particles.mass() != 0.0f) }
      , dn_scatter { dn_scatter }
      , e_min { e_min }
      , e_max { e_max }
      , log_bins { log_bins }
      , n_bins { n_bins }
      , metric { metric } {}

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        return;
      }
      real_t en;
      if constexpr (S == SimEngine::SRPIC) {
        if (is_massive) {
          en = U2GAMMA(particles.ux1(p), particles.ux2(p), particles.ux3(p)) - ONE;
        } else {
          en = NORM(particles.ux1(p), particles.ux2(p), particles.ux3(p));
        }
      } else if constexpr (S == SimEngine::GRPIC) {
        coord_t<M::Dim> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                    static_cast<real_t>(particles.dx1(p));
        x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                    static_cast<real_t>(particles.dx2(p));

        // raise full covariant 4-vector to get correct contravariant u^0
        // u^i != h^{ij} u_j
        const real_t    u_0_cov { metric.u_0(
          x_Code,
          { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          (is_massive) ? ONE : ZERO) };
        vec_t<Dim::_4D> u_cntrv_4d { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(
          x_Code,
          { u_0_cov, particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          u_cntrv_4d);
        // in GR: u^0 = Gamma/alpha
        const real_t Gamma { metric.alpha(x_Code) * u_cntrv_4d[0] };
        en = is_massive ? (Gamma - ONE) : Gamma;
      }
      if (log_bins) {
        en = math::log10(en);
      }
      size_t e_ind = 0;
      if (en <= e_min) {
        e_ind = 0;
      } else if (en >= e_max) {
        e_ind = n_bins;
      } else {
        e_ind = static_cast<size_t>(
          static_cast<real_t>(n_bins) * (en - e_min) / (e_max - e_min));
      }
      auto dn_acc    = dn_scatter.access();
      dn_acc(e_ind) += particles.weight(p);
    }
  };

} // namespace kernel

#endif // KERNELS_PARTICLE_MOMENTS_HPP
