/**
 * @file kernels/particle_moments.hpp
 * @brief Algorithm for computing different moments from particle distribution
 * @implements
 *   - kernel::ParticleMoments_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_PARTICLE_MOMENTS_HPP
#define KERNELS_PARTICLE_MOMENTS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <vector>

namespace kernel {
  using namespace ntt;

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

  template <SimEngine::type S, class M, FldsID::type F, unsigned short N>
  class ParticleMoments_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    static_assert(!((S == SimEngine::GRPIC) && (F == FldsID::V)),
                  "Bulk velocity not supported for GRPIC");
    static_assert((F == FldsID::Rho) || (F == FldsID::Charge) || (F == FldsID::N) ||
                    (F == FldsID::Nppc) || (F == FldsID::T) || (F == FldsID::V),
                  "Invalid field ID");

    const unsigned short     c1, c2;
    scatter_ndfield_t<D, N>  Buff;
    const idx_t              buff_idx;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const float              charge;
    const bool               use_weights;
    const M                  metric;
    const int                ni2;
    const unsigned short     window;

    const real_t contrib;
    const real_t smooth;
    bool         is_axis_i2min { false }, is_axis_i2max { false };

  public:
    ParticleMoments_kernel(const std::vector<unsigned short>& components,
                           const scatter_ndfield_t<D, N>&     scatter_buff,
                           idx_t                              buff_idx,
                           const array_t<int*>&               i1,
                           const array_t<int*>&               i2,
                           const array_t<int*>&               i3,
                           const array_t<prtldx_t*>&          dx1,
                           const array_t<prtldx_t*>&          dx2,
                           const array_t<prtldx_t*>&          dx3,
                           const array_t<real_t*>&            ux1,
                           const array_t<real_t*>&            ux2,
                           const array_t<real_t*>&            ux3,
                           const array_t<real_t*>&            phi,
                           const array_t<real_t*>&            weight,
                           const array_t<short*>&             tag,
                           float                              mass,
                           float                              charge,
                           bool                               use_weights,
                           const M&                           metric,
                           const boundaries_t<FldsBC>&        boundaries,
                           ncells_t                           ni2,
                           real_t                             inv_n0,
                           unsigned short                     window)
      : c1 { (components.size() > 0) ? components[0]
                                     : static_cast<unsigned short>(0) }
      , c2 { (components.size() == 2) ? components[1]
                                      : static_cast<unsigned short>(0) }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , charge { charge }
      , use_weights { use_weights }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , window { window }
      , contrib { get_contrib<F>(mass, charge) }
      , smooth { inv_n0 / (real_t)(math::pow(TWO * (real_t)window + ONE,
                                             static_cast<int>(D))) } {
      raise::ErrorIf(buff_idx >= N, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      raise::ErrorIf(((F == FldsID::Rho) || (F == FldsID::Charge)) && (mass == ZERO),
                     "Rho & Charge for massless particles not defined",
                     HERE);
      if constexpr ((M::CoordType != Coord::Cart) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == FldsID::T) {
        real_t          energy { ZERO };
        // for stress-energy tensor
        vec_t<Dim::_3D> u_Phys { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          // SR
          // stress-energy tensor for SR is computed in the tetrad (hatted) basis
          if constexpr (M::CoordType == Coord::Cart) {
            u_Phys[0] = ux1(p);
            u_Phys[1] = ux2(p);
            u_Phys[2] = ux3(p);
          } else {
            static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
            coord_t<M::PrtlDim> x_Code { ZERO };
            x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            if constexpr (D == Dim::_3D) {
              x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
            } else {
              x_Code[2] = phi(p);
            }
            metric.template transform_xyz<Idx::XYZ, Idx::T>(
              x_Code,
              { ux1(p), ux2(p), ux3(p) },
              u_Phys);
          }
          if (mass == ZERO) {
            energy = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
          } else {
            energy = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
          }
        } else {
          // GR
          // stress-energy tensor for GR is computed in contravariant basis
          static_assert(D != Dim::_1D, "GRPIC 1D");
          coord_t<D> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          }
          vec_t<Dim::_3D> u_Cntrv { ZERO };
          // compute u_i u^i for energy
          metric.template transform<Idx::D, Idx::U>(x_Code,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    u_Cntrv);
          energy = u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) + u_Cntrv[2] * ux3(p);
          if (mass == ZERO) {
            energy = math::sqrt(energy);
          } else {
            energy = math::sqrt(ONE + energy);
          }
          metric.template transform<Idx::U, Idx::PU>(x_Code, u_Cntrv, u_Phys);
        }
        // compute the corresponding moment
        coeff = ONE / energy;
#pragma unroll
        for (const auto& c : { c1, c2 }) {
          if (c == 0) {
            coeff *= energy;
          } else {
            coeff *= u_Phys[c - 1];
          }
        }
      } else if constexpr (F == FldsID::V) {
        real_t          gamma { ZERO };
        // for bulk 3vel (tetrad basis)
        vec_t<Dim::_3D> u_Phys { ZERO };
        if constexpr (M::CoordType == Coord::Cart) {
          u_Phys[0] = ux1(p);
          u_Phys[1] = ux2(p);
          u_Phys[2] = ux3(p);
        } else {
          coord_t<M::PrtlDim> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          } else {
            x_Code[2] = phi(p);
          }
          metric.template transform_xyz<Idx::XYZ, Idx::T>(x_Code,
                                                          { ux1(p), ux2(p), ux3(p) },
                                                          u_Phys);
        }
        if (mass == ZERO) {
          gamma = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
        } else {
          gamma = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
        }
        // compute the corresponding moment
        coeff = (mass == ZERO ? ONE : mass) * u_Phys[c1 - 1] / gamma;
      } else {
        // for other cases, use the `contrib` defined above
        coeff = contrib;
      }

      if constexpr (F == FldsID::V) {
        real_t          gamma { ZERO };
        // for stress-energy tensor
        vec_t<Dim::_3D> u_Phys { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          // SR
          // stress-energy tensor for SR is computed in the tetrad (hatted) basis
          if constexpr (M::CoordType == Coord::Cart) {
            u_Phys[0] = ux1(p);
            u_Phys[1] = ux2(p);
            u_Phys[2] = ux3(p);
          } else {
            static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
            coord_t<M::PrtlDim> x_Code { ZERO };
            x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            if constexpr (D == Dim::_3D) {
              x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
            } else {
              x_Code[2] = phi(p);
            }
            metric.template transform_xyz<Idx::XYZ, Idx::T>(
              x_Code,
              { ux1(p), ux2(p), ux3(p) },
              u_Phys);
          }
          if (mass == ZERO) {
            gamma = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
          } else {
            gamma = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
          }
        } else {
          // GR
          // stress-energy tensor for GR is computed in contravariant basis
          static_assert(D != Dim::_1D, "GRPIC 1D");
          coord_t<D> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          }
          vec_t<Dim::_3D> u_Cntrv { ZERO };
          // compute u_i u^i for energy
          metric.template transform<Idx::D, Idx::U>(x_Code,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    u_Cntrv);
          gamma = u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) + u_Cntrv[2] * ux3(p);
          if (mass == ZERO) {
            gamma = math::sqrt(gamma);
          } else {
            gamma = math::sqrt(ONE + gamma);
          }
          metric.template transform<Idx::U, Idx::PU>(x_Code, u_Cntrv, u_Phys);
        }
        // compute the corresponding moment
        coeff = u_Phys[c1 - 1] / gamma;
      }

      if constexpr (F != FldsID::Nppc) {
        // for nppc calculation ...
        // ... do not take volume, weights or smoothing into account
        if constexpr (D == Dim::_1D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF });
        } else if constexpr (D == Dim::_2D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF });
        } else if constexpr (D == Dim::_3D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF,
                                       static_cast<real_t>(i3(p)) + HALF });
        }
        if (use_weights) {
          coeff *= weight(p);
        }
      }
      auto buff_access = Buff.access();
      if constexpr (D == Dim::_1D) {
        for (auto di1 { -window }; di1 <= window; ++di1) {
          buff_access(i1(p) + di1 + N_GHOSTS, buff_idx) += coeff;
        }
      } else if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            if constexpr (M::CoordType == Coord::Cart) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff;
            } else {
              // reflect contribution at axes
              if (is_axis_i2min && (i2(p) + di2 < 0)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            N_GHOSTS - (i2(p) + di2),
                            buff_idx) += coeff;
              } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            buff_idx) += coeff;
              }
            }
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (auto di3 { -window }; di3 <= window; ++di3) {
          for (auto di2 { -window }; di2 <= window; ++di2) {
            for (auto di1 { -window }; di1 <= window; ++di1) {
              if constexpr (M::CoordType == Coord::Cart) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            i3(p) + di3 + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (i2(p) + di2 < 0)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              N_GHOSTS - (i2(p) + di2),
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              i2(p) + di2 + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                }
              }
            }
          }
        }
      }
    }
  };

  template <Dimension D, unsigned short N>
  class NormalizeVectorByRho_kernel {
    const ndfield_t<D, N> Rho;
    ndfield_t<D, N>       Vector;
    const unsigned short  c_rho, c_v1, c_v2, c_v3;

  public:
    NormalizeVectorByRho_kernel(const ndfield_t<D, N>& rho,
                                const ndfield_t<D, N>& vector,
                                unsigned short         crho,
                                unsigned short         cv1,
                                unsigned short         cv2,
                                unsigned short         cv3)
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

    Inline void operator()(index_t i1) const {
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

    Inline void operator()(index_t i1, index_t i2) const {
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

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
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

} // namespace kernel

#endif // KERNELS_PARTICLE_MOMENTS_HPP
