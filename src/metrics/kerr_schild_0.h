/**
 * @file metrics/kerr_schild_0.h
 * @brief
 * Kerr metric with zero spin and zero mass
 * in Kerr-Schild coordinates (rg=c=1)
 * @implements
 *   - metric::KerrSchild0<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_KERR_SCHILD_0_H
#define METRICS_KERR_SCHILD_0_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>

namespace metric {

  template <Dimension D>
  class KerrSchild0 : public MetricBase<D> {
    static_assert(D != Dim::_1D, "1D kerr_schild_0 not available");
    static_assert(D != Dim::_3D, "3D kerr_schild_0 not fully implemented");

  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t a, rg_, rh_;

  public:
    static constexpr const char* Label { "kerr_schild_0" };
    static constexpr Dimension   PrtlDim { D };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Kerr_Schild_0 };
    static constexpr ntt::Coord::type CoordType { ntt::Coord::Sph };
    using MetricBase<D>::x1_min;
    using MetricBase<D>::x1_max;
    using MetricBase<D>::x2_min;
    using MetricBase<D>::x2_max;
    using MetricBase<D>::x3_min;
    using MetricBase<D>::x3_max;
    using MetricBase<D>::nx1;
    using MetricBase<D>::nx2;
    using MetricBase<D>::nx3;
    using MetricBase<D>::set_dxMin;

    KerrSchild0(const std::vector<ncells_t>& res,
                const boundaries_t<real_t>&  ext,
                const std::map<std::string, real_t>& = {})
      : MetricBase<D> { res, ext }
      , a { ZERO }
      , rg_ { ONE }
      , rh_ { TWO }
      , dr { (x1_max - x1_min) / nx1 }
      , dtheta { (x2_max - x2_min) / nx2 }
      , dphi { (x3_max - x3_min) / nx3 }
      , dr_inv { ONE / dr }
      , dtheta_inv { ONE / dtheta }
      , dphi_inv { ONE / dphi } {
      set_dxMin(find_dxMin());
    }

    ~KerrSchild0() = default;

    [[nodiscard]]
    Inline auto spin() const -> real_t {
      return a;
    }

    [[nodiscard]]
    Inline auto rhorizon() const -> real_t {
      return rh_;
    }

    [[nodiscard]]
    Inline auto rg() const -> real_t {
      return rg_;
    }

    /**
     * minimum effective cell size for a given metric (in physical units)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      real_t min_dx { -ONE };
      for (int i { 0 }; i < nx1; ++i) {
        for (int j { 0 }; j < nx2; ++j) {
          real_t            i_ { static_cast<real_t>(i) + HALF };
          real_t            j_ { static_cast<real_t>(j) + HALF };
          coord_t<Dim::_2D> ij { i_, j_ };
          real_t            dx = ONE / std::sqrt(h<1, 1>(ij) + h<2, 2>(ij));
          if ((min_dx > dx) || (min_dx < 0.0)) {
            min_dx = dx;
          }
        }
      }
      return min_dx;
    }

    /**
     * total volume of the region described by the metric (in physical units)
     */
    [[nodiscard]]
    auto totVolume() const -> real_t override {
      if constexpr (D == Dim::_1D) {
        raise::Error("1D spherical metric not applicable", HERE);
      } else if constexpr (D == Dim::_2D) {
        return (SQR(x1_max) - SQR(x1_min)) * (x2_max - x2_min);
      } else {
        return (SQR(x1_max) - SQR(x1_min)) * (x2_max - x2_min) * (x3_max - x3_min);
      }
    }

    /**
     * metric component with lower indices: h_ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h_(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");
      if constexpr (i == 1 && j == 1) {
        // h_11
        return SQR(dr);
      } else if constexpr (i == 2 && j == 2) {
        // h_22
        return SQR(dtheta) * SQR(x[0] * dr + x1_min);
      } else if constexpr (i == 3 && j == 3) {
        // h_33
        if constexpr (D == Dim::_2D) {
          return SQR((x[0] * dr + x1_min) * math::sin(x[1] * dtheta + x2_min));
        } else {
          return SQR(dphi) *
                 SQR((x[0] * dr + x1_min) * math::sin(x[1] * dtheta + x2_min));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * metric component with upper indices: h^ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");
      if constexpr (i == 1 && j == 1) {
        // h^11
        return SQR(dr_inv);
      } else if constexpr (i == 2 && j == 2) {
        // h^22
        return SQR(dtheta_inv / (x[0] * dr + x1_min));
      } else if constexpr (i == 3 && j == 3) {
        // h^33
        if constexpr (D == Dim::_2D) {
          return ONE /
                 SQR((x[0] * dr + x1_min) * math::sin(x[1] * dtheta + x2_min));
        } else {
          return SQR(dphi_inv) /
                 SQR((x[0] * dr + x1_min) * math::sin(x[1] * dtheta + x2_min));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * lapse function
     * @param x coordinate array in code units
     */
    Inline auto alpha(const coord_t<D>&) const -> real_t {
      return ONE;
    }

    /**
     * dr derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dr_alpha(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dtheta derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dt_alpha(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto beta1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * dr derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dr_beta1(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dtheta derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dt_beta1(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dr derivative of h^11
     * @param x coordinate array in code units
     */
    Inline auto dr_h11(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dr derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dr_h22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -TWO / CUBE(r) * SQR(dtheta_inv) * dr;
    }

    /**
     * dr derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dr_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -TWO / CUBE(r) / SQR(math::sin(theta)) * dr;
    }

    /**
     * dr derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dr_h13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dtheta derivative of h^11
     * @param x coordinate array in code units
     */
    Inline auto dt_h11(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dtheta derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dt_h22(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * dtheta derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dt_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -TWO * math::cos(theta) / SQR(r) / CUBE(math::sin(theta)) * dtheta;
    }

    /**
     * dtheta derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dt_h13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      } else {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      }
    }

    /**
     * sqrt(det(h_ij)) / sin(theta)
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min);
      } else {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min);
      }
    }

    /**
     * differential area at the pole (used in axisymmetric solvers)
     * @param x1 radial coordinate along the axis (code units)
     */
    Inline auto polar_area(real_t x1) const -> real_t {
      return dr * SQR(x1 * dr + x1_min) * (ONE - math::cos(HALF * dtheta));
    }

    /**
     * component-wise coordinate conversions
     */
    template <idx_t i, Crd in, Crd out>
    Inline auto convert(real_t x_in) const -> real_t {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert((in == Crd::Cd && (out == Crd::Sph || out == Crd::Ph)) ||
                      ((in == Crd::Sph || in == Crd::Ph) && out == Crd::Cd),
                    "Invalid coordinate conversion");
      if constexpr (in == Crd::Cd && (out == Crd::Sph || out == Crd::Ph)) {
        // code -> sph/phys
        if constexpr (i == 1) {
          return x_in * dr + x1_min;
        } else if constexpr (i == 2) {
          return x_in * dtheta + x2_min;
        } else {
          if constexpr (D != Dim::_3D) {
            return x_in;
          } else {
            return x_in * dphi + x3_min;
          }
        }
      } else {
        // sph/phys -> code
        if constexpr (i == 1) {
          return (x_in - x1_min) * dr_inv;
        } else if constexpr (i == 2) {
          return (x_in - x2_min) * dtheta_inv;
        } else {
          if constexpr (D != Dim::_3D) {
            return x_in;
          } else {
            return (x_in - x3_min) * dphi_inv;
          }
        }
      }
    }

    /**
     * full coordinate conversions
     */
    template <Crd in, Crd out>
    Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(in != Crd::XYZ && out != Crd::XYZ,
                    "Invalid coordinate conversion: XYZ not allowed in GR");
      // code <-> sph/phys
      if constexpr (D == Dim::_2D) {
        x_out[0] = convert<1, in, out>(x_in[0]);
        x_out[1] = convert<2, in, out>(x_in[1]);
      } else {
        x_out[0] = convert<1, in, out>(x_in[0]);
        x_out[1] = convert<2, in, out>(x_in[1]);
        x_out[2] = convert<3, in, out>(x_in[2]);
      }
    }

    /**
     * full vector transformations
     */
    template <Idx in, Idx out>
    Inline void transform(const coord_t<D>&      xi,
                          const vec_t<Dim::_3D>& v_in,
                          vec_t<Dim::_3D>&       v_out) const {
      static_assert(in != out, "Invalid vector transformation");
      static_assert(in != Idx::XYZ && out != Idx::XYZ,
                    "Invalid vector transformation: XYZ not allowed in GR");
      if constexpr ((in == Idx::T && out == Idx::Sph) ||
                    (in == Idx::Sph && out == Idx::T)) {
        // tetrad <-> sph
        v_out[0] = v_in[0];
        v_out[1] = v_in[1];
        v_out[2] = v_in[2];
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::U) {
        // tetrad/sph -> cntrv
        v_out[0] = v_in[0] * math::sqrt(h<1, 1>(xi));
        v_out[1] = v_in[1] / math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] / math::sqrt(h_<3, 3>(xi));
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::Sph)) {
        // cntrv -> tetrad/sph
        v_out[0] = v_in[0] / math::sqrt(h<1, 1>(xi));
        v_out[1] = v_in[1] * math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] * math::sqrt(h_<3, 3>(xi));
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::D) {
        // tetrad/sph -> cov
        v_out[0] = v_in[0] / math::sqrt(h<1, 1>(xi));
        v_out[1] = v_in[1] * math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] * math::sqrt(h_<3, 3>(xi));
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::Sph)) {
        // cov -> tetrad/sph
        v_out[0] = v_in[0] * math::sqrt(h<1, 1>(xi));
        v_out[1] = v_in[1] / math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] / math::sqrt(h_<3, 3>(xi));
      } else if constexpr (in == Idx::U && out == Idx::D) {
        // cntrv -> cov
        v_out[0] = v_in[0] * h_<1, 1>(xi);
        v_out[1] = v_in[1] * h_<2, 2>(xi);
        v_out[2] = v_in[2] * h_<3, 3>(xi);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        // cov -> cntrv
        v_out[0] = v_in[0] * h<1, 1>(xi);
        v_out[1] = v_in[1] * h<2, 2>(xi);
        v_out[2] = v_in[2] * h<3, 3>(xi);
      } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                           (in == Idx::PD && out == Idx::D)) {
        // cntrv -> phys cntrv || phys cov -> cov
        v_out[0] = v_in[0] * dr;
        v_out[1] = v_in[1] * dtheta;
        if constexpr (D == Dim::_2D) {
          v_out[2] = v_in[2];
        } else {
          v_out[2] = v_in[2] * dphi;
        }
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        v_out[0] = v_in[0] * dr_inv;
        v_out[1] = v_in[1] * dtheta_inv;
        if constexpr (D == Dim::_2D) {
          v_out[2] = v_in[2];
        } else {
          v_out[2] = v_in[2] * dphi_inv;
        }
      } else {
        raise::KernelError(HERE, "Invalid transformation");
      }
    }
  };
} // namespace metric

#endif // METRICS_KERR_SCHILD_0_H
