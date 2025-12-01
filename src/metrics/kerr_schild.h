/**
 * @file metrics/kerr_schild.h
 * @brief Kerr metric in Kerr-Schild coordinates (rg=c=1)
 * @implements
 *   - metric::KerrSchild<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_KERR_SCHILD_H
#define METRICS_KERR_SCHILD_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>

namespace metric {

  template <Dimension D>
  class KerrSchild : public MetricBase<D> {
    static_assert(D != Dim::_1D, "1D kerr_schild not available");
    static_assert(D != Dim::_3D, "3D kerr_schild not fully implemented");

  private:
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t a, rg_, rh_;

    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;

    Inline auto Delta(real_t r) const -> real_t {
      return SQR(r) - TWO * r + SQR(a);
    }

    Inline auto Sigma(real_t r, real_t theta) const -> real_t {
      return SQR(r) + SQR(a) * SQR(math::cos(theta));
    }

    Inline auto A(real_t r, real_t theta) const -> real_t {
      return SQR(SQR(r) + SQR(a)) - SQR(a) * Delta(r) * SQR(math::sin(theta));
    }

    Inline auto z(real_t r, real_t theta) const -> real_t {
      return TWO * r / Sigma(r, theta);
    }

  public:
    static constexpr const char*       Label { "kerr_schild" };
    static constexpr Dimension         PrtlDim { D };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Sph };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Kerr_Schild };
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

    KerrSchild(const std::vector<ncells_t>&         res,
               const boundaries_t<real_t>&          ext,
               const std::map<std::string, real_t>& params)
      : MetricBase<D> { res, ext }
      , a { params.at("a") }
      , rg_ { ONE }
      , rh_ { ONE + math::sqrt(ONE - SQR(a)) }
      , dr { (x1_max - x1_min) / nx1 }
      , dtheta { (x2_max - x2_min) / nx2 }
      , dphi { (x3_max - x3_min) / nx3 }
      , dr_inv { ONE / dr }
      , dtheta_inv { ONE / dtheta }
      , dphi_inv { ONE / dphi } {
      set_dxMin(find_dxMin());
    }

    ~KerrSchild() = default;

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
          real_t dx = ONE / (alpha(ij) * math::sqrt(h<1, 1>(ij) + h<2, 2>(ij)) +
                             beta1(ij));
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
      // @TODO: Ask Alisa
      return ZERO;
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
        return SQR(dr) * (ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min));
      } else if constexpr (i == 2 && j == 2) {
        // h_22
        return SQR(dtheta) * Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
      } else if constexpr (i == 3 && j == 3) {
        // h_33
        if constexpr (D == Dim::_2D) {
          return A(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
                 SQR(math::sin(x[1] * dtheta + x2_min)) /
                 Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
        } else {
          return SQR(dphi) * A(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
                 SQR(math::sin(x[1] * dtheta + x2_min)) /
                 Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
        }
      } else if constexpr ((i == 1 && j == 3) || (i == 3 && j == 1)) {
        // h_13 or h_31
        if constexpr (D == Dim::_2D) {
          return -dr * a * (ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min)) *
                 SQR(math::sin(x[1] * dtheta + x2_min));
        } else {
          return -dr * dphi * a *
                 (ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min)) *
                 SQR(math::sin(x[1] * dtheta + x2_min));
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
        const real_t Sigma_ { Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) };
        return SQR(dr_inv) * A(x[0] * dr + x1_min, x[1] * dtheta + x2_min) /
               (Sigma_ * (Sigma_ + TWO * (x[0] * dr + x1_min)));
      } else if constexpr (i == 2 && j == 2) {
        // h^22
        return SQR(dtheta_inv) / Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
      } else if constexpr (i == 3 && j == 3) {
        // h^33
        if constexpr (D == Dim::_2D) {
          return ONE / (Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
                        SQR(math::sin(x[1] * dtheta + x2_min)));
        } else {
          return SQR(dphi_inv) /
                 (Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
                  SQR(math::sin(x[1] * dtheta + x2_min)));
        }
      } else if constexpr ((i == 1 && j == 3) || (i == 3 && j == 1)) {
        // h^13 or h^31
        if constexpr (D == Dim::_2D) {
          return dr_inv * a / Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
        } else {
          return dr_inv * dphi_inv * a /
                 Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min);
        }
      } else {
        return ZERO;
      }
    }

    /**
     * lapse function
     * @param x coordinate array in code units
     */
    Inline auto alpha(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return ONE / math::sqrt(ONE + z(r, theta));
    }

    /**
     * dr derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dr_alpha(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };

      return -(dr * Sigma(r, theta) - r * dr_Sigma) * CUBE(alpha(x)) /
             SQR(Sigma(r, theta));
    }

    /**
     * dtheta derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dt_alpha(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return CUBE(alpha(x)) * r * dt_Sigma(theta) / SQR(Sigma(r, theta));
    }

    /**
     * radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto beta1(const coord_t<D>& x) const -> real_t {
      const real_t z_ { z(x[0] * dr + x1_min, x[1] * dtheta + x2_min) };
      return dr_inv * z_ / (ONE + z_);
    }

    /**
     * dr derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dr_beta1(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };

      return dr_inv * TWO * (dr * Sigma(r, theta) - r * dr_Sigma) /
             SQR(Sigma(r, theta) + TWO * r);
    }

    /**
     * dtheta derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dt_beta1(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -dr_inv * TWO * r * dt_Sigma(theta) /
             SQR(Sigma(r, theta) * (ONE + z(r, theta)));
    }

    /**
     * dr derivative of h^11
     * @param x coordinate array in code units
     */
    Inline auto dr_h11(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };
      const real_t dr_Delta { TWO * dr * (r - ONE) };
      const real_t dr_A { FOUR * r * dr * (SQR(r) + SQR(a)) -
                          SQR(a) * SQR(math::sin(theta)) * dr_Delta };

      return (Sigma(r, theta) * (Sigma(r, theta) + TWO * r) * dr_A -
              TWO * A(r, theta) *
                (r * dr_Sigma + Sigma(r, theta) * (dr_Sigma + dr))) /
             (SQR(Sigma(r, theta) * (Sigma(r, theta) + TWO * r))) * SQR(dr_inv);
    }

    /**
     * dr derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dr_h22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };

      return -dr_Sigma / SQR(Sigma(r, theta)) * SQR(dtheta_inv);
    }

    /**
     * dr derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dr_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };

      return -dr_Sigma / SQR(Sigma(r, theta)) / SQR(math::sin(theta));
    }

    /**
     * dr derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dr_h13(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      const real_t dr_Sigma { TWO * r * dr };

      return -a * dr_Sigma / SQR(Sigma(r, theta)) * dr_inv;
    }

    /**
     * dtheta derivative of Sigma
     * @param x coordinate array in code units
     */
    Inline auto dt_Sigma(real_t theta) const -> real_t {
      const real_t dt_Sigma { -TWO * SQR(a) * math::sin(theta) *
                              math::cos(theta) * dtheta };
      if (cmp::AlmostZero(dt_Sigma)) {
        return ZERO;
      } else {
        return dt_Sigma;
      }
    }

    /**
     * dtheta derivative of A
     * @param x coordinate array in code units
     */
    Inline auto dt_A(real_t r, real_t theta) const -> real_t {
      const real_t dt_A { -TWO * SQR(a) * math::sin(theta) * math::cos(theta) *
                          Delta(r) * dtheta };
      if (cmp::AlmostZero(dt_A)) {
        return ZERO;
      } else {
        return dt_A;
      }
    }

    /**
     * dtheta derivative of h^11
     * @param x coordinate array in code units
     */
    Inline auto dt_h11(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return (Sigma(r, theta) * (Sigma(r, theta) + TWO * r) * dt_A(r, theta) -
              TWO * A(r, theta) * dt_Sigma(theta) * (r + Sigma(r, theta))) /
             (SQR(Sigma(r, theta) * (Sigma(r, theta) + TWO * r))) * SQR(dr_inv);
    }

    /**
     * dtheta derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dt_h22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -dt_Sigma(theta) / SQR(Sigma(r, theta)) * SQR(dtheta_inv);
    }

    /**
     * dtheta derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dt_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -TWO * dtheta * math::cos(theta) *
             (Sigma(r, theta) - SQR(a) * SQR(math::sin(theta))) /
             CUBE(math::sin(theta)) / SQR(Sigma(r, theta));
    }

    /**
     * dtheta derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dt_h13(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + x1_min };
      const real_t theta { x[1] * dtheta + x2_min };
      return -a * dt_Sigma(theta) / SQR(Sigma(r, theta)) * dr_inv;
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
               math::sin(x[1] * dtheta + x2_min) *
               math::sqrt(ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min));
      } else {
        return dr * dtheta * dphi *
               Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
               math::sin(x[1] * dtheta + x2_min) *
               math::sqrt(ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min));
      }
    }

    /**
     * sqrt(det(h_ij)) / sin(theta)
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
               math::sqrt(ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min));
      } else {
        return dr * dtheta * dphi *
               Sigma(x[0] * dr + x1_min, x[1] * dtheta + x2_min) *
               math::sqrt(ONE + z(x[0] * dr + x1_min, x[1] * dtheta + x2_min));
      }
    }

    /**
     * differential area at the pole (used in axisymmetric solvers)
     * @param x1 radial coordinate along the axis (code units)
     */
    Inline auto polar_area(real_t x1) const -> real_t {
      return dr * (SQR(x1 * dr + x1_min) + SQR(a)) *
             math::sqrt(ONE + TWO * (x1 * dr + x1_min) /
                                (SQR(x1 * dr + x1_min) + SQR(a))) *
             (ONE - math::cos(HALF * dtheta));
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
        const real_t A0 { math::sqrt(h<1, 1>(xi)) };
        v_out[0] = v_in[0] * A0;
        v_out[1] = v_in[1] / math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] / math::sqrt(h_<3, 3>(xi)) -
                   v_in[0] * A0 * h_<1, 3>(xi) / h_<3, 3>(xi);
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::Sph)) {
        // cntrv -> tetrad/sph
        v_out[0] = v_in[0] / math::sqrt(h<1, 1>(xi));
        v_out[1] = v_in[1] * math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] * math::sqrt(h_<3, 3>(xi)) +
                   v_in[0] * h_<1, 3>(xi) / math::sqrt(h_<3, 3>(xi));
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::D) {
        // tetrad/sph -> cov
        v_out[0] = v_in[0] / math::sqrt(h<1, 1>(xi)) +
                   v_in[2] * h_<1, 3>(xi) / math::sqrt(h_<3, 3>(xi));
        v_out[1] = v_in[1] * math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] * math::sqrt(h_<3, 3>(xi));
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::Sph)) {
        // cov -> tetrad/sph
        const real_t A0 { math::sqrt(h<1, 1>(xi)) };
        v_out[0] = v_in[0] * A0 - v_in[2] * A0 * h_<1, 3>(xi) / h_<3, 3>(xi);
        v_out[1] = v_in[1] / math::sqrt(h_<2, 2>(xi));
        v_out[2] = v_in[2] / math::sqrt(h_<3, 3>(xi));
      } else if constexpr (in == Idx::U && out == Idx::D) {
        // cntrv -> cov
        v_out[0] = v_in[0] * h_<1, 1>(xi) + v_in[2] * h_<1, 3>(xi);
        v_out[1] = v_in[1] * h_<2, 2>(xi);
        v_out[2] = v_in[0] * h_<1, 3>(xi) + v_in[2] * h_<3, 3>(xi);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        // cov -> cntrv
        v_out[0] = v_in[0] * h<1, 1>(xi) + v_in[2] * h<1, 3>(xi);
        v_out[1] = v_in[1] * h<2, 2>(xi);
        v_out[2] = v_in[0] * h<1, 3>(xi) + v_in[2] * h<3, 3>(xi);
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

#endif // METRICS_KERR_SCHILD_H
