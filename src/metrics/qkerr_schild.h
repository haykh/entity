/**
 * @file metrics/qkerr_schild.h
 * @brief
 * Kerr metric in qspherical Kerr-Schild coordinates (rg=c=1)
 * @implements
 *   - metric::QKerrSchild<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_QKERR_SCHILD_H
#define METRICS_QKERR_SCHILD_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

namespace metric {

  template <Dimension D>
  class QKerrSchild : public MetricBase<D> {
    static_assert(D != Dim::_1D, "1D qkerr_schild not available");
    static_assert(D != Dim::_3D, "3D qkerr_schild not fully implemented");

  private:
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t a, rg_, rh_;

    const real_t r0, h0;
    const real_t chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;

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
    static constexpr const char*       Label { "qkerr_schild" };
    static constexpr Dimension         PrtlDim { D };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::QKerr_Schild };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Qsph };
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

    QKerrSchild(const std::vector<ncells_t>&         res,
                const boundaries_t<real_t>&          ext,
                const std::map<std::string, real_t>& params)
      : MetricBase<D> { res, ext }
      , a { params.at("a") }
      , rg_ { ONE }
      , rh_ { ONE + math::sqrt(ONE - SQR(a)) }
      , r0 { params.at("r0") }
      , h0 { params.at("h") }
      , chi_min { math::log(x1_min - r0) }
      , eta_min { theta2eta(x2_min) }
      , phi_min { x3_min }
      , dchi { (math::log(x1_max - r0) - chi_min) / nx1 }
      , deta { (theta2eta(x2_max) - eta_min) / nx2 }
      , dphi { (x3_max - phi_min) / nx3 }
      , dchi_inv { ONE / dchi }
      , deta_inv { ONE / deta }
      , dphi_inv { ONE / dphi } {
      set_dxMin(find_dxMin());
    }

    ~QKerrSchild() = default;

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
          real_t dx = ONE / (alpha(ij) * std::sqrt(h<1, 1>(ij) + h<2, 2>(ij)) +
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
        return SQR(dchi) * math::exp(TWO * (x[0] * dchi + chi_min)) *
               (ONE + z(r0 + math::exp(x[0] * dchi + chi_min),
                        eta2theta(x[1] * deta + eta_min)));
      } else if constexpr (i == 2 && j == 2) {
        // h_22
        return SQR(deta) * SQR(dtheta_deta(x[1] * deta + eta_min)) *
               Sigma(r0 + math::exp(x[0] * dchi + chi_min),
                     eta2theta(x[1] * deta + eta_min));
      } else if constexpr (i == 3 && j == 3) {
        // h_33
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        if constexpr (D == Dim::_2D) {
          return A(r0 + math::exp(x[0] * dchi + chi_min), theta) *
                 SQR(math::sin(theta)) /
                 Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta);
        } else {
          return SQR(dphi) * A(r0 + math::exp(x[0] * dchi + chi_min), theta) *
                 SQR(math::sin(theta)) /
                 Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta);
        }
      } else if constexpr ((i == 1 && j == 3) || (i == 3 && j == 1)) {
        // h_13 or h_31
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        if constexpr (D == Dim::_2D) {
          return -dchi * math::exp(x[0] * dchi + chi_min) * a *
                 (ONE + z(r0 + math::exp(x[0] * dchi + chi_min), theta)) *
                 SQR(math::sin(theta));
        } else {
          return -dchi * math::exp(x[0] * dchi + chi_min) * dphi * a *
                 (ONE + z(r0 + math::exp(x[0] * dchi + chi_min), theta)) *
                 SQR(math::sin(theta));
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
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        const real_t Sigma_ { Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta) };
        return (math::exp(-TWO * (x[0] * dchi + chi_min)) / SQR(dchi)) *
               A(r0 + math::exp(x[0] * dchi + chi_min), theta) /
               (Sigma_ * (Sigma_ + TWO * (r0 + math::exp(x[0] * dchi + chi_min))));
      } else if constexpr (i == 2 && j == 2) {
        // h^22
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        return ONE / (Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta) *
                      SQR(dtheta_deta(x[1] * deta + eta_min)) * SQR(deta));
      } else if constexpr (i == 3 && j == 3) {
        // h^33
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        if constexpr (D == Dim::_2D) {
          return ONE / (Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta) *
                        SQR(math::sin(theta)));
        } else {
          return SQR(dphi_inv) /
                 (Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta) *
                  SQR(math::sin(theta)));
        }
      } else if constexpr ((i == 1 && j == 3) || (i == 3 && j == 1)) {
        const real_t theta { eta2theta(x[1] * deta + eta_min) };
        if constexpr (D == Dim::_2D) {
          return (math::exp(-(x[0] * dchi + chi_min)) * dchi_inv) * a /
                 Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta);
        } else {
          return (math::exp(-(x[0] * dchi + chi_min)) * dchi_inv) * dphi_inv *
                 a / Sigma(r0 + math::exp(x[0] * dchi + chi_min), theta);
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
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      return ONE / math::sqrt(ONE + z(r, theta));
    }

    /**
     * dr derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dr_alpha(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };
      return -(dx_r * Sigma(r, theta) - r * dr_Sigma) * CUBE(alpha(x)) /
             SQR(Sigma(r, theta));
    }

    /**
     * dtheta derivative of lapse function
     * @param x coordinate array in code units
     */
    Inline auto dt_alpha(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      const real_t dx_dt {
        deta * (ONE + TWO * h0 * static_cast<real_t>(constant::INV_PI_SQR) *
                        (TWO * THREE * SQR(eta) -
                         TWO * THREE * static_cast<real_t>(constant::PI) * eta +
                         static_cast<real_t>(constant::PI_SQR)))
      };
      const real_t dt_Sigma { -TWO * SQR(a) * math::sin(theta) *
                              math::cos(theta) * dx_dt };

      return r * dt_Sigma * CUBE(alpha(x)) / SQR(Sigma(r, theta));
    }

    /**
     * radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto beta1(const coord_t<D>& x) const -> real_t {
      const real_t chi { x[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t z_ { z(r, theta) };
      return math::exp(-chi) * dchi_inv * z_ / (ONE + z_);
    }

    /**
     * dr derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dr_beta1(const coord_t<D>& x) const -> real_t {
      const real_t chi { x[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t z_ { z(r, theta) };
      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };

      return math::exp(-chi) * dchi_inv * TWO *
               (dx_r * Sigma(r, theta) - r * dr_Sigma) /
               SQR(Sigma(r, theta) + TWO * r) -
             dchi * math::exp(-chi) * dchi_inv * z_ / (ONE + z_);
    }

    /**
     * dr derivative of radial component of shift vector
     * @param x coordinate array in code units
     */
    Inline auto dt_beta1(const coord_t<D>& x) const -> real_t {
      const real_t chi { x[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return -math::exp(-chi) * dchi_inv * TWO * r * dt_Sigma(eta) /
             SQR(Sigma(r, theta) * (ONE + z(r, theta)));
    }

    /**
     * dr derivative of h^11
     * @param x coordinate array in code units
     */
    Inline auto dr_h11(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };

      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };
      const real_t dr_Delta { TWO * dx_r * (r - ONE) };
      const real_t dr_A { FOUR * r * dx_r * (SQR(r) + SQR(a)) -
                          SQR(a) * SQR(math::sin(theta)) * dr_Delta };

      return (math::exp(-TWO * (x[0] * dchi + chi_min)) / SQR(dchi) *
              (Sigma(r, theta) * (Sigma(r, theta) + TWO * r) * dr_A -
               TWO * A(r, theta) *
                 (r * dr_Sigma + Sigma(r, theta) * (dr_Sigma + dx_r))) /
              (SQR(Sigma(r, theta) * (Sigma(r, theta) + TWO * r)))) -
             TWO * dchi * math::exp(-TWO * (x[0] * dchi + chi_min)) / SQR(dchi) *
               A(r, theta) / (Sigma(r, theta) * (Sigma(r, theta) + TWO * r));
    }

    /**
     * dr derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dr_h22(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };

      return -dr_Sigma / SQR(Sigma(r, theta)) / SQR(deta);
    }

    /**
     * dr derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dr_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };

      return -dr_Sigma / SQR(Sigma(r, theta)) / SQR(math::sin(theta));
    }

    /**
     * dr derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dr_h13(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      const real_t dx_r { dchi * math::exp(x[0] * dchi + chi_min) };
      const real_t dr_Sigma { TWO * r * dx_r };

      return -a * dr_Sigma / SQR(Sigma(r, theta)) *
               (math::exp(-(x[0] * dchi + chi_min)) * dchi_inv) -
             dchi * (math::exp(-(x[0] * dchi + chi_min)) * dchi_inv) * a /
               Sigma(r, theta);
    }

    /**
     * dtheta derivative of Sigma
     * @param x coordinate array in code units
     */
    Inline auto dt_Sigma(real_t eta) const -> real_t {
      const real_t theta { eta2theta(eta) };
      const real_t dt_Sigma { -TWO * SQR(a) * math::sin(theta) *
                              math::cos(theta) * dx_dt(eta) };
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
    Inline auto dt_A(real_t r, real_t eta) const -> real_t {
      const real_t theta { eta2theta(eta) };
      const real_t dt_A { -TWO * SQR(a) * math::sin(theta) * math::cos(theta) *
                          Delta(r) * dx_dt(eta) };
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
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return math::exp(-TWO * (x[0] * dchi + chi_min)) / SQR(dchi) *
             (Sigma(r, theta) * (Sigma(r, theta) + TWO * r) * dt_A(r, eta) -
              TWO * A(r, theta) * dt_Sigma(eta) * (r + Sigma(r, theta))) /
             (SQR(Sigma(r, theta) * (Sigma(r, theta) + TWO * r)));
    }

    /**
     * dtheta derivative of h^22
     * @param x coordinate array in code units
     */
    Inline auto dt_h22(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return -dt_Sigma(eta) / SQR(Sigma(r, theta)) / SQR(deta);
    }

    /**
     * dtheta derivative of h^33
     * @param x coordinate array in code units
     */
    Inline auto dt_h33(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return -(dt_Sigma(eta) + TWO * math::cos(theta) / math::sin(theta) *
                                 Sigma(r, theta) * dx_dt(eta)) /
             SQR(Sigma(r, theta) * math::sin(theta));
    }

    /**
     * dtheta derivative of h^13
     * @param x coordinate array in code units
     */
    Inline auto dt_h13(const coord_t<D>& x) const -> real_t {
      const real_t r { r0 + math::exp(x[0] * dchi + chi_min) };
      const real_t eta { x[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return -a * dt_Sigma(eta) / SQR(Sigma(r, theta)) *
             (math::exp(-(x[0] * dchi + chi_min)) * dchi_inv);
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      const real_t expchi { math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      if constexpr (D == Dim::_2D) {
        return dchi * expchi * dtheta_deta(x[1] * deta + eta_min) * deta *
               Sigma(r0 + expchi, theta) * math::sin(theta) *
               math::sqrt(ONE + z(r0 + expchi, theta));
      } else {
        return dchi * expchi * dtheta_deta(x[1] * deta + eta_min) * deta *
               dphi * Sigma(r0 + expchi, theta) * math::sin(theta) *
               math::sqrt(ONE + z(r0 + expchi, theta));
      }
    }

    /**
     * sqrt(det(h_ij)) divided by sin(theta).
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      const real_t expchi { math::exp(x[0] * dchi + chi_min) };
      const real_t theta { eta2theta(x[1] * deta + eta_min) };
      if constexpr (D == Dim::_2D) {
        return dchi * expchi * dtheta_deta(x[1] * deta + eta_min) * deta *
               Sigma(r0 + expchi, theta) * math::sqrt(ONE + z(r0 + expchi, theta));
      } else {
        return dchi * expchi * dtheta_deta(x[1] * deta + eta_min) * deta *
               dphi * Sigma(r0 + expchi, theta) *
               math::sqrt(ONE + z(r0 + expchi, theta));
      }
    }

    /**
     * differential area at the pole (used in axisymmetric solvers)
     * @note approximate solution for the polar area
     * @param x1 radial coordinate along the axis (code units)
     */
    Inline auto polar_area(real_t x1) const -> real_t {
      if constexpr (D != Dim::_1D) {
        return dchi * math::exp(x1 * dchi + chi_min) *
               (SQR(r0 + math::exp(x1 * dchi + chi_min)) + SQR(a)) *
               math::sqrt(
                 ONE + TWO * (r0 + math::exp(x1 * dchi + chi_min)) /
                         (SQR(r0 + math::exp(x1 * dchi + chi_min)) + SQR(a))) *
               (ONE - math::cos(eta2theta(HALF * deta)));
      }
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
          return r0 + math::exp(x_in * dchi + chi_min);
        } else if constexpr (i == 2) {
          return eta2theta(x_in * deta + eta_min);
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
          return (math::log(x_in - r0) - chi_min) * dchi_inv;
        } else if constexpr (i == 2) {
          return (theta2eta(x_in) - eta_min) * deta_inv;
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
        v_out[0] = v_in[0] * math::exp(xi[0] * dchi + chi_min) * dchi;
        v_out[1] = v_in[1] * (dtheta_deta(xi[1] * deta + eta_min) * deta);
        if constexpr (D == Dim::_2D) {
          v_out[2] = v_in[2];
        } else {
          v_out[2] = v_in[2] * dphi;
        }
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        v_out[0] = v_in[0] * dchi_inv / (math::exp(xi[0] * dchi + chi_min));
        v_out[1] = v_in[1] * deta_inv / (dtheta_deta(xi[1] * deta + eta_min));
        if constexpr (D == Dim::_2D) {
          v_out[2] = v_in[2];
        } else {
          v_out[2] = v_in[2] * dphi_inv;
        }
      } else {
        raise::KernelError(HERE, "Invalid transformation");
      }
    }

  private:
    /* Specific for Quasi-Spherical metric with angle stretching ------------ */
    /**
     * @brief d(th) / d(eta) for a given eta
     */
    Inline auto dtheta_deta(real_t eta) const -> real_t {
      if (cmp::AlmostZero(h0)) {
        return ONE;
      } else {
        return (ONE + TWO * h0 +
                static_cast<real_t>(12.0) * h0 *
                  (eta * static_cast<real_t>(constant::INV_PI)) *
                  ((eta * static_cast<real_t>(constant::INV_PI)) - ONE));
      }
    }

    /**
     * @brief quasi-spherical eta to spherical theta
     */
    Inline auto eta2theta(real_t eta) const -> real_t {
      if (cmp::AlmostZero(h0)) {
        return eta;
      } else {
        return eta + TWO * h0 * eta *
                       (static_cast<real_t>(constant::PI) - TWO * eta) *
                       (static_cast<real_t>(constant::PI) - eta) *
                       static_cast<real_t>(constant::INV_PI_SQR);
      }
    }

    /**
     * @brief quasi-spherical eta to spherical theta
     */
    Inline auto dx_dt(real_t eta) const -> real_t {
      if (cmp::AlmostZero(h0)) {
        return deta;
      } else {
        return deta *
               (ONE + TWO * h0 * static_cast<real_t>(constant::INV_PI_SQR) *
                        (TWO * THREE * SQR(eta) -
                         TWO * THREE * static_cast<real_t>(constant::PI) * eta +
                         static_cast<real_t>(constant::PI_SQR)));
      }
    }

    /**
     * @brief spherical theta to quasi-spherical eta
     */
    Inline auto theta2eta(real_t theta) const -> real_t {
      if (cmp::AlmostZero(h0)) {
        return theta;
      } else {
        using namespace constant;
        // R = (-9 h^2 (Pi - 2 y) + Sqrt[3] Sqrt[-(h^3 ((-4 + h) (Pi + 2 h Pi)^2
        // + 108 h Pi y - 108 h y^2))])^(1/3)
        double                  R { math::pow(
          -9.0 * SQR(h0) * (PI - 2.0 * theta) +
            SQRT3 * math::sqrt((CUBE(h0) * ((4.0 - h0) * SQR(PI + h0 * TWO_PI) -
                                            108.0 * h0 * PI * theta +
                                            108.0 * h0 * SQR(theta)))),
          1.0 / 3.0) };
        // eta = Pi^(2/3)(6 Pi^(1/3) + 2 2^(1/3)(h-1)(3Pi)^(2/3)/R + 2^(2/3) 3^(1/3) R / h)/12
        static constexpr double PI_TO_TWO_THIRD { 2.14502939711102560008 };
        static constexpr double PI_TO_ONE_THIRD { 1.46459188756152326302 };
        static constexpr double TWO_TO_TWO_THIRD { 1.58740105196819947475 };
        static constexpr double THREE_TO_ONE_THIRD { 1.442249570307408382321 };
        static constexpr double TWO_TO_ONE_THIRD { 1.2599210498948731647672 };
        static constexpr double THREE_PI_TO_TWO_THIRD { 4.46184094890142313715794 };
        return static_cast<real_t>(
          PI_TO_TWO_THIRD *
          (6.0 * PI_TO_ONE_THIRD +
           2.0 * TWO_TO_ONE_THIRD * (h0 - ONE) * THREE_PI_TO_TWO_THIRD / R +
           TWO_TO_TWO_THIRD * THREE_TO_ONE_THIRD * R / h0) /
          12.0);
      }
    }
  };

} // namespace metric

#endif // METRICS_QKERR_SCHILD_H
