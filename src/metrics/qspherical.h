/**
 * @file metrics/qspherical.h
 * @brief
 * Flat space-time qspherical metric class xi = log (r - r0), and eta,
 * where: theta = eta + 2h*eta * (PI - 2eta) * (PI - eta) / PI^2
 * @implements
 *   - metric::QSpherical<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_QSPHERICAL_H
#define METRICS_QSPHERICAL_H

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
  class QSpherical : public MetricBase<D> {
    static_assert(D != Dim::_1D, "1D qspherical not available");
    static_assert(D != Dim::_3D, "3D qspherical not fully implemented");

  private:
    const real_t r0, h, chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const bool   small_angle;

  public:
    static constexpr const char*       Label { "qspherical" };
    static constexpr Dimension         PrtlDim = Dim::_3D;
    static constexpr ntt::Metric::type MetricType { ntt::Metric::QSpherical };
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

    QSpherical(const std::vector<ncells_t>&         res,
               const boundaries_t<real_t>&          ext,
               const std::map<std::string, real_t>& params)
      : MetricBase<D> { res, ext }
      , r0 { params.at("r0") }
      , h { params.at("h") }
      , chi_min { math::log(x1_min - r0) }
      , eta_min { theta2eta(x2_min) }
      , phi_min { x3_min }
      , dchi { (math::log(x1_max - r0) - chi_min) / nx1 }
      , deta { (theta2eta(x2_max) - eta_min) / nx2 }
      , dphi { (x3_max - x3_min) / nx3 }
      , dchi_inv { ONE / dchi }
      , deta_inv { ONE / deta }
      , dphi_inv { ONE / dphi }
      , small_angle { eta2theta(HALF * deta) < constant::SMALL_ANGLE } {
      set_dxMin(find_dxMin());
    }

    ~QSpherical() = default;

    /**
     * minimum effective cell size for a given metric (in physical units)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      real_t min_dx { -1.0 };
      for (int i { 0 }; i < nx1; ++i) {
        for (int j { 0 }; j < nx2; ++j) {
          real_t i_ { (real_t)(i) + HALF };
          real_t j_ { (real_t)(j) + HALF };
          real_t dx1_ { h_<1, 1>({ i_, j_ }) };
          real_t dx2_ { h_<2, 2>({ i_, j_ }) };
          real_t dx = 1.0 / math::sqrt(1.0 / dx1_ + 1.0 / dx2_);
          if ((min_dx >= dx) || (min_dx < 0.0)) {
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
        return SQR(dchi) * math::exp(TWO * (x[0] * dchi + chi_min));
      } else if constexpr (i == 2 && j == 2) {
        return SQR(deta) * SQR(dtheta_deta(x[1] * deta + eta_min)) *
               SQR(r0 + math::exp(x[0] * dchi + chi_min));
      } else if constexpr (i == 3 && j == 3) {
        if constexpr (D == Dim::_2D) {
          return SQR((r0 + math::exp(x[0] * dchi + chi_min)) *
                     math::sin(eta2theta(x[1] * deta + eta_min)));
        } else {
          return SQR(dphi) * SQR((r0 + math::exp(x[0] * dchi + chi_min)) *
                                 math::sin(eta2theta(x[1] * deta + eta_min)));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * sqrt(h_ij)
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto sqrt_h_(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");
      if constexpr (i == 1 && j == 1) {
        return dchi * math::exp(x[0] * dchi + chi_min);
      } else if constexpr (i == 2 && j == 2) {
        return deta * dtheta_deta(x[1] * deta + eta_min) *
               (r0 + math::exp(x[0] * dchi + chi_min));
      } else if constexpr (i == 3 && j == 3) {
        if constexpr (D == Dim::_2D) {
          return (r0 + math::exp(x[0] * dchi + chi_min)) *
                 math::sin(eta2theta(x[1] * deta + eta_min));
        } else {
          return dphi * (r0 + math::exp(x[0] * dchi + chi_min)) *
                 math::sin(eta2theta(x[1] * deta + eta_min));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        const real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi) * math::sin(eta2theta(x[1] * deta + eta_min));
      } else if constexpr (D == Dim::_3D) {
        const real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * dphi * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi) * math::sin(eta2theta(x[1] * deta + eta_min));
      }
    }

    /**
     * sqrt(det(h_ij)) / sin(theta)
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim::_1D) {
        const real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi);
      }
    }

    /**
     * differential area at the pole (used in axisymmetric solvers)
     * @param x1 radial coordinate along the axis (code units)
     */
    Inline auto polar_area(real_t x1) const -> real_t {
      if constexpr (D != Dim::_1D) {
        const real_t exp_chi { math::exp(x1 * dchi + chi_min) };
        if (small_angle) {
          const real_t dtheta = eta2theta(HALF * deta);
          return dchi * exp_chi * SQR(r0 + exp_chi) *
                 (static_cast<real_t>(48) - SQR(dtheta)) * SQR(dtheta) /
                 static_cast<real_t>(384);
        } else {
          return dchi * exp_chi * SQR(r0 + exp_chi) *
                 (ONE - math::cos(eta2theta(HALF * deta)));
        }
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
                    "Invalid coordinate conversion: use convert_xyz");
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
     * full coordinate conversion to/from cartesian
     */
    template <Crd in, Crd out>
    Inline void convert_xyz(const coord_t<PrtlDim>& x_in,
                            coord_t<PrtlDim>&       x_out) const {
      static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                      (in == Crd::XYZ && out == Crd::Cd),
                    "Invalid coordinate conversion");
      if (in == Crd::Cd && out == Crd::XYZ) {
        // code -> cart
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(x_in[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(x_in[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(x_in[2]);
        x_out[0] = x_Sph[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]);
        x_out[1] = x_Sph[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]);
        x_out[2] = x_Sph[0] * math::cos(x_Sph[1]);
      } else {
        // cart -> code
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = math::sqrt(SQR(x_in[0]) + SQR(x_in[1]) + SQR(x_in[2]));
        x_Sph[1] = static_cast<real_t>(constant::HALF_PI) -
                   math::atan2(x_in[2], math::sqrt(SQR(x_in[0]) + SQR(x_in[1])));
        x_Sph[2] = static_cast<real_t>(constant::PI) -
                   math::atan2(x_in[1], -x_in[0]);
        x_out[0] = convert<1, Crd::Sph, Crd::Cd>(x_Sph[0]);
        x_out[1] = convert<2, Crd::Sph, Crd::Cd>(x_Sph[1]);
        x_out[2] = convert<3, Crd::Sph, Crd::Cd>(x_Sph[2]);
      }
    }

    /**
     * component-wise vector transformations
     * @note tetrad/sph <-> cntrv <-> cov
     */
    template <idx_t i, Idx in, Idx out>
    Inline auto transform(const coord_t<D>& xi, real_t v_in) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(in != out, "Invalid vector transformation");
      if constexpr ((in == Idx::T && out == Idx::Sph) ||
                    (in == Idx::Sph && out == Idx::T)) {
        // tetrad <-> sph
        return v_in;
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::U) {
        // tetrad/sph -> cntrv
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::Sph)) {
        // cntrv -> tetrad/sph
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::D) {
        // tetrad/sph -> cov
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::Sph)) {
        // cov -> tetrad/sph
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && out == Idx::D) {
        // cntrv -> cov
        return v_in * h_<i, i>(xi);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        // cov -> cntrv
        return v_in / h_<i, i>(xi);
      } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                           (in == Idx::PD && out == Idx::D)) {
        // cntrv -> phys cntrv || phys cov -> cov
        if constexpr (i == 1) {
          return v_in * math::exp(xi[0] * dchi + chi_min) * dchi;
        } else if constexpr (i == 2) {
          return v_in * (dtheta_deta(xi[1] * deta + eta_min) * deta);
        } else if constexpr (D == Dim::_2D) {
          return v_in;
        } else {
          return v_in * dphi;
        }
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        if constexpr (i == 1) {
          return v_in * dchi_inv / (math::exp(xi[0] * dchi + chi_min));
        } else if constexpr (i == 2) {
          return v_in * deta_inv / (dtheta_deta(xi[1] * deta + eta_min));
        } else if constexpr (D == Dim::_2D) {
          return v_in;
        } else {
          return v_in * dphi_inv;
        }
      } else {
        raise::KernelError(HERE, "Invalid transformation");
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
      if constexpr (in != Idx::XYZ && out != Idx::XYZ) {
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = transform<2, in, out>(xi, v_in[1]);
        v_out[2] = transform<3, in, out>(xi, v_in[2]);
      } else {
        raise::KernelError(HERE, "Invalid vector transformation");
      }
    }

    /**
     * full vector transformation to/from cartesian
     */
    template <Idx in, Idx out>
    Inline void transform_xyz(const coord_t<PrtlDim>& xi,
                              const vec_t<Dim::_3D>&  v_in,
                              vec_t<Dim::_3D>&        v_out) const {
      static_assert(in != out, "Invalid vector transformation");
      static_assert(in == Idx::XYZ || out == Idx::XYZ,
                    "Invalid vector transformation");
      if constexpr (in == Idx::T && out == Idx::XYZ) {
        // tetrad -> cart
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
        v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) -
                   v_in[2] * math::sin(x_Sph[2]);
        v_out[1] = v_in[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[2] * math::cos(x_Sph[2]);
        v_out[2] = v_in[0] * math::cos(x_Sph[1]) - v_in[1] * math::sin(x_Sph[1]);
      } else if constexpr (in == Idx::XYZ && out == Idx::T) {
        // cart -> tetrad
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
        v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[2] * math::cos(x_Sph[1]);
        v_out[1] = v_in[0] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) -
                   v_in[2] * math::sin(x_Sph[1]);
        v_out[2] = -v_in[0] * math::sin(x_Sph[2]) + v_in[1] * math::cos(x_Sph[2]);
      } else if (in == Idx::XYZ) {
        // cart -> cov/cntrv
        vec_t<Dim::_3D> v_Tetrad { ZERO };
        transform_xyz<Idx::XYZ, Idx::T>(xi, v_in, v_Tetrad);
        if constexpr (D == Dim::_2D) {
          transform<Idx::T, out>({ xi[0], xi[1] }, v_Tetrad, v_out);
        } else {
          transform<Idx::T, out>(xi, v_Tetrad, v_out);
        }
      } else if (out == Idx::XYZ) {
        // cov/cntrv -> cart
        vec_t<Dim::_3D> v_Tetrad { ZERO };
        if constexpr (D == Dim::_2D) {
          transform<in, Idx::T>({ xi[0], xi[1] }, v_in, v_Tetrad);
        } else {
          transform<in, Idx::T>(xi, v_in, v_Tetrad);
        }
        transform_xyz<Idx::T, Idx::XYZ>(xi, v_Tetrad, v_out);
      }
    }

  private:
    /* Specific for Quasi-Spherical metric with angle stretching ------------ */
    /**
     * @brief Compute d(th) / d(eta) for a given eta.
     */
    Inline auto dtheta_deta(real_t eta) const -> real_t {
      if (cmp::AlmostZero(h)) {
        return ONE;
      } else {
        return (ONE + TWO * h +
                static_cast<real_t>(12.0) * h * (eta * constant::INV_PI) *
                  ((eta * constant::INV_PI) - ONE));
      }
    }

    /**
     * @brief Convert quasi-spherical eta to spherical theta.
     */
    Inline auto eta2theta(real_t eta) const -> real_t {
      if (cmp::AlmostZero(h)) {
        return eta;
      } else {
        return eta + TWO * h * eta * (constant::PI - TWO * eta) *
                       (constant::PI - eta) * constant::INV_PI_SQR;
      }
    }

    /**
     * @brief Convert spherical theta to quasi-spherical eta.
     */
    Inline auto theta2eta(real_t theta) const -> real_t {
      if (cmp::AlmostZero(h)) {
        return theta;
      } else {
        using namespace constant;
        // R = (-9 h^2 (Pi - 2 y) + Sqrt[3] Sqrt[-(h^3 ((-4 + h) (Pi + 2 h Pi)^2
        // + 108 h Pi y - 108 h y^2))])^(1/3)
        double                  R { math::pow(
          -9.0 * SQR(h) * (PI - 2.0 * theta) +
            SQRT3 * math::sqrt((CUBE(h) * ((4.0 - h) * SQR(PI + h * TWO_PI) -
                                           108.0 * h * PI * theta +
                                           108.0 * h * SQR(theta)))),
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
           2.0 * TWO_TO_ONE_THIRD * (h - ONE) * THREE_PI_TO_TWO_THIRD / R +
           TWO_TO_TWO_THIRD * THREE_TO_ONE_THIRD * R / h) /
          12.0);
      }
    }
  };

} // namespace metric

#endif // METRICS_QSPHERICAL_H
