/**
 * @file metrics/minkowski.h
 * @brief Minkowski metric class: diag(-1, 1, 1, 1)
 * @implements
 *   - metric::Minkowski<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * @note Cell sizes in each direction dx1 = dx2 = dx3 are assumed equal
 */

#ifndef METRICS_MINKOWSKI_H
#define METRICS_MINKOWSKI_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>

namespace metric {

  template <Dimension D>
  class Minkowski : public MetricBase<D> {
    const real_t dx, dx_inv;

  public:
    static constexpr const char*       Label { "minkowski" };
    static constexpr Dimension         PrtlDim { D };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Minkowski };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Cart };
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

    Minkowski(const std::vector<ncells_t>& res,
              const boundaries_t<real_t>&  ext,
              const std::map<std::string, real_t>& = {})
      : MetricBase<D> { res, ext }
      , dx { (x1_max - x1_min) / nx1 }
      , dx_inv { ONE / dx } {
      set_dxMin(find_dxMin());
      const auto epsilon = std::numeric_limits<real_t>::epsilon() *
                           static_cast<real_t>(100.0);
      if constexpr (D != Dim::_1D) {
        raise::ErrorIf(
          not cmp::AlmostEqual((x2_max - x2_min) / (real_t)(nx2), dx, epsilon),
          "dx2 must be equal to dx1 in 2D",
          HERE);
      }
      if constexpr (D == Dim::_3D) {
        raise::ErrorIf(
          not cmp::AlmostEqual((x3_max - x3_min) / (real_t)(nx3), dx, epsilon),
          "dx3 must be equal to dx1 in 3D",
          HERE);
      }
    }

    ~Minkowski() = default;

    /**
     * minimum effective cell size for a given metric (in physical units)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      return dx / math::sqrt(static_cast<real_t>(D));
    }

    /**
     * total volume of the region described by the metric (in physical units)
     */
    [[nodiscard]]
    auto totVolume() const -> real_t override {
      if constexpr (D == Dim::_1D) {
        return x1_max - x1_min;
      } else if constexpr (D == Dim::_2D) {
        return (x1_max - x1_min) * (x2_max - x2_min);
      } else {
        return (x1_max - x1_min) * (x2_max - x2_min) * (x3_max - x3_min);
      }
    }

    /**
     * metric component with lower indices: h_ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h_(const coord_t<D>&) const -> real_t {
      static_assert(i > 0 && i <= static_cast<idx_t>(D), "Invalid index i");
      static_assert(j > 0 && j <= static_cast<idx_t>(D), "Invalid index j");
      if constexpr (i == j) {
        if constexpr (i <= static_cast<idx_t>(D)) {
          return SQR(dx);
        } else {
          return ONE;
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
    Inline auto sqrt_h_(const coord_t<D>&) const -> real_t {
      static_assert(i > 0 && i <= static_cast<idx_t>(D),
                    "Invalid coordinate index");
      static_assert(j > 0 && j <= static_cast<idx_t>(D),
                    "Invalid coordinate index");
      if constexpr (i == j) {
        if constexpr (i <= static_cast<idx_t>(D)) {
          return dx;
        } else {
          return ONE;
        }
      } else {
        return ZERO;
      }
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t {
      if constexpr (D == Dim::_1D) {
        return dx;
      } else if constexpr (D == Dim::_2D) {
        return SQR(dx);
      } else {
        return CUBE(dx);
      }
    }

    /**
     * component-wise coordinate conversions
     */
    template <idx_t i, Crd in, Crd out>
    Inline auto convert(real_t x_in) const -> real_t {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(i > 0 && i <= static_cast<idx_t>(D), "Invalid index i");
      static_assert((in == Crd::Cd && (out == Crd::XYZ || out == Crd::Ph)) ||
                      ((in == Crd::XYZ || in == Crd::Ph) && out == Crd::Cd),
                    "Invalid coordinate conversion");
      if constexpr (in == Crd::Cd && (out == Crd::XYZ || out == Crd::Ph)) {
        // code -> cart/phys
        if constexpr (i == 1) {
          return x_in * dx + x1_min;
        } else if constexpr (i == 2) {
          return x_in * dx + x2_min;
        } else {
          return x_in * dx + x3_min;
        }
      } else {
        // cart/phys -> code
        if constexpr (i == 1) {
          return (x_in - x1_min) * dx_inv;
        } else if constexpr (i == 2) {
          return (x_in - x2_min) * dx_inv;
        } else {
          return (x_in - x3_min) * dx_inv;
        }
      }
    }

    /**
     * full coordinate conversions
     */
    template <Crd in, Crd out>
    Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
      static_assert(in != out, "Invalid coordinate conversion");
      if constexpr ((in != Crd::Sph) && (out != Crd::Sph)) {
        // code <-> cart/phys
        if constexpr (D == Dim::_1D) {
          x_out[0] = convert<1, in, out>(x_in[0]);
        } else if constexpr (D == Dim::_2D) {
          x_out[0] = convert<1, in, out>(x_in[0]);
          x_out[1] = convert<2, in, out>(x_in[1]);
        } else {
          x_out[0] = convert<1, in, out>(x_in[0]);
          x_out[1] = convert<2, in, out>(x_in[1]);
          x_out[2] = convert<3, in, out>(x_in[2]);
        }
      } else {
        // sph <-> code
        static_assert((in == Crd::Sph && out == Crd::Cd) ||
                        (in == Crd::Cd && out == Crd::Sph),
                      "Invalid coordinate conversion");
        static_assert(D != Dim::_1D, "Invalid coordinate conversion");
        if constexpr (in == Crd::Sph && out == Crd::Cd) {
          // sph -> code
          if constexpr (D == Dim::_2D) {
            coord_t<Dim::_2D> x_XYZ { ZERO };
            x_XYZ[0] = x_in[0] * math::sin(x_in[1]);
            x_XYZ[1] = x_in[0] * math::cos(x_in[1]);
            convert<Crd::XYZ, Crd::Cd>(x_XYZ, x_out);
          } else {
            coord_t<Dim::_3D> x_XYZ { ZERO };
            x_XYZ[0] = x_in[0] * math::sin(x_in[1]) * math::cos(x_in[2]);
            x_XYZ[1] = x_in[0] * math::sin(x_in[1]) * math::sin(x_in[2]);
            x_XYZ[2] = x_in[0] * math::cos(x_in[1]);
            convert<Crd::XYZ, Crd::Cd>(x_XYZ, x_out);
          }
        } else {
          // code -> sph
          if constexpr (D == Dim::_2D) {
            coord_t<Dim::_2D> x_XYZ { ZERO };
            convert<Crd::Cd, Crd::XYZ>(x_in, x_XYZ);
            x_out[0] = math::sqrt(SQR(x_XYZ[0]) + SQR(x_XYZ[1]));
            x_out[1] = static_cast<real_t>(constant::HALF_PI) -
                       math::atan2(x_XYZ[1], x_XYZ[0]);
          } else {
            coord_t<Dim::_3D> x_XYZ { ZERO };
            convert<Crd::Cd, Crd::XYZ>(x_in, x_XYZ);
            x_out[0] = math::sqrt(SQR(x_XYZ[0]) + SQR(x_XYZ[1]) + SQR(x_XYZ[2]));
            x_out[1] = static_cast<real_t>(constant::HALF_PI) -
                       math::atan2(x_XYZ[2],
                                   math::sqrt(SQR(x_XYZ[0]) + SQR(x_XYZ[1])));
            x_out[2] = static_cast<real_t>(constant::PI) -
                       math::atan2(x_XYZ[1], -x_XYZ[0]);
          }
        }
      }
    }

    /**
     * full coordinate conversions to cartesian
     * @note for compatibility purposes
     */
    template <Crd in, Crd out>
    Inline void convert_xyz(const coord_t<PrtlDim>& x_in,
                            coord_t<PrtlDim>&       x_out) const {
      static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                      (in == Crd::XYZ && out == Crd::Cd),
                    "Invalid coordinate conversion");
      // code <-> cart
      convert<in, out>(x_in, x_out);
    }

    /**
     * component-wise vector transformations
     * @note tetrad/cart <-> cntrv <-> cov
     */
    template <idx_t i, Idx in, Idx out>
    Inline auto transform(const coord_t<D>& xi, real_t v_in) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(in != out, "Invalid vector transformation");
      if constexpr (i > static_cast<idx_t>(D)) {
        return v_in;
      } else if constexpr ((in == Idx::T && out == Idx::XYZ) ||
                           (in == Idx::XYZ && out == Idx::T)) {
        // tetrad <-> cart
        return v_in;
      } else if constexpr ((in == Idx::T || in == Idx::XYZ) && out == Idx::U) {
        // tetrad/cart -> cntrv
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::XYZ)) {
        // cntrv -> tetrad/cart
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr ((in == Idx::T || in == Idx::XYZ) && out == Idx::D) {
        // tetrad/cart -> cov
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::XYZ)) {
        // cov -> tetrad/cart
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
        return v_in * dx;
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        return v_in * dx_inv;
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
      v_out[0] = transform<1, in, out>(xi, v_in[0]);
      v_out[1] = transform<2, in, out>(xi, v_in[1]);
      v_out[2] = transform<3, in, out>(xi, v_in[2]);
    }

    /**
     * full vector transformations to cartesian
     * @note for compatibility purposes
     */
    template <Idx in, Idx out>
    Inline void transform_xyz(const coord_t<PrtlDim>& xi,
                              const vec_t<Dim::_3D>&  v_in,
                              vec_t<Dim::_3D>&        v_out) const {
      static_assert(in == Idx::XYZ || out == Idx::XYZ,
                    "Invalid vector transformation");
      transform<in, out>(xi, v_in, v_out);
    }
  };

} // namespace metric

#endif
