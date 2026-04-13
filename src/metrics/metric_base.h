/**
 * @file metrics/metric_base.h
 * @brief Base class for all the metrics
 * @implements
 *   - metric::MetricBase
 * @namespaces:
 *   - metric::
 * @note
 * Other metrics inherit from this class using the CRTP pattern
 * see: https://en.cppreference.com/w/cpp/language/crtp
 * @note
 * Coordinate transformations
 *
 *   +---> Cart
 *   |
 *   v
 * Code
 *   ^
 *   |
 *   +---> Sph
 *
 * Code: coordinates in code units
 * Cart: coordinates in global Cartesian basis
 * Sph: coordinates in spherical basis
 * @note
 * Vector transformations
 *
 * Cntrv (A^mu)
 *   ^  ^
 *   |  |
 *   |  v
 *   | Hat <---> Cart (A_xyz)
 *   |  ^
 *   |  |
 *   v  v
 *   Cov (A_mu)
 *
 * Cntrv: contravariant vector
 * Cov: covariant vector
 * Hat: hatted (orthonormal) basis vector
 * Cart: global Cartesian basis vector (defined for diagonal only)
 */

#ifndef METRICS_METRIC_BASE_H
#define METRICS_METRIC_BASE_H

#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include <vector>

#include "traits/metric.h"

namespace metric {

  namespace {
    template <Dimension D, unsigned short i>
    auto getNXi(const std::vector<ncells_t>& res) -> real_t {
      if constexpr (i >= static_cast<unsigned short>(D)) {
        return ONE;
      } else {
        raise::ErrorIf(res.size() <= i, "Invalid res size provided to metric", HERE);
        return static_cast<real_t>(res.at(i));
      }
    };

    template <Dimension D, unsigned short i, bool min>
    auto getExtent(const boundaries_t<real_t>& ext) -> real_t {
      if constexpr (i >= static_cast<unsigned short>(D)) {
        return ZERO;
      } else {
        raise::ErrorIf(ext.size() <= i, "Invalid ext size provided to metric", HERE);
        return min ? ext.at(i).first : ext.at(i).second;
      }
    };

    constexpr bool XMin = true;
    constexpr bool XMax = false;
  }; // namespace

  /**
   * Virtual parent metric class template: h_ij
   * Coordinates vary from `0` to `nx1` ... (code units)
   */
  template <Dimension D>
  struct MetricBase {
    static constexpr Dimension Dim { D };

    MetricBase(const std::vector<ncells_t>& res, const boundaries_t<real_t>& ext)
      : nx1 { getNXi<D, 0>(res) }
      , nx2 { getNXi<D, 1>(res) }
      , nx3 { getNXi<D, 2>(res) }
      , x1_min { getExtent<D, 0, XMin>(ext) }
      , x1_max { getExtent<D, 0, XMax>(ext) }
      , x2_min { getExtent<D, 1, XMin>(ext) }
      , x2_max { getExtent<D, 1, XMax>(ext) }
      , x3_min { getExtent<D, 2, XMin>(ext) }
      , x3_max { getExtent<D, 2, XMax>(ext) } {}

    ~MetricBase() = default;

    [[nodiscard]]
    virtual auto find_dxMin() const -> real_t = 0;

    [[nodiscard]]
    virtual auto totVolume() const -> real_t = 0;

    [[nodiscard]]
    auto dxMin() const -> real_t {
      return dx_min;
    }

    auto set_dxMin(real_t dxmin) -> void {
      dx_min = dxmin;
    }

  protected:
    real_t dx_min;

    // max of coordinates in code units
    const real_t nx1, nx2, nx3;
    // extent in `x1` in physical units
    const real_t x1_min, x1_max;
    // extent in `x2` in physical units
    const real_t x2_min, x2_max;
    // extent in `x3` in physical units
    const real_t x3_min, x3_max;
  };

} // namespace metric

template <class M>
concept Metric = traits::metric::HasD<M> and traits::metric::HasCoordType<M>;

#endif // METRICS_METRIC_BASE_H
