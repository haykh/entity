/**
 * @file metrics/traits.h
 * @brief Defines a set of traits to check if a metric class satisfies certain conditions
 * @implements
 *   - metric::traits::HasD<> - checks if metric has Dim static member
 *   - metric::traits::HasCoordType<> - checks if metric has CoordType static member
 *   - metric::traits::HasH_ij<> - checks if metric has h_ij method
 *   - metric::traits::HasHij<> - checks if metric has hij method
 *   - metric::traits::HasSqrtH_ij<> - checks if metric has sqrt_h_ij method
 *   - metric::traits::HasSqrtDetH<> - checks if metric has sqrt_det_h method
 *   - metric::traits::HasSqrtDetHTilde<> - checks if metric has sqrt_det_h_tilde method
 *   - metric::traits::HasPolarArea<> - checks if metric has polar_area method
 *   - metric::traits::HasTotVolume<> - checks if metric has totVolume method
 *   - metric::traits::HasTransform_i<> - checks if metric has transform method for single component
 *   - metric::traits::HasTransform<> - checks if metric has transform method for vector
 *   - metric::traits::HasTransformXYZ<> - checks if metric has transform_xyz method for vector
 *   - metric::traits::HasConvert_i<> - checks if metric has convert method for single component
 *   - metric::traits::HasConvert<> - checks if metric has convert method for vector
 *   - metric::traits::HasConvertXYZ<> - checks if metric has convert_xyz method for vector
 *   - metric::traits::HasAlpha<> - checks if metric has alpha method
 *   - metric::traits::HasBeta1<> - checks if metric has beta1 method
 *   - metric::traits::HasMetricDerivatives<> - checks if metric has metric derivatives methods
 * @namespaces:
 *   - metric::traits::
 */
#ifndef METRICS_TRAITS_H
#define METRICS_TRAITS_H

#include "enums.h"
#include "global.h"

namespace metric {
  namespace traits {

    template <class M>
    concept HasD = requires {
      { M::Dim } -> std::convertible_to<Dimension>;
    };

    template <class M>
    concept HasPrtlDim = requires {
      { M::PrtlDim } -> std::convertible_to<Dimension>;
    };

    template <class M>
    concept HasCoordType = requires {
      { M::CoordType } -> std::convertible_to<ntt::Coord::type>;
    };

    template <class M>
    concept HasH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template h_<1, 1>(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasHij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template h<1, 1>(xi) } -> std::convertible_to<real_t>;
    };
    template <class M>
    concept HasSqrtH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template sqrt_h_<1, 1>(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasSqrtDetH = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.sqrt_det_h(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasSqrtDetHTilde = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.sqrt_det_h_tilde(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasPolarArea = requires(const M& m, real_t xi_2) {
      { m.polar_area(xi_2) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasTotVolume = requires(const M& m) {
      { m.totVolume() } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasTransform_i = requires(const M&               m,
                                      const coord_t<M::Dim>& xi,
                                      real_t                 v_in) {
      {
        m.template transform<1, Idx::U, Idx::D>(xi, v_in)
      } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasTransform = requires(const M&               m,
                                    const coord_t<M::Dim>& xi,
                                    const vec_t<Dim::_3D>& v_in,
                                    vec_t<Dim::_3D>&       v_out) {
      {
        m.template transform<Idx::U, Idx::D>(xi, v_in, v_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasTransformXYZ = requires(const M&                   m,
                                       const coord_t<M::PrtlDim>& xi,
                                       const vec_t<Dim::_3D>&     v_in,
                                       vec_t<Dim::_3D>&           v_out) {
      {
        m.template transform_xyz<Idx::XYZ, Idx::D>(xi, v_in, v_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasConvert_i = requires(const M& m, real_t x) {
      {
        m.template convert<1, Crd::Cd, Crd::Ph>(x)
      } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasConvert = requires(const M&               m,
                                  const coord_t<M::Dim>& x_in,
                                  coord_t<M::Dim>&       x_out) {
      {
        m.template convert<Crd::Cd, Crd::Ph>(x_in, x_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasConvertXYZ = requires(const M&                   m,
                                     const coord_t<M::PrtlDim>& x_in,
                                     coord_t<M::PrtlDim>&       x_out) {
      {
        m.template convert_xyz<Crd::Cd, Crd::XYZ>(x_in, x_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasAlpha = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.alpha(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasBeta1 = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.beta1(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasMetricDerivatives = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.dr_alpha(xi) } -> std::convertible_to<real_t>;
      { m.dr_beta1(xi) } -> std::convertible_to<real_t>;
      { m.dr_h11(xi) } -> std::convertible_to<real_t>;
      { m.dr_h22(xi) } -> std::convertible_to<real_t>;
      { m.dr_h33(xi) } -> std::convertible_to<real_t>;
      { m.dr_h13(xi) } -> std::convertible_to<real_t>;
      { m.dt_alpha(xi) } -> std::convertible_to<real_t>;
      { m.dt_beta1(xi) } -> std::convertible_to<real_t>;
      { m.dt_h11(xi) } -> std::convertible_to<real_t>;
      { m.dt_h22(xi) } -> std::convertible_to<real_t>;
      { m.dt_h33(xi) } -> std::convertible_to<real_t>;
      { m.dt_h13(xi) } -> std::convertible_to<real_t>;
    };

  } // namespace traits
} // namespace metric

#endif // METRICS_TRAITS_H
