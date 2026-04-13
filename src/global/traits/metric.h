/**
 * @file traits/metric.h
 * @brief Defines a set of traits to check if a metric class satisfies certain conditions
 * @implements
 *   - traits::metric::HasD<> - checks if metric has Dim static member
 *   - traits::metric::HasCoordType<> - checks if metric has CoordType static member
 *   - traits::metric::HasH_ij<> - checks if metric has h_ij method
 *   - traits::metric::HasHij<> - checks if metric has hij method
 *   - traits::metric::HasSqrtH_ij<> - checks if metric has sqrt_h_ij method
 *   - traits::metric::HasSqrtDetH<> - checks if metric has sqrt_det_h method
 *   - traits::metric::HasSqrtDetHTilde<> - checks if metric has sqrt_det_h_tilde method
 *   - traits::metric::HasPolarArea<> - checks if metric has polar_area method
 *   - traits::metric::HasTotVolume<> - checks if metric has totVolume method
 *   - traits::metric::HasTransform_i<> - checks if metric has transform method for single component
 *   - traits::metric::HasTransform<> - checks if metric has transform method for vector
 *   - traits::metric::HasTransformXYZ<> - checks if metric has transform_xyz method for vector
 *   - traits::metric::HasConvert_i<> - checks if metric has convert method for single component
 *   - traits::metric::HasConvert<> - checks if metric has convert method for vector
 *   - traits::metric::HasConvertXYZ<> - checks if metric has convert_xyz method for vector
 *   - traits::metric::HasAlpha<> - checks if metric has alpha method
 *   - traits::metric::HasBeta1<> - checks if metric has beta1 method
 *   - traits::metric::HasMetricDerivatives<> - checks if metric has metric derivatives methods
 * @namespaces:
 *   - traits::metric::
 */
#ifndef GLOBAL_TRAITS_METRIC_H
#define GLOBAL_TRAITS_METRIC_H

#include "enums.h"
#include "global.h"

namespace traits::metric {

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
    { m.template convert<Crd::Cd, Crd::Ph>(x_in, x_out) } -> std::same_as<void>;
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

} // namespace traits::metric

#endif // GLOBAL_TRAITS_METRIC_H
