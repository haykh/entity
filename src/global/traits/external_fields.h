#ifndef GLOBAL_TRAITS_EXTERNAL_FIELDS_H
#define GLOBAL_TRAITS_EXTERNAL_FIELDS_H

#include "global.h"

namespace traits::external_fields {

  template <class F, Dimension D>
  concept HasFx1 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.fx1(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasFx2 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.fx2(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasFx3 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.fx3(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasEx1 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.ex1(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasEx2 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.ex2(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasEx3 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.ex3(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasBx1 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.bx1(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasBx2 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.bx2(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasBx3 = requires(const F& t, const coord_t<D>& x_Ph) {
    { t.bx3(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class F, Dimension D>
  concept HasExternalF = (HasFx1<F, D> or HasFx2<F, D> or HasFx3<F, D>);

  template <class F, Dimension D>
  concept HasExternalE = (HasEx1<F, D> or HasEx2<F, D> or HasEx3<F, D>);

  template <class F, Dimension D>
  concept HasExternalB = (HasBx1<F, D> or HasBx2<F, D> or HasBx3<F, D>);

} // namespace traits::external_fields

#endif // GLOBAL_TRAITS_EXTERNAL_FIELDS_H