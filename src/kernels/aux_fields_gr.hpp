/**
 * @file kernels/aux_fields_gr.hpp
 * @brief Algorithms for computing the auxiliary fields in GR
 * @implements
 *   - kernel::gr::ComputeAuxE_kernel<>
 *   - kernel::gr::ComputeAuxH_kernel<>
 * @namespaces:
 *   - kernel::gr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_AUX_FIELDS_GR_HPP
#define KERNELS_AUX_FIELDS_GR_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::gr {
  using namespace ntt;

  /**
   * @brief Kernel for computing E
   * @tparam M Metric
   */
  template <class M>
  class ComputeAuxE_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const ndfield_t<D, 6> Df;
    const ndfield_t<D, 6> Bf;
    ndfield_t<D, 6>       Ef;
    const M               metric;

  public:
    ComputeAuxE_kernel(const ndfield_t<D, 6>& Df,
                       const ndfield_t<D, 6>& Bf,
                       const ndfield_t<D, 6>& Ef,
                       const M&               metric)
      : Df { Df }
      , Bf { Bf }
      , Ef { Ef }
      , metric { metric } {}

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const real_t i1_ { COORD(i1) };
        const real_t i2_ { COORD(i2) };

        const real_t h_11_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
        const real_t h_22_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };
        const real_t h_33_00 { metric.template h_<3, 3>({ i1_, i2_ }) };
        const real_t alpha_00 { metric.alpha({ i1_, i2_ }) };
        const real_t alpha_pH0 { metric.alpha({ i1_ + HALF, i2_ }) };
        const real_t alpha_0pH { metric.alpha({ i1_, i2_ + HALF }) };

        real_t w1, w2, ww;
        real_t h_13_ij1, h_13_ij2;
        real_t sqrt_detH_ij1, sqrt_detH_ij2;
        real_t beta_ij1, beta_ij2;

        w1       = metric.sqrt_det_h_tilde({ i1_ - HALF, i2_ });
        w2       = metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ });
        ww       = TWO * metric.sqrt_det_h_tilde({ i1_, i2_ });
        h_13_ij1 = metric.template h_<1, 3>({ i1_ - HALF, i2_ });
        h_13_ij2 = metric.template h_<1, 3>({ i1_ + HALF, i2_ });
        const real_t D1_half { (w1 * h_13_ij1 * Df(i1 - 1, i2, em::dx1) +
                                w2 * h_13_ij2 * Df(i1, i2, em::dx1)) /
                               (ww) };

        sqrt_detH_ij1 = metric.sqrt_det_h({ i1_ - HALF, i2_ });
        sqrt_detH_ij2 = metric.sqrt_det_h({ i1_ + HALF, i2_ });
        beta_ij1      = metric.beta1({ i1_ - HALF, i2_ });
        beta_ij2      = metric.beta1({ i1_ + HALF, i2_ });
        const real_t B2_half {
          (w1 * sqrt_detH_ij1 * beta_ij1 * Bf(i1 - 1, i2, em::bx2) +
           w2 * sqrt_detH_ij2 * beta_ij2 * Bf(i1, i2, em::bx2)) /
          (ww)
        };

        w1            = metric.sqrt_det_h_tilde({ i1_ - HALF, i2_ + HALF });
        w2            = metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ + HALF });
        ww            = TWO * metric.sqrt_det_h_tilde({ i1_, i2_ + HALF });
        sqrt_detH_ij1 = metric.sqrt_det_h({ i1_ - HALF, i2_ + HALF });
        sqrt_detH_ij2 = metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF });
        beta_ij1      = metric.beta1({ i1_ - HALF, i2_ + HALF });
        beta_ij2      = metric.beta1({ i1_ + HALF, i2_ + HALF });
        const real_t B3_half {
          (w1 * sqrt_detH_ij1 * beta_ij1 * Bf(i1 - 1, i2, em::bx3) +
           w2 * sqrt_detH_ij2 * beta_ij2 * Bf(i1, i2, em::bx3)) /
          (ww)
        };

        w1       = metric.sqrt_det_h_tilde({ i1_, i2_ });
        w2       = metric.sqrt_det_h_tilde({ i1_ + ONE, i2_ });
        ww       = TWO * metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ });
        h_13_ij1 = metric.template h_<1, 3>({ i1_, i2_ });
        h_13_ij2 = metric.template h_<1, 3>({ i1_ + ONE, i2_ });
        const real_t D3_half { (w1 * h_13_ij1 * Df(i1, i2, em::dx3) +
                                w2 * h_13_ij2 * Df(i1 + 1, i2, em::dx3)) /
                               (ww) };

        const real_t D1_cov { h_11_pH0 * Df(i1, i2, em::dx1) + D3_half };
        const real_t D2_cov { h_22_0pH * Df(i1, i2, em::dx2) };
        const real_t D3_cov { h_33_00 * Df(i1, i2, em::dx3) + D1_half };

        Ef(i1, i2, em::ex1) = alpha_pH0 * D1_cov;
        Ef(i1, i2, em::ex2) = alpha_0pH * D2_cov - B3_half;
        Ef(i1, i2, em::ex3) = alpha_00 * D3_cov + B2_half;
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxE_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxE_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Kernel for computing H
   * @tparam M Metric
   */
  template <class M>
  class ComputeAuxH_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const ndfield_t<D, 6> Df;
    const ndfield_t<D, 6> Bf;
    ndfield_t<D, 6>       Hf;
    const M               metric;

  public:
    ComputeAuxH_kernel(const ndfield_t<D, 6>& Df,
                       const ndfield_t<D, 6>& Bf,
                       const ndfield_t<D, 6>& Hf,
                       const M&               metric)
      : Df { Df }
      , Bf { Bf }
      , Hf { Hf }
      , metric { metric } {}

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const real_t i1_ { COORD(i1) };
        const real_t i2_ { COORD(i2) };

        const real_t h_11_0pH { metric.template h_<1, 1>({ i1_, i2_ + HALF }) };
        const real_t h_22_pH0 { metric.template h_<2, 2>({ i1_ + HALF, i2_ }) };
        const real_t h_33_pHpH { metric.template h_<3, 3>(
          { i1_ + HALF, i2_ + HALF }) };
        const real_t alpha_0pH { metric.alpha({ i1_, i2_ + HALF }) };
        const real_t alpha_pH0 { metric.alpha({ i1_ + HALF, i2_ }) };
        const real_t alpha_pHpH { metric.alpha({ i1_ + HALF, i2_ + HALF }) };

        real_t w1, w2, ww;
        real_t h_13_ij1, h_13_ij2;
        real_t sqrt_detH_ij1, sqrt_detH_ij2;
        real_t beta_ij1, beta_ij2;

        w1       = metric.sqrt_det_h_tilde({ i1_, i2_ + HALF });
        w2       = metric.sqrt_det_h_tilde({ i1_ + ONE, i2_ + HALF });
        ww       = TWO * metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ + HALF });
        h_13_ij1 = metric.template h_<1, 3>({ i1_, i2_ + HALF });
        h_13_ij2 = metric.template h_<1, 3>({ i1_ + ONE, i2_ + HALF });
        const real_t B1_half { (w1 * h_13_ij1 * Bf(i1, i2, em::bx1) +
                                w2 * h_13_ij2 * Bf(i1 + 1, i2, em::bx1)) /
                               (ww) };

        sqrt_detH_ij1 = metric.sqrt_det_h({ i1_, i2_ + HALF });
        sqrt_detH_ij2 = metric.sqrt_det_h({ i1_ + ONE, i2_ + HALF });
        beta_ij1      = metric.beta1({ i1_, i2_ + HALF });
        beta_ij2      = metric.beta1({ i1_ + ONE, i2_ + HALF });
        const real_t D2_half {
          (w1 * sqrt_detH_ij1 * beta_ij1 * Df(i1, i2, em::dx2) +
           w2 * sqrt_detH_ij2 * beta_ij2 * Df(i1 + 1, i2, em::dx2)) /
          (ww)
        };

        w1            = metric.sqrt_det_h_tilde({ i1_, i2_ });
        w2            = metric.sqrt_det_h_tilde({ i1_ + ONE, i2_ });
        ww            = TWO * metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ });
        sqrt_detH_ij1 = metric.sqrt_det_h({ i1_, i2_ });
        sqrt_detH_ij2 = metric.sqrt_det_h({ i1_ + ONE, i2_ });
        beta_ij1      = metric.beta1({ i1_, i2_ });
        beta_ij2      = metric.beta1({ i1_ + ONE, i2_ });
        const real_t D3_half {
          (w1 * sqrt_detH_ij1 * beta_ij1 * Df(i1, i2, em::dx3) +
           w2 * sqrt_detH_ij2 * beta_ij2 * Df(i1 + 1, i2, em::dx3)) /
          (ww)
        };

        w1       = metric.sqrt_det_h_tilde({ i1_ - HALF, i2_ + HALF });
        w2       = metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ + HALF });
        ww       = TWO * metric.sqrt_det_h_tilde({ i1_, i2_ + HALF });
        h_13_ij1 = metric.template h_<1, 3>({ i1_ - HALF, i2_ + HALF });
        h_13_ij2 = metric.template h_<1, 3>({ i1_ + HALF, i2_ + HALF });
        const real_t B3_half { (w1 * h_13_ij1 * Bf(i1 - 1, i2, em::bx3) +
                                w2 * h_13_ij2 * Bf(i1, i2, em::bx3)) /
                               (ww) };

        const real_t B1_cov { h_11_0pH * Bf(i1, i2, em::bx1) + B3_half };
        const real_t B2_cov { h_22_pH0 * Bf(i1, i2, em::bx2) };
        const real_t B3_cov { h_33_pHpH * Bf(i1, i2, em::bx3) + B1_half };

        Hf(i1, i2, em::hx1) = alpha_0pH * B1_cov;
        Hf(i1, i2, em::hx2) = alpha_pH0 * B2_cov + D3_half;
        Hf(i1, i2, em::hx3) = alpha_pHpH * B3_cov - D2_half;
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxH_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxH_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Kernel for computing time average of B and D
   * @tparam M Metric
   */
  template <class M>
  class TimeAverageDB_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const ndfield_t<D, 6> BDf;
    ndfield_t<D, 6>       BDf0;
    const M               metric;

  public:
    TimeAverageDB_kernel(const ndfield_t<D, 6>& BDf,
                         const ndfield_t<D, 6>& BDf0,
                         const M&               metric)
      : BDf { BDf }
      , BDf0 { BDf0 }
      , metric { metric } {}

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        BDf0(i1, i2, em::bx1) = HALF *
                                (BDf0(i1, i2, em::bx1) + BDf(i1, i2, em::bx1));
        BDf0(i1, i2, em::bx2) = HALF *
                                (BDf0(i1, i2, em::bx2) + BDf(i1, i2, em::bx2));
        BDf0(i1, i2, em::bx3) = HALF *
                                (BDf0(i1, i2, em::bx3) + BDf(i1, i2, em::bx3));
        BDf0(i1, i2, em::ex1) = HALF *
                                (BDf0(i1, i2, em::ex1) + BDf(i1, i2, em::ex1));
        BDf0(i1, i2, em::ex2) = HALF *
                                (BDf0(i1, i2, em::ex2) + BDf(i1, i2, em::ex2));
        BDf0(i1, i2, em::ex3) = HALF *
                                (BDf0(i1, i2, em::ex3) + BDf(i1, i2, em::ex3));
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxH_kernel: 2D implementation called for D != 2");
      }
    }
  };

  /**
   * @brief Kernel for computing time average of J
   * @tparam M Metric
   */
  template <class M>
  class TimeAverageJ_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    ndfield_t<D, 3>       Jf;
    const ndfield_t<D, 3> Jf0;
    const M               metric;

  public:
    TimeAverageJ_kernel(const ndfield_t<D, 3>& Jf,
                        const ndfield_t<D, 3>& Jf0,
                        const M&               metric)
      : Jf { Jf }
      , Jf0 { Jf0 }
      , metric { metric } {}

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        Jf(i1, i2, cur::jx1) = HALF *
                               (Jf0(i1, i2, cur::jx1) + Jf(i1, i2, cur::jx1));
        Jf(i1, i2, cur::jx2) = HALF *
                               (Jf0(i1, i2, cur::jx2) + Jf(i1, i2, cur::jx2));
        Jf(i1, i2, cur::jx3) = HALF *
                               (Jf0(i1, i2, cur::jx3) + Jf(i1, i2, cur::jx3));
      } else {
        raise::KernelError(
          HERE,
          "ComputeAuxH_kernel: 2D implementation called for D != 2");
      }
    }
  };
} // namespace kernel::gr

#endif // KERNELS_AUX_FIELDS_GR_HPP
