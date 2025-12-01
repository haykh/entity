/**
 * @file kernels/particle_pusher_gr.h
 * @brief Implementation of the particle pusher for GR
 * @implements
 *   - kernel::gr::Pusher_kernel<>
 * @namespaces:
 *   - kernel::gr::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_PARTICLE_PUSHER_GR_HPP
#define KERNELS_PARTICLE_PUSHER_GR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    I = static_cast<int>((XI + 1)) - 1;                                        \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

#define DERIVATIVE_IN_R(func, x)                                               \
  ((func({ x[0] + epsilon, x[1] }) - func({ x[0] - epsilon, x[1] })) /         \
   (TWO * epsilon))

#define DERIVATIVE_IN_TH(func, x)                                              \
  ((func({ x[0], x[1] + epsilon }) - func({ x[0], x[1] - epsilon })) /         \
   (TWO * epsilon))

/* -------------------------------------------------------------------------- */

namespace kernel::gr {
  using namespace ntt;

  struct Massive_t {};

  struct Massless_t {};

  /**
   * @brief Algorithm for the Particle pusher
   * @tparam M Metric
   */
  template <class M>
  class Pusher_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

  private:
    const randacc_ndfield_t<D, 6> DB;
    const randacc_ndfield_t<D, 6> DB0;
    array_t<int*>                 i1, i2, i3;
    array_t<int*>                 i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>            dx1, dx2, dx3;
    array_t<prtldx_t*>            dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>              ux1, ux2, ux3;
    array_t<real_t*>              phi;
    array_t<short*>               tag;
    const M                       metric;

    const real_t         coeff, dt;
    const int            ni1, ni2, ni3;
    const real_t         epsilon;
    const unsigned short niter;

    bool is_axis_i2min { false }, is_axis_i2max { false };
    bool is_absorb_i1min { false }, is_absorb_i1max { false };

  public:
    Pusher_kernel(const ndfield_t<D, 6>&      DB,
                  const ndfield_t<D, 6>&      DB0,
                  array_t<int*>&              i1,
                  array_t<int*>&              i2,
                  array_t<int*>&              i3,
                  array_t<int*>&              i1_prev,
                  array_t<int*>&              i2_prev,
                  array_t<int*>&              i3_prev,
                  array_t<prtldx_t*>&         dx1,
                  array_t<prtldx_t*>&         dx2,
                  array_t<prtldx_t*>&         dx3,
                  array_t<prtldx_t*>&         dx1_prev,
                  array_t<prtldx_t*>&         dx2_prev,
                  array_t<prtldx_t*>&         dx3_prev,
                  array_t<real_t*>&           ux1,
                  array_t<real_t*>&           ux2,
                  array_t<real_t*>&           ux3,
                  array_t<real_t*>&           phi,
                  array_t<short*>&            tag,
                  const M&                    metric,
                  real_t                      coeff,
                  real_t                      dt,
                  int                         ni1,
                  int                         ni2,
                  int                         ni3,
                  real_t                      epsilon,
                  unsigned short              niter,
                  const boundaries_t<PrtlBC>& boundaries)
      : DB { DB }
      , DB0 { DB0 }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , tag { tag }
      , metric { metric }
      , coeff { coeff }
      , dt { dt }
      , ni1 { ni1 }
      , ni2 { ni2 }
      , ni3 { ni3 }
      , epsilon { epsilon }
      , niter { niter } {

      raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
      is_absorb_i1min = (boundaries[0].first == PrtlBC::ABSORB) ||
                        (boundaries[0].first == PrtlBC::HORIZON);
      is_absorb_i1max = (boundaries[0].second == PrtlBC::ABSORB) ||
                        (boundaries[0].second == PrtlBC::HORIZON);
      is_axis_i2min = (boundaries[1].first == PrtlBC::AXIS);
      is_axis_i2max = (boundaries[1].second == PrtlBC::AXIS);
    }

    /**
     * @brief Main pusher subroutine for photon particles.
     */
    Inline void operator()(Massless_t, index_t) const;

    /**
     * @brief Main pusher subroutine for massive particles.
     */
    Inline void operator()(Massive_t, index_t) const;

    /**
     * @brief Iterative geodesic pusher substep for momentum only.
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicMomentumPush(T,
                                     const coord_t<D>&      xp,
                                     const vec_t<Dim::_3D>& vp,
                                     vec_t<Dim::_3D>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep for coordinate only.
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    template <typename T>
    Inline void GeodesicCoordinatePush(T,
                                       const coord_t<D>&      xp,
                                       const vec_t<Dim::_3D>& vp,
                                       coord_t<D>&            xp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicFullPush(T,
                                 const coord_t<D>&      xp,
                                 const vec_t<Dim::_3D>& vp,
                                 coord_t<D>&            xp_upd,
                                 vec_t<Dim::_3D>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate (at n + 1/2 for leapfrog, at n for old scheme).
     * @param vp particle velocity (at n + 1/2 for leapfrog, at n for old scheme).
     * @param phi updated phi [return].
     */
    template <typename T>
    Inline void UpdatePhi(T,
                          const coord_t<D>&      xp,
                          const vec_t<Dim::_3D>& vp,
                          real_t                 phi) const;

    /**
     * @brief EM pusher (Boris) substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param Dp_hat hatted electric field at the particle position.
     * @param Bp_hat hatted magnetic field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EMHalfPush(const coord_t<D>&      xp,
                           const vec_t<Dim::_3D>& vp,
                           const vec_t<Dim::_3D>& Dp_hat,
                           const vec_t<Dim::_3D>& Bp_hat,
                           vec_t<Dim::_3D>&       vp_upd) const {
      vec_t<Dim::_3D> D0 { Dp_hat[0], Dp_hat[1], Dp_hat[2] };
      vec_t<Dim::_3D> B0 { Bp_hat[0], Bp_hat[1], Bp_hat[2] };
      vec_t<Dim::_3D> vp_hat { ZERO }, vp_upd_hat { ZERO };
      metric.template transform<Idx::D, Idx::T>(xp, vp, vp_upd_hat);

      // this is a half-push
      real_t COEFF { coeff * HALF * metric.alpha(xp) };

      D0[0] *= COEFF;
      D0[1] *= COEFF;
      D0[2] *= COEFF;

      vp_upd_hat[0] += D0[0];
      vp_upd_hat[1] += D0[1];
      vp_upd_hat[2] += D0[2];

      COEFF *= ONE / math::sqrt(ONE + SQR(vp_upd_hat[0]) + SQR(vp_upd_hat[1]) +
                                SQR(vp_upd_hat[2]));
      B0[0] *= COEFF;
      B0[1] *= COEFF;
      B0[2] *= COEFF;
      COEFF  = TWO / (ONE + SQR(B0[0]) + SQR(B0[1]) + SQR(B0[2]));

      vp_hat[0] = (vp_upd_hat[0] + vp_upd_hat[1] * B0[2] - vp_upd_hat[2] * B0[1]) *
                  COEFF;
      vp_hat[1] = (vp_upd_hat[1] + vp_upd_hat[2] * B0[0] - vp_upd_hat[0] * B0[2]) *
                  COEFF;
      vp_hat[2] = (vp_upd_hat[2] + vp_upd_hat[0] * B0[1] - vp_upd_hat[1] * B0[0]) *
                  COEFF;

      vp_upd_hat[0] += vp_hat[1] * B0[2] - vp_hat[2] * B0[1] + D0[0];
      vp_upd_hat[1] += vp_hat[2] * B0[0] - vp_hat[0] * B0[2] + D0[1];
      vp_upd_hat[2] += vp_hat[0] * B0[1] - vp_hat[1] * B0[0] + D0[2];

      metric.template transform<Idx::T, Idx::D>(xp, vp_upd_hat, vp_upd);
    }

    // Helper functions

    /**
     * @brief First order Yee mesh field interpolation to particle position.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    Inline void interpolateFields(index_t          p,
                                  vec_t<Dim::_3D>& e,
                                  vec_t<Dim::_3D>& b) const;

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(u_i u_j h^ij) for massless particles.
     */
    Inline auto computeGamma(const Massless_t&,
                             const vec_t<Dim::_3D>& u_cov,
                             const vec_t<Dim::_3D>& u_cntrv) const -> real_t {
      return math::sqrt(
        u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] + u_cov[2] * u_cntrv[2]);
    }

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(1 + u_i u_j h^ij) for massive particles.
     */
    Inline auto computeGamma(const Massive_t&,
                             const vec_t<Dim::_3D>& u_cov,
                             const vec_t<Dim::_3D>& u_cntrv) const -> real_t {
      return math::sqrt(ONE + u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] +
                        u_cov[2] * u_cntrv[2]);
    }

    // Extra
    Inline void boundaryConditions(index_t) const;
  };

  /* -------------------------------------------------------------------------- */
  /*                               Geodesic pusher */
  /* -------------------------------------------------------------------------- */

  template <class M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicMomentumPush(T,
                                                     const coord_t<D>&      xp,
                                                     const vec_t<Dim::_3D>& vp,
                                                     vec_t<Dim::_3D>& vp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "1D not applicable");
    } else if constexpr (D == Dim::_2D) {
      // initialize midpoint values & updated values
      vec_t<Dim::_3D> vp_mid { ZERO };
      vec_t<Dim::_3D> vp_mid_cntrv { ZERO };
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < niter; ++i) {
        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpointы
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) / metric.alpha(xp) };

        // find updated velocity
        // vp_upd[0] =
        //   vp[0] +
        //   dt *
        //     (-metric.alpha(xp) * u0 * DERIVATIVE_IN_R(metric.alpha, xp) +
        //      vp_mid[0] * DERIVATIVE_IN_R(metric.beta1, xp) -
        //      (HALF / u0) *
        //        (DERIVATIVE_IN_R((metric.template h<1, 1>), xp) * SQR(vp_mid[0]) +
        //         DERIVATIVE_IN_R((metric.template h<2, 2>), xp) * SQR(vp_mid[1]) +
        //         DERIVATIVE_IN_R((metric.template h<3, 3>), xp) * SQR(vp_mid[2]) +
        //         TWO * DERIVATIVE_IN_R((metric.template h<1, 3>), xp) *
        //           vp_mid[0] * vp_mid[2]));
        // vp_upd[1] =
        //   vp[1] +
        //   dt *
        //     (-metric.alpha(xp) * u0 * DERIVATIVE_IN_TH(metric.alpha, xp) +
        //      vp_mid[0] * DERIVATIVE_IN_TH(metric.beta1, xp) -
        //      (HALF / u0) *
        //        (DERIVATIVE_IN_TH((metric.template h<1, 1>), xp) * SQR(vp_mid[0]) +
        //         DERIVATIVE_IN_TH((metric.template h<2, 2>), xp) * SQR(vp_mid[1]) +
        //         DERIVATIVE_IN_TH((metric.template h<3, 3>), xp) * SQR(vp_mid[2]) +
        //         TWO * DERIVATIVE_IN_TH((metric.template h<1, 3>), xp) *
        //           vp_mid[0] * vp_mid[2]));
        vp_upd[0] = vp[0] +
                    dt * (-metric.alpha(xp) * u0 * metric.dr_alpha(xp) +
                          vp_mid[0] * metric.dr_beta1(xp) -
                          (HALF / u0) *
                            (metric.dr_h11(xp) * SQR(vp_mid[0]) +
                             metric.dr_h22(xp) * SQR(vp_mid[1]) +
                             metric.dr_h33(xp) * SQR(vp_mid[2]) +
                             TWO * metric.dr_h13(xp) * vp_mid[0] * vp_mid[2]));
        vp_upd[1] = vp[1] +
                    dt * (-metric.alpha(xp) * u0 * metric.dt_alpha(xp) +
                          vp_mid[0] * metric.dt_beta1(xp) -
                          (HALF / u0) *
                            (metric.dt_h11(xp) * SQR(vp_mid[0]) +
                             metric.dt_h22(xp) * SQR(vp_mid[1]) +
                             metric.dt_h33(xp) * SQR(vp_mid[2]) +
                             TWO * metric.dt_h13(xp) * vp_mid[0] * vp_mid[2]));
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <class M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicCoordinatePush(T,
                                                       const coord_t<D>& xp,
                                                       const vec_t<Dim::_3D>& vp,
                                                       coord_t<D>& xp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "GeodesicCoordinatePush: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      vec_t<Dim::_3D>   vp_cntrv { ZERO };
      coord_t<Dim::_2D> xp_mid { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];

      for (auto i { 0 }; i < niter; ++i) {
        // find midpoint values
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp_mid, vp, vp_cntrv);
        real_t gamma = computeGamma(T {}, vp, vp_cntrv);

        // find midpoint coefficients
        real_t u0 { gamma / metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + dt * (vp_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + dt * (vp_cntrv[1] / u0);
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <class M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicFullPush(T,
                                                 const coord_t<D>&      xp,
                                                 const vec_t<Dim::_3D>& vp,
                                                 coord_t<D>&            xp_upd,
                                                 vec_t<Dim::_3D>& vp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "GeodesicFullPush: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      // initialize midpoint values & updated values
      vec_t<Dim::_2D> xp_mid { ZERO };
      vec_t<Dim::_3D> vp_mid { ZERO }, vp_mid_cntrv { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < niter; ++i) {
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp_mid, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpoint
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) /
                    metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + dt * (vp_mid_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + dt * (vp_mid_cntrv[1] / u0);

        // find updated velocity
        vp_upd[0] =
          vp[0] +
          dt *
            (-metric.alpha(xp_mid) * u0 * DERIVATIVE_IN_R(metric.alpha, xp_mid) +
             vp_mid[0] * DERIVATIVE_IN_R(metric.beta1, xp_mid) -
             (HALF / u0) *
               (DERIVATIVE_IN_R((metric.template h<1, 1>), xp_mid) * SQR(vp_mid[0]) +
                DERIVATIVE_IN_R((metric.template h<2, 2>), xp_mid) * SQR(vp_mid[1]) +
                DERIVATIVE_IN_R((metric.template h<3, 3>), xp_mid) * SQR(vp_mid[2]) +
                TWO * DERIVATIVE_IN_R((metric.template h<1, 3>), xp_mid) *
                  vp_mid[0] * vp_mid[2]));
        vp_upd[1] = vp[1] +
                    dt *
                      (-metric.alpha(xp_mid) * u0 *
                         DERIVATIVE_IN_TH(metric.alpha, xp_mid) +
                       vp_mid[0] * DERIVATIVE_IN_TH(metric.beta1, xp_mid) -
                       (HALF / u0) *
                         (DERIVATIVE_IN_TH((metric.template h<1, 1>), xp_mid) *
                            SQR(vp_mid[0]) +
                          DERIVATIVE_IN_TH((metric.template h<2, 2>), xp_mid) *
                            SQR(vp_mid[1]) +
                          DERIVATIVE_IN_TH((metric.template h<3, 3>), xp_mid) *
                            SQR(vp_mid[2]) +
                          TWO * DERIVATIVE_IN_TH((metric.template h<1, 3>), xp_mid) *
                            vp_mid[0] * vp_mid[2]));
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  /* -------------------------------------------------------------------------- */
  /*                                 Phi pusher */
  /* -------------------------------------------------------------------------- */
  template <class M>
  template <typename T>
  Inline void Pusher_kernel<M>::UpdatePhi(T,
                                          const coord_t<D>&      xp,
                                          const vec_t<Dim::_3D>& vp,
                                          real_t                 phi) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "UpdatePhi: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      vec_t<Dim::_3D> vp_cntrv { ZERO };
      metric.template transform<Idx::D, Idx::U>(xp, vp, vp_cntrv);
      real_t u0 { computeGamma(T {}, vp, vp_cntrv) / metric.alpha(xp) };
      phi += dt * vp_cntrv[2] / u0;
      if (phi >= constant::TWO_PI) {
        phi -= constant::TWO_PI;
      } else if (phi < ZERO) {
        phi += constant::TWO_PI;
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <class M>
  Inline void Pusher_kernel<M>::interpolateFields(index_t          p,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "interpolateFields: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      // first order interpolation
      // Using fields em::e = D(t = n), and em::b0 = B(t = n)

      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };
      const auto dx2_ { static_cast<real_t>(dx2(p)) };

      // first order
      real_t c000, c100, c010, c110, c00, c10;

      // Ex1
      // interpolate to nodes
      c000  = HALF * (DB(i, j, em::dx1) + DB(i - 1, j, em::dx1));
      c100  = HALF * (DB(i, j, em::dx1) + DB(i + 1, j, em::dx1));
      c010  = HALF * (DB(i, j + 1, em::dx1) + DB(i - 1, j + 1, em::dx1));
      c110  = HALF * (DB(i, j + 1, em::dx1) + DB(i + 1, j + 1, em::dx1));
      // interpolate from nodes to the particle position
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex2
      c000  = HALF * (DB(i, j, em::dx2) + DB(i, j - 1, em::dx2));
      c100  = HALF * (DB(i + 1, j, em::dx2) + DB(i + 1, j - 1, em::dx2));
      c010  = HALF * (DB(i, j, em::dx2) + DB(i, j + 1, em::dx2));
      c110  = HALF * (DB(i + 1, j, em::dx2) + DB(i + 1, j + 1, em::dx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex3
      c000  = DB(i, j, em::dx3);
      c100  = DB(i + 1, j, em::dx3);
      c010  = DB(i, j + 1, em::dx3);
      c110  = DB(i + 1, j + 1, em::dx3);
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[2] = c00 * (ONE - dx2_) + c10 * dx2_;

      // Bx1
      c000  = HALF * (DB0(i, j, em::bx1) + DB0(i, j - 1, em::bx1));
      c100  = HALF * (DB0(i + 1, j, em::bx1) + DB0(i + 1, j - 1, em::bx1));
      c010  = HALF * (DB0(i, j, em::bx1) + DB0(i, j + 1, em::bx1));
      c110  = HALF * (DB0(i + 1, j, em::bx1) + DB0(i + 1, j + 1, em::bx1));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx2
      c000  = HALF * (DB0(i - 1, j, em::bx2) + DB0(i, j, em::bx2));
      c100  = HALF * (DB0(i, j, em::bx2) + DB0(i + 1, j, em::bx2));
      c010  = HALF * (DB0(i - 1, j + 1, em::bx2) + DB0(i, j + 1, em::bx2));
      c110  = HALF * (DB0(i, j + 1, em::bx2) + DB0(i + 1, j + 1, em::bx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx3
      c000  = INV_4 * (DB0(i - 1, j - 1, em::bx3) + DB0(i - 1, j, em::bx3) +
                      DB0(i, j - 1, em::bx3) + DB0(i, j, em::bx3));
      c100  = INV_4 * (DB0(i, j - 1, em::bx3) + DB0(i, j, em::bx3) +
                      DB0(i + 1, j - 1, em::bx3) + DB0(i + 1, j, em::bx3));
      c010  = INV_4 * (DB0(i - 1, j, em::bx3) + DB0(i - 1, j + 1, em::bx3) +
                      DB0(i, j, em::bx3) + DB0(i, j + 1, em::bx3));
      c110  = INV_4 * (DB0(i, j, em::bx3) + DB0(i, j + 1, em::bx3) +
                      DB0(i + 1, j, em::bx3) + DB0(i + 1, j + 1, em::bx3));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[2] = c00 * (ONE - dx2_) + c10 * dx2_;
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  /* ------------------------------ Photon pusher ----------------------------- */

  template <class M>
  Inline void Pusher_kernel<M>::operator()(Massless_t, index_t p) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "Photon pusher not implemented for 1D");
    } else if constexpr (D == Dim::_2D) {
      if (tag(p) != ParticleTag::alive) {
        if (tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      // record previous coordinate
      i1_prev(p)  = i1(p);
      i2_prev(p)  = i2(p);
      dx1_prev(p) = dx1(p);
      dx2_prev(p) = dx2(p);

      coord_t<Dim::_2D> xp { ZERO };
      vec_t<Dim::_3D>   vp { ux1(p), ux2(p), ux3(p) };

      xp[0] = i_di_to_Xi(i1(p), dx1(p));
      xp[1] = i_di_to_Xi(i2(p), dx2(p));

      /* ----------------------------- Leapfrog pusher ---------------------------- */
      // u_i(n - 1/2) -> u_i(n + 1/2)
      vec_t<Dim::_3D> vp_upd { ZERO };
      GeodesicMomentumPush<Massless_t>(Massless_t {}, xp, vp, vp_upd);
      // x^i(n) -> x^i(n + 1)
      coord_t<Dim::_2D> xp_upd { ZERO };
      GeodesicCoordinatePush<Massless_t>(Massless_t {}, xp, vp_upd, xp_upd);
      // update phi
      UpdatePhi<Massless_t>(
        Massless_t {},
        { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
        vp_upd,
        phi(p));

      // update coordinate
      int      i1_, i2_;
      prtldx_t dx1_, dx2_;
      from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
      from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
      i1(p)  = i1_;
      dx1(p) = dx1_;
      i2(p)  = i2_;
      dx2(p) = dx2_;

      // update velocity
      ux1(p) = vp_upd[0];
      ux2(p) = vp_upd[1];
      ux3(p) = vp_upd[2];

      boundaryConditions(p);
    } else if constexpr (D == Dim::_3D) {
      raise::KernelError(HERE, "3D not implemented");
    }
  }

  /* ------------------------- Massive particle pusher ------------------------ */

  template <class M>
  Inline void Pusher_kernel<M>::operator()(Massive_t, index_t p) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "Massive pusher not implemented for 1D");
    } else if constexpr (D == Dim::_2D) {
      if (tag(p) != ParticleTag::alive) {
        if (tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      // record previous coordinate
      i1_prev(p)  = i1(p);
      i2_prev(p)  = i2(p);
      dx1_prev(p) = dx1(p);
      dx2_prev(p) = dx2(p);

      coord_t<Dim::_2D> xp { ZERO };

      xp[0] = i_di_to_Xi(i1(p), dx1(p));
      xp[1] = i_di_to_Xi(i2(p), dx2(p));

      coord_t<Dim::_2D> xp_ { ZERO };
      xp_[0] = xp[0];
      real_t       theta_Cd { xp[1] };
      const real_t theta_Ph { metric.template convert<2, Crd::Cd, Crd::Ph>(
        theta_Cd) };
      const real_t small_angle { constant::SMALL_ANGLE_GR };
      const auto   large_angle { constant::PI - small_angle };
      if (theta_Ph < small_angle) {
        theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(small_angle);
      } else if (theta_Ph >= large_angle) {
        theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(large_angle);
      }
      xp_[1] = theta_Cd;

      vec_t<Dim::_3D> Dp_cntrv { ZERO }, Bp_cntrv { ZERO }, Dp_hat { ZERO },
        Bp_hat { ZERO };
      interpolateFields(p, Dp_cntrv, Bp_cntrv);
      metric.template transform<Idx::U, Idx::T>(xp, Dp_cntrv, Dp_hat);
      metric.template transform<Idx::U, Idx::T>(xp, Bp_cntrv, Bp_hat);

      vec_t<Dim::_3D> vp { ux1(p), ux2(p), ux3(p) };

      /* -------------------------------- Leapfrog -------------------------------- */
      /* u_i(n - 1/2) -> u*_i(n) */
      vec_t<Dim::_3D> vp_upd { ZERO };
      EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      /* u*_i(n) -> u**_i(n) */
      vp[0] = vp_upd[0];
      vp[1] = vp_upd[1];
      vp[2] = vp_upd[2];
      GeodesicMomentumPush<Massive_t>(Massive_t {}, xp_, vp, vp_upd);
      /* u**_i(n) -> u_i(n + 1/2) */
      vp[0] = vp_upd[0];
      vp[1] = vp_upd[1];
      vp[2] = vp_upd[2];
      EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      /* x^i(n) -> x^i(n + 1) */
      coord_t<Dim::_2D> xp_upd { ZERO };
      GeodesicCoordinatePush<Massive_t>(Massive_t {}, xp, vp_upd, xp_upd);

      // update phi
      UpdatePhi<Massive_t>(
        Massive_t {},
        { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
        vp_upd,
        phi(p));

      // update coordinate
      int      i1_, i2_;
      prtldx_t dx1_, dx2_;
      from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
      from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
      i1(p)  = i1_;
      dx1(p) = dx1_;
      i2(p)  = i2_;
      dx2(p) = dx2_;

      // update velocity
      ux1(p) = vp_upd[0];
      ux2(p) = vp_upd[1];
      ux3(p) = vp_upd[2];

      boundaryConditions(p);
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  // Boundary conditions

  template <class M>
  Inline void Pusher_kernel<M>::boundaryConditions(index_t p) const {
    if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
      if (i1(p) < 0 && is_absorb_i1min) {
        tag(p) = ParticleTag::dead;
      } else if (i1(p) >= ni1 && is_absorb_i1max) {
        tag(p) = ParticleTag::dead;
      }
    }
    if constexpr (D == Dim::_2D || D == Dim::_3D) {
      if (i2(p) < 0) {
        if (is_axis_i2min) {
          i2(p)  = 0;
          dx2(p) = ONE - dx2(p);
          ux2(p) = -ux2(p);
        }
      } else if (i2(p) >= ni2) {
        if (is_axis_i2max) {
          i2(p)  = ni2 - 1;
          dx2(p) = ONE - dx2(p);
          ux2(p) = -ux2(p);
        }
      }
    }
    if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
#if defined(MPI_ENABLED)
    if constexpr (D == Dim::_1D) {
      tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= ni1);
    } else if constexpr (D == Dim::_2D) {
      tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2);
    } else if constexpr (D == Dim::_3D) {
      tag(p) = mpi::SendTag(tag(p),
                            i1(p) < 0,
                            i1(p) >= ni1,
                            i2(p) < 0,
                            i2(p) >= ni2,
                            i3(p) < 0,
                            i3(p) >= ni3);
    }
#endif
  }

} // namespace kernel::gr

#undef DERIVATIVE_IN_TH
#undef DERIVATIVE_IN_R

#undef i_di_to_Xi
#undef from_Xi_to_i_di
#undef from_Xi_to_i

#endif // KERNELS_PARTICLE_PUSHER_GR_HPP
