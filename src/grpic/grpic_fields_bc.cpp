#include "global.h"
#include "grpic.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t& t, const short& s) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    (void)t;
    auto mblock {this->m_mblock};
    // BC for D field
    if (s == 0) {
      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i - 1, j, em::ex1) = mblock.em0(i, j, em::ex1);

          // mblock.em0(i, j, em::ex2) = mblock.em0(i + 1, j, em::ex2);
          mblock.em0(i - 1, j, em::ex2) = mblock.em0(i, j, em::ex2);

          mblock.em0(i, j, em::ex3) = mblock.em0(i + 1, j, em::ex3);
          mblock.em0(i - 1, j, em::ex3) = mblock.em0(i, j, em::ex3);

          mblock.em(i - 1, j, em::ex1) = mblock.em(i, j, em::ex1);

          // mblock.em(i, j, em::ex2) = mblock.em(i + 1, j, em::ex2);
          mblock.em(i - 1, j, em::ex2) = mblock.em(i, j, em::ex2);

          mblock.em(i, j, em::ex3) = mblock.em(i + 1, j, em::ex3);
          mblock.em(i - 1, j, em::ex3) = mblock.em(i, j, em::ex3);
        });

      // Absorbing boundary
      auto r_absorb {m_sim_params.metric_parameters()[2]};
      auto r_max {m_mblock.metric.x1_max};
      auto pGen {this->m_pGen};
      Kokkos::parallel_for(
        "2d_absorbing bc", m_mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
          real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
          real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

          // i
          vec_t<Dimension::TWO_D> rth_;
          mblock.metric.x_Code2Sph({i_, j_}, rth_);
          real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
          // real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
          real_t sigma_r1 {ONE - std::exp(-HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1)};

          // i + 1/2
          mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
          real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
          // real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};
          real_t sigma_r2 {ONE - std::exp(-HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2)};

          mblock.em0(i, j, em::ex1) = (ONE - sigma_r2) * mblock.em0(i, j, em::ex1);
          mblock.em0(i, j, em::ex2) = (ONE - sigma_r1) * mblock.em0(i, j, em::ex2);
          mblock.em0(i, j, em::ex3) = (ONE - sigma_r1) * mblock.em0(i, j, em::ex3);

          mblock.em(i, j, em::ex1) = (ONE - sigma_r2) * mblock.em(i, j, em::ex1);
          mblock.em(i, j, em::ex2) = (ONE - sigma_r1) * mblock.em(i, j, em::ex2);
          mblock.em(i, j, em::ex3) = (ONE - sigma_r1) * mblock.em(i, j, em::ex3);
        });

      // r = rmax
      Kokkos::parallel_for(
        "2d_bc_rmax",
        NTTRange<Dimension::TWO_D>({mblock.i_max(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i, j, em::ex3) = ZERO;
          mblock.em0(i, j, em::ex2) = ZERO;

          mblock.em(i, j, em::ex3) = ZERO;
          mblock.em(i, j, em::ex2) = ZERO;
        });

      // BC for B field
    } else if (s == 1) {

      // theta = 0 boundary
      Kokkos::parallel_for(
        "2d_bc_theta0",
        // NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
        NTTRange<Dimension::TWO_D>({mblock.i_min() - 1, mblock.j_min()}, {mblock.i_max(), mblock.j_min() + 1}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i, j, em::bx2) = ZERO;
          mblock.em(i, j, em::bx2) = ZERO;
        });

      // theta = pi boundary
      Kokkos::parallel_for(
        "2d_bc_thetaPi",
        // NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() +
        // N_GHOSTS}),
        NTTRange<Dimension::TWO_D>({mblock.i_min() - 1, m_mblock.j_max()}, {m_mblock.i_max(), m_mblock.j_max() + 1}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i, j, em::bx2) = ZERO;
          mblock.em(i, j, em::bx2) = ZERO;
        });

      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i, j, em::bx1) = mblock.em0(i + 1, j, em::bx1);
          mblock.em0(i - 1, j, em::bx1) = mblock.em0(i, j, em::bx1);
          mblock.em0(i - 1, j, em::bx2) = mblock.em0(i, j, em::bx2);
          mblock.em0(i - 1, j, em::bx3) = mblock.em0(i, j, em::bx3);

          mblock.em(i, j, em::bx1) = mblock.em(i + 1, j, em::bx1);
          mblock.em(i - 1, j, em::bx1) = mblock.em(i, j, em::bx1);
          mblock.em(i - 1, j, em::bx2) = mblock.em(i, j, em::bx2);
          mblock.em(i - 1, j, em::bx3) = mblock.em(i, j, em::bx3);
        });

      auto r_absorb {m_sim_params.metric_parameters()[2]};
      auto r_max {m_mblock.metric.x1_max};
      auto pGen {this->m_pGen};
      Kokkos::parallel_for(
        "2d_absorbing bc", m_mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
          real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
          real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

          // i
          vec_t<Dimension::TWO_D> rth_;
          mblock.metric.x_Code2Sph({i_, j_}, rth_);
          real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
          // real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
          real_t sigma_r1 {ONE - std::exp(-HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1)};

          // i + 1/2
          mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
          real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
          // real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};
          real_t sigma_r2 {ONE - std::exp(-HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2)};

          real_t br_target {pGen.userTargetField_br_cntrv(mblock, {i_, j_ + HALF})};
          real_t bth_target {pGen.userTargetField_bth_cntrv(mblock, {i_ + HALF, j_})};
          real_t bx1_source_cntr {mblock.em0(i, j, em::bx1)};
          real_t bx2_source_cntr {mblock.em0(i, j, em::bx2)};
          real_t br_interm {(ONE - sigma_r1) * bx1_source_cntr + sigma_r1 * br_target};
          real_t bth_interm {(ONE - sigma_r2) * bx2_source_cntr + sigma_r2 * bth_target};

          mblock.em0(i, j, em::bx1) = br_interm;
          mblock.em0(i, j, em::bx2) = bth_interm;
          mblock.em0(i, j, em::bx3) = (ONE - sigma_r2) * mblock.em0(i, j, em::bx3);

          bx1_source_cntr = mblock.em(i, j, em::bx1);
          bx2_source_cntr = mblock.em(i, j, em::bx2);
          br_interm = (ONE - sigma_r1) * bx1_source_cntr + sigma_r1 * br_target;
          bth_interm = (ONE - sigma_r2) * bx2_source_cntr + sigma_r2 * bth_target;

          mblock.em(i, j, em::bx1) = br_interm;
          mblock.em(i, j, em::bx2) = bth_interm;
          mblock.em(i, j, em::bx3) = (ONE - sigma_r2) * mblock.em(i, j, em::bx3);
        });

      // r = rmax
      Kokkos::parallel_for(
        "2d_bc_rmax",
        NTTRange<Dimension::TWO_D>({mblock.i_max(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.em0(i, j, em::bx1) = mblock.em0(i - 1, j, em::bx1);
          mblock.em(i, j, em::bx1) = mblock.em(i - 1, j, em::bx1);
        });

    } else {
      NTTError("Only two options: 0 and 1");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&, const short&) {
    NTTError("not implemented");
  }

  template <>
  void GRPIC<Dimension::TWO_D>::AuxiliaryBoundaryConditions(const real_t& t, const short& s) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    (void)t;

    auto mblock {this->m_mblock};

    // BC for E field
    if (s == 0) {

      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.aux(i - 1, j, em::ex1) = mblock.aux(i, j, em::ex1);

          // mblock.aux(i, j, em::ex2) = mblock.aux(i + 1, j, em::ex2);
          mblock.aux(i - 1, j, em::ex2) = mblock.aux(i, j, em::ex2);

          mblock.aux(i, j, em::ex3) = mblock.aux(i + 1, j, em::ex3);
          mblock.aux(i - 1, j, em::ex3) = mblock.aux(i, j, em::ex3);
        });

      // BC for H field
    } else if (s == 1) {

      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
        Lambda(index_t i, index_t j) {
          mblock.aux(i, j, em::bx1) = mblock.aux(i + 1, j, em::bx1);
          mblock.aux(i - 1, j, em::bx1) = mblock.aux(i, j, em::bx1);

          mblock.aux(i - 1, j, em::bx2) = mblock.aux(i, j, em::bx2);

          // mblock.aux(i , j, em::bx3) = mblock.aux(i + 1, j, em::bx3);
          mblock.aux(i - 1, j, em::bx3) = mblock.aux(i, j, em::bx3);
        });
    } else {
      NTTError("Only two options: 0 and 1");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::AuxiliaryBoundaryConditions(const real_t&, const short&) {
    NTTError("not implemented");
  }

} // namespace ntt

// // theta = 0 boundary
// Kokkos::parallel_for(
//   "2d_bc_theta0",
//   NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
//   Lambda(index_t i, index_t j) {
//     // mblock.em0(i, j, em::ex3) = ZERO;
//     // mblock.em(i, j, em::ex3) = ZERO;
//     });

// // theta = pi boundary
// Kokkos::parallel_for(
//   "2d_bc_thetaPi",
//   NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() +
//   N_GHOSTS}), Lambda(index_t i, index_t j) {
//     mblock.em0(i, j, em::ex3) = ZERO;

//     mblock.em(i, j, em::ex3) = ZERO;
//     });
