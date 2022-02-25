#include "global.h"
#include "grpic.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t& t, const short& s) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;

    // r = rmin boundary
    if (m_mblock.boundaries[0] == BoundaryCondition::USER) {
      m_pGen.userBCFields(t, m_sim_params, m_mblock);
    } else {
      NTTError("2d non-user boundary condition not implemented for curvilinear");
    }

    if (s == 0) {

    auto mblock {this->m_mblock};
    // theta = 0 boundary
    Kokkos::parallel_for(
      "2d_bc_theta0",
      NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.em0(i, j, em::bx2) = ZERO;
        mblock.em0(i, j, em::ex3) = ZERO;

        mblock.em0(i, j + 1, em::bx3) = ZERO; //mblock.em0(i, j, em::bx3);

        });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em0(i, j, em::bx2) = ZERO;
        mblock.em0(i, j, em::ex3) = ZERO;

        mblock.em0(i, j - 2, em::bx3) = ZERO; //mblock.em0(i, j - 1, em::bx3);

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
        real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
        // real_t sigma_r1 {ONE - std::exp(- 10.0 * HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1)};

        // i + 1/2
        mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
        real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};
        // real_t sigma_r2 {ONE - std::exp(- 10.0 * HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2)};

        mblock.em0(i, j, em::ex1) = (ONE - sigma_r1) * mblock.em0(i, j, em::ex1);
        // mblock.em0(i, j, em::bx2) = (ONE - sigma_r1) * mblock.em0(i, j, em::bx2);
        mblock.em0(i, j, em::bx3) = (ONE - sigma_r1) * mblock.em0(i, j, em::bx3);

        // real_t br_target_hat  {pGen.userTargetField_br_hat(mblock, {i_, j_ + HALF})};
        // real_t bth_target_hat {pGen.userTargetField_bth_hat(mblock, {i_ + HALF, j_})};
        // real_t bx1_source_cntr {mblock.em(i, j, em::bx1)};
        // real_t bx2_source_cntr {mblock.em(i, j, em::bx2)};
        // vec_t<Dimension::THREE_D> b_source_hat;
        // mblock.metric.v_Cntrv2Hat({i_, j_ + HALF}, {bx1_source_cntr, bx2_source_cntr, ZERO}, b_source_hat);
        // real_t br_interm_hat  {(ONE - sigma_r2) * b_so5urce_hat[0] + sigma_r2 * br_target_hat};
        // real_t bth_interm_hat {(ONE - sigma_r1) * b_source_hat[1] + sigma_r1 * bth_target_hat};
        // vec_t<Dimension::THREE_D> b_interm_cntr;
        // mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_interm_hat, bth_interm_hat, ZERO}, b_interm_cntr);

        // real_t br_target_hat  {pGen.userTargetField_br_hat(mblock, {i_, j_ + HALF})};
        // real_t bx1_source_cntr {mblock.em(i, j, em::bx1)};
        // vec_t<Dimension::THREE_D> b_source_hat;
        // mblock.metric.v_Cntrv2Hat({i_, j_ + HALF}, {bx1_source_cntr, ZERO, ZERO}, b_source_hat);
        // real_t br_interm_hat  {(ONE - sigma_r2) * b_source_hat[0] + sigma_r2 * br_target_hat};
        // vec_t<Dimension::THREE_D> b_interm_cntr;
        // mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_interm_hat, ZERO, ZERO}, b_interm_cntr);

        real_t br_target  {pGen.userTargetField_br_cntrv(mblock, {i_, j_ + HALF})};
        real_t bth_target  {pGen.userTargetField_bth_cntrv(mblock, {i_, j_ + HALF})};
        real_t bx1_source_cntr {mblock.em0(i, j, em::bx1)};
        real_t bx2_source_cntr {mblock.em0(i, j, em::bx2)};
        real_t br_interm {(ONE - sigma_r2) * bx1_source_cntr + sigma_r2 * br_target};
        real_t bth_interm {(ONE - sigma_r1) * bx2_source_cntr + sigma_r1 * bth_target};


    // coord_t<Dimension::TWO_D> rth_;
    // coord_t<Dimension::TWO_D> x {i_, j_};
    // m_mblock.metric.x_Code2Sph(x, rth_);
    // printf("ALONG R %f %f\n", bx1_source_cntr, ONE * std::cos(rth_[1]));
    // printf("ALONG R %f %f\n", bx2_source_cntr, ONE * std::cos(rth_[1]));

        mblock.em0(i, j, em::bx1) = br_interm;
        mblock.em0(i, j, em::bx2) = bth_interm;
        mblock.em0(i, j, em::ex2) = (ONE - sigma_r2) * mblock.em0(i, j, em::ex2);
        mblock.em0(i, j, em::ex3) = (ONE - sigma_r2) * mblock.em0(i, j, em::ex3);

      });
  }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&, const short& s) {
    NTTError("not implemented");
  }

  template <>
  void GRPIC<Dimension::TWO_D>::AuxiliaryBoundaryConditions(const real_t& t) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;

    auto mblock {this->m_mblock};
    // theta = 0 boundary

    Kokkos::parallel_for(
      "2d_bc_theta0",
      NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.aux(i, j, em::bx2) = ZERO;
        mblock.aux(i, j, em::ex3) = ZERO;

        mblock.aux(i, j + 1, em::bx3) = mblock.aux(i, j, em::bx3);

        });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.aux(i, j, em::bx2) = ZERO;
        mblock.aux(i, j, em::ex3) = ZERO;
  
        mblock.aux(i, j - 2, em::bx3) = mblock.aux(i, j - 1, em::bx3);

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
        real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
        // real_t sigma_r1 {ONE - std::exp(- 10.0 * HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1)};

        // i + 1/2
        mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
        real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};
        // real_t sigma_r2 {ONE - std::exp(- 10.0 * HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2)};

      //  printf("BEFORE %f %f \n", m_mblock.aux(i, j, em::ex3), sigma_r2);

        mblock.aux(i, j, em::bx3) = (ONE - sigma_r1) * mblock.aux(i, j, em::bx3);
        mblock.aux(i, j, em::ex1) = (ONE - sigma_r1) * mblock.aux(i, j, em::ex1);
        mblock.aux(i, j, em::ex2) = (ONE - sigma_r2) * mblock.aux(i, j, em::ex2);
        mblock.aux(i, j, em::ex3) = (ONE - sigma_r2) * mblock.aux(i, j, em::ex3);

      //  printf("after %f %f \n", m_mblock.aux(i, j, em::ex3), sigma_r2);

      });

    }

  template <>
  void GRPIC<Dimension::THREE_D>::AuxiliaryBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt