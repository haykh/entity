#include "global.h"
#include "grpic.h"
#include "grpic_ampere.hpp"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::ampereSubstep(const real_t&, const real_t& fraction, const gr_ampere& f) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    auto         range {
      CreateRangePolicy<Dimension::TWO_D>({m_mblock.i1_min(), m_mblock.i2_min() + 1}, {m_mblock.i1_max(), m_mblock.i2_max()})};
    auto range_pole {CreateRangePolicy<Dimension::ONE_D>({m_mblock.i1_min()}, {m_mblock.i1_max()})};
    if (f == gr_ampere::aux) {
      Kokkos::parallel_for("ampere", range, AmpereGR_aux<Dimension::TWO_D>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR_aux<Dimension::TWO_D>(m_mblock, coeff));
    } else if (f == gr_ampere::main) {
      Kokkos::parallel_for("ampere", range, AmpereGR<Dimension::TWO_D>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR<Dimension::TWO_D>(m_mblock, coeff));
    } else if (f == gr_ampere::init) {
      Kokkos::parallel_for("ampere", range, AmpereGR_init<Dimension::TWO_D>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR_init<Dimension::TWO_D>(m_mblock, coeff));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::ampereSubstep(const real_t&, const real_t&, const gr_ampere&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt