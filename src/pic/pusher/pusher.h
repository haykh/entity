#ifndef PIC_PUSHER_H
#define PIC_PUSHER_H

#include "global.h"
#include "meshblock.h"
#include "particles.h"

#include <utility>

namespace ntt {

template <Dimension D>
class Pusher {
public:
  Meshblock<D> m_meshblock;
  Particles<D> m_particles;
  real_t coeff;
  real_t dt;
  using index_t = NTTArray<real_t*>::size_type;

  Pusher(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : m_meshblock(m_meshblock_), m_particles(m_particles_), coeff(coeff_), dt(dt_) {}

  // clang-format off
  void interpolateFields(const index_t&,
                         real_t&, real_t&, real_t&,
                         real_t&, real_t&, real_t&) const;
  void positionUpdate(const index_t&) const;
  virtual void velocityUpdate(const index_t&,
                              real_t&, real_t&, real_t&,
                              real_t&, real_t&, real_t&) const {}
  Inline void operator()(const index_t p) const {
    real_t e0_x1, e0_x2, e0_x3;
    real_t b0_x1, b0_x2, b0_x3;
    // TODO:
    // fix this to cartesian
    // 1. interpolate fields
    // 2. convert E,B and X,U to cartesian
    // 3. update U
    // 4. update X
    // 5. convert back
    interpolateFields(p,
                      e0_x1, e0_x2, e0_x3,
                      b0_x1, b0_x2, b0_x3);
    velocityUpdate(p,
                   e0_x1, e0_x2, e0_x3,
                   b0_x1, b0_x2, b0_x3);
    positionUpdate(p);
  }
  // clang-format on
};

// * * * * Position update * * * * * * * * * * * * * *
template <>
void Pusher<ONE_D>::positionUpdate(const index_t& p) const {
  // TESTPERF: faster sqrt?
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
}

template <>
void Pusher<TWO_D>::positionUpdate(const index_t& p) const {
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
}

template <>
void Pusher<THREE_D>::positionUpdate(const index_t& p) const {
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
  m_particles.m_x3(p) += dt * m_particles.m_ux3(p) * inv_gamma0;
}

// * * * * Field interpolation * * * * * * * * * * * *
template <>
Inline void Pusher<ONE_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  e0_x1 = m_meshblock.em_fields(i, fld::ex1);
  e0_x2 = m_meshblock.em_fields(i, fld::ex2);
  e0_x3 = m_meshblock.em_fields(i, fld::ex3);

  b0_x1 = m_meshblock.em_fields(i, fld::bx1);
  b0_x2 = m_meshblock.em_fields(i, fld::bx2);
  b0_x3 = m_meshblock.em_fields(i, fld::bx3);
}

template <>
Inline void Pusher<TWO_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  const auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));

  // first order
  real_t c000, c100, c010, c110, c00, c10;

  // clang-format off
  // Ex1
  // interpolate to nodes
  c000 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::ex1) + m_meshblock.em_fields(i - 1,     j, fld::ex1));
  c100 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::ex1) + m_meshblock.em_fields(i + 1,     j, fld::ex1));
  c010 = 0.5 * (m_meshblock.em_fields(    i, j + 1, fld::ex1) + m_meshblock.em_fields(i - 1, j + 1, fld::ex1));
  c110 = 0.5 * (m_meshblock.em_fields(    i, j + 1, fld::ex1) + m_meshblock.em_fields(i + 1, j + 1, fld::ex1));
  // interpolate from nodes to the particle position
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x1 = c00 * (ONE - dx2) + c10 * dx2;
  // Ex2
  c000 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::ex2) + m_meshblock.em_fields(    i, j - 1, fld::ex2));
  c100 = 0.5 * (m_meshblock.em_fields(i + 1,     j, fld::ex2) + m_meshblock.em_fields(i + 1, j - 1, fld::ex2));
  c010 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::ex2) + m_meshblock.em_fields(    i, j + 1, fld::ex2));
  c110 = 0.5 * (m_meshblock.em_fields(i + 1,     j, fld::ex2) + m_meshblock.em_fields(i + 1, j + 1, fld::ex2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x2 = c00 * (ONE - dx2) + c10 * dx2;
  // Ex3
  c000 = m_meshblock.em_fields(    i,     j, fld::ex3);
  c100 = m_meshblock.em_fields(i + 1,     j, fld::ex3);
  c010 = m_meshblock.em_fields(    i, j + 1, fld::ex3);
  c110 = m_meshblock.em_fields(i + 1, j + 1, fld::ex3);
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x3 = c00 * (ONE - dx2) + c10 * dx2;

  // Bx1
  c000 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::bx1) + m_meshblock.em_fields(    i, j - 1, fld::bx1));
  c100 = 0.5 * (m_meshblock.em_fields(i + 1,     j, fld::bx1) + m_meshblock.em_fields(i + 1, j - 1, fld::bx1));
  c010 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::bx1) + m_meshblock.em_fields(    i, j + 1, fld::bx1));
  c110 = 0.5 * (m_meshblock.em_fields(i + 1,     j, fld::bx1) + m_meshblock.em_fields(i + 1, j + 1, fld::bx1));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x1 = c00 * (ONE - dx2) + c10 * dx2;
  // Bx2
  c000 = 0.5 * (m_meshblock.em_fields(i - 1,     j, fld::bx2) + m_meshblock.em_fields(    i,     j, fld::bx2));
  c100 = 0.5 * (m_meshblock.em_fields(    i,     j, fld::bx2) + m_meshblock.em_fields(i + 1,     j, fld::bx2));
  c010 = 0.5 * (m_meshblock.em_fields(i - 1, j + 1, fld::bx2) + m_meshblock.em_fields(    i, j + 1, fld::bx2));
  c110 = 0.5 * (m_meshblock.em_fields(    i, j + 1, fld::bx2) + m_meshblock.em_fields(i + 1, j + 1, fld::bx2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x2 = c00 * (ONE - dx2) + c10 * dx2;
  // Bx3
  c000 = 0.25 * (m_meshblock.em_fields(i - 1, j - 1, fld::bx3) + m_meshblock.em_fields(i - 1,     j, fld::bx3) + m_meshblock.em_fields(    i, j - 1, fld::bx3) + m_meshblock.em_fields(    i,     j, fld::bx3));
  c100 = 0.25 * (m_meshblock.em_fields(    i, j - 1, fld::bx3) + m_meshblock.em_fields(    i,     j, fld::bx3) + m_meshblock.em_fields(i + 1, j - 1, fld::bx3) + m_meshblock.em_fields(i + 1,     j, fld::bx3));
  c010 = 0.25 * (m_meshblock.em_fields(i - 1,     j, fld::bx3) + m_meshblock.em_fields(i - 1, j + 1, fld::bx3) + m_meshblock.em_fields(    i,     j, fld::bx3) + m_meshblock.em_fields(    i, j + 1, fld::bx3));
  c110 = 0.25 * (m_meshblock.em_fields(    i,     j, fld::bx3) + m_meshblock.em_fields(    i, j + 1, fld::bx3) + m_meshblock.em_fields(i + 1,     j, fld::bx3) + m_meshblock.em_fields(i + 1, j + 1, fld::bx3));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x3 = c00 * (ONE - dx2) + c10 * dx2;
  // clang-format on
}

template <>
Inline void Pusher<THREE_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  const auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));
  const auto [k, dx3] = convert_x3TOkdx3(m_meshblock, m_particles.m_x3(p));
  e0_x1 = m_meshblock.em_fields(i, j, k, fld::ex1);
  e0_x2 = m_meshblock.em_fields(i, j, k, fld::ex2);
  e0_x3 = m_meshblock.em_fields(i, j, k, fld::ex3);

  b0_x1 = m_meshblock.em_fields(i, j, k, fld::bx1);
  b0_x2 = m_meshblock.em_fields(i, j, k, fld::bx2);
  b0_x3 = m_meshblock.em_fields(i, j, k, fld::bx3);
}

} // namespace ntt

template class ntt::Pusher<ntt::ONE_D>;
template class ntt::Pusher<ntt::TWO_D>;
template class ntt::Pusher<ntt::THREE_D>;

#endif
