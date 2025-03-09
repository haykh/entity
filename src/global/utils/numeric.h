/**
 * @file utils/numeric.h
 * @brief Basic numerical constants and utilities
 * @implements
 *   - macro IMIN
 *   - macro IMAX
 *   - macro SIGN
 *   - macro HEAVISIDE
 *   - macro SQR
 *   - macro CUBE
 *   - macro DOT
 *   - macro NORM_SQR
 *   - macro NORM
 *   - macro CROSS_x1
 *   - macro CROSS_x2
 *   - macro CROSS_x3
 *   - literal real-valued numbers
 * @namespaces:
 *   - constant::
 * @macros:
 *   - SINGLE_PRECISION
 * !TODO:
 *   - potentially use math::signbit instead of SIGN
 */

#ifndef GLOBAL_UTILS_NUMERIC_H
#define GLOBAL_UTILS_NUMERIC_H

#include "arch/kokkos_aliases.h"

#include <cstdint>

#if defined(SINGLE_PRECISION)
inline constexpr float ONE    = 1.0f;
inline constexpr float TWO    = 2.0f;
inline constexpr float THREE  = 3.0f;
inline constexpr float FOUR   = 4.0f;
inline constexpr float FIVE   = 5.0f;
inline constexpr float ZERO   = 0.0f;
inline constexpr float HALF   = 0.5f;
inline constexpr float INV_2  = 0.5f;
inline constexpr float INV_4  = 0.25f;
inline constexpr float INV_8  = 0.125f;
inline constexpr float INV_16 = 0.0625f;
inline constexpr float INV_32 = 0.03125f;
inline constexpr float INV_64 = 0.015625f;
#else
inline constexpr double ONE    = 1.0;
inline constexpr double TWO    = 2.0;
inline constexpr double THREE  = 3.0;
inline constexpr double FOUR   = 4.0;
inline constexpr double FIVE   = 5.0;
inline constexpr double ZERO   = 0.0;
inline constexpr double HALF   = 0.5;
inline constexpr double INV_2  = 0.5;
inline constexpr double INV_4  = 0.25;
inline constexpr double INV_8  = 0.125;
inline constexpr double INV_16 = 0.0625;
inline constexpr double INV_32 = 0.03125;
inline constexpr double INV_64 = 0.015625;
#endif

#define IMIN(a, b)   ((a) < (b) ? (a) : (b))
#define IMAX(a, b)   ((a) > (b) ? (a) : (b))
#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

#define DOT(ax1, ax2, ax3, bx1, bx2, bx3)                                      \
  ((ax1) * (bx1) + (ax2) * (bx2) + (ax3) * (bx3))
#define NORM_SQR(ax1, ax2, ax3)                (DOT((ax1), (ax2), (ax3), (ax1), (ax2), (ax3)))
#define U2GAMMA_SQR(ax1, ax2, ax3)             (ONE + NORM_SQR((ax1), (ax2), (ax3)))
#define U2GAMMA(ax1, ax2, ax3)                 (math::sqrt(U2GAMMA_SQR((ax1), (ax2), (ax3))))
#define NORM(ax1, ax2, ax3)                    (math::sqrt(NORM_SQR((ax1), (ax2), (ax3))))
#define CROSS_x1(ax1, ax2, ax3, bx1, bx2, bx3) ((ax2) * (bx3) - (ax3) * (bx2))
#define CROSS_x2(ax1, ax2, ax3, bx1, bx2, bx3) ((ax3) * (bx1) - (ax1) * (bx3))
#define CROSS_x3(ax1, ax2, ax3, bx1, bx2, bx3) ((ax1) * (bx2) - (ax2) * (bx1))

namespace constant {
  inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;
  inline constexpr double        HALF_PI    = 1.57079632679489661923;
  inline constexpr double        PI         = 3.14159265358979323846;
  inline constexpr double        INV_PI     = 0.31830988618379067154;
  inline constexpr double        PI_SQR     = 9.86960440108935861882;
  inline constexpr double        INV_PI_SQR = 0.10132118364233777144;
  inline constexpr double        TWO_PI     = 6.28318530717958647692;
  inline constexpr double        E          = 2.71828182845904523536;
  inline constexpr double        SQRT2      = 1.41421356237309504880;
  inline constexpr double        INV_SQRT2  = 0.70710678118654752440;
  inline constexpr double        SQRT3      = 1.73205080756887729352;
} // namespace constant

namespace convert {
  inline constexpr double deg2rad = constant::PI / 180.0;
} // namespace convert

#endif // GLOBAL_UTILS_NUMERIC_H
