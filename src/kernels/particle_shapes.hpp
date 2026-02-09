/**
 * @file kernels/particle_shapes.hpp
 * @brief Functions to compute particle shapes at specific locations on the grid.
 * @implements:
 *   - order_2<> -> void
 * @namespaces:
 *   - prtl_shape::
 */

#ifndef KERNELS_PARTICLE_SHAPES_HPP
#define KERNELS_PARTICLE_SHAPES_HPP

#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

namespace prtl_shape {

  // clang-format off
  //        115/192 - (5/8) * |x|^2 + (1/4) * |x|^4                                                  |x| < 1/2
  // S(x) = 55/96 + (5/24) * |x| - (5/4) * |x|^2 + (5/6) * |x|^3 - (1/6) * |x|^4               1/2 ≤ |x| < 3/2
  //        625/384 - (125/48) * |x| + (25/16) * |x|^2 - (5/12) * |x|^3 + (1/24) * |x|^4       3/2 ≤ |x| < 5/2
  //        0.0                                                                                      |x| ≥ 5/2
  // clang-format on
  Inline real_t S3(const real_t x) {
    if (x < ONE) {
      return static_cast<real_t>(2.0 / 3.0) - SQR(x) + HALF * CUBE(x);
    } else if (x < TWO) {
      return static_cast<real_t>(4.0 / 3.0) - TWO * x + SQR(x) -
             static_cast<real_t>(1.0 / 6.0) * CUBE(x);
    } else {
      return ZERO;
    }
  }

  // clang-format off
  //        115/192 - (5/8) * |x|^2 + (1/4) * |x|^4                                                  |x| < 1/2
  // S(x) = 55/96 + (5/24) * |x| - (5/4) * |x|^2 + (5/6) * |x|^3 - (1/6) * |x|^4               1/2 ≤ |x| < 3/2
  //        625/384 - (125/48) * |x| + (25/16) * |x|^2 - (5/12) * |x|^3 + (1/24) * |x|^4       3/2 ≤ |x| < 5/2
  //        0.0                                                                                      |x| ≥ 5/2
  // clang-format on
  Inline real_t S4(const real_t x) {
    if (x < HALF) {
      return static_cast<real_t>(115.0 / 192.0) -
             static_cast<real_t>(5.0 / 8.0) * SQR(x) + INV_4 * SQR(SQR(x));
    } else if (x < THREE_HALFS) {
      return static_cast<real_t>(55.0 / 96.0) +
             static_cast<real_t>(5.0 / 24.0) * x -
             static_cast<real_t>(5.0 / 4.0) * SQR(x) +
             static_cast<real_t>(5.0 / 6.0) * CUBE(x) -
             static_cast<real_t>(1.0 / 6.0) * SQR(SQR(x));
    } else if (x < static_cast<real_t>(2.5)) {
      return static_cast<real_t>(625.0 / 384.0) -
             static_cast<real_t>(125.0 / 48.0) * x +
             static_cast<real_t>(25.0 / 16.0) * SQR(x) -
             static_cast<real_t>(5.0 / 12.0) * CUBE(x) +
             static_cast<real_t>(1.0 / 24.0) * SQR(SQR(x));
    } else {
      return ZERO;
    }
  }

  // clang-format off
      //  S5(x) = 
      //   11/20 - (1/2) * |x|^2 + (1/4) * |x|^4 - (1/12) * |x|^5                                        if |x| ≤ 1
      //   17/40 + (5/8) * |x| - (7/4) * |x|^2 + (5/4) * |x|^3 - (3/8) * |x|^4 + (1/24) * |x|^5      if 1 < |x| < 2
      //   81/40 - (27/8) * |x| + (9/4) * |x|^2 - (3/4) * |x|^3 + (1/8) * |x|^4 - (1/120) * |x|^5    if 2 ≤ |x| < 3
      //   0.0                                                                                           if |x| > 3
  // clang-format on

  Inline real_t S5(const real_t x) {
    if (x <= ONE) {
      return static_cast<real_t>(11.0 / 20.0) - HALF * SQR(x) +
             INV_4 * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 12.0) * CUBE(x) * SQR(x);
    } else if (x < TWO) {
      return static_cast<real_t>(17.0 / 40.0) + static_cast<real_t>(5.0 / 8.0) * x -
             static_cast<real_t>(7.0 / 4.0) * SQR(x) +
             static_cast<real_t>(5.0 / 4.0) * CUBE(x) -
             static_cast<real_t>(3.0 / 8.0) * SQR(SQR(x)) +
             static_cast<real_t>(1.0 / 24.0) * CUBE(x) * SQR(x);
    } else if (x < THREE) {
      return static_cast<real_t>(81.0 / 40.0) -
             static_cast<real_t>(27.0 / 8.0) * x +
             static_cast<real_t>(9.0 / 4.0) * SQR(x) - THREE_FOURTHS * CUBE(x) +
             INV_8 * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 120.0) * CUBE(x) * SQR(x);
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S6(x) = 
  //   5887/11520 - (77/192) * |x|^2 + (7/48) * |x|^4 - (1/36) * |x|^6                     if |x| ≤ 1/2
  //   7861/15360 - (7/768) * |x| - (91/256) * |x|^2 - (35/288) * |x|^3 + (21/64) * |x|^4 
  //                 - (7/48) * |x|^5 + (1/48) * |x|^6                                     if 1/2 < |x| < 3/2
  //   1379/7680 + (1267/960) * |x| - (329/128) * |x|^2 + (133/72) * |x|^3 
  //                 - (21/32) * |x|^4 + (7/60) * |x|^5 - (1/120) * |x|^6                  if 3/2 ≤ |x| < 5/2
  //   117649/46080 - (16807/3840) * |x| + (2401/768) * |x|^2 - (343/288) * |x|^3 
  //                 + (49/192) * |x|^4 - (7/240) * |x|^5 + (1/720) * |x|^6                if 5/2 ≤ |x| < 7/2
  //   0.0                                                                                 if |x| ≥ 7/2
  // clang-format on
  Inline real_t S6(const real_t x) {
    if (x <= HALF) {
      return static_cast<real_t>(5887.0 / 11520.0) -
             static_cast<real_t>(77.0 / 192.0) * SQR(x) +
             static_cast<real_t>(7.0 / 48.0) * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 36.0) * SQR(CUBE(x));
    } else if (x < THREE_HALFS) {
      return static_cast<real_t>(7861.0 / 15360.0) -
             static_cast<real_t>(7.0 / 768.0) * x -
             static_cast<real_t>(91.0 / 256.0) * SQR(x) -
             static_cast<real_t>(35.0 / 288.0) * CUBE(x) +
             static_cast<real_t>(21.0 / 64.0) * SQR(SQR(x)) -
             static_cast<real_t>(7.0 / 48.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(1.0 / 48.0) * SQR(CUBE(x));
    } else if (x < static_cast<real_t>(2.5)) {
      return static_cast<real_t>(1379.0 / 7680.0) +
             static_cast<real_t>(1267.0 / 960.0) * x -
             static_cast<real_t>(329.0 / 128.0) * SQR(x) +
             static_cast<real_t>(133.0 / 72.0) * CUBE(x) -
             static_cast<real_t>(21.0 / 32.0) * SQR(SQR(x)) +
             static_cast<real_t>(7.0 / 60.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(1.0 / 120.0) * SQR(CUBE(x));
    } else if (x < static_cast<real_t>(3.5)) {
      return static_cast<real_t>(117649.0 / 46080.0) -
             static_cast<real_t>(16807.0 / 3840.0) * x +
             static_cast<real_t>(2401.0 / 768.0) * SQR(x) -
             static_cast<real_t>(343.0 / 288.0) * CUBE(x) +
             static_cast<real_t>(49.0 / 192.0) * SQR(SQR(x)) -
             static_cast<real_t>(7.0 / 240.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(1.0 / 720.0) * SQR(CUBE(x));
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S7(x) = 
  //   151/315 - (1/3) * |x|^2 + (1/9) * |x|^4 - (1/36) * |x|^6 + (1/144) * |x|^7       if |x| < 1
  //   103/210 - (7/90) * |x| - (1/10) * |x|^2 - (7/18) * |x|^3 + (1/2) * |x|^4 
  //             - (7/30) * |x|^5 + (1/20) * |x|^6 - (1/240) * |x|^7                   if 1 ≤ |x| ≤ 2
  //   (217/90) * |x| - (23/6) * |x|^2 + (49/18) * |x|^3 - (19/18) * |x|^4 
  //             + (7/30) * |x|^5 - (1/36) * |x|^6 + (1/720) * |x|^7 - (139/630)       if 2 < |x| < 3
  //   1024/315 - (256/45) * |x| + (64/15) * |x|^2 - (16/9) * |x|^3 + (4/9) * |x|^4 
  //             - (1/15) * |x|^5 + (1/180) * |x|^6 - (1/5040) * |x|^7                 if 3 ≤ |x| < 4
  //   0.0                                                                             if |x| ≥ 4
  // clang-format on
  Inline real_t S7(const real_t x) {
    if (x < ONE) {
      return static_cast<real_t>(151.0 / 315.0) - THIRD * SQR(x) +
             static_cast<real_t>(1.0 / 9.0) * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 36.0) * SQR(SQR(x)) * SQR(x) +
             static_cast<real_t>(1.0 / 144.0) * SQR(SQR(x)) * CUBE(x);
    } else if (x <= TWO) {
      return static_cast<real_t>(103.0 / 210.0) -
             static_cast<real_t>(7.0 / 90.0) * x -
             static_cast<real_t>(1.0 / 10.0) * SQR(x) -
             static_cast<real_t>(7.0 / 18.0) * CUBE(x) + HALF * SQR(SQR(x)) -
             static_cast<real_t>(7.0 / 30.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(1.0 / 20.0) * SQR(SQR(x)) * SQR(x) -
             static_cast<real_t>(1.0 / 240.0) * SQR(SQR(x)) * CUBE(x);
    } else if (x < THREE) {
      return static_cast<real_t>(217.0 / 90.0) * x -
             static_cast<real_t>(23.0 / 6.0) * SQR(x) +
             static_cast<real_t>(49.0 / 18.0) * CUBE(x) -
             static_cast<real_t>(19.0 / 18.0) * SQR(SQR(x)) +
             static_cast<real_t>(7.0 / 30.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(1.0 / 36.0) * SQR(SQR(x)) * SQR(x) +
             static_cast<real_t>(1.0 / 720.0) * SQR(SQR(x)) * CUBE(x) -
             static_cast<real_t>(139.0 / 630.0);
    } else if (x < FOUR) {
      return static_cast<real_t>(1024.0 / 315.0) -
             static_cast<real_t>(256.0 / 45.0) * x +
             static_cast<real_t>(64.0 / 15.0) * SQR(x) -
             static_cast<real_t>(16.0 / 9.0) * CUBE(x) +
             static_cast<real_t>(4.0 / 9.0) * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 15.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(1.0 / 180.0) * SQR(SQR(x)) * SQR(x) -
             static_cast<real_t>(1.0 / 5040.0) * SQR(SQR(x)) * CUBE(x);
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S8(x) = 
  //   259723/573440 - (289/1024) * |x|^2 + (43/512) * |x|^4 - (1/64) * |x|^6 + (1/576) * |x|^8                     if |x| < 1/2
  //   64929/143360 + (1/5120) * |x| - (363/1280) * |x|^2 + (7/1280) * |x|^3 + (9/128) * |x|^4 
  //                  + (7/320) * |x|^5 - (3/80) * |x|^6 + (1/80) * |x|^7 - (1/720) * |x|^8                        if 1/2 ≤ |x| ≤ 3/2
  //   145167/286720 - (1457/5120) * |x| + (195/512) * |x|^2 - (1127/1280) * |x|^3 + (207/256) * |x|^4 
  //                  - (119/320) * |x|^5 + (3/32) * |x|^6 - (1/80) * |x|^7 + (1/1440) * |x|^8                     if 3/2 < |x| < 2.5
  //   (146051/35840) * |x| - (1465/256) * |x|^2 + (5123/1280) * |x|^3 - (209/128) * |x|^4 
  //                  + (131/320) * |x|^5 - (1/16) * |x|^6 + (3/560) * |x|^7 - (1/5040) * |x|^8 - (122729/143360)  if 2.5 ≤ |x| < 3.5
  //   4782969/1146880 - (531441/71680) * |x| + (59049/10240) * |x|^2 - (6561/2560) * |x|^3 + (729/1024) * |x|^4 
  //                  - (81/640) * |x|^5 + (9/640) * |x|^6 - (1/1120) * |x|^7 + (1/40320) * |x|^8                  if 3.5 ≤ |x| < 4.5
  //   0.0
  // clang-format on
  Inline real_t S8(const real_t x) {
    if (x < HALF) {
      return static_cast<real_t>(259723.0 / 573440.0) -
             static_cast<real_t>(289.0 / 1024.0) * SQR(x) +
             static_cast<real_t>(43.0 / 512.0) * SQR(SQR(x)) -
             static_cast<real_t>(1.0 / 64.0) * SQR(SQR(x)) * SQR(x) +
             static_cast<real_t>(1.0 / 576.0) * SQR(SQR(SQR(x)));
    } else if (x <= THREE_HALFS) {
      return static_cast<real_t>(64929.0 / 143360.0) +
             static_cast<real_t>(1.0 / 5120.0) * x -
             static_cast<real_t>(363.0 / 1280.0) * SQR(x) +
             static_cast<real_t>(7.0 / 1280.0) * CUBE(x) +
             static_cast<real_t>(9.0 / 128.0) * SQR(SQR(x)) +
             static_cast<real_t>(7.0 / 320.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(3.0 / 80.0) * SQR(CUBE(x)) +
             static_cast<real_t>(1.0 / 80.0) * SQR(SQR(x)) * CUBE(x) -
             static_cast<real_t>(1.0 / 720.0) * SQR(SQR(SQR(x)));
    } else if (x < static_cast<real_t>(2.5)) {
      return static_cast<real_t>(145167.0 / 286720.0) -
             static_cast<real_t>(1457.0 / 5120.0) * x +
             static_cast<real_t>(195.0 / 512.0) * SQR(x) -
             static_cast<real_t>(1127.0 / 1280.0) * CUBE(x) +
             static_cast<real_t>(207.0 / 256.0) * SQR(SQR(x)) -
             static_cast<real_t>(119.0 / 320.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(3.0 / 32.0) * SQR(CUBE(x)) -
             static_cast<real_t>(1.0 / 80.0) * SQR(SQR(x)) * CUBE(x) +
             static_cast<real_t>(1.0 / 1440.0) * SQR(SQR(SQR(x)));
    } else if (x < static_cast<real_t>(3.5)) {
      return static_cast<real_t>(146051.0 / 35840.0) * x -
             static_cast<real_t>(1465.0 / 256.0) * SQR(x) +
             static_cast<real_t>(5123.0 / 1280.0) * CUBE(x) -
             static_cast<real_t>(209.0 / 128.0) * SQR(SQR(x)) +
             static_cast<real_t>(131.0 / 320.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(1.0 / 16.0) * SQR(CUBE(x)) +
             static_cast<real_t>(3.0 / 560.0) * SQR(SQR(x)) * CUBE(x) -
             static_cast<real_t>(1.0 / 5040.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(122729.0 / 143360.0);
    } else if (x < static_cast<real_t>(4.5)) {
      return static_cast<real_t>(4782969.0 / 1146880.0) -
             static_cast<real_t>(531441.0 / 71680.0) * x +
             static_cast<real_t>(59049.0 / 10240.0) * SQR(x) -
             static_cast<real_t>(6561.0 / 2560.0) * CUBE(x) +
             static_cast<real_t>(729.0 / 1024.0) * SQR(SQR(x)) -
             static_cast<real_t>(81.0 / 640.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(9.0 / 640.0) * SQR(CUBE(x)) -
             static_cast<real_t>(1.0 / 1120.0) * SQR(SQR(x)) * CUBE(x) +
             static_cast<real_t>(1.0 / 40320.0) * SQR(SQR(SQR(x)));
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S9(x) = 
  //   15619/36288 - (35/144) * |x|^2 + (19/288) * |x|^4 - (5/432) * |x|^6 + (1/576) * |x|^8 - (1/2880) * |x|^9          if |x| ≤ 1
  //   7799/18144 + (1/192) * |x| - (19/72) * |x|^2 + (7/144) * |x|^3 - (1/144) * |x|^4 + (7/96) * |x|^5 
  //                - (13/216) * |x|^6 + (1/48) * |x|^7 - (1/288) * |x|^8 + (1/4320) * |x|^9                            if 1 < |x| < 2
  //   1553/2592 - (339/448) * |x| + (635/504) * |x|^2 - (83/48) * |x|^3 + (191/144) * |x|^4 - (19/32) * |x|^5 
  //                + (35/216) * |x|^6 - (3/112) * |x|^7 + (5/2016) * |x|^8 - (1/10080) * |x|^9                         if 2 ≤ |x| < 3
  //   (5883/896) * |x| - (2449/288) * |x|^2 + (563/96) * |x|^3 - (1423/576) * |x|^4 + (43/64) * |x|^5 
  //                - (103/864) * |x|^6 + (3/224) * |x|^7 - (1/1152) * |x|^8 + (1/40320) * |x|^9 - (133663/72576)       if 3 ≤ |x| < 4
  //   390625/72576 - (78125/8064) * |x| + (15625/2016) * |x|^2 - (3125/864) * |x|^3 + (625/576) * |x|^4 
  //                - (125/576) * |x|^5 + (25/864) * |x|^6 - (5/2016) * |x|^7 + (1/8064) * |x|^8 - (1/362880) * |x|^9   if 4 ≤ |x| < 5
  //   0.0                                                                                                              if |x| ≥ 5
  // clang-format on
  Inline real_t S9(const real_t x) {
    if (x <= ONE) {
      return static_cast<real_t>(15619.0 / 36288.0) -
             static_cast<real_t>(35.0 / 144.0) * SQR(x) +
             static_cast<real_t>(19.0 / 288.0) * SQR(SQR(x)) -
             static_cast<real_t>(5.0 / 432.0) * SQR(CUBE(x)) +
             static_cast<real_t>(1.0 / 576.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 2880.0) * SQR(SQR(SQR(x))) * x;
    } else if (x < TWO) {
      return static_cast<real_t>(7799.0 / 18144.0) +
             static_cast<real_t>(1.0 / 192.0) * x -
             static_cast<real_t>(19.0 / 72.0) * SQR(x) +
             static_cast<real_t>(7.0 / 144.0) * CUBE(x) -
             static_cast<real_t>(1.0 / 144.0) * SQR(SQR(x)) +
             static_cast<real_t>(7.0 / 96.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(13.0 / 216.0) * SQR(CUBE(x)) +
             static_cast<real_t>(1.0 / 48.0) * SQR(SQR(x)) * CUBE(x) -
             static_cast<real_t>(1.0 / 288.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(1.0 / 4320.0) * CUBE(CUBE(x));
    } else if (x <= THREE) {
      return static_cast<real_t>(1553.0 / 2592.0) -
             static_cast<real_t>(339.0 / 448.0) * x +
             static_cast<real_t>(635.0 / 504.0) * SQR(x) -
             static_cast<real_t>(83.0 / 48.0) * CUBE(x) +
             static_cast<real_t>(191.0 / 144.0) * SQR(SQR(x)) -
             static_cast<real_t>(19.0 / 32.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(35.0 / 216.0) * SQR(CUBE(x)) -
             static_cast<real_t>(3.0 / 112.0) * SQR(SQR(x)) * CUBE(x) +
             static_cast<real_t>(5.0 / 2016.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 10080.0) * CUBE(CUBE(x));
    } else if (x < FOUR) {
      return static_cast<real_t>(5883.0 / 896.0) * x -
             static_cast<real_t>(2449.0 / 288.0) * SQR(x) +
             static_cast<real_t>(563.0 / 96.0) * CUBE(x) -
             static_cast<real_t>(1423.0 / 576.0) * SQR(SQR(x)) +
             static_cast<real_t>(43.0 / 64.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(103.0 / 864.0) * SQR(CUBE(x)) +
             static_cast<real_t>(3.0 / 224.0) * SQR(SQR(x)) * CUBE(x) -
             static_cast<real_t>(1.0 / 1152.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(1.0 / 40320.0) * CUBE(CUBE(x)) -
             static_cast<real_t>(133663.0 / 72576.0);
    } else if (x < FIVE) {
      return static_cast<real_t>(390625.0 / 72576.0) -
             static_cast<real_t>(78125.0 / 8064.0) * x +
             static_cast<real_t>(15625.0 / 2016.0) * SQR(x) -
             static_cast<real_t>(3125.0 / 864.0) * CUBE(x) +
             static_cast<real_t>(625.0 / 576.0) * SQR(SQR(x)) -
             static_cast<real_t>(125.0 / 576.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(25.0 / 864.0) * SQR(CUBE(x)) -
             static_cast<real_t>(5.0 / 2016.0) * SQR(SQR(x)) * CUBE(x) +
             static_cast<real_t>(1.0 / 8064.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 362880.0) * CUBE(CUBE(x));
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S10(x) =
  //   381773117/928972800 - (156409/737280) * |x|^2 + (14597/276480) * |x|^4 - (583/69120) * |x|^6 + (11/11520) * |x|^8 - (1/14400) * |x|^10                     if |x| ≤ 0.5
  //   152709293/371589120 - (11/4423680) * |x| - (62557/294912) * |x|^2 - (11/92160) * |x|^3 + (5885/110592) * |x|^4 - (77/76800) * |x|^5 - 
  //              (187/27648) * |x|^6 - (11/5760) * |x|^7 + (11/4608) * |x|^8 - (11/17280) * |x|^9 + (1/17280) * |x|^10                                           if 0.5 < |x| ≤ 1.5
  //   37690169/92897280 + (135311/3870720) * |x| - (163603/516096) * |x|^2 + (7513/40320) * |x|^3  - (4543/27648) * |x|^4  
  //              + (1661/9600) * |x|^5 - (715/6912) * |x|^6 + (11/315) * |x|^7 - (55/8064) * |x|^8 + (11/15120) * |x|^9 - (1/30240) * |x|^10                     if 1.5 < |x| ≤ 2.5
  //   623786977/743178240 - (11695211/6881280) * |x| + (1654543/589824) * |x|^2 - (1352153/430080) * |x|^3 + (479281/221184) * |x|^4 
  //              - (48433/51200) * |x|^5 + (14905/55296) * |x|^6 - (451/8960) * |x|^7 + (55/9216) * |x|^8 - (11/26880) * |x|^9 + (1/80640) * |x|^10              if 2.5 < |x| ≤ 3.5
  //   -1241720381/371589120 + (237959711/23224320) * |x| - (3702215/294912) * |x|^2 + (2070343/241920) * |x|^3 - (407429/110592) * |x|^4 
  //              + (61061/57600) * |x|^5 - (5753/27648) * |x|^6 + (209/7560) * |x|^7 - (11/4608) * |x|^8 + (11/90720) * |x|^9 - (1/362880) * |x|^10              if 3.5 < |x| ≤ 4.5
  //   25937424601/3715891200 - (2357947691/185794560) * |x| + (214358881/20643840) * |x|^2 - (19487171/3870720) * |x|^3 + (1771561/1105920) * |x|^4 
  //              - (161051/460800) * |x|^5 + (14641/276480) * |x|^6 - (1331/241920) * |x|^7 + (121/322560) * |x|^8 - (11/725760) * |x|^9 + (1/3628800) * |x|^10  if 4.5 < |x| ≤ 5.5
  //   0.0                                                                                                                                                        otherwise
  // clang-format on
  Inline real_t S10(const real_t x) {
    if (x < HALF) {
      return static_cast<real_t>(381773117.0 / 928972800.0) -
             static_cast<real_t>(156409.0 / 737280.0) * SQR(x) +
             static_cast<real_t>(14597.0 / 276480.0) * SQR(SQR(x)) -
             static_cast<real_t>(583.0 / 69120.0) * SQR(CUBE(x)) +
             static_cast<real_t>(11.0 / 11520.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 14400.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else if (x < THREE_HALFS) {
      return static_cast<real_t>(152709293.0 / 371589120.0) -
             static_cast<real_t>(11.0 / 4423680.0) * x -
             static_cast<real_t>(62557.0 / 294912.0) * SQR(x) -
             static_cast<real_t>(11.0 / 92160.0) * CUBE(x) +
             static_cast<real_t>(5885.0 / 110592.0) * SQR(SQR(x)) -
             static_cast<real_t>(77.0 / 76800.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(187.0 / 27648.0) * SQR(CUBE(x)) -
             static_cast<real_t>(11.0 / 5760.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(11.0 / 4608.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(11.0 / 17280.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 17280.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else if (x <= static_cast<real_t>(2.5)) {
      return static_cast<real_t>(37690169.0 / 92897280.0) +
             static_cast<real_t>(135311.0 / 3870720.0) * x -
             static_cast<real_t>(163603.0 / 516096.0) * SQR(x) +
             static_cast<real_t>(7513.0 / 40320.0) * CUBE(x) -
             static_cast<real_t>(4543.0 / 27648.0) * SQR(SQR(x)) +
             static_cast<real_t>(1661.0 / 9600.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(715.0 / 6912.0) * SQR(CUBE(x)) +
             static_cast<real_t>(11.0 / 315.0) * SQR(CUBE(x)) * x -
             static_cast<real_t>(55.0 / 8064.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(11.0 / 15120.0) * SQR(SQR(SQR(x))) * x -
             static_cast<real_t>(1.0 / 30240.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else if (x < static_cast<real_t>(3.5)) {
      return static_cast<real_t>(623786977.0 / 743178240.0) -
             static_cast<real_t>(11695211.0 / 6881280.0) * x +
             static_cast<real_t>(1654543.0 / 589824.0) * SQR(x) -
             static_cast<real_t>(1352153.0 / 430080.0) * CUBE(x) +
             static_cast<real_t>(479281.0 / 221184.0) * SQR(SQR(x)) -
             static_cast<real_t>(48433.0 / 51200.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(14905.0 / 55296.0) * SQR(CUBE(x)) -
             static_cast<real_t>(451.0 / 8960.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(55.0 / 9216.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(11.0 / 26880.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 80640.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else if (x < static_cast<real_t>(4.5)) {
      return static_cast<real_t>(-1241720381.0 / 371589120.0) +
             static_cast<real_t>(237959711.0 / 23224320.0) * x -
             static_cast<real_t>(3702215.0 / 294912.0) * SQR(x) +
             static_cast<real_t>(2070343.0 / 241920.0) * CUBE(x) -
             static_cast<real_t>(407429.0 / 110592.0) * SQR(SQR(x)) +
             static_cast<real_t>(61061.0 / 57600.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(5753.0 / 27648.0) * SQR(CUBE(x)) +
             static_cast<real_t>(209.0 / 7560.0) * SQR(CUBE(x)) * x -
             static_cast<real_t>(11.0 / 4608.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(11.0 / 90720.0) * SQR(SQR(SQR(x))) * x -
             static_cast<real_t>(1.0 / 362880.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else if (x < static_cast<real_t>(5.5)) {
      return static_cast<real_t>(25937424601.0 / 3715891200.0) -
             static_cast<real_t>(2357947691.0 / 185794560.0) * x +
             static_cast<real_t>(214358881.0 / 20643840.0) * SQR(x) -
             static_cast<real_t>(19487171.0 / 3870720.0) * CUBE(x) +
             static_cast<real_t>(1771561.0 / 1105920.0) * SQR(SQR(x)) -
             static_cast<real_t>(161051.0 / 460800.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(14641.0 / 276480.0) * SQR(CUBE(x)) -
             static_cast<real_t>(1331.0 / 241920.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(121.0 / 322560.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(11.0 / 725760.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 3628800.0) * SQR(SQR(SQR(x))) * SQR(x);
    } else {
      return ZERO;
    }
  }

  // clang-format off
  // S11(x) =
  //   655177/1663200 - (809/4320) * |x|^2 + (31/720) * |x|^4 - (23/3600) * |x|^6 
  //                  + (1/1440) * |x|^8 - (1/14400) * |x|^10 + (1/86400) * |x|^11                        if |x| < 1
  //   65521/166320 - (11/50400) * |x| - (563/3024) * |x|^2 - (11/3360) * |x|^3 
  //                  + (25/504) * |x|^4 - (11/1200) * |x|^5 + (1/360) * |x|^6 
  //                  - (11/1680) * |x|^7 + (1/252) * |x|^8 - (11/10080) * |x|^9 
  //                  + (1/6720) * |x|^10 - (1/120960) * |x|^11                                          if 1 ≤ |x| ≤ 2
  //   61297/166320 + (781/5600) * |x| - (1619/3024) * |x|^2 + (583/1120) * |x|^3 
  //                  - (239/504) * |x|^4 + (143/400) * |x|^5 - (13/72) * |x|^6 
  //                  + (33/560) * |x|^7 - (25/2016) * |x|^8 + (11/6720) * |x|^9 
  //                  - (1/8064) * |x|^10 + (1/241920) * |x|^11                                         if 2 < |x| < 3
  //   894727/665280 - (38533/11200) * |x| + (9385/1728) * |x|^2 - (12199/2240) * |x|^3 
  //                  + (1009/288) * |x|^4 - (1199/800) * |x|^5 + (631/1440) * |x|^6 
  //                  - (99/1120) * |x|^7 + (7/576) * |x|^8 - (11/10080) * |x|^9 
  //                  + (1/17280) * |x|^10 - (1/725760) * |x|^11                                        if 3 ≤ |x| ≤ 4
  //  -18595037/3326400 + (4726777/302400) * |x| - (1113317/60480) * |x|^2 + (250657/20160) * |x|^3 
  //                  - (54797/10080) * |x|^4 + (11737/7200) * |x|^5 - (2477/7200) * |x|^6 
  //                  + (517/10080) * |x|^7 - (107/20160) * |x|^8 + (11/30240) * |x|^9 
  //                  - (1/67200) * |x|^10 + (1/3628800) * |x|^11                                      if 4 < |x| < 5
  //   17496/1925 - (2916/175) * |x| + (486/35) * |x|^2 - (243/35) * |x|^3 
  //                  + (81/35) * |x|^4 - (27/50) * |x|^5 + (9/100) * |x|^6 
  //                  - (3/280) * |x|^7 + (1/1120) * |x|^8 - (1/20160) * |x|^9 
  //                  + (1/604800) * |x|^10 - (1/39916800) * |x|^11                                     if 5 ≤ |x| < 6
  //   0.0                                                                                              otherwise
  // clang-format on
  Inline real_t S11(const real_t x) {
    if (x < ONE) {
      return static_cast<real_t>(655177.0 / 1663200.0) -
             static_cast<real_t>(809.0 / 4320.0) * SQR(x) +
             static_cast<real_t>(31.0 / 720.0) * SQR(SQR(x)) -
             static_cast<real_t>(23.0 / 3600.0) * CUBE(SQR(x)) +
             static_cast<real_t>(1.0 / 1440.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 14400.0) * SQR(SQR(SQR(x))) * SQR(x) +
             static_cast<real_t>(1.0 / 86400.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else if (x <= TWO) {
      return static_cast<real_t>(65521.0 / 166320.0) -
             static_cast<real_t>(11.0 / 50400.0) * x -
             static_cast<real_t>(563.0 / 3024.0) * SQR(x) -
             static_cast<real_t>(11.0 / 3360.0) * CUBE(x) +
             static_cast<real_t>(25.0 / 504.0) * SQR(SQR(x)) -
             static_cast<real_t>(11.0 / 1200.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(1.0 / 360.0) * SQR(CUBE(x)) -
             static_cast<real_t>(11.0 / 1680.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(1.0 / 252.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(11.0 / 10080.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 6720.0) * SQR(SQR(SQR(x))) * SQR(x) -
             static_cast<real_t>(1.0 / 120960.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else if (x < THREE) {
      return static_cast<real_t>(61297.0 / 166320.0) +
             static_cast<real_t>(781.0 / 5600.0) * x -
             static_cast<real_t>(1619.0 / 3024.0) * SQR(x) +
             static_cast<real_t>(583.0 / 1120.0) * CUBE(x) -
             static_cast<real_t>(239.0 / 504.0) * SQR(SQR(x)) +
             static_cast<real_t>(143.0 / 400.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(13.0 / 72.0) * SQR(CUBE(x)) +
             static_cast<real_t>(33.0 / 560.0) * SQR(CUBE(x)) * x -
             static_cast<real_t>(25.0 / 2016.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(11.0 / 6720.0) * SQR(SQR(SQR(x))) * x -
             static_cast<real_t>(1.0 / 8064.0) * SQR(SQR(SQR(x))) * SQR(x) +
             static_cast<real_t>(1.0 / 241920.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else if (x <= FOUR) {
      return static_cast<real_t>(894727.0 / 665280.0) -
             static_cast<real_t>(38533.0 / 11200.0) * x +
             static_cast<real_t>(9385.0 / 1728.0) * SQR(x) -
             static_cast<real_t>(12199.0 / 2240.0) * CUBE(x) +
             static_cast<real_t>(1009.0 / 288.0) * SQR(SQR(x)) -
             static_cast<real_t>(1199.0 / 800.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(631.0 / 1440.0) * SQR(CUBE(x)) -
             static_cast<real_t>(99.0 / 1120.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(7.0 / 576.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(11.0 / 10080.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 17280.0) * SQR(SQR(SQR(x))) * SQR(x) -
             static_cast<real_t>(1.0 / 725760.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else if (x < FIVE) {
      return -static_cast<real_t>(18595037.0 / 3326400.0) +
             static_cast<real_t>(4726777.0 / 302400.0) * x -
             static_cast<real_t>(1113317.0 / 60480.0) * SQR(x) +
             static_cast<real_t>(250657.0 / 20160.0) * CUBE(x) -
             static_cast<real_t>(54797.0 / 10080.0) * SQR(SQR(x)) +
             static_cast<real_t>(11737.0 / 7200.0) * CUBE(x) * SQR(x) -
             static_cast<real_t>(2477.0 / 7200.0) * SQR(CUBE(x)) +
             static_cast<real_t>(517.0 / 10080.0) * SQR(CUBE(x)) * x -
             static_cast<real_t>(107.0 / 20160.0) * SQR(SQR(SQR(x))) +
             static_cast<real_t>(11.0 / 30240.0) * SQR(SQR(SQR(x))) * x -
             static_cast<real_t>(1.0 / 67200.0) * SQR(SQR(SQR(x))) * SQR(x) +
             static_cast<real_t>(1.0 / 3628800.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else if (x < SIX) {
      return static_cast<real_t>(17496.0 / 1925.0) -
             static_cast<real_t>(2916.0 / 175.0) * x +
             static_cast<real_t>(486.0 / 35.0) * SQR(x) -
             static_cast<real_t>(243.0 / 35.0) * CUBE(x) +
             static_cast<real_t>(81.0 / 35.0) * SQR(SQR(x)) -
             static_cast<real_t>(27.0 / 50.0) * CUBE(x) * SQR(x) +
             static_cast<real_t>(9.0 / 100.0) * SQR(CUBE(x)) -
             static_cast<real_t>(3.0 / 280.0) * SQR(CUBE(x)) * x +
             static_cast<real_t>(1.0 / 1120.0) * SQR(SQR(SQR(x))) -
             static_cast<real_t>(1.0 / 20160.0) * SQR(SQR(SQR(x))) * x +
             static_cast<real_t>(1.0 / 604800.0) * SQR(SQR(SQR(x))) * SQR(x) -
             static_cast<real_t>(1.0 / 39916800.0) * SQR(SQR(SQR(x))) * SQR(x) * x;
    } else {
      return ZERO;
    }
  }

  template <bool STAGGERED, unsigned short O>
  Inline void order(const int& i, const real_t& di, int& i_min, real_t S[O + 1]) {
    if constexpr (O == 1u) {
      // S(x) = 1 - |x|      |x| < 1
      //        0.0          |x| ≥ 1
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i;
        S[0]  = ONE - di;
        S[1]  = di;
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 1;
          S[0]  = HALF - di;
          S[1]  = ONE - S[0];
        } else {
          i_min = i;
          S[0]  = THREE_HALFS - di;
          S[1]  = ONE - S[0];
        }
      } // staggered
    } else if constexpr (O == 2u) {
      //        3/4 - |x|^2                   |x| < 1/2
      // S(x) = 1/2 * (3/2 - |x|)^2     1/2 ≤ |x| < 3/2
      //        0.0                           |x| ≥ 3/2
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 1;
          S[0]  = HALF * SQR(HALF - di);
          S[1]  = THREE_FOURTHS - SQR(di);
          S[2]  = ONE - S[0] - S[1];
        } else {
          i_min = i;
          S[0]  = HALF * SQR(THREE_HALFS - di);
          S[1]  = THREE_FOURTHS - SQR(ONE - di);
          S[2]  = ONE - S[0] - S[1];
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 1;
        S[0]  = HALF * SQR(ONE - di);
        S[2]  = HALF * SQR(di);
        S[1]  = ONE - S[0] - S[2];
      } // staggered
    } else if constexpr (O == 3u) {
      //        2/3 - x^2 + 1/2 * x^3      |x| < 1
      // S(x) = 1/6 * (2 - |x|)^3      1 ≤ |x| < 2
      //        0.0                        |x| ≥ 2
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 1;

#pragma unroll
        for (int n = 0; n < 4; n++) {
          S[n] = S3(Kokkos::fabs(ONE + di - static_cast<real_t>(n)));
        }
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 2;

#pragma unroll
          for (int n = 0; n < 4; n++) {
            S[n] = S3(Kokkos::fabs(
              static_cast<real_t>(1.5) + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 1;

#pragma unroll
          for (int n = 0; n < 4; n++) {
            S[n] = S3(Kokkos::fabs(HALF + di - static_cast<real_t>(n)));
          }
        }
      } // staggered
    } else if constexpr (O == 4u) {
      // clang-format off
      //        115/192 - (5/8) * |x|^2 + (1/4) * |x|^4                                                  |x| < 1/2
      // S(x) = 55/96 + (5/24) * |x| - (5/4) * |x|^2 + (5/6) * |x|^3 - (1/6) * |x|^4               1/2 ≤ |x| < 3/2
      //        625/384 - (125/48) * |x| + (25/16) * |x|^2 - (5/12) * |x|^3 + (1/24) * |x|^4       3/2 ≤ |x| < 5/2
      //        0.0                                                                                      |x| ≥ 5/2
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions

        if (di < HALF) {
          i_min = i - 2;

#pragma unroll
          for (int n = 0; n < 5; n++) {
            S[n] = S4(Kokkos::fabs(TWO + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 1;

#pragma unroll
          for (int n = 0; n < 5; n++) {
            S[n] = S4(Kokkos::fabs(ONE + di - static_cast<real_t>(n)));
          }
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 2;

#pragma unroll
        for (int n = 0; n < 5; n++) {
          S[n] = S4(Kokkos::fabs(THREE_HALFS + di - static_cast<real_t>(n)));
        }
      } // staggered
    } else if constexpr (O == 5u) {
      // clang-format off
      //  S5(x) = 
      //   11/20 - (1/2) * |x|^2 + (1/4) * |x|^4 - (1/12) * |x|^5                                        if |x| ≤ 1
      //   17/40 + (5/8) * |x| - (7/4) * |x|^2 + (5/4) * |x|^3 - (3/8) * |x|^4 + (1/24) * |x|^5      if 1 < |x| < 2
      //   81/40 - (27/8) * |x| + (9/4) * |x|^2 - (3/4) * |x|^3 + (1/8) * |x|^4 - (1/120) * |x|^5    if 2 ≤ |x| < 3
      //   0.0                                                                                           if |x| > 3
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 2;

#pragma unroll
        for (int n = 0; n < 6; n++) {
          S[n] = S5(Kokkos::fabs(TWO + di - static_cast<real_t>(n)));
        }
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 3;

#pragma unroll
          for (int n = 0; n < 6; n++) {
            S[n] = S5(Kokkos::fabs(
              static_cast<real_t>(2.5) + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 2;

#pragma unroll
          for (int n = 0; n < 6; n++) {
            S[n] = S5(Kokkos::fabs(THREE_HALFS + di - static_cast<real_t>(n)));
          }
        }
      } // staggered
    } else if constexpr (O == 6u) {
      // clang-format off
      // S6(x) = 
      //   5887/11520 - (77/192) * |x|^2 + (7/48) * |x|^4 - (1/36) * |x|^6                     if |x| ≤ 1/2
      //   7861/15360 - (7/768) * |x| - (91/256) * |x|^2 - (35/288) * |x|^3 + (21/64) * |x|^4 
      //                 - (7/48) * |x|^5 + (1/48) * |x|^6                                     if 1/2 < |x| < 3/2
      //   1379/7680 + (1267/960) * |x| - (329/128) * |x|^2 + (133/72) * |x|^3 
      //                 - (21/32) * |x|^4 + (7/60) * |x|^5 - (1/120) * |x|^6                  if 3/2 ≤ |x| < 5/2
      //   117649/46080 - (16807/3840) * |x| + (2401/768) * |x|^2 - (343/288) * |x|^3 
      //                 + (49/192) * |x|^4 - (7/240) * |x|^5 + (1/720) * |x|^6                if 5/2 ≤ |x| < 7/2
      //   0.0                                                                                 if |x| ≥ 7/2
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions

        if (di < HALF) {
          i_min = i - 3;

#pragma unroll
          for (int n = 0; n < 7; n++) {
            S[n] = S6(Kokkos::fabs(THREE + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 2;

#pragma unroll
          for (int n = 0; n < 7; n++) {
            S[n] = S6(Kokkos::fabs(TWO + di - static_cast<real_t>(n)));
          }
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 3;

#pragma unroll
        for (int n = 0; n < 7; n++) {
          S[n] = S6(
            Kokkos::fabs(static_cast<real_t>(2.5) + di - static_cast<real_t>(n)));
        }
      } // staggered
    } else if constexpr (O == 7u) {
      // clang-format off
      // S7(x) = 
      //   151/315 - (1/3) * |x|^2 + (1/9) * |x|^4 - (1/36) * |x|^6 + (1/144) * |x|^7       if |x| < 1
      //   103/210 - (7/90) * |x| - (1/10) * |x|^2 - (7/18) * |x|^3 + (1/2) * |x|^4 
      //             - (7/30) * |x|^5 + (1/20) * |x|^6 - (1/240) * |x|^7                   if 1 ≤ |x| ≤ 2
      //   (217/90) * |x| - (23/6) * |x|^2 + (49/18) * |x|^3 - (19/18) * |x|^4 
      //             + (7/30) * |x|^5 - (1/36) * |x|^6 + (1/720) * |x|^7 - (139/630)       if 2 < |x| < 3
      //   1024/315 - (256/45) * |x| + (64/15) * |x|^2 - (16/9) * |x|^3 + (4/9) * |x|^4 
      //             - (1/15) * |x|^5 + (1/180) * |x|^6 - (1/5040) * |x|^7                 if 3 ≤ |x| < 4
      //   0.0                                                                             if |x| ≥ 4
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 3;

#pragma unroll
        for (int n = 0; n < 8; n++) {
          S[n] = S7(Kokkos::fabs(THREE + di - static_cast<real_t>(n)));
        }
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 4;

          for (int n = 0; n < 8; n++) {
            S[n] = S7(Kokkos::fabs(
              static_cast<real_t>(3.5) + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 3;

#pragma unroll
          for (int n = 0; n < 8; n++) {
            S[n] = S7(Kokkos::fabs(
              static_cast<real_t>(2.5) + di - static_cast<real_t>(n)));
          }
        }
      } // staggered
    } else if constexpr (O == 8u) {
      // clang-format off
      // S8(x) = 
      //   259723/573440 - (289/1024) * |x|^2 + (43/512) * |x|^4 - (1/64) * |x|^6 + (1/576) * |x|^8                     if |x| < 1/2
      //   64929/143360 + (1/5120) * |x| - (363/1280) * |x|^2 + (7/1280) * |x|^3 + (9/128) * |x|^4 
      //                  + (7/320) * |x|^5 - (3/80) * |x|^6 + (1/80) * |x|^7 - (1/720) * |x|^8                        if 1/2 ≤ |x| ≤ 3/2
      //   145167/286720 - (1457/5120) * |x| + (195/512) * |x|^2 - (1127/1280) * |x|^3 + (207/256) * |x|^4 
      //                  - (119/320) * |x|^5 + (3/32) * |x|^6 - (1/80) * |x|^7 + (1/1440) * |x|^8                     if 3/2 < |x| < 2.5
      //   (146051/35840) * |x| - (1465/256) * |x|^2 + (5123/1280) * |x|^3 - (209/128) * |x|^4 
      //                  + (131/320) * |x|^5 - (1/16) * |x|^6 + (3/560) * |x|^7 - (1/5040) * |x|^8 - (122729/143360)  if 2.5 ≤ |x| < 3.5
      //   4782969/1146880 - (531441/71680) * |x| + (59049/10240) * |x|^2 - (6561/2560) * |x|^3 + (729/1024) * |x|^4 
      //                  - (81/640) * |x|^5 + (9/640) * |x|^6 - (1/1120) * |x|^7 + (1/40320) * |x|^8                  if 3.5 ≤ |x| < 4.5
      //   0.0
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 4;

#pragma unroll
          for (int n = 0; n < 9; n++) {
            S[n] = S8(Kokkos::fabs(FOUR + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 3;

#pragma unroll
          for (int n = 0; n < 9; n++) {
            S[n] = S8(Kokkos::fabs(THREE + di - static_cast<real_t>(n)));
          }
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 4;

#pragma unroll
        for (int n = 0; n < 9; n++) {
          S[n] = S8(
            Kokkos::fabs(static_cast<real_t>(3.5) + di - static_cast<real_t>(n)));
        }
      } // staggered
    } else if constexpr (O == 9u) {
      // clang-format off
      // S9(x) = 
      //   15619/36288 - (35/144) * |x|^2 + (19/288) * |x|^4 - (5/432) * |x|^6 + (1/576) * |x|^8 - (1/2880) * |x|^9       if |x| ≤ 1
      //   7799/18144 + (1/192) * |x| - (19/72) * |x|^2 + (7/144) * |x|^3 - (1/144) * |x|^4 + (7/96) * |x|^5 
      //                - (13/216) * |x|^6 + (1/48) * |x|^7 - (1/288) * |x|^8 + (1/4320) * |x|^9                         if 1 < |x| < 2
      //   1553/2592 - (339/448) * |x| + (635/504) * |x|^2 - (83/48) * |x|^3 + (191/144) * |x|^4 - (19/32) * |x|^5 
      //                + (35/216) * |x|^6 - (3/112) * |x|^7 + (5/2016) * |x|^8 - (1/10080) * |x|^9                      if 2 ≤ |x| < 3
      //   (5883/896) * |x| - (2449/288) * |x|^2 + (563/96) * |x|^3 - (1423/576) * |x|^4 + (43/64) * |x|^5 
      //                - (103/864) * |x|^6 + (3/224) * |x|^7 - (1/1152) * |x|^8 + (1/40320) * |x|^9 - (133663/72576)    if 3 ≤ |x| < 4
      //   390625/72576 - (78125/8064) * |x| + (15625/2016) * |x|^2 - (3125/864) * |x|^3 + (625/576) * |x|^4 
      //                - (125/576) * |x|^5 + (25/864) * |x|^6 - (5/2016) * |x|^7 + (1/8064) * |x|^8 - (1/362880) * |x|^9 if 4 ≤ |x| < 5
      //   0.0                                                                                                           if |x| ≥ 5
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 4;

#pragma unroll
        for (int n = 0; n < 10; n++) {
          S[n] = S9(Kokkos::fabs(FOUR + di - static_cast<real_t>(n)));
        }
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 5;

          for (int n = 0; n < 10; n++) {
            S[n] = S9(Kokkos::fabs(
              static_cast<real_t>(4.5) + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 4;

#pragma unroll
          for (int n = 0; n < 10; n++) {
            S[n] = S9(Kokkos::fabs(
              static_cast<real_t>(3.5) + di - static_cast<real_t>(n)));
          }
        }
      } // staggered
    } else if constexpr (O == 10u) {
      // clang-format off
      // S10(x) =
      //   381773117/928972800 - (156409/737280) * |x|^2 + (14597/276480) * |x|^4 - (583/69120) * |x|^6 + (11/11520) * |x|^8 - (1/14400) * |x|^10                     if |x| ≤ 0.5
      //   152709293/371589120 - (11/4423680) * |x| - (62557/294912) * |x|^2 - (11/92160) * |x|^3 + (5885/110592) * |x|^4 - (77/76800) * |x|^5 - 
      //              (187/27648) * |x|^6 - (11/5760) * |x|^7 + (11/4608) * |x|^8 - (11/17280) * |x|^9 + (1/17280) * |x|^10                                           if 0.5 < |x| ≤ 1.5
      //   37690169/92897280 + (135311/3870720) * |x| - (163603/516096) * |x|^2 + (7513/40320) * |x|^3  - (4543/27648) * |x|^4  
      //              + (1661/9600) * |x|^5 - (715/6912) * |x|^6 + (11/315) * |x|^7 - (55/8064) * |x|^8 + (11/15120) * |x|^9 - (1/30240) * |x|^10                     if 1.5 < |x| ≤ 2.5
      //   623786977/743178240 - (11695211/6881280) * |x| + (1654543/589824) * |x|^2 - (1352153/430080) * |x|^3 + (479281/221184) * |x|^4 
      //              - (48433/51200) * |x|^5 + (14905/55296) * |x|^6 - (451/8960) * |x|^7 + (55/9216) * |x|^8 - (11/26880) * |x|^9 + (1/80640) * |x|^10              if 2.5 < |x| ≤ 3.5
      //   -1241720381/371589120 + (237959711/23224320) * |x| - (3702215/294912) * |x|^2 + (2070343/241920) * |x|^3 - (407429/110592) * |x|^4 
      //              + (61061/57600) * |x|^5 - (5753/27648) * |x|^6 + (209/7560) * |x|^7 - (11/4608) * |x|^8 + (11/90720) * |x|^9 - (1/362880) * |x|^10              if 3.5 < |x| ≤ 4.5
      //   25937424601/3715891200 - (2357947691/185794560) * |x| + (214358881/20643840) * |x|^2 - (19487171/3870720) * |x|^3 + (1771561/1105920) * |x|^4 
      //              - (161051/460800) * |x|^5 + (14641/276480) * |x|^6 - (1331/241920) * |x|^7 + (121/322560) * |x|^8 - (11/725760) * |x|^9 + (1/3628800) * |x|^10  if 4.5 < |x| ≤ 5.5
      //   0.0                                                                                                                                                        otherwise
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 5;

#pragma unroll
          for (int n = 0; n < 11; n++) {
            S[n] = S10(Kokkos::fabs(FIVE + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 4;

#pragma unroll
          for (int n = 0; n < 11; n++) {
            S[n] = S10(Kokkos::fabs(FOUR + di - static_cast<real_t>(n)));
          }
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 5;

#pragma unroll
        for (int n = 0; n < 11; n++) {
          S[n] = S10(
            Kokkos::fabs(static_cast<real_t>(4.5) + di - static_cast<real_t>(n)));
        }
      } // staggered
    } else if constexpr (O == 11u) {
      // clang-format off
      // S11(x) =
      //   655177/1663200 - (809/4320) * |x|^2 + (31/720) * |x|^4 - (23/3600) * |x|^6 
      //                  + (1/1440) * |x|^8 - (1/14400) * |x|^10 + (1/86400) * |x|^11                        if |x| < 1
      //   65521/166320 - (11/50400) * |x| - (563/3024) * |x|^2 - (11/3360) * |x|^3 
      //                  + (25/504) * |x|^4 - (11/1200) * |x|^5 + (1/360) * |x|^6 
      //                  - (11/1680) * |x|^7 + (1/252) * |x|^8 - (11/10080) * |x|^9 
      //                  + (1/6720) * |x|^10 - (1/120960) * |x|^11                                          if 1 ≤ |x| ≤ 2
      //   61297/166320 + (781/5600) * |x| - (1619/3024) * |x|^2 + (583/1120) * |x|^3 
      //                  - (239/504) * |x|^4 + (143/400) * |x|^5 - (13/72) * |x|^6 
      //                  + (33/560) * |x|^7 - (25/2016) * |x|^8 + (11/6720) * |x|^9 
      //                  - (1/8064) * |x|^10 + (1/241920) * |x|^11                                         if 2 < |x| < 3
      //   894727/665280 - (38533/11200) * |x| + (9385/1728) * |x|^2 - (12199/2240) * |x|^3 
      //                  + (1009/288) * |x|^4 - (1199/800) * |x|^5 + (631/1440) * |x|^6 
      //                  - (99/1120) * |x|^7 + (7/576) * |x|^8 - (11/10080) * |x|^9 
      //                  + (1/17280) * |x|^10 - (1/725760) * |x|^11                                        if 3 ≤ |x| ≤ 4
      //  -18595037/3326400 + (4726777/302400) * |x| - (1113317/60480) * |x|^2 + (250657/20160) * |x|^3 
      //                  - (54797/10080) * |x|^4 + (11737/7200) * |x|^5 - (2477/7200) * |x|^6 
      //                  + (517/10080) * |x|^7 - (107/20160) * |x|^8 + (11/30240) * |x|^9 
      //                  - (1/67200) * |x|^10 + (1/3628800) * |x|^11                                      if 4 < |x| < 5
      //   17496/1925 - (2916/175) * |x| + (486/35) * |x|^2 - (243/35) * |x|^3 
      //                  + (81/35) * |x|^4 - (27/50) * |x|^5 + (9/100) * |x|^6 
      //                  - (3/280) * |x|^7 + (1/1120) * |x|^8 - (1/20160) * |x|^9 
      //                  + (1/604800) * |x|^10 - (1/39916800) * |x|^11                                     if 5 ≤ |x| < 6
      //   0.0                                                                                              otherwise
      // clang-format on
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 5;

#pragma unroll
        for (int n = 0; n < 12; n++) {
          S[n] = S11(Kokkos::fabs(FIVE + di - static_cast<real_t>(n)));
        }
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 6;

          for (int n = 0; n < 12; n++) {
            S[n] = S11(Kokkos::fabs(
              static_cast<real_t>(5.5) + di - static_cast<real_t>(n)));
          }
        } else {
          i_min = i - 5;

#pragma unroll
          for (int n = 0; n < 12; n++) {
            S[n] = S11(Kokkos::fabs(
              static_cast<real_t>(4.5) + di - static_cast<real_t>(n)));
          }
        }
      } // staggered
    } else {
      raise::KernelError(
        HERE,
        "Unsupported interpolation order. O > 11 not supported. Seriously. "
        "What are you even doing here? Entity already goes to 11!");
    }
  }

  template <int O>
  Inline void for_deposit(const int&    i_init,
                          const real_t& di_init,
                          const int&    i_fin,
                          const real_t& di_fin,
                          int&          i_min,
                          int&          i_max,
                          real_t        iS[O + 2],
                          real_t        fS[O + 2]) {

    /*
    The N-th order shape function per particle is a N+2 element array
    where the shape function contributes to only N+1 elements.
    We need to find which indices are contributing to the shape function

    Let * be the particle position at the current timestep
    Let x be the particle position at the previous timestep


              0      1    (...)    N     N+1
          ___________________________________
          |  x*  |  x*  |  ... |  x*  |      |   // i_init_min = i_fin_min
          |______|______|______|______|______|
          |  x   |  x*  |  ... |  x*  |  *   |   // i_init_min < i_fin_min
          |______|______|______|______|______|
          |  *   |  x*  |  ... |  x*  |  x   |   // i_init_min > i_fin_min
          |______|______|______|______|______|
    */

    int i_init_min, i_fin_min;

    // temporary shape function arrays
    real_t iS_[O + 1], fS_[O + 1];

    // fill shape function arrays
    order<false, O>(i_init, di_init, i_init_min, iS_);
    order<false, O>(i_fin, di_fin, i_fin_min, fS_);

    // determine i_min and i_max and fill final shape function arrays
    if (i_init_min < i_fin_min) {
      // case two from above graphics
      i_min = i_init_min;
      i_max = i_min + O + 1;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

      fS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j + 1] = fS_[j];
      }

    } else if (i_init_min > i_fin_min) {
      // case three from above graphics
      i_min = i_fin_min;
      i_max = i_min + O + 1;

      iS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j + 1] = iS_[j];
      }

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;

    } else {
      // case one from above graphics
      i_min = i_init_min;
      i_max = i_min + O;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;
    }
  }
} // namespace prtl_shape

#endif // KERNELS_PARTICLE_SHAPES_HPP
