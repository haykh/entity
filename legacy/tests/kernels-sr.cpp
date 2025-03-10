#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

#include "wrapper.h"

#include METRIC_HEADER
#include PGEN_HEADER

#include "kernels/particle_pusher_sr.hpp"

#include "particle_macros.h"

template <typename T>
void put_value(ntt::array_t<T*>& arr, T value, int i) {
  auto arr_h = Kokkos::create_mirror_view(arr);
  arr_h(i)   = value;
  Kokkos::deep_copy(arr, arr_h);
}

template <typename T>
auto get_value(const ntt::array_t<T*>& arr, int i) -> T {
  auto arr_h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(arr_h, arr);
  return arr_h(i);
}

template <class M>
auto get_cartesian_coord(const int                      p,
                         const ntt::array_t<int*>&      i1,
                         const ntt::array_t<int*>&      i2,
                         const ntt::array_t<prtldx_t*>& dx1,
                         const ntt::array_t<prtldx_t*>& dx2,
                         const ntt::array_t<real_t*>&   phi,
                         const M& metric) -> std::pair<real_t, real_t> {
  std::pair<real_t, real_t> xy;
#ifdef MINKOWSKI_METRIC
  ntt::coord_t<ntt::Dim2> xC { ZERO };
  ntt::coord_t<ntt::Dim2> xyz { ZERO };
  xC[0] = i_di_to_Xi(get_value(i1, p), get_value(dx1, p));
  xC[1] = i_di_to_Xi(get_value(i2, p), get_value(dx2, p));
  (void)phi;
  metric.x_Code2Cart(xC, xyz);
  xy.first  = xyz[0];
  xy.second = xyz[1];
#else
  ntt::coord_t<ntt::Dim3> xC { ZERO };
  ntt::coord_t<ntt::Dim3> xyz { ZERO };
  xC[0] = i_di_to_Xi(get_value(i1, p), get_value(dx1, p));
  xC[1] = i_di_to_Xi(get_value(i2, p), get_value(dx2, p));
  xC[2] = get_value(phi, p);
  metric.x_Code2Cart(xC, xyz);
  xy.first  = xyz[0];
  xy.second = xyz[2];
#endif
  return xy;
}

auto dummy_metric(const unsigned int nx1, const unsigned int nx2)
  -> ntt::Metric<ntt::Dim2> {
  const auto resolution = std::vector<unsigned int>({ nx1, nx2 });
#ifdef MINKOWSKI_METRIC
  const auto extent = std::vector<real_t>({ 1.0, 100.0, -49.5, 49.5 });
#else
  const auto extent = std::vector<real_t>({ 1.0, 100.0, ZERO, ntt::constant::PI });
#endif
  // optional for Qspherical
  const auto qsph_r0 = (real_t)(0.0);
  const auto qsph_h  = (real_t)(0.25);

  auto params = new real_t[6];
  params[0]   = qsph_r0;
  params[1]   = qsph_h;
  ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
  delete[] params;
  return metric;
}

auto dummy_pgen() -> ntt::ProblemGenerator<ntt::Dim2, ntt::PICEngine> {
  return ntt::ProblemGenerator<ntt::Dim2, ntt::PICEngine>();
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    constexpr auto nx1 = 10, nx2 = 10;
    auto           metric = dummy_metric(nx1, nx2);
    auto           pgen   = dummy_pgen();
    {
      /* -------------------------------------------------------------------------- */
      /*                                   pusher */
      /* -------------------------------------------------------------------------- */
      ntt::ndfield_t<ntt::Dim2, 6> EB { "EB", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
      ntt::array_t<int*>      i1 { "i1", 10 };
      ntt::array_t<int*>      i2 { "i2", 10 };
      ntt::array_t<int*>      i3 { "i3", 10 };
      ntt::array_t<int*>      i1_prev { "i1_prev", 10 };
      ntt::array_t<int*>      i2_prev { "i2_prev", 10 };
      ntt::array_t<int*>      i3_prev { "i3_prev", 10 };
      ntt::array_t<prtldx_t*> dx1 { "dx1", 10 };
      ntt::array_t<prtldx_t*> dx2 { "dx2", 10 };
      ntt::array_t<prtldx_t*> dx3 { "dx3", 10 };
      ntt::array_t<prtldx_t*> dx1_prev { "dx1_prev", 10 };
      ntt::array_t<prtldx_t*> dx2_prev { "dx2_prev", 10 };
      ntt::array_t<prtldx_t*> dx3_prev { "dx3_prev", 10 };
      ntt::array_t<real_t*>   ux1 { "ux1", 10 };
      ntt::array_t<real_t*>   ux2 { "ux2", 10 };
      ntt::array_t<real_t*>   ux3 { "ux3", 10 };
      ntt::array_t<real_t*>   phi { "phi", 10 };
      ntt::array_t<real_t*>   weight { "weight", 10 };
      ntt::array_t<short*>    tag { "tag", 10 };

      int          i1_0, i2_0;
      real_t       dx1_0, dx2_0;
      const real_t ux1_0 = 1.5;

      const real_t x0 = 50.5, y0 = 1.0;

#ifdef MINKOWSKI_METRIC
      const real_t            ux2_0 = -1.2, ux3_0 = 0.5;
      ntt::coord_t<ntt::Dim2> xyz0 { x0, y0 };
      ntt::coord_t<ntt::Dim2> xC_0 { ZERO };
#else
      const real_t            ux3_0 = -1.2, ux2_0 = 0.5;
      ntt::coord_t<ntt::Dim3> xyz0 { x0, ZERO, y0 };
      ntt::coord_t<ntt::Dim3> xC_0 { ZERO };
#endif

      const real_t gamma = math::sqrt(ONE + SQR(ux1_0) + SQR(ux2_0) + SQR(ux3_0));
      metric.x_Cart2Code(xyz0, xC_0);
      from_Xi_to_i_di(xC_0[0], i1_0, dx1_0);
      from_Xi_to_i_di(xC_0[1], i2_0, dx2_0);

      put_value<int>(i1, i1_0, 0);
      put_value<int>(i2, i2_0, 0);
      put_value<prtldx_t>(dx1, dx1_0, 0);
      put_value<prtldx_t>(dx2, dx2_0, 0);
      put_value<prtldx_t>(ux1, ux1_0, 0);
      put_value<prtldx_t>(ux2, ux2_0, 0);
      put_value<prtldx_t>(ux3, ux3_0, 0);
      put_value<short>(tag, ntt::ParticleTag::alive, 0);

      std::vector<std::vector<ntt::BoundaryCondition>> boundaries;
      boundaries.push_back(
        std::vector<ntt::BoundaryCondition>(2, ntt::BoundaryCondition::PERIODIC));
      boundaries.push_back(
        std::vector<ntt::BoundaryCondition>(2, ntt::BoundaryCondition::PERIODIC));

      auto kernel = ntt::Pusher_kernel<ntt::Dim2,
                                       ntt::Metric<ntt::Dim2>,
                                       ntt::ProblemGenerator<ntt::Dim2, ntt::PICEngine>,
                                       ntt::Boris_t,
                                       false>(EB,
                                              i1,
                                              i2,
                                              i3,
                                              i1_prev,
                                              i2_prev,
                                              i3_prev,
                                              dx1,
                                              dx2,
                                              dx3,
                                              dx1_prev,
                                              dx2_prev,
                                              dx3_prev,
                                              ux1,
                                              ux2,
                                              ux3,
                                              phi,
                                              tag,
                                              metric,
                                              pgen,
                                              ZERO,
                                              ONE,
                                              ONE,
                                              nx1,
                                              nx2,
                                              1,
                                              boundaries,
                                              ZERO,
                                              ZERO,
                                              ZERO);
      Kokkos::parallel_for("ParticlesPush",
                           Kokkos::RangePolicy<AccelExeSpace, ntt::Boris_t>(0, 1),
                           kernel);
      auto [xa, ya] = get_cartesian_coord(0, i1, i2, dx1, dx2, phi, metric);

      if (!ntt::AlmostEqual(xa,
                            static_cast<real_t>(x0 + ux1_0 / gamma),
                            static_cast<real_t>(1e-4))) {
        throw std::runtime_error("x coordinate is not correct");
      }

      if (!ntt::AlmostEqual(ya,
                            static_cast<real_t>(y0 - 1.2 / gamma),
                            static_cast<real_t>(1e-4))) {
        throw std::runtime_error("y/z coordinate is not correct");
      }

      if (!ntt::AlmostEqual(get_value(ux1, 0), ux1_0)) {
        throw std::runtime_error("ux1 is not correct");
      }

      if (!ntt::AlmostEqual(get_value(ux2, 0), ux2_0)) {
        throw std::runtime_error("ux2 is not correct");
      }

      if (!ntt::AlmostEqual(get_value(ux3, 0), ux3_0)) {
        throw std::runtime_error("ux3 is not correct");
      }
    }
  }

  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }

  ntt::GlobalFinalize();

  return 0;
}
