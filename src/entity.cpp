#include "enums.h"

#include "arch/traits.h"
#include "utils/error.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/simulation.h"

#include "engines/grpic.hpp"
#include "engines/srpic.hpp"
#include "pgen.hpp"

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
static constexpr bool should_compile {
  traits::check_compatibility<S>::value(user::PGen<S, M<D>>::engines) &&
  traits::check_compatibility<M<D>::MetricType>::value(user::PGen<S, M<D>>::metrics) &&
  traits::check_compatibility<D>::value(user::PGen<S, M<D>>::dimensions)
};

template <template <class> class E, template <Dimension> class M, Dimension D>
void shouldCompile(ntt::Simulation& sim) {
  if constexpr (should_compile<E<M<D>>::S, M, D>) {
    sim.run<E, M, D>();
  }
}

auto main(int argc, char* argv[]) -> int {
  ntt::Simulation sim { argc, argv };
  const auto      is_srpic = sim.requested_engine() == ntt::SimEngine::SRPIC;
  const auto      is_grpic = sim.requested_engine() == ntt::SimEngine::GRPIC;
  const auto is_minkowski  = sim.requested_metric() == ntt::Metric::Minkowski;
  const auto is_spherical  = sim.requested_metric() == ntt::Metric::Spherical;
  const auto is_qspherical = sim.requested_metric() == ntt::Metric::QSpherical;
  const auto is_kerr_schild = sim.requested_metric() == ntt::Metric::Kerr_Schild;
  const auto is_qkerr_schild = sim.requested_metric() == ntt::Metric::QKerr_Schild;
  const auto is_kerr_schild_0 = sim.requested_metric() ==
                                ntt::Metric::Kerr_Schild_0;
  const auto is_1d = sim.requested_dimension() == Dim::_1D;
  const auto is_2d = sim.requested_dimension() == Dim::_2D;
  const auto is_3d = sim.requested_dimension() == Dim::_3D;

  // sanity checks
  if (not is_srpic and not is_grpic) {
    raise::Fatal("Invalid engine", HERE);
  }

  if (not is_minkowski and not is_spherical and not is_qspherical and
      not is_kerr_schild and not is_qkerr_schild and not is_kerr_schild_0) {
    raise::Fatal("Invalid metric", HERE);
  }

  if (not is_1d and not is_2d and not is_3d) {
    raise::Fatal("Invalid dimension", HERE);
  }

  if (is_srpic and not(is_minkowski or is_spherical or is_qspherical)) {
    raise::Fatal("Invalid metric for SRPIC", HERE);
  }

  if (is_grpic and not(is_kerr_schild or is_qkerr_schild or is_kerr_schild_0)) {
    raise::Fatal("Invalid metric for GRPIC", HERE);
  }

  if ((is_spherical or is_qspherical or is_kerr_schild or is_qkerr_schild or
       is_kerr_schild_0) and
      (is_1d or is_3d)) {
    raise::Fatal("Invalid dimension for metric", HERE);
  }

  if (is_srpic and is_minkowski and is_1d) {
    shouldCompile<ntt::SRPICEngine, metric::Minkowski, Dim::_1D>(sim);
    return 0;
  }

  if (is_srpic and is_minkowski and is_2d) {
    shouldCompile<ntt::SRPICEngine, metric::Minkowski, Dim::_2D>(sim);
    return 0;
  }

  if (is_srpic and is_minkowski and is_3d) {
    shouldCompile<ntt::SRPICEngine, metric::Minkowski, Dim::_3D>(sim);
    return 0;
  }

  if (is_srpic and is_spherical and is_2d) {
    shouldCompile<ntt::SRPICEngine, metric::Spherical, Dim::_2D>(sim);
    return 0;
  }

  if (is_srpic and is_qspherical and is_2d) {
    shouldCompile<ntt::SRPICEngine, metric::QSpherical, Dim::_2D>(sim);
    return 0;
  }

  if (is_grpic and is_kerr_schild and is_2d) {
    shouldCompile<ntt::GRPICEngine, metric::KerrSchild, Dim::_2D>(sim);
    return 0;
  }

  if (is_grpic and is_qkerr_schild and is_2d) {
    shouldCompile<ntt::GRPICEngine, metric::QKerrSchild, Dim::_2D>(sim);
    return 0;
  }

  if (is_grpic and is_kerr_schild_0 and is_2d) {
    shouldCompile<ntt::GRPICEngine, metric::KerrSchild0, Dim::_2D>(sim);
    return 0;
  }

  return 0;
}
