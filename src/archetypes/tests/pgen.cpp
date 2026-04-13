#include "enums.h"

#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/traits.h"

#include <Kokkos_Core.hpp>

#include <iostream>

using namespace ntt;

template <Dimension D>
struct CustomFieldsetter {
  Inline auto ex1(const coord_t<D>&) const -> real_t {
    return ZERO;
  }
};

template <Dimension D>
struct ExtForce {
  Inline auto fx1(const coord_t<D>&) const -> real_t {
    return ZERO;
  }

  Inline auto ex1(const coord_t<D>&) const -> real_t {
    return ZERO;
  }

  Inline auto bx3(const coord_t<D>&) const -> real_t {
    return ZERO;
  }
};

template <Dimension D>
struct ExtCurrent {
  Inline auto jx1(const coord_t<D>&) const -> real_t {
    return ZERO;
  }
};

template <SimEngine::type S, class M>
struct CustomPgen : public arch::ProblemGenerator<S, M> {
  CustomPgen(const SimulationParams& params = {})
    : arch::ProblemGenerator<S, M> { params } {}

  CustomFieldsetter<M::Dim> init_flds {};
  ExtCurrent<M::Dim>        ext_current {};

  void InitPrtls(Domain<S, M>&) {}

  auto AtmFields(simtime_t) const -> CustomFieldsetter<M::Dim> {
    return init_flds;
  }

  auto MatchFields(simtime_t) const -> CustomFieldsetter<M::Dim> {
    return init_flds;
  }

  auto MatchFieldsInX1(simtime_t) const -> CustomFieldsetter<M::Dim> {
    return init_flds;
  }

  auto MatchFieldsInX2(simtime_t) const -> CustomFieldsetter<M::Dim> {
    return init_flds;
  }

  auto MatchFieldsInX3(simtime_t) const -> CustomFieldsetter<M::Dim> {
    return init_flds;
  }

  auto ExternalFields(simtime_t, spidx_t, const Domain<S, M>&) const
    -> std::pair<bool, ExtForce<M::Dim>> {
    return { true, ExtForce<M::Dim> {} };
  }

  auto FixFieldsConst(simtime_t, const bc_in&, ntt::em) const
    -> std::pair<real_t, bool> {
    return { ZERO, false };
  }

  void CustomPostStep(timestep_t, simtime_t, Domain<S, M>&) {}

  void CustomFieldOutput(const std::string&,
                         ndfield_t<M::Dim, 6>&,
                         index_t,
                         timestep_t,
                         simtime_t,
                         const Domain<S, M>&) {}

  auto CustomStat(const std::string&, timestep_t, simtime_t, const Domain<S, M>&)
    -> real_t {
    return ZERO;
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    auto custom_pgen = CustomPgen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>> {};

    if constexpr (not arch::traits::pgen::HasInitFlds<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have init_flds");
    }
    if constexpr (not arch::traits::pgen::HasInitPrtls<
                    decltype(custom_pgen),
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      throw std::runtime_error("CustomPgen should have InitPrtls");
    }
    if constexpr (not arch::traits::pgen::HasExternalFields<
                    decltype(custom_pgen),
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      throw std::runtime_error("CustomPgen should have ext_fields");
    }
    if constexpr (not arch::traits::pgen::HasExtCurrent<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have ext_current");
    }
    if constexpr (not arch::traits::pgen::HasAtmFields<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have AtmFields");
    }
    if constexpr (not arch::traits::pgen::HasMatchFields<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have MatchFields");
    }
    if constexpr (
      not arch::traits::pgen::HasMatchFieldsInX1<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have MatchFieldsInX1");
    }
    if constexpr (
      not arch::traits::pgen::HasMatchFieldsInX2<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have MatchFieldsInX2");
    }
    if constexpr (
      not arch::traits::pgen::HasMatchFieldsInX3<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have MatchFieldsInX3");
    }
    if constexpr (not arch::traits::pgen::HasFixFieldsConst<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have FixFieldsConst");
    }
    if constexpr (not arch::traits::pgen::HasCustomPostStep<
                    decltype(custom_pgen),
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      throw std::runtime_error("CustomPgen should have CustomPostStep");
    }
    if constexpr (not arch::traits::pgen::HasCustomFieldOutput<
                    decltype(custom_pgen),
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      throw std::runtime_error("CustomPgen should have CustomFieldOutput");
    }
    if constexpr (not arch::traits::pgen::HasCustomStatOutput<
                    decltype(custom_pgen),
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      throw std::runtime_error("CustomPgen should have CustomStat");
    }
    auto domain = Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>> {
      false, 0u, { 0u }, { 0u }, { 10u }, { { (real_t)0.0, (real_t)1.0 } },
      {},    {}
    };
    auto [apply_extfields,
          ext_fields] = custom_pgen.ExternalFields(ZERO, 0, domain);
    if constexpr (
      not kernel::traits::pusher::external::HasFx1<decltype(ext_fields), Dim::_1D>) {
      throw std::runtime_error("CustomPgen's ext_fields should have fx1");
    }
    if constexpr (
      not kernel::traits::pusher::external::HasEx1<decltype(ext_fields), Dim::_1D>) {
      throw std::runtime_error("CustomPgen's ext_fields should have ex1");
    }
    if constexpr (
      kernel::traits::pusher::external::HasBx1<decltype(ext_fields), Dim::_1D>) {
      throw std::runtime_error("CustomPgen's ext_fields should not have bx1");
    }
    if constexpr (
      not kernel::traits::pusher::external::HasBx3<decltype(ext_fields), Dim::_1D>) {
      throw std::runtime_error("CustomPgen's ext_current should have bx3");
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
