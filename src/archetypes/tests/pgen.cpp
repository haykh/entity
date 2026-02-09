#include "enums.h"

#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

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
  Inline auto fx1(spidx_t, simtime_t, const coord_t<D>&) const -> real_t {
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
  ExtForce<M::Dim>          ext_force {};
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

  auto FixFieldsConst(const bc_in&, const ntt::em&) const
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
    if constexpr (not arch::traits::pgen::HasExtForce<decltype(custom_pgen)>) {
      throw std::runtime_error("CustomPgen should have ext_force");
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

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
