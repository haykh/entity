#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "metrics/minkowski.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include <iostream>

struct InitFields {};

template <ntt::SimEngine::type S, class M>
struct PGen : public arch::ProblemGenerator<S, M> {
  // compatibility traits for the problem generator
  static constexpr auto engines {
    traits::compatible_with<ntt::SimEngine::SRPIC>::value
  };
  static constexpr auto metrics {
    traits::compatible_with<ntt::Metric::Minkowski>::value
  };
  static constexpr auto dimensions { traits::compatible_with<Dim::_1D>::value };

  using arch::ProblemGenerator<S, M>::D;
  using arch::ProblemGenerator<S, M>::C;
  using arch::ProblemGenerator<S, M>::params;

  inline PGen(const ntt::SimulationParams& p, const ntt::Metadomain<S, M>&)
    : arch::ProblemGenerator<S, M> { p }
    , init_flds {} {}

  InitFields init_flds;

  inline void InitPrtls(ntt::Domain<S, M>&) {}

  inline auto AtmFields(simtime_t) const -> InitFields {
    return init_flds;
  }

  inline auto MatchFields(simtime_t) const -> InitFields {
    return init_flds;
  }

  inline auto MatchFieldsInX1(simtime_t) const -> InitFields {
    return init_flds;
  }

  inline auto MatchFieldsInX2(simtime_t) const -> InitFields {
    return init_flds;
  }

  inline auto MatchFieldsInX3(simtime_t) const -> InitFields {
    return init_flds;
  }

  inline auto FixFieldsConst(const bc_in&, const ntt::em&) const
    -> std::pair<real_t, bool> {
    return { ZERO, false };
  }

  inline void CustomPostStep(timestep_t, simtime_t, ntt::Domain<S, M>&) {}

  inline void CustomFieldOutput(const std::string&,
                                ndfield_t<D, 6>&,
                                index_t,
                                timestep_t,
                                simtime_t,
                                const Domain<S, M>&) {}

  inline auto CustomStat(const std::string&,
                         timestep_t,
                         simtime_t,
                         const Domain<S, M>&) -> real_t {
    return ZERO;
  }
};

auto main(int argc, char** argv) -> int {
  Kokkos::initialize(argc, argv);
  try {
    using namespace ntt;
    if constexpr (
      not traits::pgen::HasInitFlds<PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing InitFlds method", HERE);
    }

    if constexpr (not traits::pgen::HasInitPrtls<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>,
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing InitPrtls method", HERE);
    }

    if constexpr (
      not traits::pgen::HasAtmFields<PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing AtmFields method", HERE);
    }

    if constexpr (not traits::pgen::HasMatchFields<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing MatchFields method", HERE);
    }

    if constexpr (not traits::pgen::HasMatchFieldsInX1<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing MatchFieldsInX1 method", HERE);
    }

    if constexpr (not traits::pgen::HasMatchFieldsInX2<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing MatchFieldsInX2 method", HERE);
    }

    if constexpr (not traits::pgen::HasMatchFieldsInX3<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing MatchFieldsInX3 method", HERE);
    }

    if constexpr (not traits::pgen::HasFixFieldsConst<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing FixFieldsConst method", HERE);
    }

    if constexpr (not traits::pgen::HasCustomPostStep<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>,
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing CustomPostStep method", HERE);
    }

    if constexpr (not traits::pgen::HasCustomFieldOutput<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>,
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing CustomFieldOutput method", HERE);
    }

    if constexpr (not traits::pgen::HasCustomStatOutput<
                    PGen<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>,
                    Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>>) {
      raise::Error("PGen is missing CustomStatOutput method", HERE);
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
