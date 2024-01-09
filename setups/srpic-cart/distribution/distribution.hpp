#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WeibelInit : public EnergyDistribution<D, S> {
    WeibelInit(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      drift_p { params.get<real_t>("problem", "drift_p", 0.0) },
      drift_b { params.get<real_t>("problem", "drift_b", 0.0) },
      temp_1 { params.get<real_t>("problem", "temperature_1", 0.0) },
      temp_2 { params.get<real_t>("problem", "temperature_2", 0.0) },
      temp_3 { params.get<real_t>("problem", "temperature_3", 0.0) },
      temp_4 { params.get<real_t>("problem", "temperature_4", 0.0) } {}

    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
        maxwellian(v, temp_1, drift_p, -dir::z);
      } else if (species == 2) {
        maxwellian(v, temp_2, drift_p, -dir::z);
      } else if (species == 3) {
        maxwellian(v, temp_3, drift_b, dir::z);
      } else if (species == 4) {
        maxwellian(v, temp_4, drift_b, dir::z);
      }
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           drift_p, drift_b, temp_1, temp_2, temp_3, temp_4;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}

    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override;
  };

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitParticles(
    const SimulationParams& params,
    Meshblock<D, S>&        mblock) {
    InjectUniform<D, S, WeibelInit>(params, mblock, { 1, 2 }, params.ppc0() * 0.25);
    InjectUniform<D, S, WeibelInit>(params, mblock, { 3, 4 }, params.ppc0() * 0.25);
  }
} // namespace ntt

#endif
