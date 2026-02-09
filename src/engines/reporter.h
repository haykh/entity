#ifndef ENGINES_REPORTER_H
#define ENGINES_REPORTER_H

#include "enums.h"

#include "framework/parameters/parameters.h"

#include <string>
#include <vector>

namespace ntt {

  auto ReportSimulationConfig(const SimulationParams&,
                              const std::string&,
                              SimEngine,
                              Metric,
                              real_t,
                              simtime_t,
                              timestep_t,
                              const std::vector<unsigned int>&,
                              unsigned int) -> std::string;

} // namespace ntt

#endif // ENGINES_REPORTER_H
