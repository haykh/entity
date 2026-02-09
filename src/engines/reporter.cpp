#include "engines/reporter.h"

#include "enums.h"

#include "utils/reporter.h"

#include "framework/parameters/parameters.h"

#include <string>
#include <vector>

namespace ntt {

  auto ReportSimulationConfig(const SimulationParams&          params,
                              const std::string&               pgen,
                              SimEngine                        S,
                              Metric                           M,
                              real_t                           dt,
                              simtime_t                        runtime,
                              timestep_t                       max_steps,
                              const std::vector<unsigned int>& ndomains_per_dim,
                              unsigned int ndomains) -> std::string {
    std::string report = "";
    /*
     * Simulation configs
     */
    reporter::AddCategory(report, 4, "Configuration");
    reporter::AddParam(report,
                       4,
                       "Name",
                       "%s",
                       params.template get<std::string>("simulation.name").c_str());
    reporter::AddParam(report, 4, "Problem generator", "%s", pgen.c_str());
    reporter::AddParam(report, 4, "Engine", "%s", S.to_string());
    reporter::AddParam(report, 4, "Metric", "%s", M.to_string());
#if SHAPE_ORDER == 0
    reporter::AddParam(report, 4, "Deposit", "%s", "zigzag");
#else
    reporter::AddParam(report, 4, "Deposit", "%s", "esirkepov");
    reporter::AddParam(report, 4, "Interpolation order", "%i", SHAPE_ORDER);
#endif
    reporter::AddParam(report, 4, "Timestep [dt]", "%.3e", dt);
    reporter::AddParam(report, 4, "Runtime", "%.3e [%d steps]", runtime, max_steps);
    report += "\n";
    reporter::AddCategory(report, 4, "Global domain");
    reporter::AddParam(
      report,
      4,
      "Resolution",
      "%s",
      params.template stringize<ncells_t>("grid.resolution").c_str());
    reporter::AddParam(report,
                       4,
                       "Extent",
                       "%s",
                       params.template stringize<real_t>("grid.extent").c_str());
    reporter::AddParam(report,
                       4,
                       "Fiducial cell size [dx0]",
                       "%.3e",
                       params.template get<real_t>("scales.dx0"));
    reporter::AddSubcategory(report, 4, "Boundary conditions");
    reporter::AddParam(
      report,
      6,
      "Fields",
      "%s",
      params.template stringize<FldsBC>("grid.boundaries.fields").c_str());
    reporter::AddParam(
      report,
      6,
      "Particles",
      "%s",
      params.template stringize<PrtlBC>("grid.boundaries.particles").c_str());
    reporter::AddParam(report,
                       4,
                       "Domain decomposition",
                       "%s [%d total]",
                       fmt::formatVector(ndomains_per_dim).c_str(),
                       ndomains);
    report += "\n";
    reporter::AddCategory(report, 4, "Fiducial parameters");
    reporter::AddParam(report,
                       4,
                       "Particles per cell [ppc0]",
                       "%.1f",
                       params.template get<real_t>("particles.ppc0"));
    reporter::AddParam(report,
                       4,
                       "Larmor radius [larmor0]",
                       "%.3e [%.3f dx0]",
                       params.template get<real_t>("scales.larmor0"),
                       params.template get<real_t>("scales.larmor0") /
                         params.template get<real_t>("scales.dx0"));
    reporter::AddParam(report,
                       4,
                       "Larmor frequency [omegaB0 * dt]",
                       "%.3e",
                       params.template get<real_t>("scales.omegaB0") *
                         params.template get<real_t>("algorithms.timestep.dt"));
    reporter::AddParam(report,
                       4,
                       "Skin depth [skindepth0]",
                       "%.3e [%.3f dx0]",
                       params.template get<real_t>("scales.skindepth0"),
                       params.template get<real_t>("scales.skindepth0") /
                         params.template get<real_t>("scales.dx0"));
    reporter::AddParam(report,
                       4,
                       "Plasma frequency [omp0 * dt]",
                       "%.3e",
                       params.template get<real_t>("algorithms.timestep.dt") /
                         params.template get<real_t>("scales.skindepth0"));
    reporter::AddParam(report,
                       4,
                       "Magnetization [sigma0]",
                       "%.3e",
                       params.template get<real_t>("scales.sigma0"));
    return report;
  }

} // namespace ntt
