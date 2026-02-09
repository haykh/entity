/**
 * @file engines/engine.hpp
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI
 *   - DEBUG
 *   - OUTPUT_ENABLED
 *   - CUDA_ENABLED
 */

#ifndef ENGINES_ENGINE_H
#define ENGINES_ENGINE_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/mpi_aliases.h"
#include "utils/colors.h"
#include "utils/diag.h"
#include "utils/formatting.h"
#include "utils/timer.h"
#include "utils/toml.h"

#include "metrics/traits.h"

#include "archetypes/field_setter.h"
#include "archetypes/traits.h"
#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include "pgen.hpp"

#if defined(CUDA_ENABLED)
  #include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
  #include <hip/hip_runtime.h>
#endif

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
#endif

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>
#include <vector>

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
  #include <adios2/cxx/KokkosView.h>
#endif // OUTPUT_ENABLED

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <map>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  concept IsCompatibleWithEngine =
    metric::traits::HasD<M> and
    arch::traits::pgen::check_compatibility<S>::value(user::PGen<S, M>::engines) and
    arch::traits::pgen::check_compatibility<M::MetricType>::value(
      user::PGen<S, M>::metrics) and
    arch::traits::pgen::check_compatibility<M::Dim>::value(
      user::PGen<S, M>::dimensions);

  template <SimEngine::type S, class M>
    requires IsCompatibleWithEngine<S, M>
  class Engine {

  protected:
#if defined(OUTPUT_ENABLED)
  #if defined(MPI_ENABLED)
    adios2::ADIOS m_adios { MPI_COMM_WORLD };
  #else
    adios2::ADIOS m_adios;
  #endif
#endif

    SimulationParams m_params;
    Metadomain<S, M> m_metadomain;
    user::PGen<S, M> m_pgen;

    const bool       is_resuming;
    const simtime_t  runtime;
    const real_t     dt;
    const timestep_t max_steps;
    const timestep_t start_step;
    const simtime_t  start_time;
    simtime_t        time;
    timestep_t       step;

  public:
    static constexpr Dimension D { M::Dim };

    Engine(const SimulationParams& params)
      : m_params { params }
      , m_metadomain { m_params.get<unsigned int>("simulation.domain.number"),
                       m_params.get<std::vector<int>>(
                         "simulation.domain.decomposition"),
                       m_params.get<std::vector<ncells_t>>("grid.resolution"),
                       m_params.get<boundaries_t<real_t>>("grid.extent"),
                       m_params.get<boundaries_t<FldsBC>>(
                         "grid.boundaries.fields"),
                       m_params.get<boundaries_t<PrtlBC>>(
                         "grid.boundaries.particles"),
                       m_params.get<std::map<std::string, real_t>>(
                         "grid.metric.params"),
                       m_params.get<std::vector<ParticleSpecies>>(
                         "particles.species") }
      , m_pgen { m_params, m_metadomain }
      , is_resuming { m_params.get<bool>("checkpoint.is_resuming") }
      , runtime { m_params.get<simtime_t>("simulation.runtime") }
      , dt { m_params.get<real_t>("algorithms.timestep.dt") }
      , max_steps { static_cast<timestep_t>(runtime / dt) }
      , start_step { m_params.get<timestep_t>("checkpoint.start_step") }
      , start_time { m_params.get<simtime_t>("checkpoint.start_time") }
      , time { start_time }
      , step { start_step } {}

    ~Engine() = default;

    void init();
    void print_report() const;

    virtual void step_forward(timer::Timers&, Domain<S, M>&) = 0;

    void run();
  };

  template <SimEngine::type S, class M>
    requires IsCompatibleWithEngine<S, M>
  void Engine<S, M>::init() {
    m_metadomain.InitStatsWriter(m_params, is_resuming);
#if defined(OUTPUT_ENABLED)
    m_metadomain.InitWriter(&m_adios, m_params);
    m_metadomain.InitCheckpointWriter(&m_adios, m_params);
#endif
    logger::Checkpoint("Initializing Engine", HERE);
    if (not is_resuming) {
      // start a new simulation with initial conditions
      logger::Checkpoint("Loading initial conditions", HERE);
      if constexpr (arch::traits::pgen::HasInitFlds<user::PGen<S, M>>) {
        logger::Checkpoint("Initializing fields from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          Kokkos::parallel_for(
            "InitFields",
            loc_dom.mesh.rangeActiveCells(),
            arch::SetEMFields_kernel<decltype(m_pgen.init_flds), S, M> {
              loc_dom.fields.em,
              m_pgen.init_flds,
              loc_dom.mesh.metric });
        });
      }
      if constexpr (
        arch::traits::pgen::HasInitPrtls<user::PGen<S, M>, Domain<S, M>>) {
        logger::Checkpoint("Initializing particles from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          m_pgen.InitPrtls(loc_dom);
        });
      }
    } else {
#if defined(OUTPUT_ENABLED)
      // read simulation data from the checkpoint
      raise::ErrorIf(
        m_params.template get<timestep_t>("checkpoint.start_step") == 0,
        "Resuming simulation from a checkpoint requires a valid start_step",
        HERE);
      logger::Checkpoint("Resuming simulation from a checkpoint", HERE);
      m_metadomain.ContinueFromCheckpoint(&m_adios, m_params);
#else
      raise::Error(
        "Resuming simulation from a checkpoint requires -D output=ON",
        HERE);
#endif
    }
    print_report();
  }

  namespace {
    void add_header(std::string&                    report,
                    const std::vector<std::string>& lines,
                    const std::vector<const char*>& colors) {
      report += fmt::format("%s╔%s╗%s\n",
                            color::BRIGHT_BLACK,
                            fmt::repeat("═", 58).c_str(),
                            color::RESET);
      for (auto i { 0u }; i < lines.size(); ++i) {
        report += fmt::format("%s║%s %s%s%s%s%s║%s\n",
                              color::BRIGHT_BLACK,
                              color::RESET,
                              colors[i],
                              lines[i].c_str(),
                              color::RESET,
                              fmt::repeat(" ", 57 - lines[i].size()).c_str(),
                              color::BRIGHT_BLACK,
                              color::RESET);
      }
      report += fmt::format("%s╚%s╝%s\n",
                            color::BRIGHT_BLACK,
                            fmt::repeat("═", 58).c_str(),
                            color::RESET);
    }

    void add_category(std::string& report, unsigned short indent, const char* name) {
      report += fmt::format("%s%s%s%s\n",
                            std::string(indent, ' ').c_str(),
                            color::BLUE,
                            name,
                            color::RESET);
    }

    void add_subcategory(std::string& report, unsigned short indent, const char* name) {
      report += fmt::format("%s%s-%s %s:\n",
                            std::string(indent, ' ').c_str(),
                            color::BRIGHT_BLACK,
                            color::RESET,
                            name);
    }

    void add_label(std::string& report, unsigned short indent, const char* label) {
      report += fmt::format("%s%s\n", std::string(indent, ' ').c_str(), label);
    }

    template <typename... Args>
    void add_param(std::string&   report,
                   unsigned short indent,
                   const char*    name,
                   const char*    format,
                   Args... args) {
      report += fmt::format("%s%s-%s %s: %s%s%s\n",
                            std::string(indent, ' ').c_str(),
                            color::BRIGHT_BLACK,
                            color::RESET,
                            name,
                            color::BRIGHT_YELLOW,
                            fmt::format(format, args...).c_str(),
                            color::RESET);
    }

    template <typename... Args>
    void add_unlabeled_param(std::string&   report,
                             unsigned short indent,
                             const char*    name,
                             const char*    format,
                             Args... args) {
      report += fmt::format("%s%s: %s%s%s\n",
                            std::string(indent, ' ').c_str(),
                            name,
                            color::BRIGHT_YELLOW,
                            fmt::format(format, args...).c_str(),
                            color::RESET);
    }

    auto bytes_to_human_readable(std::size_t bytes)
      -> std::pair<double, std::string> {
      const std::vector<std::string> units { "B", "KB", "MB", "GB", "TB" };
      idx_t                          unit_idx = 0;
      auto                           size     = static_cast<double>(bytes);
      while ((size >= 1024.0) and (unit_idx < units.size() - 1)) {
        size /= 1024.0;
        ++unit_idx;
      }
      return { size, units[unit_idx] };
    }
  } // namespace

  template <SimEngine::type S, class M>
    requires IsCompatibleWithEngine<S, M>
  void Engine<S, M>::print_report() const {
    const auto colored_stdout = m_params.template get<bool>(
      "diagnostics.colored_stdout");
    std::string report = "";
    CallOnce(
      [&](auto& metadomain, auto& params) {
#if defined(MPI_ENABLED)
        int mpi_v, mpi_subv;
        MPI_Get_version(&mpi_v, &mpi_subv);
        const std::string mpi_version = fmt::format("%d.%d", mpi_v, mpi_subv);
#else  // not MPI_ENABLED
        const std::string mpi_version = "OFF";
#endif // MPI_ENABLED

        const auto entity_version = "Entity v" + std::string(ENTITY_VERSION);
        const auto hash           = std::string(ENTITY_GIT_HASH);
        const auto pgen           = std::string(PGEN);
        const auto nspec          = metadomain.species_params().size();
        const auto precision      = (sizeof(real_t) == 4) ? "single" : "double";

#if defined(__clang__)
        const std::string ccx = "Clang/LLVM " __clang_version__;
#elif defined(__ICC) || defined(__INTEL_COMPILER)
        const std::string ccx = "Intel ICC/ICPC " __VERSION__;
#elif defined(__GNUC__) || defined(__GNUG__)
        const std::string ccx = "GNU GCC/G++ " __VERSION__;
#elif defined(__HP_cc) || defined(__HP_aCC)
        const std::string ccx = "Hewlett-Packard C/aC++ " __HP_aCC;
#elif defined(__IBMC__) || defined(__IBMCPP__)
        const std::string ccx = "IBM XL C/C++ " __IBMCPP__;
#elif defined(_MSC_VER)
        const std::string ccx = "Microsoft Visual Studio " _MSC_VER;
#else
        const std::string ccx = "Unknown compiler";
#endif
        std::string cpp_standard;
        if (__cplusplus == 202101L) {
          cpp_standard = "C++23";
        } else if (__cplusplus == 202002L) {
          cpp_standard = "C++20";
        } else if (__cplusplus == 201703L) {
          cpp_standard = "C++17";
        } else if (__cplusplus == 201402L) {
          cpp_standard = "C++14";
        } else if (__cplusplus == 201103L) {
          cpp_standard = "C++11";
        } else if (__cplusplus == 199711L) {
          cpp_standard = "C++98";
        } else {
          cpp_standard = "pre-standard " + std::to_string(__cplusplus);
        }

#if defined(CUDA_ENABLED)
        int cuda_v;
        cudaRuntimeGetVersion(&cuda_v);
        const auto major { cuda_v / 1000 };
        const auto minor { cuda_v % 1000 / 10 };
        const auto patch { cuda_v % 10 };
        const auto cuda_version = fmt::format("%d.%d.%d", major, minor, patch);
#elif defined(HIP_ENABLED)
        int  hip_v;
        auto status = hipDriverGetVersion(&hip_v);
        raise::ErrorIf(status != hipSuccess,
                       "hipDriverGetVersion failed with error code %d",
                       HERE);
        const auto major { hip_v / 10000000 };
        const auto minor { (hip_v % 10000000) / 100000 };
        const auto patch { hip_v % 100000 };
        const auto hip_version = fmt::format("%d.%d.%d", major, minor, patch);
#endif

        const auto kokkos_version = fmt::format("%d.%d.%d",
                                                KOKKOS_VERSION / 10000,
                                                KOKKOS_VERSION / 100 % 100,
                                                KOKKOS_VERSION % 100);

#if defined(OUTPUT_ENABLED)
        const std::string adios2_version = fmt::format("%d.%d.%d",
                                                       ADIOS2_VERSION / 10000,
                                                       ADIOS2_VERSION / 100 % 100,
                                                       ADIOS2_VERSION % 100);
#else // not OUTPUT_ENABLED
        const std::string adios2_version = "OFF";
#endif

#if defined(DEBUG)
        const std::string dbg = "ON";
#else // not DEBUG
        const std::string dbg = "OFF";
#endif

        report += "\n\n";
        add_header(report, { entity_version }, { color::BRIGHT_GREEN });
        report += "\n";

        /*
         * Backend
         */
        add_category(report, 4, "Backend");
        add_param(report, 4, "Build hash", "%s", hash.c_str());
        add_param(report, 4, "CXX", "%s [%s]", ccx.c_str(), cpp_standard.c_str());
#if defined(CUDA_ENABLED)
        add_param(report, 4, "CUDA", "%s", cuda_version.c_str());
#elif defined(HIP_VERSION)
        add_param(report, 4, "HIP", "%s", hip_version.c_str());
#endif
        add_param(report, 4, "MPI", "%s", mpi_version.c_str());
#if defined(MPI_ENABLED) && defined(DEVICE_ENABLED)
  #if defined(GPU_AWARE_MPI)
        const std::string gpu_aware_mpi = "ON";
  #else
        const std::string gpu_aware_mpi = "OFF";
  #endif
        add_param(report, 4, "GPU-aware MPI", "%s", gpu_aware_mpi.c_str());
#endif
        add_param(report, 4, "Kokkos", "%s", kokkos_version.c_str());
        add_param(report, 4, "ADIOS2", "%s", adios2_version.c_str());
        add_param(report, 4, "Precision", "%s", precision);
        add_param(report, 4, "Debug", "%s", dbg.c_str());
        report += "\n";

        /*
         * Compilation flags
         */
        add_category(report, 4, "Compilation flags");
#if defined(SINGLE_PRECISION)
        add_param(report, 4, "SINGLE_PRECISION", "%s", "ON");
#else
        add_param(report, 4, "SINGLE_PRECISION", "%s", "OFF");
#endif

#if defined(OUTPUT_ENABLED)
        add_param(report, 4, "OUTPUT_ENABLED", "%s", "ON");
#else
        add_param(report, 4, "OUTPUT_ENABLED", "%s", "OFF");
#endif

#if defined(DEBUG)
        add_param(report, 4, "DEBUG", "%s", "ON");
#else
        add_param(report, 4, "DEBUG", "%s", "OFF");
#endif

#if defined(CUDA_ENABLED)
        add_param(report, 4, "CUDA_ENABLED", "%s", "ON");
#else
        add_param(report, 4, "CUDA_ENABLED", "%s", "OFF");
#endif

#if defined(HIP_ENABLED)
        add_param(report, 4, "HIP_ENABLED", "%s", "ON");
#else
        add_param(report, 4, "HIP_ENABLED", "%s", "OFF");
#endif

#if defined(DEVICE_ENABLED)
        add_param(report, 4, "DEVICE_ENABLED", "%s", "ON");
#else
        add_param(report, 4, "DEVICE_ENABLED", "%s", "OFF");
#endif

#if defined(MPI_ENABLED)
        add_param(report, 4, "MPI_ENABLED", "%s", "ON");
#else
        add_param(report, 4, "MPI_ENABLED", "%s", "OFF");
#endif

#if defined(GPU_AWARE_MPI)
        add_param(report, 4, "GPU_AWARE_MPI", "%s", "ON");
#else
        add_param(report, 4, "GPU_AWARE_MPI", "%s", "OFF");
#endif
        report += "\n";

        /*
         * Simulation configs
         */
        add_category(report, 4, "Configuration");
        add_param(report,
                  4,
                  "Name",
                  "%s",
                  params.template get<std::string>("simulation.name").c_str());
        add_param(report, 4, "Problem generator", "%s", pgen.c_str());
        add_param(report, 4, "Engine", "%s", SimEngine(S).to_string());
        add_param(report, 4, "Metric", "%s", Metric(M::MetricType).to_string());
#if SHAPE_ORDER == 0
        add_param(report, 4, "Deposit", "%s", "zigzag");
#else
        add_param(report, 4, "Deposit", "%s", "esirkepov");
        add_param(report, 4, "Interpolation order", "%i", SHAPE_ORDER);
#endif
        add_param(report, 4, "Timestep [dt]", "%.3e", dt);
        add_param(report, 4, "Runtime", "%.3e [%d steps]", runtime, max_steps);
        report += "\n";
        add_category(report, 4, "Global domain");
        add_param(report,
                  4,
                  "Resolution",
                  "%s",
                  params.template stringize<ncells_t>("grid.resolution").c_str());
        add_param(report,
                  4,
                  "Extent",
                  "%s",
                  params.template stringize<real_t>("grid.extent").c_str());
        add_param(report,
                  4,
                  "Fiducial cell size [dx0]",
                  "%.3e",
                  params.template get<real_t>("scales.dx0"));
        add_subcategory(report, 4, "Boundary conditions");
        add_param(
          report,
          6,
          "Fields",
          "%s",
          params.template stringize<FldsBC>("grid.boundaries.fields").c_str());
        add_param(
          report,
          6,
          "Particles",
          "%s",
          params.template stringize<PrtlBC>("grid.boundaries.particles").c_str());
        add_param(report,
                  4,
                  "Domain decomposition",
                  "%s [%d total]",
                  fmt::formatVector(m_metadomain.ndomains_per_dim()).c_str(),
                  m_metadomain.ndomains());
        report += "\n";
        add_category(report, 4, "Fiducial parameters");
        add_param(report,
                  4,
                  "Particles per cell [ppc0]",
                  "%.1f",
                  params.template get<real_t>("particles.ppc0"));
        add_param(report,
                  4,
                  "Larmor radius [larmor0]",
                  "%.3e [%.3f dx0]",
                  params.template get<real_t>("scales.larmor0"),
                  params.template get<real_t>("scales.larmor0") /
                    params.template get<real_t>("scales.dx0"));
        add_param(report,
                  4,
                  "Larmor frequency [omegaB0 * dt]",
                  "%.3e",
                  params.template get<real_t>("scales.omegaB0") *
                    params.template get<real_t>("algorithms.timestep.dt"));
        add_param(report,
                  4,
                  "Skin depth [skindepth0]",
                  "%.3e [%.3f dx0]",
                  params.template get<real_t>("scales.skindepth0"),
                  params.template get<real_t>("scales.skindepth0") /
                    params.template get<real_t>("scales.dx0"));
        add_param(report,
                  4,
                  "Plasma frequency [omp0 * dt]",
                  "%.3e",
                  params.template get<real_t>("algorithms.timestep.dt") /
                    params.template get<real_t>("scales.skindepth0"));
        add_param(report,
                  4,
                  "Magnetization [sigma0]",
                  "%.3e",
                  params.template get<real_t>("scales.sigma0"));

        if (nspec > 0) {
          report += "\n";
          add_category(report, 4, "Particles");
        }
        for (const auto& species : metadomain.species_params()) {
          add_subcategory(report,
                          4,
                          fmt::format("Species #%d", species.index()).c_str());
          add_param(report, 6, "Label", "%s", species.label().c_str());
          add_param(report, 6, "Mass", "%.1f", species.mass());
          add_param(report, 6, "Charge", "%.1f", species.charge());
          add_param(report, 6, "Max #", "%d [per domain]", species.maxnpart());
          add_param(report,
                    6,
                    "Pusher",
                    "%s",
                    ParticlePusher::to_string(species.pusher()).c_str());
          add_param(
            report,
            6,
            "Radiative drag",
            "%s",
            RadiativeDrag::to_string(species.radiative_drag_flags()).c_str());
          add_param(report, 6, "# of real-value payloads", "%d", species.npld_r());
          add_param(report, 6, "# of integer-value payloads", "%d", species.npld_i());
        }
        report.pop_back();
      },
      m_metadomain,
      m_params);
    info::Print(report, colored_stdout);

    report = "\n";
    CallOnce([&]() {
      add_category(report, 4, "Domains");
      report.pop_back();
    });
    info::Print(report, colored_stdout);

    for (unsigned int idx { 0 }; idx < m_metadomain.ndomains(); ++idx) {
      auto is_local = false;
      for (const auto& lidx : m_metadomain.l_subdomain_indices()) {
        is_local |= (idx == lidx);
      }
      if (is_local) {
        report             = "";
        const auto& domain = m_metadomain.subdomain(idx);
        add_subcategory(report,
                        4,
                        fmt::format("Domain #%d", domain.index()).c_str());
#if defined(MPI_ENABLED)
        add_param(report, 6, "Rank", "%d", domain.mpi_rank());
#endif
        add_param(report,
                  6,
                  "Resolution",
                  "%s",
                  fmt::formatVector(domain.mesh.n_active()).c_str());
        add_param(report,
                  6,
                  "Extent",
                  "%s",
                  fmt::formatVector(domain.mesh.extent()).c_str());
        add_subcategory(report, 6, "Boundary conditions");

        add_label(
          report,
          8 + 2 + 2 * M::Dim,
          fmt::format("%-10s  %-10s  %-10s", "[flds]", "[prtl]", "[neighbor]").c_str());
        for (auto& direction : dir::Directions<M::Dim>::all) {
          const auto flds_bc      = domain.mesh.flds_bc_in(direction);
          const auto prtl_bc      = domain.mesh.prtl_bc_in(direction);
          bool       has_sync     = false;
          auto       neighbor_idx = domain.neighbor_idx_in(direction);
          if (flds_bc == FldsBC::SYNC || prtl_bc == PrtlBC::SYNC) {
            has_sync = true;
          }
          add_unlabeled_param(report,
                              8,
                              direction.to_string().c_str(),
                              "%-10s  %-10s  %-10s",
                              flds_bc.to_string(),
                              prtl_bc.to_string(),
                              has_sync ? std::to_string(neighbor_idx).c_str()
                                       : ".");
        }
        add_subcategory(report, 6, "Memory footprint");
        auto flds_footprint         = domain.fields.memory_footprint();
        auto [flds_size, flds_unit] = bytes_to_human_readable(flds_footprint);
        add_param(report, 8, "Fields", "%.2f %s", flds_size, flds_unit.c_str());
        if (domain.species.size() > 0) {
          add_subcategory(report, 8, "Particles");
        }
        for (auto& species : domain.species) {
          const auto str = fmt::format("Species #%d (%s)",
                                       species.index(),
                                       species.label().c_str());
          auto [size, unit] = bytes_to_human_readable(species.memory_footprint());
          add_param(report, 10, str.c_str(), "%.2f %s", size, unit.c_str());
        }
        report.pop_back();
        if (idx == m_metadomain.ndomains() - 1) {
          report += "\n\n";
        }
        info::Print(report, colored_stdout, true, false);
      }
#if defined(MPI_ENABLED)
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
  }

  template <SimEngine::type S, class M>
    requires IsCompatibleWithEngine<S, M>
  void Engine<S, M>::run() {
    init();

    auto timers = timer::Timers {
      { "FieldSolver",
       "CurrentFiltering", "CurrentDeposit",
       "ParticlePusher", "FieldBoundaries",
       "ParticleBoundaries", "Communications",
       "Injector", "Custom",
       "PrtlClear", "Output",
       "Checkpoint" },
      []() {
        Kokkos::fence();
       },
      m_params.get<bool>("diagnostics.blocking_timers")
    };
    const auto diag_interval = m_params.template get<timestep_t>(
      "diagnostics.interval");

    auto       time_history   = pbar::DurationHistory { 1000 };
    const auto clear_interval = m_params.template get<timestep_t>(
      "particles.clear_interval");

    // main algorithm loop
    while (step < max_steps) {
      // run the engine-dependent algorithm step
      m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
        step_forward(timers, dom);
      });
      // poststep (if defined)
      if constexpr (
        arch::traits::pgen::HasCustomPostStep<decltype(m_pgen), Domain<S, M>>) {
        timers.start("Custom");
        m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
          m_pgen.CustomPostStep(step, time, dom);
        });
        timers.stop("Custom");
      }
      auto print_prtl_clear = (clear_interval > 0 and
                               step % clear_interval == 0 and step > 0);

      // advance time & step
      time += dt;
      ++step;

      auto print_output     = false;
      auto print_checkpoint = false;
#if defined(OUTPUT_ENABLED)
      timers.start("Output");
      if constexpr (
        arch::traits::pgen::HasCustomFieldOutput<decltype(m_pgen), Domain<S, M>>) {
        auto lambda_custom_field_output = [&](const std::string&    name,
                                              ndfield_t<M::Dim, 6>& buff,
                                              index_t               idx,
                                              timestep_t            step,
                                              simtime_t             time,
                                              const Domain<S, M>&   dom) {
          m_pgen.CustomFieldOutput(name, buff, idx, step, time, dom);
        };
        print_output &= m_metadomain.Write(m_params,
                                           step,
                                           step - 1,
                                           time,
                                           time - dt,
                                           lambda_custom_field_output);
      } else {
        print_output &= m_metadomain.Write(m_params, step, step - 1, time, time - dt);
      }
      if constexpr (
        arch::traits::pgen::HasCustomStatOutput<decltype(m_pgen), Domain<S, M>>) {
        auto lambda_custom_stat = [&](const std::string&  name,
                                      timestep_t          step,
                                      simtime_t           time,
                                      const Domain<S, M>& dom) -> real_t {
          return m_pgen.CustomStat(name, step, time, dom);
        };
        print_output &= m_metadomain.WriteStats(m_params,
                                                step,
                                                step - 1,
                                                time,
                                                time - dt,
                                                lambda_custom_stat);
      } else {
        print_output &= m_metadomain.WriteStats(m_params,
                                                step,
                                                step - 1,
                                                time,
                                                time - dt);
      }
      timers.stop("Output");

      timers.start("Checkpoint");
      print_checkpoint = m_metadomain.WriteCheckpoint(m_params,
                                                      step,
                                                      step - 1,
                                                      time,
                                                      time - dt);
      timers.stop("Checkpoint");
#endif

      // advance time_history
      time_history.tick();
      // print timestep report
      if (diag_interval > 0 and step % diag_interval == 0) {
        diag::printDiagnostics(
          step - 1,
          max_steps,
          time - dt,
          dt,
          timers,
          time_history,
          m_metadomain.l_ncells(),
          m_metadomain.species_labels(),
          m_metadomain.l_npart_perspec(),
          m_metadomain.l_maxnpart_perspec(),
          print_prtl_clear,
          print_output,
          print_checkpoint,
          m_params.get<bool>("diagnostics.colored_stdout"));
      }
      timers.resetAll();
    }
  }

} // namespace ntt

#endif // ENGINES_ENGINE_H
