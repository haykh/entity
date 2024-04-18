#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/mpi_aliases.h"
#include "utils/formatting.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/engine.h"

#if defined(GPU_ENABLED)
  #include <cuda_runtime.h>
#endif

#if defined(OUTPUT_ENABLED)
  #include <H5public.h>
  #include <adios2.h>
#endif

#include <string>

namespace ntt {

  namespace {
    void add_header(std::string&                    report,
                    const std::vector<std::string>& lines,
                    const std::vector<const char*>& colors) {
      report += fmt::format("%s╔%s╗%s\n",
                            color::BRIGHT_BLACK,
                            fmt::repeat("═", 58).c_str(),
                            color::RESET);
      for (std::size_t i { 0 }; i < lines.size(); ++i) {
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

#if defined(MPI_ENABLED)
    void add_label(std::string& report, unsigned short indent, const char* label) {
      report += fmt::format("%s%s\n", std::string(indent, ' ').c_str(), label);
    }
#endif // MPI_ENABLED

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
    void add_directional_param(std::string&   report,
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
  } // namespace

  template <SimEngine::type S, class M>
  void Engine<S, M>::print_report() const {
#if defined(MPI_ENABLED)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank != MPI_ROOT_RANK) {
      return;
    }
    int mpi_v, mpi_subv;
    MPI_Get_version(&mpi_v, &mpi_subv);
    const std::string mpi_version = fmt::format("%d.%d", mpi_v, mpi_subv);
#else  // not MPI_ENABLED
    const std::string mpi_version = "OFF";
#endif // MPI_ENABLED

    const auto entity_version = "Entity v" + std::string(ENTITY_VERSION);
    const auto hash           = std::string(ENTITY_GIT_HASH);
    const auto pgen           = std::string(PGEN);
    const auto nspec          = m_metadomain.species_params().size();

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

#if defined(GPU_ENABLED)
    int cuda_v;
    cudaRuntimeGetVersion(&cuda_v);
    const auto major { cuda_v / 1000 };
    const auto minor { cuda_v % 1000 / 10 };
    const auto patch { cuda_v % 10 };
    const auto cuda_version = fmt::format("%d.%d.%d", major, minor, patch);
#else // not GPU_ENABLED
    const std::string cuda_version = "OFF";
#endif

    const auto kokkos_version = fmt::format("%d.%d.%d",
                                            KOKKOS_VERSION / 10000,
                                            KOKKOS_VERSION / 100 % 100,
                                            KOKKOS_VERSION % 100);

#if defined(OUTPUT_ENABLED)
    unsigned h5_major, h5_minor, h5_release;
    H5get_libversion(&h5_major, &h5_minor, &h5_release);
    const std::string hdf5_version   = fmt::format("%d.%d.%d",
                                                 h5_major,
                                                 h5_minor,
                                                 h5_release);
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

    // clang-format off
    std::string report {"\n\n"};
    add_header(report, {entity_version}, {color::BRIGHT_GREEN});
    report += "\n";
    add_category(report, 4, "Backend");
    add_param(report, 4, "Build hash", "%s", hash.c_str());
    add_param(report, 4, "CXX", "%s [%s]", ccx.c_str(), cpp_standard.c_str());
    add_param(report, 4, "CUDA", "%s", cuda_version.c_str());
    add_param(report, 4, "MPI", "%s", mpi_version.c_str());
    add_param(report, 4, "HDF5", "%s", hdf5_version.c_str());
    add_param(report, 4, "Kokkos", "%s", kokkos_version.c_str());
    add_param(report, 4, "ADIOS2", "%s", adios2_version.c_str());
    add_param(report, 4, "Debug", "%s", dbg.c_str());
    report += "\n";
    add_category(report, 4, "Configuration");
    add_param(report, 4, "Name", "%s", m_params.get<std::string>("simulation.name").c_str());
    add_param(report, 4, "Problem generator", "%s", pgen.c_str());
    add_param(report, 4, "Engine", "%s", SimEngine(S).to_string());
    add_param(report, 4, "Metric", "%s", Metric(M::MetricType).to_string());
    add_param(report, 4, "Timestep [dt]", "%.3e", dt);
    add_param(report, 4, "Runtime", "%.3Le [%d steps]", runtime, max_steps);
    report += "\n";
    add_category(report, 4, "Global domain");
    add_param(report, 4, "Resolution", "%s", m_params.stringize<std::size_t>("grid.resolution").c_str());
    add_param(report, 4, "Extent", "%s", m_params.stringize<real_t>("grid.extent").c_str());
    add_param(report, 4, "Fiducial cell size [dx0]", "%.3e", m_params.get<real_t>("scales.dx0"));
    add_subcategory(report, 4, "Boundary conditions");
    add_param(report, 6, "Fields", "%s", m_params.stringize<FldsBC>("grid.boundaries.fields").c_str());
    add_param(report, 6, "Particles", "%s", m_params.stringize<PrtlBC>("grid.boundaries.particles").c_str());
    add_param(report, 4, "Domain decomposition", "%s [%d total]", fmt::formatVector(m_metadomain.ndomains_per_dim()).c_str(), m_metadomain.ndomains());
    report += "\n";
    add_category(report, 4, "Fiducial parameters");
    add_param(report, 4, "Particles per cell [ppc0]", "%.1f", m_params.get<real_t>("particles.ppc0"));
    add_param(report, 4, "Larmor radius [larmor0]", "%.3e [%.3f dx0]", m_params.get<real_t>("scales.larmor0"), m_params.get<real_t>("scales.larmor0") / m_params.get<real_t>("scales.dx0"));
    add_param(report, 4, "Larmor frequency [omegaB0 * dt]", "%.3e", m_params.get<real_t>("scales.omegaB0") * m_params.get<real_t>("algorithms.timestep.dt"));
    add_param(report, 4, "Skin depth [skindepth0]", "%.3e [%.3f dx0]", m_params.get<real_t>("scales.skindepth0"), m_params.get<real_t>("scales.skindepth0") / m_params.get<real_t>("scales.dx0"));
    add_param(report, 4, "Plasma frequency [omp0 * dt]", "%.3e", m_params.get<real_t>("algorithms.timestep.dt") / m_params.get<real_t>("scales.skindepth0"));
    add_param(report, 4, "Magnetization [sigma0]", "%.3e", m_params.get<real_t>("scales.sigma0"));

    if (nspec > 0) {
      report += "\n";
      add_category(report, 4, "Particles");
    }
    for (const auto& species : m_metadomain.species_params()) {
      add_subcategory(report, 4, fmt::format("Species #%d", species.index()).c_str());
      add_param(report, 6, "Label", "%s", species.label().c_str());
      add_param(report, 6, "Mass", "%.1f", species.mass());
      add_param(report, 6, "Charge", "%.1f", species.charge());
      add_param(report, 6, "Max #", "%d [per domain]", species.maxnpart());
      add_param(report, 6, "Pusher", "%s", species.pusher().to_string());
      add_param(report, 6, "Cooling", "%s", species.cooling().to_string());
      add_param(report, 6, "# of payloads", "%d", species.npld());
    }
    // clang-format on

#if defined(MPI_ENABLED)
    report += "\n";
    add_category(report, 4, "MPI");
    for (unsigned int idx { 0 }; idx < m_metadomain.ndomains(); ++idx) {
      const auto& domain = m_metadomain.idx2subdomain(idx);
      add_subcategory(report, 4, fmt::format("Domain #%d", domain.index()).c_str());
      add_param(report, 6, "Rank", "%d", domain.mpi_rank());
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
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        const auto flds_bc      = domain.mesh.flds_bc_in(direction);
        const auto prtl_bc      = domain.mesh.prtl_bc_in(direction);
        bool       has_sync     = false;
        auto       neighbor_idx = domain.neighbor_idx_in(direction);
        if (flds_bc == FldsBC::SYNC || prtl_bc == PrtlBC::SYNC) {
          has_sync = true;
        }
        add_directional_param(report,
                              8,
                              direction.to_string().c_str(),
                              "%-10s  %-10s  %-10s",
                              flds_bc.to_string(),
                              prtl_bc.to_string(),
                              has_sync ? std::to_string(neighbor_idx).c_str()
                                       : ".");
      }
    }
#endif // MPI_ENABLED
    report += "\n\n";
    info::Print(report);
  }

  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template class Engine<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;

} // namespace ntt