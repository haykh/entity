#include "utils/reporter.h"

#include "utils/colors.h"
#include "utils/error.h"
#include "utils/formatting.h"

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

namespace reporter {
  void AddHeader(std::string&                    report,
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

  void AddCategory(std::string& report, unsigned short indent, const char* name) {
    report += fmt::format("%s%s%s%s\n",
                          std::string(indent, ' ').c_str(),
                          color::BLUE,
                          name,
                          color::RESET);
  }

  void AddSubcategory(std::string& report, unsigned short indent, const char* name) {
    report += fmt::format("%s%s-%s %s:\n",
                          std::string(indent, ' ').c_str(),
                          color::BRIGHT_BLACK,
                          color::RESET,
                          name);
  }

  void AddLabel(std::string& report, unsigned short indent, const char* label) {
    report += fmt::format("%s%s\n", std::string(indent, ' ').c_str(), label);
  }

  auto Bytes2HumanReadable(std::size_t bytes) -> std::pair<double, std::string> {
    const std::vector<std::string> units { "B", "KB", "MB", "GB", "TB" };
    idx_t                          unit_idx = 0;
    auto                           size     = static_cast<double>(bytes);
    while ((size >= 1024.0) and (unit_idx < units.size() - 1)) {
      size /= 1024.0;
      ++unit_idx;
    }
    return { size, units[unit_idx] };
  }

  auto Backend() -> std::string {
    std::string report = "";
#if defined(MPI_ENABLED)
    int mpi_v, mpi_subv;
    MPI_Get_version(&mpi_v, &mpi_subv);
    const std::string mpi_version = fmt::format("%d.%d", mpi_v, mpi_subv);
#else  // not MPI_ENABLED
    const std::string mpi_version = "OFF";
#endif // MPI_ENABLED

    const auto entity_version = "Entity v" + std::string(ENTITY_VERSION);
    const auto hash           = std::string(ENTITY_GIT_HASH);
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
    AddHeader(report, { entity_version }, { color::BRIGHT_GREEN });
    report += "\n";

    /*
     * Backend
     */
    AddCategory(report, 4, "Backend");
    AddParam(report, 4, "Build hash", "%s", hash.c_str());
    AddParam(report, 4, "CXX", "%s [%s]", ccx.c_str(), cpp_standard.c_str());
#if defined(CUDA_ENABLED)
    AddParam(report, 4, "CUDA", "%s", cuda_version.c_str());
#elif defined(HIP_VERSION)
    AddParam(report, 4, "HIP", "%s", hip_version.c_str());
#endif
    AddParam(report, 4, "MPI", "%s", mpi_version.c_str());
#if defined(MPI_ENABLED) && defined(DEVICE_ENABLED)
  #if defined(GPU_AWARE_MPI)
    const std::string gpu_aware_mpi = "ON";
  #else
    const std::string gpu_aware_mpi = "OFF";
  #endif
    AddParam(report, 4, "GPU-aware MPI", "%s", gpu_aware_mpi.c_str());
#endif
    AddParam(report, 4, "Kokkos", "%s", kokkos_version.c_str());
    AddParam(report, 4, "ADIOS2", "%s", adios2_version.c_str());
    AddParam(report, 4, "Precision", "%s", precision);
    AddParam(report, 4, "Debug", "%s", dbg.c_str());
    report += "\n";

    /*
     * Compilation flags
     */
    AddCategory(report, 4, "Compilation flags");
#if defined(SINGLE_PRECISION)
    AddParam(report, 4, "SINGLE_PRECISION", "%s", "ON");
#else
    AddParam(report, 4, "SINGLE_PRECISION", "%s", "OFF");
#endif

#if defined(OUTPUT_ENABLED)
    AddParam(report, 4, "OUTPUT_ENABLED", "%s", "ON");
#else
    AddParam(report, 4, "OUTPUT_ENABLED", "%s", "OFF");
#endif

#if defined(DEBUG)
    AddParam(report, 4, "DEBUG", "%s", "ON");
#else
    AddParam(report, 4, "DEBUG", "%s", "OFF");
#endif

#if defined(CUDA_ENABLED)
    AddParam(report, 4, "CUDA_ENABLED", "%s", "ON");
#else
    AddParam(report, 4, "CUDA_ENABLED", "%s", "OFF");
#endif

#if defined(HIP_ENABLED)
    AddParam(report, 4, "HIP_ENABLED", "%s", "ON");
#else
    AddParam(report, 4, "HIP_ENABLED", "%s", "OFF");
#endif

#if defined(DEVICE_ENABLED)
    AddParam(report, 4, "DEVICE_ENABLED", "%s", "ON");
#else
    AddParam(report, 4, "DEVICE_ENABLED", "%s", "OFF");
#endif

#if defined(MPI_ENABLED)
    AddParam(report, 4, "MPI_ENABLED", "%s", "ON");
#else
    AddParam(report, 4, "MPI_ENABLED", "%s", "OFF");
#endif

#if defined(GPU_AWARE_MPI)
    AddParam(report, 4, "GPU_AWARE_MPI", "%s", "ON");
#else
    AddParam(report, 4, "GPU_AWARE_MPI", "%s", "OFF");
#endif
    report += "\n";
    return report;
  }

} // namespace reporter
