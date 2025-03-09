#include "utils/progressbar.h"

#include "utils/error.h"
#include "utils/formatting.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pbar {

  auto normalize_duration_fmt(long double t, const std::string& u)
    -> std::pair<long double, std::string> {
    const std::vector<std::pair<std::string, long double>> units {
      {"µs",   1e0},
      { "ms",   1e3},
      {  "s",   1e6},
      {"min",   6e7},
      { "hr", 3.6e9}
    };
    auto it    = std::find_if(units.begin(), units.end(), [&u](const auto& pr) {
      return pr.first == u;
    });
    int  u_idx = (it != units.end()) ? std::distance(units.begin(), it) : -1;
    raise::ErrorIf(u_idx < 0, "Invalid unit", HERE);
    int shift = 0;
    if (t < 1) {
      shift = -1;
    } else if (1e3 <= t && t < 1e6) {
      shift = 1;
    } else if (1e6 <= t && t < 6e7) {
      shift += 2;
    } else if (6e7 <= t && t < 3.6e9) {
      shift += 3;
    } else if (3.6e9 <= t) {
      shift += 4;
    }
    auto newu_idx = std::min(std::max(0, u_idx + shift),
                             static_cast<int>(units.size()));
    return { t * (units[u_idx].second / units[newu_idx].second),
             units[newu_idx].first };
  }

  auto to_human_readable(long double t, const std::string& u) -> std::string {
    const auto [tt, tu]   = normalize_duration_fmt(t, u);
    const auto t1         = static_cast<int>(tt);
    const auto t2         = tt - static_cast<long double>(t1);
    const auto [tt2, tu2] = normalize_duration_fmt(t2, tu);
    return fmt::format("%d%s %d%s", t1, tu.c_str(), static_cast<int>(tt2), tu2.c_str());
  }

  auto ProgressBar(const DurationHistory& history,
                   std::size_t            step,
                   std::size_t            max_steps,
                   DiagFlags&             flags) -> std::string {
    auto avg_duration = history.average();

#if defined(MPI_ENABLED)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<long double> mpi_avg_durations(size, 0.0);
    MPI_Gather(&avg_duration,
               1,
               mpi::get_type<long double>(),
               mpi_avg_durations.data(),
               1,
               mpi::get_type<long double>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank != MPI_ROOT_RANK) {
      return "";
    }
    avg_duration = *std::max_element(mpi_avg_durations.begin(),
                                     mpi_avg_durations.end());
#endif

    const auto avg     = to_human_readable(avg_duration, "µs");
    const auto elapsed = to_human_readable(history.elapsed(), "µs");
    const auto remain  = to_human_readable(
      static_cast<long double>(max_steps - step) * avg_duration,
      "µs");

    const auto pct = static_cast<long double>(step) /
                     static_cast<long double>(max_steps);
    const int nfilled = std::min(static_cast<int>(pct * params::width),
                                 params::width);
    const int nempty  = params::width - nfilled;
    const auto c_bmagenta = color::get_color("bmagenta", flags & Diag::Colorful);
    const auto c_reset = color::get_color("reset", flags & Diag::Colorful);

    std::stringstream ss;

    ss << "Timestep duration: " << c_bmagenta << avg << c_reset << std::endl;
    ss << "Remaining time: " << c_bmagenta << remain << c_reset << std::endl;
    ss << "Elapsed time: " << c_bmagenta << elapsed << c_reset << std::endl;
    ss << params::start;
    for (auto i { 0 }; i < nfilled; ++i) {
      ss << params::fill;
    }
    for (auto i { 0 }; i < nempty; ++i) {
      ss << params::empty;
    }
    ss << params::end << " " << std::fixed << std::setprecision(2)
       << std::setfill(' ') << std::setw(6) << std::right << pct * 100.0 << "%\n";

    return ss.str();
  }

} // namespace pbar
