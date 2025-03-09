/**
 * @file utils/progressbar.h
 * @brief Progress bar for logging the simulation progress
 * @implements
 *   - pbar::ProgressBar -> void
 * @cpp:
 *   - progressbar.cpp
 * @namespaces:
 *   - pbar::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - remove outlying values from the average (excl. output)
 */

#ifndef GLOBAL_UTILS_PROGRESSBAR_H
#define GLOBAL_UTILS_PROGRESSBAR_H

#include "utils/colors.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include <chrono>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pbar {
  namespace params {
    inline constexpr int       width { 70 };
    constexpr std::string_view fill { "■" };
    constexpr std::string_view empty { " " };
    constexpr std::string_view start { "[" };
    constexpr std::string_view end { "]" };
  } // namespace params

  class DurationHistory {
    std::size_t                                              capacity;
    std::vector<long double>                                 durations;
    const std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock>       prev_start;

  public:
    DurationHistory(std::size_t cap)
      : capacity { cap }
      , start { std::chrono::system_clock::now() }
      , prev_start { start } {}

    ~DurationHistory() = default;

    void tick() {
      const auto now = std::chrono::system_clock::now();
      if (durations.size() >= capacity) {
        durations.erase(durations.begin());
      }
      durations.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(now - prev_start).count());
      prev_start = now;
    }

    auto average() const -> long double {
      if (durations.size() > 0) {
        return std::accumulate(durations.begin(), durations.end(), 0.0) /
               static_cast<long double>(durations.size());
      } else {
        return 0.0;
      }
    }

    auto elapsed() const -> long double {
      return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now() - start)
        .count();
    }
  };

  auto normalize_duration_fmt(long double t, const std::string& u)
    -> std::pair<long double, std::string>;

  auto to_human_readable(long double t, const std::string& u) -> std::string;

  auto ProgressBar(const DurationHistory& history,
                   std::size_t            step,
                   std::size_t            max_steps,
                   DiagFlags&             flags) -> std::string;

} // namespace pbar

#endif // GLOBAL_UTILS_PROGRESSBAR_H
