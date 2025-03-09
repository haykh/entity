/**
 * @file utils/timer.h
 * @brief Basic timekeeping functionality with fancy printing
 * @implements
 *   - timer::Timers
 *   - enum timer::TimerFlags
 * @cpp:
 *   - timer.cpp
 * @namespces:
 *   - timer::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_UTILS_TIMER_H
#define GLOBAL_UTILS_TIMER_H

#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/colors.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"
#include "utils/tools.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

#include <chrono>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace timer {
  using timestamp  = std::chrono::time_point<std::chrono::system_clock>;
  using duration_t = long double;

  inline void convertTime(duration_t& value, std::string& units) {
    if (value > 1e6) {
      value /= 1e6;
      units  = " s";
    } else if (value > 1e3) {
      value /= 1e3;
      units  = "ms";
    } else if (value < 1e-2) {
      value *= 1e3;
      units  = "ns";
    }
  }

  class Timers {
    std::map<std::string, std::pair<timestamp, duration_t>> m_timers;
    std::vector<std::string>                                m_names;
    const bool                                              m_blocking;
    const std::function<void(void)>                         m_synchronize;

  public:
    Timers(std::initializer_list<std::string> names,
           const std::function<void(void)>&   synchronize = nullptr,
           const bool&                        blocking    = false)
      : m_blocking { blocking }
      , m_synchronize { synchronize } {
      raise::ErrorIf((synchronize == nullptr) && blocking,
                     "Synchronize function not provided",
                     HERE);
      for (const auto& name : names) {
        m_timers.insert({
          name,
          {std::chrono::system_clock::now(), 0.0}
        });
        m_names.push_back(name);
      }
    }

    ~Timers() = default;

    void start(const std::string& name) {
      m_timers[name].first = std::chrono::system_clock::now();
    }

    void stop(const std::string& name) {
      if (m_blocking) {
        m_synchronize();
#if defined(MPI_ENABLED)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
      auto end     = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                       end - m_timers[name].first)
                       .count();
      m_timers[name].second += elapsed;
    }

    void reset(const std::string& name) {
      m_timers[name].second = 0.0;
    }

    void resetAll() {
      for (auto& [name, _] : m_timers) {
        reset(name);
      }
    }

    [[nodiscard]]
    auto get(const std::string& name) const -> duration_t {
      if (name == "Total") {
        duration_t total = 0.0;
        for (auto& timer : m_timers) {
          total += timer.second.second;
        }
        return total;
      } else {
        raise::ErrorIf(m_timers.find(name) == m_timers.end(), "Timer not found", HERE);
        return m_timers.at(name).second;
      }
    }

    /**
     * @brief Gather all timers from all ranks
     * @param ignore_in_tot: vector of timer names to ignore in computing the total
     * @return map:
     *    key: timer name
     *    value: vector of numbers
     *        - max duration across ranks
     *        - max duration per particle
     *        - max duration per cell
     *        - max duration as % of total on that rank
     *        - imbalance % of the given timer
     */
    [[nodiscard]]
    auto gather(const std::vector<std::string>& ignore_in_tot,
                std::size_t                     npart,
                std::size_t                     ncells) const
      -> std::map<std::string,
                  std::tuple<duration_t, duration_t, duration_t, unsigned short, unsigned short>>;

    [[nodiscard]]
    auto printAll(TimerFlags  flags  = Timer::Default,
                  std::size_t npart  = 0,
                  std::size_t ncells = 0) const -> std::string;
  };
} // namespace timer

#endif // GLOBAL_UTILS_TIMER_H
