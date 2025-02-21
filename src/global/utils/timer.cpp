#include "utils/timer.h"

#include "utils/colors.h"
#include "utils/formatting.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace timer {

  auto Timers::gather(const std::vector<std::string>& ignore_in_tot,
                      std::size_t                     npart,
                      std::size_t                     ncells) const
    -> std::map<std::string,
                std::tuple<duration_t, duration_t, duration_t, unsigned short, unsigned short>> {
    auto timer_stats = std::map<
      std::string,
      std::tuple<duration_t, duration_t, duration_t, unsigned short, unsigned short>> {};
#if defined(MPI_ENABLED)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::map<std::string, std::vector<duration_t>> all_timers {};

    // accumulate timers from MPI blocks
    for (auto& [name, timer] : m_timers) {
      all_timers.insert({ name, std::vector<duration_t>(size, 0.0) });
      MPI_Gather(&timer.second,
                 1,
                 mpi::get_type<duration_t>(),
                 all_timers[name].data(),
                 1,
                 mpi::get_type<duration_t>(),
                 MPI_ROOT_RANK,
                 MPI_COMM_WORLD);
    }
    // accumulate nparts and ncells from MPI blocks
    auto all_nparts = std::vector<std::size_t>(size, 0);
    auto all_ncells = std::vector<std::size_t>(size, 0);
    MPI_Gather(&npart,
               1,
               mpi::get_type<std::size_t>(),
               all_nparts.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    MPI_Gather(&ncells,
               1,
               mpi::get_type<std::size_t>(),
               all_ncells.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank != MPI_ROOT_RANK) {
      return {};
    }
    std::vector<duration_t> all_totals(size, 0.0);
    for (auto i { 0u }; i < size; ++i) {
      for (auto& [name, timer] : m_timers) {
        if (std::find(ignore_in_tot.begin(), ignore_in_tot.end(), name) ==
            ignore_in_tot.end()) {
          all_totals[i] += all_timers[name][i];
        }
      }
    }
    for (auto& [name, timer] : m_timers) {
      const auto max_time = *std::max_element(all_timers[name].begin(),
                                              all_timers[name].end());
      const auto max_idx  = std::distance(
        all_timers[name].begin(),
        std::max_element(all_timers[name].begin(), all_timers[name].end()));

      const auto per_npart  = all_nparts[max_idx] > 0
                                ? max_time /
                                   static_cast<duration_t>(all_nparts[max_idx])
                                : 0.0;
      const auto per_ncells = all_ncells[max_idx] > 0
                                ? max_time /
                                    static_cast<duration_t>(all_ncells[max_idx])
                                : 0.0;
      const auto pcent      = static_cast<unsigned short>(
        (max_time / all_totals[max_idx]) * 100.0);
      timer_stats.insert(
        { name,
          std::make_tuple(max_time,
                          per_npart,
                          per_ncells,
                          pcent,
                          tools::ArrayImbalance<duration_t>(all_timers[name])) });
    }
    const auto max_tot = *std::max_element(all_totals.begin(), all_totals.end());
    const auto tot_imb = tools::ArrayImbalance<duration_t>(all_totals);
    timer_stats.insert(
      { "Total", std::make_tuple(max_tot, 0.0, 0.0, 100u, tot_imb) });
#else
    duration_t local_tot = 0.0;
    for (auto& [name, timer] : m_timers) {
      if (std::find(ignore_in_tot.begin(), ignore_in_tot.end(), name) ==
          ignore_in_tot.end()) {
        local_tot += timer.second;
      }
    }
    for (auto& [name, timer] : m_timers) {
      const auto pcent = static_cast<unsigned short>(
        (timer.second / local_tot) * 100.0);
      timer_stats.insert(
        { name,
          std::make_tuple(
            timer.second,
            npart > 0 ? timer.second / static_cast<duration_t>(npart) : 0.0,
            timer.second / static_cast<duration_t>(ncells),
            pcent,
            0u) });
    }
    timer_stats.insert({ "Total", std::make_tuple(local_tot, 0.0, 0.0, 100u, 0u) });
#endif
    return timer_stats;
  }

  auto Timers::printAll(TimerFlags  flags,
                        std::size_t npart,
                        std::size_t ncells) const -> std::string {
    const std::vector<std::string> extras { "PrtlClear", "Output", "Checkpoint" };
    const auto stats = gather(extras, npart, ncells);
    if (stats.empty()) {
      return "";
    }

#if defined(MPI_ENABLED)
    const auto multi_rank = true;
#else
    const auto multi_rank = false;
#endif

    std::stringstream ss;

    const auto c_bblack  = color::get_color("bblack");
    const auto c_reset   = color::get_color("reset");
    const auto c_byellow = color::get_color("byellow");
    const auto c_blue    = color::get_color("blue");
    const auto c_red     = color::get_color("red");
    const auto c_yellow  = color::get_color("yellow");
    const auto c_green   = color::get_color("green");

    if (multi_rank and flags & Timer::PrintTitle) {
      ss << fmt::alignedTable(
        { "[SUBSTEP]", "[MAX DURATION]", "[% TOT", "VAR]", "[per PRTL", "CELL]" },
        { c_bblack, c_bblack, c_bblack, c_bblack, c_bblack, c_bblack },
        { 0, 37, 45, -48, 63, -66 },
        { ' ', '.', ' ', ':', ' ', ':' },
        c_bblack,
        c_reset);
    } else {
      ss << fmt::alignedTable(
        { "[SUBSTEP]", "[DURATION]", "[% TOT]", "[per PRTL", "CELL]" },
        { c_bblack, c_bblack, c_bblack, c_bblack, c_bblack },
        { 0, 37, 45, 55, -58 },
        { ' ', '.', ' ', ' ', ':' },
        c_bblack,
        c_reset);
    }

    for (auto& [name, timers] : m_timers) {
      if (std::find(extras.begin(), extras.end(), name) != extras.end()) {
        continue;
      }
      std::string units = "µs", units_npart = "µs", units_ncells = "µs";
      auto        time       = std::get<0>(stats.at(name));
      auto        per_npart  = std::get<1>(stats.at(name));
      auto        per_ncells = std::get<2>(stats.at(name));
      const auto  tot_pct    = std::get<3>(stats.at(name));
      const auto  var_pct    = std::get<4>(stats.at(name));
      if (flags & Timer::AutoConvert) {
        convertTime(time, units);
        convertTime(per_npart, units_npart);
        convertTime(per_ncells, units_ncells);
      }

      if (multi_rank) {
        ss << fmt::alignedTable(
          { name,
            fmt::format("%.2Lf", time) + " " + units,
            std::to_string(tot_pct) + "%",
            std::to_string(var_pct) + "%",
            fmt::format("%.2Lf", per_npart) + " " + units_npart,
            fmt::format("%.2Lf", per_ncells) + " " + units_ncells },
          { c_reset,
            c_yellow,
            ((tot_pct > 60) ? c_red : ((tot_pct > 40) ? c_yellow : c_green)),
            ((var_pct > 50) ? c_red : ((var_pct > 30) ? c_yellow : c_green)),
            c_yellow,
            c_yellow },
          { -2, 37, 45, -48, 63, -66 },
          { ' ', '.', ' ', ':', ' ', ':' },
          c_bblack,
          c_reset);
      } else {
        ss << fmt::alignedTable(
          { name,
            fmt::format("%.2Lf", time) + " " + units,
            std::to_string(tot_pct) + "%",
            fmt::format("%.2Lf", per_npart) + " " + units_npart,
            fmt::format("%.2Lf", per_ncells) + " " + units_ncells },
          { c_reset,
            c_yellow,
            ((tot_pct > 60) ? c_red : ((tot_pct > 40) ? c_yellow : c_green)),
            c_yellow,
            c_yellow },
          { -2, 37, 45, 55, -58 },
          { ' ', '.', ' ', ' ', ':' },
          c_bblack,
          c_reset);
      }
    }

    // total
    if (flags & Timer::PrintTotal) {
      std::string units   = "µs";
      auto        time    = std::get<0>(stats.at("Total"));
      const auto  var_pct = std::get<4>(stats.at("Total"));
      if (flags & Timer::AutoConvert) {
        convertTime(time, units);
      }
      if (multi_rank) {
        ss << fmt::alignedTable(
          { "Total",
            fmt::format("%.2Lf", time) + " " + units,
            std::to_string(var_pct) + "%" },
          { c_reset,
            c_blue,
            ((var_pct > 50) ? c_red : ((var_pct > 30) ? c_yellow : c_green)) },
          { 0, 37, -48 },
          { ' ', ' ', ' ' },
          c_bblack,
          c_reset);
      } else {
        ss << fmt::alignedTable(
          { "Total", fmt::format("%.2Lf", time) + " " + units },
          { c_reset, c_blue },
          { 0, 37 },
          { ' ', ' ' },
          c_bblack,
          c_reset);
      }
    }

    // print extra timers for output/checkpoint/prtlClear
    const std::vector<TimerFlags> extras_f { Timer::PrintPrtlClear,
                                             Timer::PrintOutput,
                                             Timer::PrintCheckpoint };
    for (auto i { 0u }; i < extras.size(); ++i) {
      const auto  name    = extras[i];
      const auto  active  = flags & extras_f[i];
      std::string units   = "µs";
      auto        time    = std::get<0>(stats.at(name));
      const auto  tot_pct = std::get<3>(stats.at(name));
      if (flags & Timer::AutoConvert) {
        convertTime(time, units);
      }
      ss << fmt::alignedTable({ name,
                                fmt::format("%.2Lf", time) + " " + units,
                                std::to_string(tot_pct) + "%" },
                              { (active ? c_reset : c_bblack),
                                (active ? c_byellow : c_bblack),
                                (active ? c_byellow : c_bblack) },
                              { -2, 37, 45 },
                              { ' ', '.', ' ' },
                              c_bblack,
                              c_reset);
    }
    return ss.str();
  }

} // namespace timer
