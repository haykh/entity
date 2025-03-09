#include "utils/diag.h"

#include "global.h"

#include "utils/colors.h"
#include "utils/formatting.h"
#include "utils/progressbar.h"
#include "utils/timer.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace diag {
  auto npart_stats(
    std::size_t npart,
    std::size_t maxnpart) -> std::vector<std::pair<std::size_t, unsigned short>> {
    auto stats = std::vector<std::pair<std::size_t, unsigned short>>();
#if !defined(MPI_ENABLED)
    stats.push_back(
      { npart,
        static_cast<unsigned short>(
          100.0f * static_cast<float>(npart) / static_cast<float>(maxnpart)) });
#else
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::size_t> mpi_npart(size, 0);
    std::vector<std::size_t> mpi_maxnpart(size, 0);
    MPI_Gather(&npart,
               1,
               mpi::get_type<std::size_t>(),
               mpi_npart.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    MPI_Gather(&maxnpart,
               1,
               mpi::get_type<std::size_t>(),
               mpi_maxnpart.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank != MPI_ROOT_RANK) {
      return stats;
    }
    auto tot_npart     = std::accumulate(mpi_npart.begin(), mpi_npart.end(), 0);
    const auto max_idx = std::distance(
      mpi_npart.begin(),
      std::max_element(mpi_npart.begin(), mpi_npart.end()));
    const auto min_idx = std::distance(
      mpi_npart.begin(),
      std::min_element(mpi_npart.begin(), mpi_npart.end()));
    stats.push_back({ tot_npart, 0u });
    stats.push_back({ mpi_npart[min_idx],
                      static_cast<unsigned short>(
                        100.0f * static_cast<float>(mpi_npart[min_idx]) /
                        static_cast<float>(mpi_maxnpart[min_idx])) });
    stats.push_back({ mpi_npart[max_idx],
                      static_cast<unsigned short>(
                        100.0f * static_cast<float>(mpi_npart[max_idx]) /
                        static_cast<float>(mpi_maxnpart[max_idx])) });
#endif
    return stats;
  }

  void printDiagnostics(std::size_t                     step,
                        std::size_t                     tot_steps,
                        long double                     time,
                        long double                     dt,
                        timer::Timers&                  timers,
                        pbar::DurationHistory&          time_history,
                        std::size_t                     ncells,
                        const std::vector<std::string>& species_labels,
                        const std::vector<std::size_t>& species_npart,
                        const std::vector<std::size_t>& species_maxnpart,
                        bool                            print_prtl_clear,
                        bool                            print_output,
                        bool                            print_checkpoint,
                        bool                            print_colors) {
    DiagFlags  diag_flags  = Diag::Default;
    TimerFlags timer_flags = Timer::Default;
    if (not print_colors) {
      diag_flags ^= Diag::Colorful;
    }
    if (species_labels.size() == 0) {
      diag_flags ^= Diag::Species;
    }
    if (print_prtl_clear) {
      timer_flags |= Timer::PrintPrtlClear;
    }
    if (print_output) {
      timer_flags |= Timer::PrintOutput;
    }
    if (print_checkpoint) {
      timer_flags |= Timer::PrintCheckpoint;
    }

    std::stringstream ss;

    const auto c_red    = color::get_color("red");
    const auto c_yellow = color::get_color("yellow");
    const auto c_green  = color::get_color("green");
    const auto c_bgreen = color::get_color("bgreen");
    const auto c_bblack = color::get_color("bblack");
    const auto c_reset  = color::get_color("reset");

    // basic info
    CallOnce([&]() {
      ss << fmt::alignedTable(
        { "Step:", fmt::format("%lu", step), fmt::format("[of %lu]", tot_steps) },
        { c_reset, c_bgreen, c_bblack },
        { 0, -6, -15 },
        { ' ', ' ', ' ' },
        c_bblack,
        c_reset);

      ss << fmt::alignedTable(
        { "Time:", fmt::format("%.4Lf", time), fmt::format("[Δt = %.4Lf]", dt) },
        { c_reset, c_bgreen, c_bblack },
        { 0, -6, -15 },
        { ' ', ' ', ' ' },
        c_bblack,
        c_reset);
    });

    // substep timers
    if (diag_flags & Diag::Timers) {
      const auto total_npart = std::accumulate(species_npart.begin(),
                                               species_npart.end(),
                                               0);
      const auto timer_diag = timers.printAll(timer_flags, total_npart, ncells);
      CallOnce([&]() {
        ss << std::endl << timer_diag << std::endl;
      });
    }

    // particle counts
    if (diag_flags & Diag::Species) {
#if defined(MPI_ENABLED)
      CallOnce([&]() {
        ss << fmt::alignedTable(
          { "[PARTICLE SPECIES]", "[TOTAL]", "[% MIN", "MAX]", "[MIN", "MAX]" },
          { c_bblack, c_bblack, c_bblack, c_bblack, c_bblack, c_bblack },
          { 0, 37, 45, -48, 63, -66 },
          { ' ', ' ', ' ', ':', ' ', ':' },
          c_bblack,
          c_reset);
      });
#else
      CallOnce([&]() {
        ss << fmt::alignedTable({ "[PARTICLE SPECIES]", "[TOTAL]", "[% TOT]" },
                                { c_bblack, c_bblack, c_bblack },
                                { 0, 37, 45 },
                                { ' ', ' ', ' ' },
                                c_bblack,
                                c_reset);
      });
#endif
      for (std::size_t i = 0; i < species_labels.size(); ++i) {
        const auto part_stats = npart_stats(species_npart[i], species_maxnpart[i]);
        if (part_stats.size() == 0) {
          continue;
        }
        const auto tot_npart = part_stats[0].first;
#if defined(MPI_ENABLED)
        const auto min_npart = part_stats[1].first;
        const auto min_pct   = part_stats[1].second;
        const auto max_npart = part_stats[2].first;
        const auto max_pct   = part_stats[2].second;
        ss << fmt::alignedTable(
          {
            fmt::format("species %2lu (%s)", i, species_labels[i].c_str()),
            tot_npart > 9999 ? fmt::format("%.2Le", (long double)tot_npart)
                             : std::to_string(tot_npart),
            std::to_string(min_pct) + "%",
            std::to_string(max_pct) + "%",
            min_npart > 9999 ? fmt::format("%.2Le", (long double)min_npart)
                             : std::to_string(min_npart),
            max_npart > 9999 ? fmt::format("%.2Le", (long double)max_npart)
                             : std::to_string(max_npart),
          },
          {
            c_reset,
            c_reset,
            (min_pct > 80) ? c_red : ((min_pct > 50) ? c_yellow : c_green),
            (max_pct > 80) ? c_red : ((max_pct > 50) ? c_yellow : c_green),
            c_reset,
            c_reset,
          },
          { -2, 37, 45, -48, 63, -66 },
          { ' ', '.', ' ', ':', ' ', ':' },
          c_bblack,
          c_reset);
#else
        const auto tot_pct = part_stats[0].second;
        ss << fmt::alignedTable(
          {
            fmt::format("species %2lu (%s)", i, species_labels[i].c_str()),
            tot_npart > 9999 ? fmt::format("%.2Le", (long double)tot_npart)
                             : std::to_string(tot_npart),
            std::to_string(tot_pct) + "%",
          },
          {
            c_reset,
            c_reset,
            (tot_pct > 80) ? c_red : ((tot_pct > 50) ? c_yellow : c_green),
          },
          { -2, 37, 45 },
          { ' ', '.', ' ' },
          c_bblack,
          c_reset);
#endif
      }
      CallOnce([&]() {
        ss << std::endl;
      });
    }

    // progress bar
    if (diag_flags & Diag::Progress) {
      const auto progbar = pbar::ProgressBar(time_history, step, tot_steps, diag_flags);
      CallOnce([&]() {
        ss << progbar;
      });
    }

    // separator
    CallOnce([&]() {
      ss << std::setw(80) << std::setfill('.') << "" << std::endl << std::endl;
    });

    std::cout << ((diag_flags & Diag::Colorful) ? ss.str()
                                                : color::strip(ss.str()));
  }
} // namespace diag
