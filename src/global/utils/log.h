/**
 * @file utils/log.h
 * @brief Global logging utilities with proper fences and MPI blockings
 * @implements
 *   - macro HERE
 *   - raise::Warning -> void
 *   - logger::Checkpoint -> void
 *   - info::Print -> void
 * @namespaces:
 *   - logger::
 *   - info::
 *   - raise::
 * @macros:
 *   - MPI_ENABLED
 *   - DEBUG
 */

#ifndef GLOBAL_LOG_H
#define GLOBAL_LOG_H

#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/colors.h"

#include <Kokkos_Core.hpp>
#include <plog/Log.h>

#include <iostream>
#include <string>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace raise {
  using namespace files;

  inline void Warning(const std::string& msg,
                      const std::string& file,
                      const std::string& func,
                      int                line,
                      bool               once = true) {
    if (once) {
      CallOnce(
        [](auto& msg, auto& file, auto& func, auto& line) {
          PLOGW_(LogFile) << "Warning: " << file << ":" << line << " @ " << func;
          PLOGW_(ErrFile) << "Warning: " << file << ":" << line << " @ " << func;
#if defined(MPI_ENABLED)
          int rank;
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          PLOGW_(ErrFile) << ": rank : " << rank;
#endif
          PLOGW_(ErrFile) << ": message : " << msg;
          PLOGW << msg;
          PLOGW << "see the `*.err` file for more details";
        },
        msg,
        file,
        func,
        line);
    } else {
      PLOGW_(LogFile) << "Warning: " << file << ":" << line << " @ " << func;
      PLOGW_(ErrFile) << "Warning: " << file << ":" << line << " @ " << func;
#if defined(MPI_ENABLED)
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      PLOGW_(ErrFile) << ": rank : " << rank;
#endif
      PLOGW_(ErrFile) << ": message : " << msg;
      PLOGW << msg;
      PLOGW << "see the `*.err` file for more details";
    }
  }
} // namespace raise

namespace logger {
  using namespace files;

  inline void Checkpoint(const std::string& file, const std::string& func, int line) {
#if defined(DEBUG)
    Kokkos::fence();
  #if defined(MPI_ENABLED)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif
    CallOnce(
      [](auto& file, auto& func, auto& line) {
        PLOGV_(LogFile) << "Checkpoint: " << file << ":" << line << " @ " << func;
      },
      file,
      func,
      line);
  }

  inline void Checkpoint(const std::string& msg,
                         const std::string& file,
                         const std::string& func,
                         int                line) {
#if defined(DEBUG)
    Kokkos::fence();
  #if defined(MPI_ENABLED)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif
    CallOnce(
      [](auto& msg, auto& file, auto& func, auto& line) {
        PLOGV_(LogFile) << "Checkpoint: " << file << ":" << line << " @ " << func;
        PLOGV_(LogFile) << " : message : " << msg;
      },
      msg,
      file,
      func,
      line);
  }

} // namespace logger

namespace info {
  using namespace files;

  inline void Print(const std::string& msg,
                    bool               colored = true,
                    bool               stdout  = true,
                    bool               once    = true) {
    auto msg_nocol = color::strip(msg);
    if (once) {
      CallOnce(
        [](auto& msg, auto& msg_nocol, auto& stdout, auto& colored) {
          PLOGN_(InfoFile) << msg_nocol << std::flush;
          if (stdout) {
            if (colored) {
              std::cout << msg << std::endl;
            } else {
              std::cout << msg_nocol << std::endl;
            }
          }
        },
        msg,
        msg_nocol,
        stdout,
        colored);
    } else {
      PLOGN_(InfoFile) << msg_nocol << std::flush;
      if (stdout) {
        if (colored) {
          std::cout << msg << std::endl;
        } else {
          std::cout << msg_nocol << std::endl;
        }
      }
    }
  }

} // namespace info

#endif // GLOBAL_LOG_H
