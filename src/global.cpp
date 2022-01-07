#include "global.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
#include <cassert>
#include <string>
#include <iomanip>

namespace ntt {

  auto stringifySimulationType(SimulationType sim) -> std::string {
    switch (sim) {
    case SimulationType::PIC:
      return "PIC";
    case SimulationType::FORCE_FREE:
      return "FF";
    case SimulationType::MHD:
      return "MHD";
    default:
      return "N/A";
    }
  }

  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::OPEN:
      return "Open";
    case BoundaryCondition::USER:
      return "User";
    default:
      return "N/A";
    }
  }

  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string {
    switch (pusher) {
    case ParticlePusher::BORIS:
      return "Boris";
    case ParticlePusher::VAY:
      return "Vay";
    case ParticlePusher::PHOTON:
      return "Photon";
    default:
      return "N/A";
    }
  }

  template <>
  auto NTTRange<Dimension::ONE_D>(const long int (&i1)[1],
                                  const long int (&i2)[1]) -> RangeND<Dimension::ONE_D> {
    return Kokkos::RangePolicy<AccelExeSpace>(static_cast<range_t>(i1[0]), static_cast<range_t>(i2[0]));
  }

  template <>
  auto NTTRange<Dimension::TWO_D>(const long int (&i1)[2], const long int (&i2)[2]) -> RangeND<Dimension::TWO_D> {
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
      {static_cast<range_t>(i1[0]), static_cast<range_t>(i1[1])},
      {static_cast<range_t>(i2[0]), static_cast<range_t>(i2[1])});
  }
  template <>
  auto NTTRange<Dimension::THREE_D>(const long int (&i1)[3], const long int (&i2)[3]) -> RangeND<Dimension::THREE_D> {
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
      {static_cast<range_t>(i1[0]), static_cast<range_t>(i1[1]), static_cast<range_t>(i1[2])},
      {static_cast<range_t>(i2[0]), static_cast<range_t>(i2[1]), static_cast<range_t>(i2[2])});
  }

  } // namespace ntt

namespace plog {

  auto NTTFormatter::header() -> util::nstring { return util::nstring(); }
  auto NTTFormatter::format(const Record& record) -> util::nstring {
    util::nostringstream ss;
#ifdef DEBUG
    if (record.getSeverity() == plog::debug) {
      ss << PLOG_NSTR("\n") << record.getFunc() << PLOG_NSTR(" @ ") << record.getLine()
         << PLOG_NSTR("\n");
    }
#endif
    ss << std::setw(9) << std::left << severityToString(record.getSeverity()) << PLOG_NSTR(": ");
    ss << record.getMessage() << PLOG_NSTR("\n");
    return ss.str();
  }

} // namespace plog
