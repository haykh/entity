/**
 * @file enums.h
 * @brief Special enum variables describing the simulation
 * @implements
 *   - enum ntt::Coord             // Cart, Sph, Qsph
 *   - enum ntt::Metric            // Minkowski, Spherical, QSpherical,
 *                                    Kerr_Schild, QKerr_Schild, Kerr_Schild_0
 *   - enum ntt::SimEngine         // SRPIC, GRPIC
 *   - enum ntt::PrtlBC            // periodic, absorb, atmosphere, custom,
 *                                    reflect, horizon, axis, sync
 *   - enum ntt::FldsBC            // periodic, match, fixed, atmosphere,
 *                                    custom, horizon, axis, sync
 *   - enum ntt::PrtlPusher        // boris, vay, photon, none
 *   - enum ntt::Cooling           // synchrotron, none
 *   - enum ntt::FldsID            // e, dive, d, divd, b, h, j,
 *                                    a, t, rho, charge, n, nppc, custom
 * @namespaces:
 *   - ntt::
 * @note Enums of the same type can be compared with each other and with strings
 * @note
 * Comparison with strings is case-sensitive(!) since the latter are defined as
 * string literals example: "srpIC" != SimEngine::SRPIC
 * @note
 * To convert an enum to a string, use its std::string()
 * example: SimEngine(SimEngine::SRPIC).to_string() [return "srpic"]
 * @note
 * To check if a string is a valid option, use the contains() function
 * example: PrtlPusher::contains("vay") == true
 * @note
 * To get the proper enum instance from a string, use the pick() function
 * example: PrtlPusher::pick("vay") [returns PrtlPusher(PrtlPusher::VAY)]
 * @note
 * To get the total number of enum instances, use the total variable
 * example: Cooling::total == 2
 * @note
 * To iterate over all enum instances, use the variants array
 * example: for (Cooling c : Cooling::variants) { ... }
 */

#ifndef GLOBAL_ENUMS_H
#define GLOBAL_ENUMS_H

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include <cstring>
#include <string>

namespace ntt {

  namespace enums_hidden {
    template <typename T>
    constexpr auto basePick(const enum T::type arr[],
                            const char* const* arr_c,
                            const std::size_t  n,
                            const char*        elem) -> T {
      for (std::size_t i = 0; i < n; ++i) {
        if (strcmp(arr_c[i], elem) == 0) {
          return (T)(arr[i]);
        }
      }
      raise::Error(fmt::format("Invalid enum value: %s for %s", elem, T::label),
                   HERE);
      return T::INVALID;
    }

    template <typename T>
    constexpr auto baseContains(const char* const* arr_c,
                                const std::size_t  n,
                                const char*        elem) -> bool {
      for (std::size_t i = 0; i < n; ++i) {
        if (strcmp(arr_c[i], elem) == 0) {
          return true;
        }
      }
      return false;
    }

    template <class T>
    class BaseEnum {
    public:
      constexpr bool operator==(BaseEnum<T> other) const {
        return value == other.value;
      }

      constexpr bool operator!=(BaseEnum<T> other) const {
        return value != other.value;
      }

      constexpr bool operator==(uint8_t other) const {
        return value == other;
      }

      constexpr bool operator!=(uint8_t other) const {
        return value != other;
      }

      constexpr bool operator==(const char* other) const {
        return strcmp(T::lookup[value - 1], other) == 0;
      }

      constexpr bool operator!=(const char* other) const {
        return strcmp(T::lookup[value - 1], other) != 0;
      }

      static constexpr auto pick(const char* c) -> T {
        return basePick<T>(T::variants, T::lookup, T::total, fmt::toLower(c).c_str());
      }

      static constexpr auto contains(const char* c) -> bool {
        return baseContains<T>(T::lookup, T::total, c);
      }

      constexpr auto to_string() const -> const char* {
        return T::lookup[value - 1];
      }

      BaseEnum() = default;

      constexpr BaseEnum(uint8_t v) : value(v) {}

    protected:
      uint8_t value;
    };
  } // namespace enums_hidden

  struct Coord : public enums_hidden::BaseEnum<Coord> {
    static constexpr const char* label = "coord";

    enum type : uint8_t {
      INVALID = 0,
      Cart    = 1,
      Sph     = 2,
      Qsph    = 3,
    };

    constexpr Coord(uint8_t c) : enums_hidden::BaseEnum<Coord> { c } {}

    static constexpr type        variants[] = { Cart, Sph, Qsph };
    static constexpr const char* lookup[]   = { "cart", "sph", "qsph" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct Metric : public enums_hidden::BaseEnum<Metric> {
    static constexpr const char* label = "metric";

    enum type : uint8_t {
      INVALID       = 0,
      Minkowski     = 1,
      Spherical     = 2,
      QSpherical    = 3,
      Kerr_Schild   = 4,
      QKerr_Schild  = 5,
      Kerr_Schild_0 = 6,
    };

    constexpr Metric(uint8_t c) : enums_hidden::BaseEnum<Metric> { c } {}

    static constexpr type        variants[] = { Minkowski,    Spherical,
                                                QSpherical,   Kerr_Schild,
                                                QKerr_Schild, Kerr_Schild_0 };
    static constexpr const char* lookup[]   = { "minkowski",    "spherical",
                                                "qspherical",   "kerr_schild",
                                                "qkerr_schild", "kerr_schild_0" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct SimEngine : public enums_hidden::BaseEnum<SimEngine> {
    static constexpr const char* label = "sim_engine";

    enum type : uint8_t {
      INVALID = 0,
      SRPIC   = 1,
      GRPIC   = 2,
    };

    constexpr SimEngine(uint8_t c) : enums_hidden::BaseEnum<SimEngine> { c } {}

    static constexpr type        variants[] = { SRPIC, GRPIC };
    static constexpr const char* lookup[]   = { "srpic", "grpic" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct PrtlBC : public enums_hidden::BaseEnum<PrtlBC> {
    static constexpr const char* label = "prtl_bc";

    enum type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      ABSORB     = 2,
      ATMOSPHERE = 3,
      CUSTOM     = 4,
      REFLECT    = 5,
      HORIZON    = 6,
      AXIS       = 7,
      SYNC       = 8,
    };

    constexpr PrtlBC(uint8_t c) : enums_hidden::BaseEnum<PrtlBC> { c } {}

    static constexpr type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE, CUSTOM,
                                         REFLECT,  HORIZON, AXIS,       SYNC };
    static constexpr const char* lookup[] = { "periodic",   "absorb",
                                              "atmosphere", "custom",
                                              "reflect",    "horizon",
                                              "axis",       "sync" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct FldsBC : public enums_hidden::BaseEnum<FldsBC> {
    static constexpr const char* label = "flds_bc";

    enum type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      MATCH      = 2,
      FIXED      = 3,
      ATMOSPHERE = 4,
      CUSTOM     = 5,
      HORIZON    = 6,
      AXIS       = 7,
      SYNC       = 8, // <- SYNC means synchronization with other domains
    };

    constexpr FldsBC(uint8_t c) : enums_hidden::BaseEnum<FldsBC> { c } {}

    static constexpr type variants[] = { PERIODIC, MATCH,   FIXED, ATMOSPHERE,
                                         CUSTOM,   HORIZON, AXIS,  SYNC };
    static constexpr const char* lookup[] = { "periodic",   "match",  "fixed",
                                              "atmosphere", "custom", "horizon",
                                              "axis",       "sync" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct PrtlPusher : public enums_hidden::BaseEnum<PrtlPusher> {
    static constexpr const char* label = "prtl_pusher";

    enum type : uint8_t {
      INVALID = 0,
      BORIS   = 1,
      VAY     = 2,
      PHOTON  = 3,
      NONE    = 4,
    };

    constexpr PrtlPusher(uint8_t c)
      : enums_hidden::BaseEnum<PrtlPusher> { c } {}

    static constexpr type variants[] = { BORIS, VAY, PHOTON, NONE };
    static constexpr const char* lookup[] = { "boris", "vay", "photon", "none" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct Cooling : public enums_hidden::BaseEnum<Cooling> {
    static constexpr const char* label = "cooling";

    enum type : uint8_t {
      INVALID     = 0,
      SYNCHROTRON = 1,
      NONE        = 2,
    };

    constexpr Cooling(uint8_t c) : enums_hidden::BaseEnum<Cooling> { c } {}

    static constexpr type        variants[] = { SYNCHROTRON, NONE };
    static constexpr const char* lookup[]   = { "synchrotron", "none" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct FldsID : public enums_hidden::BaseEnum<FldsID> {
    static constexpr const char* label = "out_flds";

    enum type : uint8_t {
      INVALID = 0,
      E       = 1,
      divE    = 2,
      D       = 3,
      divD    = 4,
      B       = 5,
      H       = 6,
      J       = 7,
      A       = 8,
      T       = 9,
      Rho     = 10,
      Charge  = 11,
      N       = 12,
      Nppc    = 13,
      Custom  = 14,
    };

    constexpr FldsID(uint8_t c) : enums_hidden::BaseEnum<FldsID> { c } {}

    static constexpr type variants[]      = { E, divE, D,   divD,   B, H,    J,
                                              A, T,    Rho, Charge, N, Nppc, Custom };
    static constexpr const char* lookup[] = { "e",    "dive",  "d",      "divd",
                                              "b",    "h",     "j",      "a",
                                              "t",    "rho",   "charge", "n",
                                              "nppc", "custom" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

} // namespace ntt

#endif // GLOBAL_ENUMS_H
