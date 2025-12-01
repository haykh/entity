#include "enums.h"

#include <stdexcept>
#include <string>
#include <type_traits>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

using enum_str_t = const std::vector<std::string>;

template <typename T>
void checkEnum(const enum_str_t& all) {
  for (const auto& c : T::variants) {
    errorIf(not T::contains(T(c).to_string()),
            "Enum does not contain " + std::string(T(c).to_string()));
  }
  for (const auto& c : all) {
    errorIf(not T::contains(c.c_str()), "Enum does not contain " + std::string(c));
    errorIf(T::pick(c.c_str()) == T::INVALID,
            "Enum::pick(" + std::string(c) + ") == Enum::INVALID");
  }
  errorIf(all.size() != T::total, "Enum::total is incorrect");
}

auto main() -> int {
  using namespace ntt;
  static_assert(em::ex1 == 0);
  static_assert(em::ex2 == 1);
  static_assert(em::ex3 == 2);

  static_assert(em::dx1 == 0);
  static_assert(em::dx2 == 1);
  static_assert(em::dx3 == 2);

  static_assert(em::bx1 == 3);
  static_assert(em::bx2 == 4);
  static_assert(em::bx3 == 5);

  static_assert(em::hx1 == 3);
  static_assert(em::hx2 == 4);
  static_assert(em::hx3 == 5);

  static_assert(cur::jx1 == 0);
  static_assert(cur::jx2 == 1);
  static_assert(cur::jx3 == 2);

  static_assert(ParticleTag::dead == 0);
  static_assert(ParticleTag::alive == 1);

  static_assert(std::is_convertible_v<ParticleTag, short>);

  using enum_str_t = const std::vector<std::string>;

  enum_str_t all_coords  = { "cart", "sph", "qsph" };
  enum_str_t all_metrics = { "minkowski",   "spherical",    "qspherical",
                             "kerr_schild", "qkerr_schild", "kerr_schild_0" };
  enum_str_t all_simulation_engines = { "srpic", "grpic" };
  enum_str_t all_particle_bcs = { "periodic", "absorb",  "atmosphere", "custom",
                                  "reflect",  "horizon", "axis",       "sync" };
  enum_str_t all_fields_bcs   = { "periodic",   "match",     "fixed",
                                  "atmosphere", "custom",    "horizon",
                                  "axis",       "conductor", "sync" };
  enum_str_t all_particle_pushers = { "boris", "vay", "photon", "none" };
  enum_str_t all_coolings         = { "synchrotron", "compton", "none" };

  enum_str_t all_out_flds = { "e",      "dive", "d",    "divd", "b",
                              "h",      "j",    "a",    "t",    "rho",
                              "charge", "n",    "nppc", "v",    "custom" };

  enum_str_t all_out_stats = { "b^2", "e^2",    "exb", "j.e",   "t",
                               "rho", "charge", "n",   "npart", "custom" };

  checkEnum<Coord>(all_coords);
  checkEnum<Metric>(all_metrics);
  checkEnum<SimEngine>(all_simulation_engines);
  checkEnum<PrtlBC>(all_particle_bcs);
  checkEnum<FldsBC>(all_fields_bcs);
  checkEnum<PrtlPusher>(all_particle_pushers);
  checkEnum<Cooling>(all_coolings);
  checkEnum<FldsID>(all_out_flds);
  checkEnum<StatsID>(all_out_stats);

  return 0;
}
