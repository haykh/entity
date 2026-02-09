#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include <iostream>
#include <string>

template <Dimension D, ntt::Coord::type C>
void testParticles(
  int                      index,
  const std::string&       label,
  float                    m,
  float                    ch,
  std::size_t              maxnpart,
  ntt::ParticlePusherFlags pusher,
  bool                     use_tracking,
  ntt::RadiativeDragFlags  radiative_drag_flags,
  ntt::EmissionTypeFlag    emission_policy_flag = ntt::EmissionType::NONE,
  unsigned short           npld_r               = 0,
  unsigned short           npld_i               = 0) {
  using namespace ntt;
  auto p = Particles<D, C>(index,
                           label,
                           m,
                           ch,
                           maxnpart,
                           pusher,
                           use_tracking,
                           radiative_drag_flags,
                           emission_policy_flag,
                           npld_r,
                           npld_i);
  raise::ErrorIf(p.index() != index, "Index mismatch", HERE);
  raise::ErrorIf(p.label() != label, "Label mismatch", HERE);
  raise::ErrorIf(p.mass() != m, "Mass mismatch", HERE);
  raise::ErrorIf(p.charge() != ch, "Charge mismatch", HERE);
  raise::ErrorIf(p.maxnpart() != maxnpart, "Max number of particles mismatch", HERE);
  raise::ErrorIf(p.pusher() != pusher, "Pusher mismatch", HERE);
  raise::ErrorIf(p.radiative_drag_flags() != radiative_drag_flags,
                 "Radiative drag mismatch",
                 HERE);
  raise::ErrorIf(p.emission_policy_flag() != emission_policy_flag,
                 "Emission policy mismatch",
                 HERE);
  raise::ErrorIf(p.npart() != 0, "Number of particles mismatch", HERE);

  raise::ErrorIf(p.i1.extent(0) != maxnpart, "i1 incorrectly allocated", HERE);
  raise::ErrorIf(p.dx1.extent(0) != maxnpart, "dx1 incorrectly allocated", HERE);
  raise::ErrorIf(p.i1_prev.extent(0) != maxnpart,
                 "i1_prev incorrectly allocated",
                 HERE);
  raise::ErrorIf(p.dx1_prev.extent(0) != maxnpart,
                 "dx1_prev incorrectly allocated",
                 HERE);
  raise::ErrorIf(p.ux1.extent(0) != maxnpart, "ux1 incorrectly allocated", HERE);
  raise::ErrorIf(p.ux2.extent(0) != maxnpart, "ux2 incorrectly allocated", HERE);
  raise::ErrorIf(p.ux3.extent(0) != maxnpart, "ux3 incorrectly allocated", HERE);

  raise::ErrorIf(p.tag.extent(0) != maxnpart, "tag incorrectly allocated", HERE);
  raise::ErrorIf(p.weight.extent(0) != maxnpart, "weight incorrectly allocated", HERE);

  if (npld_r > 0) {
    raise::ErrorIf(p.pld_r.extent(0) != maxnpart, "pld_r incorrectly allocated", HERE);
    raise::ErrorIf(p.pld_r.extent(1) != npld_r, "pld_r incorrectly allocated", HERE);
  }
  if (npld_i > 0) {
    raise::ErrorIf(p.pld_i.extent(0) != maxnpart, "pld_i incorrectly allocated", HERE);
    raise::ErrorIf(p.pld_i.extent(1) != npld_i, "pld_i incorrectly allocated", HERE);
  }

  if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
    raise::ErrorIf(p.i2.extent(0) != maxnpart, "i2 incorrectly allocated", HERE);
    raise::ErrorIf(p.dx2.extent(0) != maxnpart, "dx2 incorrectly allocated", HERE);

    raise::ErrorIf(p.i2_prev.extent(0) != maxnpart,
                   "i2_prev incorrectly allocated",
                   HERE);
    raise::ErrorIf(p.dx2_prev.extent(0) != maxnpart,
                   "dx2_prev incorrectly allocated",
                   HERE);
  } else {
    raise::ErrorIf(p.i2.extent(0) != 0, "i2 incorrectly allocated", HERE);
    raise::ErrorIf(p.dx2.extent(0) != 0, "dx2 incorrectly allocated", HERE);

    raise::ErrorIf(p.i2_prev.extent(0) != 0, "i2_prev incorrectly allocated", HERE);
    raise::ErrorIf(p.dx2_prev.extent(0) != 0, "dx2_prev incorrectly allocated", HERE);
  }
  if constexpr (D == Dim::_3D) {
    raise::ErrorIf(p.i3.extent(0) != maxnpart, "i3 incorrectly allocated", HERE);
    raise::ErrorIf(p.dx3.extent(0) != maxnpart, "dx3 incorrectly allocated", HERE);

    raise::ErrorIf(p.i3_prev.extent(0) != maxnpart,
                   "i3_prev incorrectly allocated",
                   HERE);
    raise::ErrorIf(p.dx3_prev.extent(0) != maxnpart,
                   "dx3_prev incorrectly allocated",
                   HERE);
  } else {
    raise::ErrorIf(p.i3.extent(0) != 0, "i3 incorrectly allocated", HERE);
    raise::ErrorIf(p.dx3.extent(0) != 0, "dx3 incorrectly allocated", HERE);

    raise::ErrorIf(p.i3_prev.extent(0) != 0, "i3_prev incorrectly allocated", HERE);
    raise::ErrorIf(p.dx3_prev.extent(0) != 0, "dx3_prev incorrectly allocated", HERE);
  }

  if ((D == Dim::_2D) && (C != Coord::Cart)) {
    raise::ErrorIf(p.phi.extent(0) != maxnpart, "phi incorrectly allocated", HERE);
  } else {
    raise::ErrorIf(p.phi.extent(0) != 0, "phi incorrectly allocated", HERE);
  }
}

auto main(int argc, char** argv) -> int {
  Kokkos::initialize(argc, argv);
  try {
    using namespace ntt;
    testParticles<Dim::_1D, Coord::Cart>(1,
                                         "e-",
                                         1.0,
                                         -1.0,
                                         100,
                                         ParticlePusher::BORIS,
                                         false,
                                         RadiativeDrag::SYNCHROTRON);
    testParticles<Dim::_2D, Coord::Cart>(2,
                                         "p+",
                                         100.0,
                                         -1.0,
                                         1000,
                                         ParticlePusher::VAY,
                                         true,
                                         RadiativeDrag::SYNCHROTRON |
                                           RadiativeDrag::COMPTON,
                                         EmissionType::SYNCHROTRON,
                                         2,
                                         1);
    testParticles<Dim::_3D, Coord::Cart>(3,
                                         "ph",
                                         0.0,
                                         0.0,
                                         100,
                                         ParticlePusher::PHOTON,
                                         false,
                                         RadiativeDrag::NONE,
                                         EmissionType::COMPTON,
                                         5);
    testParticles<Dim::_2D, Coord::Sph>(4,
                                        "e+",
                                        1.0,
                                        1.0,
                                        100,
                                        ParticlePusher::BORIS,
                                        true,
                                        RadiativeDrag::NONE,
                                        EmissionType::NONE,
                                        2,
                                        3);
    testParticles<Dim::_2D, Coord::Qsph>(5,
                                         "e+",
                                         1.0,
                                         1.0,
                                         100,
                                         ParticlePusher::BORIS,
                                         false,
                                         RadiativeDrag::NONE,
                                         EmissionType::NONE,
                                         1,
                                         2);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
