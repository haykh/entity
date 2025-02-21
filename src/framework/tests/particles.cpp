#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <Dimension D, ntt::Coord::type C>
void testParticles(const int&             index,
                   const std::string&     label,
                   const float&           m,
                   const float&           ch,
                   const std::size_t&     maxnpart,
                   const ntt::PrtlPusher& pusher,
                   const ntt::Cooling&    cooling,
                   const unsigned short&  npld = 0) {
  using namespace ntt;
  auto p = Particles<D, C>(index, label, m, ch, maxnpart, pusher, false, cooling, npld);
  raise::ErrorIf(p.index() != index, "Index mismatch", HERE);
  raise::ErrorIf(p.label() != label, "Label mismatch", HERE);
  raise::ErrorIf(p.mass() != m, "Mass mismatch", HERE);
  raise::ErrorIf(p.charge() != ch, "Charge mismatch", HERE);
  raise::ErrorIf(p.maxnpart() != maxnpart, "Max number of particles mismatch", HERE);
  raise::ErrorIf(p.pusher() != pusher, "Pusher mismatch", HERE);
  raise::ErrorIf(p.cooling() != cooling, "Cooling mismatch", HERE);
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

  if (npld > 0) {
    raise::ErrorIf(p.pld.extent(0) != maxnpart, "pld incorrectly allocated", HERE);
    raise::ErrorIf(p.pld.extent(1) != npld, "pld incorrectly allocated", HERE);
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
                                         PrtlPusher::BORIS,
                                         Cooling::SYNCHROTRON);
    testParticles<Dim::_2D, Coord::Cart>(2,
                                         "p+",
                                         100.0,
                                         -1.0,
                                         1000,
                                         PrtlPusher::VAY,
                                         Cooling::SYNCHROTRON);
    testParticles<Dim::_3D, Coord::Cart>(3,
                                         "ph",
                                         0.0,
                                         0.0,
                                         100,
                                         PrtlPusher::PHOTON,
                                         Cooling::NONE,
                                         5);
    testParticles<Dim::_2D, Coord::Sph>(4,
                                        "e+",
                                        1.0,
                                        1.0,
                                        100,
                                        PrtlPusher::BORIS,
                                        Cooling::NONE);
    testParticles<Dim::_2D, Coord::Qsph>(5,
                                         "e+",
                                         1.0,
                                         1.0,
                                         100,
                                         PrtlPusher::BORIS,
                                         Cooling::NONE,
                                         1);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
