#include "global.h"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
  // #include <hip/hip_runtime.h>
#endif // MPI_ENABLED

void ntt::GlobalInitialize(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
#endif // MPI_ENABLED
}

void ntt::GlobalFinalize() {
#if defined(MPI_ENABLED)
  MPI_Finalize();
#endif // MPI_ENABLED
  Kokkos::finalize();
}
