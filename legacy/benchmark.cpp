#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "metrics/metric_base.h"
#include "metrics/minkowski.h"

#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <Kokkos_Random.hpp>

#include "framework/domain/communications.cpp"
#include "mpi.h"
#include "mpi-ext.h"

#define TIMER_START(label)                                                     \
  Kokkos::fence();                                                             \
  auto start_##label = std::chrono::high_resolution_clock::now();

#define TIMER_STOP(label)                                                        \
  Kokkos::fence();                                                               \
  auto stop_##label     = std::chrono::high_resolution_clock::now();             \
  auto duration_##label = std::chrono::duration_cast<std::chrono::microseconds>( \
                            stop_##label - start_##label)                        \
                            .count();                                            \
  std::cout << "Timer [" #label "]: " << duration_##label << " microseconds"     \
            << std::endl;

/*
  Test to check the performance of the new particle allocation scheme
    - Create a metadomain object main()
    - Set npart + initialize tags InitializeParticleArrays()
    - 'Push' the particles by randomly updating the tags PushParticles()
    - Communicate particles to neighbors and time the communication
    - Compute the time taken for best of N iterations for the communication
 */
using namespace ntt;

// Set npart and set the particle tags to alive
template <SimEngine::type S, class M>
void InitializeParticleArrays(Domain<S, M>& domain, const int npart) {
  raise::ErrorIf(npart > domain.species[0].maxnpart(),
                 "Npart cannot be greater than maxnpart",
                 HERE);
  const auto nspecies = domain.species.size();
  for (int i_spec = 0; i_spec < nspecies; i_spec++) {
    domain.species[i_spec].set_npart(npart);
    domain.species[i_spec].SyncHostDevice();
    auto& this_tag = domain.species[i_spec].tag;
    Kokkos::parallel_for(
      "Initialize particles",
      npart,
      Lambda(const std::size_t i) { this_tag(i) = ParticleTag::alive; });
  }
  return;
}

// Randomly reassign tags to particles for a fraction of particles
template <SimEngine::type S, class M>
void PushParticles(Domain<S, M>& domain,
                   const double  send_frac,
                   const int     seed_ind,
                   const int     seed_tag) {
  raise::ErrorIf(send_frac > 1.0, "send_frac cannot be greater than 1.0", HERE);
  const auto nspecies = domain.species.size();
  for (int i_spec = 0; i_spec < nspecies; i_spec++) {
    domain.species[i_spec].set_unsorted();
    const auto nparticles         = domain.species[i_spec].npart();
    const auto nparticles_to_send = static_cast<int>(send_frac * nparticles);
    // Generate random indices to send
    // Kokkos::Random_XorShift64_Pool<> random_pool(seed_ind);
    Kokkos::View<int*> indices_to_send("indices_to_send", nparticles_to_send);
    Kokkos::fill_random(indices_to_send, domain.random_pool, 0, nparticles);
    // Generate random tags to send
    // Kokkos::Random_XorShift64_Pool<> random_pool_tag(seed_tag);
    Kokkos::View<int*> tags_to_send("tags_to_send", nparticles_to_send);
    Kokkos::fill_random(tags_to_send,
                        domain.random_pool,
                        0,
                        domain.species[i_spec].ntags());
    auto& this_tag = domain.species[i_spec].tag;
    Kokkos::parallel_for(
      "Push particles",
      nparticles_to_send,
      Lambda(const std::size_t i) {
        auto prtl_to_send      = indices_to_send(i);
        auto tag_to_send       = tags_to_send(i);
        this_tag(prtl_to_send) = tag_to_send;
      });
    domain.species[i_spec].npart_per_tag();
    domain.species[i_spec].SyncHostDevice();
  }
  return;
}

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  {
    /*
      MPI checks
    */
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

  /*
    Test to send and receive Kokkos arrays
  */
  int sender_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &sender_rank);

  int neighbor_rank = 0;
  if (sender_rank == 0) {
    neighbor_rank = 1;
  }
  else if (sender_rank == 1) {
    neighbor_rank = 0;
  }
  else {
    raise::Error("This test is only for 2 ranks", HERE);
  }
  Kokkos::View<int*> send_array("send_array", 10);
  Kokkos::View<int*> recv_array("recv_array", 10);
  if (sender_rank == 0) {
    Kokkos::deep_copy(send_array, 10);
  }
  else {
    Kokkos::deep_copy(send_array, 20);
  }

  auto send_array_host = Kokkos::create_mirror_view(send_array);
  Kokkos::deep_copy(send_array_host, send_array);
  auto host_recv_array = Kokkos::create_mirror_view(recv_array);

  MPI_Sendrecv(send_array.data(), send_array.extent(0), MPI_INT, neighbor_rank, 0,
               recv_array.data(), recv_array.extent(0), MPI_INT, neighbor_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Print the received array
  Kokkos::deep_copy(host_recv_array, recv_array);
  for (int i = 0; i < 10; ++i) {
    printf("Rank %d: Received %d\n", sender_rank, host_recv_array(i));
  }


    std::cout << "Constructing the domain" << std::endl;
    // Create a Metadomain object
    const unsigned int     ndomains             = 2;
    const std::vector<int> global_decomposition = {
      {-1, -1, -1}
    };
    const std::vector<std::size_t> global_ncells = { 32, 32, 32 };
    const boundaries_t<real_t>     global_extent = {
      {0.0, 3.0},
      {0.0, 3.0},
      {0.0, 3.0}
    };
    const boundaries_t<FldsBC> global_flds_bc = {
      {FldsBC::PERIODIC, FldsBC::PERIODIC},
      {FldsBC::PERIODIC, FldsBC::PERIODIC},
      {FldsBC::PERIODIC, FldsBC::PERIODIC}
    };
    const boundaries_t<PrtlBC> global_prtl_bc = {
      {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
      {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
      {PrtlBC::PERIODIC, PrtlBC::PERIODIC}
    };
    const std::map<std::string, real_t> metric_params = {};
    const int    maxnpart           = argc > 1 ? std::stoi(argv[1]) : 1000;
    const double npart_to_send_frac = 0.01;
    const int npart = static_cast<int>(maxnpart * (1 - 2 * npart_to_send_frac));
    auto      species = ntt::ParticleSpecies(1u,
                                        "test_e",
                                        1.0f,
                                        1.0f,
                                        maxnpart,
                                        ntt::PrtlPusher::BORIS,
                                        false,
                                        ntt::Cooling::NONE);
    auto metadomain = Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>(
      ndomains,
      global_decomposition,
      global_ncells,
      global_extent,
      global_flds_bc,
      global_prtl_bc,
      metric_params,
      { species });

    const auto local_subdomain_idx = metadomain.l_subdomain_indices()[0];
    auto       local_domain = metadomain.subdomain_ptr(local_subdomain_idx);
    auto       timers = timer::Timers { { "Communication" }, nullptr, false };
    InitializeParticleArrays(*local_domain, npart);
    // Timers for both the communication routines
    auto total_time_elapsed_old = 0;
    auto total_time_elapsed_new = 0;

    int seed_ind = 0;
    int seed_tag = 1;
    Kokkos::fence();

    for (int i = 0; i < 10; ++i) {
      {
        // Push
        seed_ind += 2;
        seed_tag += 3;
        PushParticles(*local_domain, npart_to_send_frac, seed_ind, seed_tag);
        // Sort new
        Kokkos::fence();
        auto start_new = std::chrono::high_resolution_clock::now();
        metadomain.CommunicateParticlesBuffer(*local_domain, &timers);
        auto stop_new = std::chrono::high_resolution_clock::now();
        auto duration_new = std::chrono::duration_cast<std::chrono::microseconds>(
                              stop_new - start_new)
                              .count();
        total_time_elapsed_new += duration_new;
        Kokkos::fence();
      }
      {
        // Push
        seed_ind += 2;
        seed_tag += 3;
        PushParticles(*local_domain, npart_to_send_frac, seed_ind, seed_tag);
        // Sort old
        Kokkos::fence();
        auto start_old = std::chrono::high_resolution_clock::now();
        metadomain.CommunicateParticles(*local_domain, &timers);
        auto stop_old = std::chrono::high_resolution_clock::now();
        auto duration_old = std::chrono::duration_cast<std::chrono::microseconds>(
                              stop_old - start_old)
                              .count();
        total_time_elapsed_old += duration_old;
        Kokkos::fence();
      }
    }
    printf("Total time elapsed for old: %f us : %f us/prtl\n",
           total_time_elapsed_old / 10.0,
           total_time_elapsed_old / 10.0 * 1000 / npart);
    printf("Total time elapsed for new: %f us : %f us/prtl\n",
           total_time_elapsed_new / 10.0,
           total_time_elapsed_new / 10.0 * 1000 / npart);
  }
  GlobalFinalize();
  return 0;
}

/*
  Buggy behavior:
  Consider a single domain with a single mpi rank
  Particle tag arrays is set to [0, 0, 1, 1, 2, 3, ...] for a single domain
  CommunicateParticles() discounts all the dead particles and reassigns the
  other tags to alive
  CommunicateParticlesBuffer() only keeps the ParticleTag::Alive particles
  and discounts the rest
*/
