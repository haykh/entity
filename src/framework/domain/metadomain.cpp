#include "framework/domain/metadomain.h"

#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/tools.h"

#include "framework/domain/domain.h"
#include "framework/domain/domain.h"
#include "framework/specialization_registry.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  Metadomain<S, M>::Metadomain(unsigned int            global_ndomains,
                               const std::vector<int>& global_decomposition,
                               const std::vector<ncells_t>& global_ncells,
                               const boundaries_t<real_t>&  global_extent,
                               const boundaries_t<FldsBC>&  global_flds_bc,
                               const boundaries_t<PrtlBC>&  global_prtl_bc,
                               const std::map<std::string, real_t>& metric_params,
                               const std::vector<ParticleSpecies>& species_params)
    : g_ndomains { global_ndomains }
    , g_decomposition { global_decomposition }
    , g_mesh { global_ncells, global_extent, metric_params, global_flds_bc, global_prtl_bc }
    , g_metric_params { metric_params }
    , g_species_params { species_params } {
#if defined(MPI_ENABLED)
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    raise::ErrorIf(global_ndomains != (unsigned int)g_mpi_size,
                   "Exactly 1 domain per MPI rank is allowed",
                   HERE);
#endif
    initialValidityCheck();

    createEmptyDomains();
    redefineNeighbors();
    redefineBoundaries();

    finalValidityCheck();
    metricCompatibilityCheck();
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::initialValidityCheck() const {
    // ensure everything has the correct shape
    raise::ErrorIf(g_decomposition.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_decomposition",
                   HERE);
    raise::ErrorIf(g_mesh.n_active().size() != (std::size_t)D,
                   "Invalid number of dimensions in g_mesh.n_all()",
                   HERE);
    raise::ErrorIf(g_mesh.extent().size() != (std::size_t)D,
                   "Invalid number of dimensions in g_mesh.extent()",
                   HERE);
    raise::ErrorIf(g_mesh.flds_bc().size() != (std::size_t)D,
                   "Invalid number of dimensions in g_mesh.flds_bc()",
                   HERE);
    raise::ErrorIf(g_mesh.prtl_bc().size() != (std::size_t)D,
                   "Invalid number of dimensions in g_mesh.prtl_bc()",
                   HERE);
#if defined(MPI_ENABLED)
    int init_flag;
    int status = MPI_Initialized(&init_flag);
    raise::ErrorIf((status != MPI_SUCCESS) || (init_flag != 1),
                   "MPI not initialized",
                   HERE);
    raise::ErrorIf((unsigned int)g_mpi_size != g_ndomains,
                   "ndomains != g_mpi_size is not implemented with MPI",
                   HERE);

#else // not MPI_ENABLED
  #if !defined(DEBUG)
    raise::ErrorIf(g_ndomains != 1,
                   "ndomains > 1 is not implemented for non-DEBUG purposes",
                   HERE);
  #endif
#endif // MPI_ENABLED
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::createEmptyDomains() {
    /* decompose and compute cell & domain offsets ------------------------ */
    auto d_ncells = tools::Decompose(g_ndomains, g_mesh.n_active(), g_decomposition);
    raise::ErrorIf(d_ncells.size() != (std::size_t)D,
                   "Invalid number of dimensions received",
                   HERE);
    auto d_offset_ncells = std::vector<std::vector<ncells_t>> {};
    auto d_offset_ndoms  = std::vector<std::vector<unsigned int>> {};
    for (auto& d : d_ncells) {
      g_ndomains_per_dim.push_back(d.size());
      auto offset_ncell = std::vector<ncells_t> { 0 };
      auto offset_ndom  = std::vector<unsigned int> { 0 };
      for (auto i { 1u }; i < d.size(); ++i) {
        auto di = d[i - 1];
        offset_ncell.push_back(offset_ncell.back() + di);
        offset_ndom.push_back(offset_ndom.back() + 1);
      }
      d_offset_ncells.push_back(offset_ncell);
      d_offset_ndoms.push_back(offset_ndom);
    }

    /* compute tensor products of the domain decompositions --------------- */
    // works similar to np.meshgrid()
    const auto domain_ncells        = tools::TensorProduct<ncells_t>(d_ncells);
    const auto domain_offset_ncells = tools::TensorProduct<ncells_t>(
      d_offset_ncells);
    const auto domain_offset_ndoms = tools::TensorProduct<unsigned int>(
      d_offset_ndoms);

    g_domain_offsets = domain_offset_ndoms;

    /* create the domains ------------------------------------------------- */
    if (not g_subdomains.empty()) {
      g_subdomains.clear();
    }
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      auto                 l_offset_ndomains = domain_offset_ndoms[idx];
      auto                 l_ncells          = domain_ncells[idx];
      auto                 l_offset_ncells   = domain_offset_ncells[idx];
      boundaries_t<real_t> l_extent;
      coord_t<D>           low_corner_Code { ZERO }, up_corner_Code { ZERO };
      coord_t<D>           low_corner_Phys { ZERO }, up_corner_Phys { ZERO };
      for (auto d { 0u }; d < D; d++) {
        low_corner_Code[d] = (real_t)l_offset_ncells[d];
        up_corner_Code[d]  = (real_t)(l_offset_ncells[d] + l_ncells[d]);
      }
      g_mesh.metric.template convert<Crd::Cd, Crd::Ph>(low_corner_Code,
                                                       low_corner_Phys);
      g_mesh.metric.template convert<Crd::Cd, Crd::Ph>(up_corner_Code,
                                                       up_corner_Phys);
      for (auto d { 0 }; d < (short)D; ++d) {
        l_extent.push_back({ low_corner_Phys[d], up_corner_Phys[d] });
      }

#if defined(MPI_ENABLED)
      // !TODO: need to change to support multiple domains per rank
      // assuming ONE local subdomain
      const auto local = ((int)idx == g_mpi_rank);
      if (not local) {
        g_subdomains.emplace_back(false,
                                  idx,
                                  l_offset_ndomains,
                                  l_offset_ncells,
                                  l_ncells,
                                  l_extent,
                                  g_metric_params,
                                  g_species_params);
      } else {
        g_subdomains.emplace_back(idx,
                                  l_offset_ndomains,
                                  l_offset_ncells,
                                  l_ncells,
                                  l_extent,
                                  g_metric_params,
                                  g_species_params);
      }
      g_subdomains.back().set_mpi_rank(idx);
      if (g_subdomains.back().mpi_rank() == g_mpi_rank) {
        g_local_subdomain_indices.push_back(idx);
      }
#else  // not MPI_ENABLED
      g_subdomains.emplace_back(idx,
                                l_offset_ndomains,
                                l_offset_ncells,
                                l_ncells,
                                l_extent,
                                g_metric_params,
                                g_species_params);
      g_local_subdomain_indices.push_back(idx);
#endif // MPI_ENABLED
      g_domain_offset2index[l_offset_ndomains] = idx;
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::redefineNeighbors() {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      // offset of the subdomain[idx]
      auto&      current_domain = g_subdomains[idx];
      const auto current_offset = current_domain.offset_ndomains();
      for (const auto& direction : dir::Directions<D>::all) {
        // find the neighbor by its offset in specific direction
        auto nghbr_offset = current_offset;
        for (auto d { 0 }; d < (short)D; ++d) { // loop over all dimensions
          auto dir = direction[d];
          // find the neighbor offset in the direction dir, dimension d
          if ((dir == -1) && (current_offset[d] == 0)) {
            // left edge
            // rollover from left wall
            nghbr_offset[d] = g_ndomains_per_dim[d] - 1;
          } else if ((dir == 1) &&
                     (current_offset[d] == g_ndomains_per_dim[d] - 1)) {
            // rollover from right wall
            nghbr_offset[d] = 0;
          } else {
            // otherwise just shift the offset in the right direction
            // to get the neighbor
            nghbr_offset[d] += dir;
          }
        }
        // pointer to the neighbor candidate
        raise::ErrorIf(g_domain_offset2index.find(nghbr_offset) ==
                         g_domain_offset2index.end(),
                       "Neighbor candidate not found",
                       HERE);
        const auto idx = g_domain_offset2index.at(nghbr_offset);
        raise::ErrorIf(idx >= g_subdomains.size(),
                       "Neighbor candidate not found",
                       HERE);
        current_domain.set_neighbor_idx(direction, idx);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::redefineBoundaries() {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      // offset of the subdomain[idx]
      auto&      current_domain = g_subdomains[idx];
      const auto current_offset = current_domain.offset_ndomains();
      for (auto direction : dir::Directions<D>::orth) {
        FldsBC flds_bc { FldsBC::INVALID };
        PrtlBC prtl_bc { PrtlBC::INVALID };
        for (auto d { 0 }; d < (short)D; ++d) {
          auto dir = direction[d];
          if (dir == -1) {
            if (current_offset[d] == 0) {
              // left edge
              flds_bc = g_mesh.flds_bc()[d].first;
              prtl_bc = g_mesh.prtl_bc()[d].first;
            } else {
              // not left edge
              flds_bc = FldsBC::SYNC;
              prtl_bc = PrtlBC::SYNC;
            }
            break;
          } else if (dir == 1) {
            if (current_offset[d] == g_ndomains_per_dim[d] - 1) {
              // right edge
              flds_bc = g_mesh.flds_bc()[d].second;
              prtl_bc = g_mesh.prtl_bc()[d].second;
            } else {
              // not right edge
              flds_bc = FldsBC::SYNC;
              prtl_bc = PrtlBC::SYNC;
            }
            break;
          }
        }
        // if sending to a different domain & periodic, then sync
        const auto& neighbor = g_subdomains[current_domain.neighbor_idx_in(direction)];
        if (neighbor.index() != idx) {
          if (flds_bc == FldsBC::PERIODIC) {
            flds_bc = FldsBC::SYNC;
          }
          if (prtl_bc == PrtlBC::PERIODIC) {
            prtl_bc = PrtlBC::SYNC;
          }
        }
        current_domain.mesh.set_flds_bc(direction, flds_bc);
        current_domain.mesh.set_prtl_bc(direction, prtl_bc);
      }
      // setting boundaries in non-orthogonal (corner) directions
      for (auto direction : dir::Directions<D>::all) {
        auto assoc_orth = direction.get_assoc_orth();
        if (assoc_orth.size() == 1) {
          // skip the orthogonal directions
          continue;
        }
        // if one of the boundaries is not periodic, then use it
        // otherwise, use periodic
        FldsBC flds_bc { FldsBC::INVALID };
        for (auto dir : assoc_orth) {
          const auto fldsbc_in_dir = current_domain.mesh.flds_bc_in(dir);
          if (fldsbc_in_dir != FldsBC::PERIODIC) {
            flds_bc = fldsbc_in_dir;
            break;
          } else {
            flds_bc = FldsBC::PERIODIC;
          }
        }
        PrtlBC prtl_bc { PrtlBC::INVALID };
        for (auto dir : assoc_orth) {
          const auto prtlbc_in_dir = current_domain.mesh.prtl_bc_in(dir);
          if (prtlbc_in_dir != PrtlBC::PERIODIC) {
            prtl_bc = prtlbc_in_dir;
            break;
          } else {
            prtl_bc = PrtlBC::PERIODIC;
          }
        }
        raise::ErrorIf(flds_bc == FldsBC::INVALID,
                       "Invalid boundary condition for fields",
                       HERE);
        raise::ErrorIf(prtl_bc == PrtlBC::INVALID,
                       "Invalid boundary condition for particles",
                       HERE);
        current_domain.mesh.set_flds_bc(direction, flds_bc);
        current_domain.mesh.set_prtl_bc(direction, prtl_bc);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::finalValidityCheck() const {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto& current_domain = g_subdomains[idx];
      // check that all neighbors are set properly
      for (const auto& direction : dir::Directions<D>::all) {
        // check neighbors
        const auto  neighbor_idx = current_domain.neighbor_idx_in(direction);
        const auto& neighbor     = g_subdomains[neighbor_idx];
        const auto  self_idx     = neighbor.neighbor_idx_in(-direction);
        raise::ErrorIf(self_idx != idx, "Neighbor not set properly", HERE);
      }
      // check that all boundaries are set properly
      for (const auto& direction : dir::Directions<D>::all) {
        raise::ErrorIf(current_domain.mesh.flds_bc_in(direction) == FldsBC::INVALID,
                       "Invalid boundary condition for fields",
                       HERE);
        raise::ErrorIf(current_domain.mesh.prtl_bc_in(direction) == PrtlBC::INVALID,
                       "Invalid boundary condition for particles",
                       HERE);
      }
      // check that local subdomains are contained in g_local_subdomain_indices
      auto contained_in_local = false;
      for (const auto& gidx : l_subdomain_indices()) {
        contained_in_local |= (idx == gidx);
      }
#if defined(MPI_ENABLED)
      const auto is_same_rank = current_domain.mpi_rank() == g_mpi_rank;
      raise::ErrorIf(is_same_rank != contained_in_local,
                     "local subdomains not set properly",
                     HERE);
#else  // not MPI_ENABLED
      raise::ErrorIf(not contained_in_local,
                     "local subdomains not set properly",
                     HERE);
#endif // MPI_ENABLED
      // check that non-local subdomains do not allocate memory
      if (not contained_in_local) {
        raise::ErrorIf(not current_domain.is_placeholder(),
                       "Non-local domain has memory allocated",
                       HERE);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::metricCompatibilityCheck() const {
    const auto epsilon = std::numeric_limits<real_t>::epsilon() *
                         static_cast<real_t>(100.0);
    const auto dx_min              = g_mesh.metric.dxMin();
    auto       dx_min_from_domains = std::numeric_limits<real_t>::infinity();
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto& current_domain = g_subdomains[idx];
      const auto  current_dx_min = current_domain.mesh.metric.dxMin();
      dx_min_from_domains = std::min(dx_min_from_domains, current_dx_min);
    }
    raise::ErrorIf(
      not cmp::AlmostEqual_host(dx_min / dx_min_from_domains, ONE, epsilon),
      "dx_min is not the same across all domains: " + std::to_string(dx_min) +
        " " + std::to_string(dx_min_from_domains),
      HERE);
#if defined(MPI_ENABLED)
    auto dx_mins        = std::vector<real_t>(g_ndomains);
    dx_mins[g_mpi_rank] = dx_min;
    MPI_Allgather(&dx_min,
                  1,
                  mpi::get_type<real_t>(),
                  dx_mins.data(),
                  1,
                  mpi::get_type<real_t>(),
                  MPI_COMM_WORLD);
    for (const auto& dx : dx_mins) {
      raise::ErrorIf(not cmp::AlmostEqual_host(dx / dx_min, ONE, epsilon),
                     "dx_min is not the same across all MPI ranks",
                     HERE);
    }
#endif
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::setFldsBC(const bc_in& dir, const FldsBC& new_bcs) {
    if (dir == bc_in::Mx1) {
      if constexpr (M::Dim == Dim::_1D) {
        g_mesh.set_flds_bc({ -1 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_flds_bc({ -1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ -1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px1) {
      if constexpr (M::Dim == Dim::_1D) {
        g_mesh.set_flds_bc({ +1 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_flds_bc({ +1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ +1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Mx2) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set -x2 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_flds_bc({ -1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ -1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px2) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set +x2 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_flds_bc({ +1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ +1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Mx3) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set -x3 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        raise::Error("Cannot set -x3 BCs for 2D", HERE);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ 0, 0, -1 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px3) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set +x3 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        raise::Error("Cannot set +x3 BCs for 2D", HERE);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_flds_bc({ 0, 0, +1 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else {
      raise::Error("Invalid direction", HERE);
    }
    redefineBoundaries();
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::setPrtlBC(const bc_in& dir, const PrtlBC& new_bcs) {
    if (dir == bc_in::Mx1) {
      if constexpr (M::Dim == Dim::_1D) {
        g_mesh.set_prtl_bc({ -1 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_prtl_bc({ -1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ -1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px1) {
      if constexpr (M::Dim == Dim::_1D) {
        g_mesh.set_prtl_bc({ +1 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_prtl_bc({ +1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ +1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Mx2) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set -x2 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_prtl_bc({ -1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ -1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px2) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set +x2 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        g_mesh.set_prtl_bc({ +1, 0 }, new_bcs);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ +1, 0, 0 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Mx3) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set -x3 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        raise::Error("Cannot set -x3 BCs for 2D", HERE);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ 0, 0, -1 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (dir == bc_in::Px3) {
      if constexpr (M::Dim == Dim::_1D) {
        raise::Error("Cannot set +x3 BCs for 1D", HERE);
      } else if constexpr (M::Dim == Dim::_2D) {
        raise::Error("Cannot set +x3 BCs for 2D", HERE);
      } else if constexpr (M::Dim == Dim::_3D) {
        g_mesh.set_prtl_bc({ 0, 0, +1 }, new_bcs);
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else {
      raise::Error("Invalid direction", HERE);
    }
    redefineBoundaries();
  }

#define METADOMAIN_STRUCT(E, M, D)                                              \
  template struct Metadomain<E<M<D>>::S, M<D>>;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_STRUCT)

#undef METADOMAIN_STRUCT

} // namespace ntt
