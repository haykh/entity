/**
 * @file framework/domain/domain.h
 * @brief
 * Domain class containing information about the local meshblock
 * including the Mesh object itself, fields, particle species,
 * as well as pointers to neighboring domains.
 * @implements
 *   - ntt::Domain<>
 * @macros:
 *   - MPI_ENABLED
 * @note
 * Illustration below shows the structure of a metadomain with 2D decomposition.
 * Class Domain defines a single element of this global metadomain.
 *
 * Metadomain
 * |---------|--------|------------------|------------------|
 * | D       | D      | D                | D                |
 * |         |        |                  |                  |
 * |---------|--------|------------------X------------------|
 * | D       | D      | Domain {index}   |\                 |
 * |         |        |                  | (extent[1],      |
 * |         |        |                  |  extent[3])      |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |<- offset_ncells->|<---- ncells ---->|                  |
 * |---------|--------X------------------|------------------|
 * | D       | D      |\                 | D                |
 * |         |        | (extent[0],      |                  |
 * |         |        |  extent[2])      |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |---------|--------|------------------|------------------|
 * ^                  ^
 * |--offset_ndomains-|
 */

#ifndef FRAMEWORK_DOMAIN_DOMAIN_H
#define FRAMEWORK_DOMAIN_DOMAIN_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/formatting.h"
#include "utils/numeric.h"
#include "utils/reporter.h"

#include "metrics/traits.h"

#include "framework/containers/fields.h"
#include "framework/containers/particles.h"
#include "framework/containers/species.h"
#include "framework/domain/mesh.h"

#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
    requires metric::traits::HasD<M>
  struct Domain {
    static constexpr Dimension D { M::Dim };

    Mesh<M>                                 mesh;
    Fields<D, S>                            fields;
    std::vector<Particles<D, M::CoordType>> species;
    random_number_pool_t                    random_pool;

    /**
     * @brief constructor for "empty" allocation of non-local domain placeholders
     */
    Domain(bool,
           unsigned int                         index,
           const std::vector<unsigned int>&     offset_ndomains,
           const std::vector<ncells_t>&         offset_ncells,
           const std::vector<ncells_t>&         ncells,
           const boundaries_t<real_t>&          extent,
           const std::map<std::string, real_t>& metric_params,
           const std::vector<ParticleSpecies>&)
      : mesh { ncells, extent, metric_params }
      , fields {}
      , species {}
      , random_pool { constant::RandomSeed }
      , m_index { index }
      , m_offset_ndomains { offset_ndomains }
      , m_offset_ncells { offset_ncells } {}

    Domain(unsigned int                         index,
           const std::vector<unsigned int>&     offset_ndomains,
           const std::vector<ncells_t>&         offset_ncells,
           const std::vector<ncells_t>&         ncells,
           const boundaries_t<real_t>&          extent,
           const std::map<std::string, real_t>& metric_params,
           const std::vector<ParticleSpecies>&  species_params)
      : mesh { ncells, extent, metric_params }
      , fields { ncells }
      , species { species_params.begin(), species_params.end() }
      , random_pool { constant::RandomSeed + static_cast<std::uint64_t>(index) }
      , m_index { index }
      , m_offset_ndomains { offset_ndomains }
      , m_offset_ncells { offset_ncells } {}

#if defined(MPI_ENABLED)
    [[nodiscard]]
    auto mpi_rank() const -> int {
      return m_mpi_rank;
    }

    void set_mpi_rank(int rank) {
      m_mpi_rank = rank;
    }
#endif // MPI_ENABLED

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto index() const -> unsigned int {
      return m_index;
    }

    [[nodiscard]]
    auto offset_ndomains() const -> std::vector<unsigned int> {
      return m_offset_ndomains;
    }

    [[nodiscard]]
    auto offset_ncells() const -> std::vector<ncells_t> {
      return m_offset_ncells;
    }

    [[nodiscard]]
    auto neighbor_idx_in(const dir::direction_t<D>& dir) const -> unsigned int {
      raise::ErrorIf(m_neighbor_idx.find(dir) == m_neighbor_idx.end(),
                     "neighbor_in() failed",
                     HERE);
      return m_neighbor_idx.at(dir);
    }

    [[nodiscard]]
    auto is_placeholder() const -> bool {
      std::size_t sp_footprint { 0 };
      for (auto& sp : species) {
        sp_footprint += sp.memory_footprint();
      }
      return fields.memory_footprint() == 0 and sp_footprint == 0;
    }

    /* setters -------------------------------------------------------------- */
    auto set_neighbor_idx(const dir::direction_t<D>& dir, unsigned int idx)
      -> void {
      m_neighbor_idx[dir] = idx;
    }

    /* printer overload ----------------------------------------------------- */
    auto Report() const -> std::string {
      std::string report = "";
      reporter::AddSubcategory(report,
                               4,
                               fmt::format("Domain #%d", index()).c_str());
#if defined(MPI_ENABLED)
      reporter::AddParam(report, 6, "Rank", "%d", mpi_rank());
#endif
      reporter::AddParam(report,
                         6,
                         "Resolution",
                         "%s",
                         fmt::formatVector(mesh.n_active()).c_str());
      reporter::AddParam(report,
                         6,
                         "Extent",
                         "%s",
                         fmt::formatVector(mesh.extent()).c_str());
      reporter::AddSubcategory(report, 6, "Boundary conditions");

      reporter::AddLabel(
        report,
        8 + 2 + 2 * M::Dim,
        fmt::format("%-10s  %-10s  %-10s", "[flds]", "[prtl]", "[neighbor]").c_str());
      for (auto& direction : dir::Directions<M::Dim>::all) {
        const auto flds_bc      = mesh.flds_bc_in(direction);
        const auto prtl_bc      = mesh.prtl_bc_in(direction);
        bool       has_sync     = false;
        auto       neighbor_idx = neighbor_idx_in(direction);
        if (flds_bc == FldsBC::SYNC || prtl_bc == PrtlBC::SYNC) {
          has_sync = true;
        }
        reporter::AddUnlabeledParam(
          report,
          8,
          direction.to_string().c_str(),
          "%-10s  %-10s  %-10s",
          flds_bc.to_string(),
          prtl_bc.to_string(),
          has_sync ? std::to_string(neighbor_idx).c_str() : ".");
      }
      reporter::AddSubcategory(report, 6, "Memory footprint");
      auto flds_footprint = fields.memory_footprint();
      auto [flds_size, flds_unit] = reporter::Bytes2HumanReadable(flds_footprint);
      reporter::AddParam(report, 8, "Fields", "%.2f %s", flds_size, flds_unit.c_str());
      if (species.size() > 0) {
        reporter::AddSubcategory(report, 8, "Particles");
      }
      for (auto& species : species) {
        const auto str = fmt::format("Species #%d (%s)",
                                     species.index(),
                                     species.label().c_str());
        auto [size,
              unit] = reporter::Bytes2HumanReadable(species.memory_footprint());
        reporter::AddParam(report, 10, str.c_str(), "%.2f %s", size, unit.c_str());
      }
      report.pop_back();
      return report;
    }

  private:
    // index of the domain in the metadomain
    unsigned int                m_index;
    // offset of the domain in # of domains
    std::vector<unsigned int>   m_offset_ndomains;
    // offset of the domain in cells (# of cells in each dimension)
    std::vector<ncells_t>       m_offset_ncells;
    // neighboring domain indices
    dir::map_t<D, unsigned int> m_neighbor_idx;
    // MPI rank of the domain (used only when MPI enabled)
    int                         m_mpi_rank;
  };

  template <SimEngine::type S, class M>
  inline auto operator<<(std::ostream& os, const Domain<S, M>& domain)
    -> std::ostream& {
    os << "Domain #" << domain.index();
#if defined(MPI_ENABLED)
    os << " [MPI rank: " << domain.mpi_rank() << "]";
#endif
    os << ":\n";
    os << std::setw(19) << std::left << "  engine: " << SimEngine(S).to_string()
       << "\n";
    os << std::setw(19) << std::left << "  global offset: ";
    for (auto& off_nd : domain.offset_ndomains()) {
      os << std::setw(15) << std::left << off_nd;
    }
    os << "\n";
    os << std::setw(19) << std::left << "  cells offset: ";
    for (auto& off_nc : domain.offset_ncells()) {
      os << std::setw(15) << std::left << off_nc;
    }
    os << "\n";
    os << std::setw(19) << std::left << "  physical extent: ";
    for (auto dim { 0u }; dim < M::Dim; ++dim) {
      os << std::setw(15) << std::left
         << fmt::format("{%.2f; %.2f}",
                        domain.mesh.extent(static_cast<in>(dim)).first,
                        domain.mesh.extent(static_cast<in>(dim)).second);
    }
    os << "\n  neighbors:\n";
    for (auto& direction : dir::Directions<M::Dim>::all) {
      auto neighbor_idx = domain.neighbor_idx_in(direction);
      os << "   " << direction << " -> #" << neighbor_idx << "\n";
    }
    os << "  field boundaries:\n";
    for (auto& direction : dir::Directions<M::Dim>::orth) {
      os << "   " << direction;
      os << " -> " << domain.mesh.flds_bc_in(direction).to_string() << "\n";
    }
    os << "  particle boundaries:\n";
    for (auto& direction : dir::Directions<M::Dim>::orth) {
      os << "   " << direction;
      os << " -> " << domain.mesh.prtl_bc_in(direction).to_string() << "\n";
    }
    return os;
  }

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_DOMAIN_H
