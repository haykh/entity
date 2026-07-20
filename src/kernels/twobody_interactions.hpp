/**
 * @file kernels/twobody_interactions.hpp
 * @brief Generic two-body interaction kernel that can be used to implement various
 * types of collisions between species, e.g. Compton scattering, Breit-Wheeler pair production, etc.
 * @implements
 *   - kernel::mink::TwoBodyInteraction<>
 * @namespaces:
 *   - arch::mink::
 */
#ifndef KERNELS_TWOBODY_INTERACTIONS_HPP
#define KERNELS_TWOBODY_INTERACTIONS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/policies.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <cstdint>
#include <vector>

namespace kernel::mink {
  using namespace ntt;

  namespace {
    struct CollisionSpecies {
      const spidx_t sp;
      const npart_t npart;
      ncells_t      num_tiles { 0u };

      array_t<ncells_t*> tileidx;
      array_t<npart_t*>  num_ppt;

      CollisionSpecies(spidx_t                   sp,
                       npart_t                   npart,
                       const array_t<ncells_t*>& tileidx,
                       const array_t<npart_t*>&  num_ppt,
                       ncells_t                  num_tiles)
        : sp { sp }
        , npart { npart }
        , tileidx { tileidx }
        , num_ppt { num_ppt }
        , num_tiles { num_tiles } {}
    };

    template <Dimension D>
    Inline void UnravelTileIdx(ncells_t  tile_idx,
                               ncells_t  ntx2,
                               ncells_t  ntx3,
                               ncells_t& ti,
                               ncells_t& tj,
                               ncells_t& tk) {
      if constexpr (D == Dim::_1D) {
        ti = tile_idx;
      } else if constexpr (D == Dim::_2D) {
        ti = tile_idx / ntx2;
        tj = tile_idx % ntx2;
      } else if constexpr (D == Dim::_3D) {
        ti             = tile_idx / (ntx2 * ntx3);
        const auto rem = tile_idx % (ntx2 * ntx3);
        tj             = rem / ntx3;
        tk             = rem % ntx3;
      } else {
        raise::KernelError(HERE, "Wrong D in TileIdxUnravel");
      }
    }

    template <Dimension D>
    Inline auto NCellsOnTile(ncells_t tile_idx,
                             ncells_t tile_size,
                             ncells_t ntx2,
                             ncells_t ntx3,
                             ncells_t nx1,
                             ncells_t nx2,
                             ncells_t nx3) -> ncells_t {
      ncells_t ncells_on_tile { 1 };
      ncells_t ti { 0u }, tj { 0u }, tk { 0u };
      UnravelTileIdx<D>(tile_idx, ntx2, ntx3, ti, tj, tk);
      if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
        const auto i1_min  = ti * tile_size;
        const auto i1_max  = math::min(i1_min + tile_size, nx1);
        ncells_on_tile    *= (i1_max - i1_min);
      }
      if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
        const auto i2_min  = tj * tile_size;
        const auto i2_max  = math::min(i2_min + tile_size, nx2);
        ncells_on_tile    *= (i2_max - i2_min);
      }
      if constexpr (D == Dim::_3D) {
        const auto i3_min  = tk * tile_size;
        const auto i3_max  = math::min(i3_min + tile_size, nx3);
        ncells_on_tile    *= (i3_max - i3_min);
      }
      return ncells_on_tile;
    }

    namespace {
      struct PackIndex {
        const npart_t          offset;
        const CollisionSpecies species;
        array_t<uint64_t*>     combined_idx;
        array_t<ncells_t*>     combined_tileidx;

        PackIndex(npart_t                 offset,
                  const CollisionSpecies& species,
                  array_t<uint64_t*>&     combined_idx,
                  array_t<ncells_t*>&     combined_tileidx)
          : offset { offset }
          , species { species }
          , combined_idx { combined_idx }
          , combined_tileidx { combined_tileidx } {}

        Inline void operator()(prtlidx_t p) const {
          // pack species idx into top 8 bits + prtl index into the remaining 56 bits
          combined_idx(offset + p) = (static_cast<uint64_t>(species.sp) << 56) |
                                     static_cast<uint64_t>(p);
          combined_tileidx(offset + p) = species.tileidx(p);
        }
      };

      struct CombineNumPpt {
        const CollisionSpecies species;
        array_t<npart_t*>      combined_num_ppt;

        CombineNumPpt(const CollisionSpecies& species,
                      array_t<npart_t*>&      combined_num_ppt)
          : species { species }
          , combined_num_ppt { combined_num_ppt } {}

        Inline void operator()(cellidx_t t) const {
          combined_num_ppt(t) += species.num_ppt(t);
        }
      };

      struct PackRandom {
        const array_t<ncells_t*> combined_tileidx;
        array_t<uint64_t*>       shuffle_key;
        random_number_pool_t     random_pool;

        PackRandom(const array_t<ncells_t*>& combined_tileidx,
                   array_t<uint64_t*>&       shuffle_key,
                   random_number_pool_t&     random_pool)
          : combined_tileidx { combined_tileidx }
          , shuffle_key { shuffle_key }
          , random_pool { random_pool } {}

        Inline void operator()(prtlidx_t p) const {
          auto       gen = random_pool.get_state();
          const auto rnd = static_cast<uint64_t>(gen.urand());
          random_pool.free_state(gen);
          const auto tile_idx = static_cast<uint64_t>(combined_tileidx(p));
          // packing top 32 bits with tile index, and the rest -- random
          shuffle_key(p)      = (tile_idx << 32) | rnd;
        }
      };

      struct TileOffsets {
        array_t<npart_t*> tile_offsets;
        array_t<npart_t*> combined_num_ppt;

        TileOffsets(array_t<npart_t*>&       tile_offsets,
                    const array_t<npart_t*>& combined_num_ppt)
          : tile_offsets { tile_offsets }
          , combined_num_ppt { combined_num_ppt } {}

        Inline void operator()(cellidx_t t, npart_t& acc, const bool is_final) const {
          if (is_final) {
            tile_offsets(t) = acc;
          }
          acc += combined_num_ppt(t);
        }
      };
    } // namespace

    template <Dimension D>
    struct CollisionGroup {
      std::vector<CollisionSpecies> group;

      array_t<uint64_t*> combined_idx;
      array_t<ncells_t*> combined_tileidx;
      array_t<npart_t*>  combined_num_ppt;
      array_t<npart_t*>  tile_offsets;

      ncells_t num_tiles { 0u };

      CollisionGroup(const std::vector<Particles<D, Coord::Cartesian>*>& particles,
                     const std::vector<ncells_t>& ncells,
                     ncells_t                     tile_size,
                     random_number_pool_t&        random_pool) {
        for (const auto* species : particles) {
          const auto         npart_s = species->npart();
          array_t<ncells_t*> tileidx { "tile_idx", npart_s };
          auto tile_indexing_kernel = sort::PositionToTileIndex<D, true>(
            species->i1,
            species->i2,
            species->i3,
            species->tag,
            tileidx,
            ncells,
            tile_size);
          Kokkos::parallel_for("TileIndexing", species->npart(), tile_indexing_kernel);
          group.emplace_back(species->sp,
                             npart_s,
                             tileidx,
                             tile_indexing_kernel.num_ppt,
                             tile_indexing_kernel.total_tiles);
          if (num_tiles == 0u) {
            num_tiles = group.back().num_tiles;
          } else if (num_tiles != group.back().num_tiles) {
            raise::Error("unequal num_tiles across species in group", HERE);
          }
          raise::ErrorIf(group.back().tileidx.extent(0) != species->npart(),
                         "tileidx must have the same extent as npart for all "
                         "species in group",
                         HERE);
        }

        npart_t tot_npart = 0u;
        for (const auto& species : group) {
          tot_npart += species.npart;
        }

        combined_idx     = array_t<uint64_t*> { "combined_idx", tot_npart };
        combined_tileidx = array_t<ncells_t*> { "combined_tileidx", tot_npart };
        combined_num_ppt = array_t<npart_t*> { "combined_num_ppt", num_tiles };
        tile_offsets     = array_t<npart_t*> { "tile_offsets", num_tiles };

        {
          // combine particle indices in the group & compute total number in each tile
          npart_t offset = 0u;
          for (const auto& species : group) {
            Kokkos::parallel_for(
              "CombineInGroup",
              species.npart,
              PackIndex { offset, species, combined_idx, combined_tileidx });
            offset += species.npart;
            Kokkos::parallel_for("CombineNumPpt",
                                 species.num_tiles,
                                 CombineNumPpt { species, combined_num_ppt });
            Kokkos::fence();
          }
        }
        {
          // randomly shuffle particles within each tile and sort by tiles
          array_t<uint64_t*> shuffle_key { "shuffle_key", tot_npart };
          Kokkos::parallel_for(
            "PackRandom",
            tot_npart,
            PackRandom { combined_tileidx, shuffle_key, random_pool });
          Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace {},
                                            shuffle_key,
                                            combined_idx);
        }
        {
          // compute index offsets for each tile
          Kokkos::parallel_scan("TileOffsets",
                                num_tiles,
                                TileOffsets { tile_offsets, combined_num_ppt });
        }
      }
    };
  } // namespace

  template <Dimension D, TwoBodyInteractionPolicyClass I>
  void TwoBodyInteraction(
    const std::vector<Particles<D, Coord::Cartesian>*>& species1,
    const std::vector<Particles<D, Coord::Cartesian>*>& species2,
    const std::vector<ncells_t>&                        ncells,
    const boundaries_t<real_t>&                         domain_extent,
    ncells_t                                            tile_size,
    real_t                                              ppc0,
    random_number_pool_t&                               random_pool,
    I&                                                  interaction_policy) {
    raise::ErrorIf(species1.empty() or species2.empty(),
                   "species groups must be non-empty",
                   HERE);
    raise::ErrorIf(ncells.size() != static_cast<std::size_t>(D),
                   "ncells size must match D",
                   HERE);
    raise::ErrorIf(domain_extent.size() != static_cast<std::size_t>(D),
                   "domain_extent size must match D",
                   HERE);
    // compute base cell volume in physical units
    real_t cell_volume { ONE };
    for (int d = 0; d < static_cast<int>(D); ++d) {
      cell_volume *= static_cast<real_t>(
                       domain_extent[d].second - domain_extent[d].first) /
                     static_cast<real_t>(ncells[d]);
    }

    const auto group1 = CollisionGroup<D>(species1, ncells, tile_size, random_pool);
    const auto group2 = CollisionGroup<D>(species2, ncells, tile_size, random_pool);
    raise::ErrorIf(group1.num_tiles != group2.num_tiles,
                   "number of tiles differ in group1 vs group2",
                   HERE);
    const auto num_tiles = group1.num_tiles;

    const auto& combined_idx1     = group1.combined_idx;
    const auto& combined_idx2     = group2.combined_idx;
    const auto& combined_num_ppt1 = group1.combined_num_ppt;
    const auto& combined_num_ppt2 = group2.combined_num_ppt;
    const auto& tile_offsets1     = group1.tile_offsets;
    const auto& tile_offsets2     = group2.tile_offsets;

    // fill species in the interaction policy
    for (auto& sp1 : species1) {
      interaction_policy.species[sp1->sp - 1] = static_cast<const ParticleArrays&>(
        *sp1);
    }
    for (auto& sp2 : species2) {
      interaction_policy.species[sp2->sp - 1] = static_cast<const ParticleArrays&>(
        *sp2);
    }

    // total particle weight on each tile
    auto weights_on_tile1 = array_t<real_t*> { "weights_on_tile1", num_tiles };
    auto weights_on_tile2 = array_t<real_t*> { "weights_on_tile2", num_tiles };
    Kokkos::parallel_for(
      "ComputeWeightsOnTiles",
      Kokkos::TeamPolicy<>(num_tiles, Kokkos::AUTO),
      Lambda(const Kokkos::TeamPolicy<>::member_type& team) {
        const ncells_t t = team.league_rank();

        const auto o1              = tile_offsets1(t);
        const auto num_ppt1        = combined_num_ppt1(t);
        real_t     weight_on_tile1 = ZERO;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, num_ppt1),
          [&](prtlidx_t i, real_t& lsum) {
            const auto sp1 = static_cast<spidx_t>(combined_idx1(o1 + i) >> 56);
            const auto p1  = static_cast<npart_t>(combined_idx1(o1 + i) &
                                                 ((1ull << 56) - 1));

            lsum += (interaction_policy.species[sp1 - 1].tag(p1) ==
                     ParticleTag::alive)
                      ? interaction_policy.species[sp1 - 1].weight(p1)
                      : ZERO;
          },
          weight_on_tile1);
        weights_on_tile1(t) = weight_on_tile1;

        const auto o2              = tile_offsets2(t);
        const auto num_ppt2        = combined_num_ppt2(t);
        real_t     weight_on_tile2 = ZERO;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, num_ppt2),
          [&](prtlidx_t i, real_t& lsum) {
            const auto sp2 = static_cast<spidx_t>(combined_idx2(o2 + i) >> 56);
            const auto p2  = static_cast<npart_t>(combined_idx2(o2 + i) &
                                                 ((1ull << 56) - 1));

            lsum += (interaction_policy.species[sp2 - 1].tag(p2) ==
                     ParticleTag::alive)
                      ? interaction_policy.species[sp2 - 1].weight(p2)
                      : ZERO;
          },
          weight_on_tile2);
        weights_on_tile2(t) = weight_on_tile2;
      });

    // number of cells in each direction
    ncells_t nx1 { 1u }, nx2 { 1u }, nx3 { 1u };
    if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
      nx1 = ncells[0];
    }
    if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
      nx2 = ncells[1];
    }
    if constexpr (D == Dim::_3D) {
      nx3 = ncells[2];
    }
    ncells_t ntx2 { 1u }, ntx3 { 1u };
    if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
      ntx2 = static_cast<ncells_t>(
        math::ceil(static_cast<double>(nx2) / static_cast<double>(tile_size)));
    }
    if constexpr (D == Dim::_3D) {
      ntx3 = static_cast<ncells_t>(
        math::ceil(static_cast<double>(nx3) / static_cast<double>(tile_size)));
    }

    array_t<uint64_t* [2]> interaction_pairs { "interaction_pairs",
                                               combined_idx1.extent(0) };
    array_t<npart_t>       counter { "counter" };

    Kokkos::parallel_for(
      "PopulateInteractionPairs",
      Kokkos::TeamPolicy<>(num_tiles, Kokkos::AUTO),
      Lambda(const Kokkos::TeamPolicy<>::member_type& team) {
        const ncells_t t = team.league_rank();
        const auto     tile_weight =
          math::max(weights_on_tile1(t), weights_on_tile2(t)) /
          (NCellsOnTile<D>(t, tile_size, ntx2, ntx3, nx1, nx2, nx3) * ppc0);

        const auto k  = math::min(combined_num_ppt1(t), combined_num_ppt2(t));
        const auto o1 = tile_offsets1(t);
        const auto o2 = tile_offsets2(t);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k), [&](prtlidx_t i) {
          // unpack the higher 8 bits
          const auto sp1 = static_cast<spidx_t>(combined_idx1(o1 + i) >> 56);
          const auto sp2 = static_cast<spidx_t>(combined_idx2(o2 + i) >> 56);

          // unpack the lower 56 bits
          const auto p1 = static_cast<npart_t>(combined_idx1(o1 + i) &
                                               ((1ull << 56) - 1));
          const auto p2 = static_cast<npart_t>(combined_idx2(o2 + i) &
                                               ((1ull << 56) - 1));
          if ((interaction_policy.species[sp1 - 1].tag(p1) == ParticleTag::alive) and
              (interaction_policy.species[sp2 - 1].tag(p2) == ParticleTag::alive)) {
            if (interaction_policy.should_interact(sp1, p1, sp2, p2, tile_weight)) {
              const auto idx = Kokkos::atomic_fetch_add(&counter(), 1);
              interaction_pairs(idx, 0) = combined_idx1(o1 + i);
              interaction_pairs(idx, 1) = combined_idx2(o2 + i);
            }
          }
        });
      });
    auto counter_h = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(counter_h, counter);
    Kokkos::parallel_for(
      "ProcessInteractions",
      counter_h(),
      Lambda(prtlidx_t idx) {
        const auto sp1 = static_cast<spidx_t>(interaction_pairs(idx, 0) >> 56);
        const auto sp2 = static_cast<spidx_t>(interaction_pairs(idx, 1) >> 56);
        const auto p1  = static_cast<npart_t>(interaction_pairs(idx, 0) &
                                             ((1ull << 56) - 1));
        const auto p2  = static_cast<npart_t>(interaction_pairs(idx, 1) &
                                             ((1ull << 56) - 1));
        interaction_policy(sp1, p1, sp2, p2);
      });
  }

} // namespace kernel::mink

#endif // KERNELS_TWOBODY_INTERACTIONS_HPP
