#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "checkpoint/reader.h"
#include "checkpoint/writer.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::InitCheckpointWriter(adios2::ADIOS*          ptr_adios,
                                              const SimulationParams& params) {
    raise::ErrorIf(ptr_adios == nullptr, "adios == nullptr", HERE);
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Checkpoint writing for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);

    std::vector<std::size_t> glob_shape_with_ghosts, off_ncells_with_ghosts;
    for (auto d { 0u }; d < M::Dim; ++d) {
      off_ncells_with_ghosts.push_back(
        local_domain->offset_ncells()[d] +
        2 * N_GHOSTS * local_domain->offset_ndomains()[d]);
      glob_shape_with_ghosts.push_back(
        mesh().n_active()[d] + 2 * N_GHOSTS * ndomains_per_dim()[d]);
    }
    auto loc_shape_with_ghosts = local_domain->mesh.n_all();

    std::vector<unsigned short> nplds;
    for (auto s { 0u }; s < local_domain->species.size(); ++s) {
      nplds.push_back(local_domain->species[s].npld());
    }

    g_checkpoint_writer.init(
      ptr_adios,
      params.template get<std::size_t>("checkpoint.interval"),
      params.template get<long double>("checkpoint.interval_time"),
      params.template get<int>("checkpoint.keep"));
    if (g_checkpoint_writer.enabled()) {
      g_checkpoint_writer.defineFieldVariables(S,
                                               glob_shape_with_ghosts,
                                               off_ncells_with_ghosts,
                                               loc_shape_with_ghosts);
      g_checkpoint_writer.defineParticleVariables(M::CoordType,
                                                  M::Dim,
                                                  local_domain->species.size(),
                                                  nplds);
    }
  }

  template <SimEngine::type S, class M>
  auto Metadomain<S, M>::WriteCheckpoint(const SimulationParams& params,
                                         std::size_t             current_step,
                                         std::size_t             finished_step,
                                         long double             current_time,
                                         long double finished_time) -> bool {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Checkpointing for now is only supported for one subdomain per rank",
      HERE);
    if (not g_checkpoint_writer.shouldSave(finished_step, finished_time) or
        finished_step <= 1) {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing checkpoint", HERE);
    g_checkpoint_writer.beginSaving(current_step, current_time);
    {
      g_checkpoint_writer.saveAttrs(params, current_time);
      g_checkpoint_writer.saveField<M::Dim, 6>("em", local_domain->fields.em);
      if constexpr (S == SimEngine::GRPIC) {
        g_checkpoint_writer.saveField<M::Dim, 6>("em0", local_domain->fields.em0);
        g_checkpoint_writer.saveField<M::Dim, 3>("cur0", local_domain->fields.cur0);
      }
      std::size_t dom_offset = 0, dom_tot = 1;
#if defined(MPI_ENABLED)
      dom_offset = g_mpi_rank;
      dom_tot    = g_mpi_size;
#endif // MPI_ENABLED

      for (auto s { 0u }; s < local_domain->species.size(); ++s) {
        auto        npart    = local_domain->species[s].npart();
        std::size_t offset   = 0;
        auto        glob_tot = npart;
#if defined(MPI_ENABLED)
        auto glob_npart = std::vector<std::size_t>(g_ndomains);
        MPI_Allgather(&npart,
                      1,
                      mpi::get_type<std::size_t>(),
                      glob_npart.data(),
                      1,
                      mpi::get_type<std::size_t>(),
                      MPI_COMM_WORLD);
        glob_tot = 0;
        for (auto r = 0; r < g_mpi_size; ++r) {
          if (r < g_mpi_rank) {
            offset += glob_npart[r];
          }
          glob_tot += glob_npart[r];
        }
#endif // MPI_ENABLED
        g_checkpoint_writer.savePerDomainVariable<std::size_t>(
          fmt::format("s%d_npart", s + 1),
          dom_tot,
          dom_offset,
          npart);
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i1", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i1);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx1", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx1);
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i1_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i1_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx1_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx1_prev);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i2", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i2);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx2", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx2);
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i2_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i2_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx2_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx2_prev);
        }
        if constexpr (M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i3", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i3);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx3", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx3);
          g_checkpoint_writer.saveParticleQuantity<int>(
            fmt::format("s%d_i3_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].i3_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            fmt::format("s%d_dx3_prev", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].dx3_prev);
        }
        if constexpr (M::Dim == Dim::_2D and M::CoordType != Coord::Cart) {
          g_checkpoint_writer.saveParticleQuantity<real_t>(
            fmt::format("s%d_phi", s + 1),
            glob_tot,
            offset,
            npart,
            local_domain->species[s].phi);
        }
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          fmt::format("s%d_ux1", s + 1),
          glob_tot,
          offset,
          npart,
          local_domain->species[s].ux1);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          fmt::format("s%d_ux2", s + 1),
          glob_tot,
          offset,
          npart,
          local_domain->species[s].ux2);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          fmt::format("s%d_ux3", s + 1),
          glob_tot,
          offset,
          npart,
          local_domain->species[s].ux3);
        g_checkpoint_writer.saveParticleQuantity<short>(
          fmt::format("s%d_tag", s + 1),
          glob_tot,
          offset,
          npart,
          local_domain->species[s].tag);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          fmt::format("s%d_weight", s + 1),
          glob_tot,
          offset,
          npart,
          local_domain->species[s].weight);

        auto nplds = local_domain->species[s].npld();
        if (nplds > 0) {
          g_checkpoint_writer.saveParticlePayloads(fmt::format("s%d_plds", s + 1),
                                                   nplds,
                                                   glob_tot,
                                                   offset,
                                                   npart,
                                                   local_domain->species[s].pld);
        }
      }
    }
    g_checkpoint_writer.endSaving();
    logger::Checkpoint("Checkpoint written", HERE);
    return true;
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::ContinueFromCheckpoint(adios2::ADIOS* ptr_adios,
                                                const SimulationParams& params) {
    raise::ErrorIf(ptr_adios == nullptr, "adios == nullptr", HERE);
    auto fname = fmt::format(
      "checkpoints/step-%08lu.bp",
      params.template get<std::size_t>("checkpoint.start_step"));
    logger::Checkpoint(fmt::format("Reading checkpoint from %s", fname.c_str()),
                       HERE);

    adios2::IO io = ptr_adios->DeclareIO("Entity::CheckpointRead");
    io.SetEngine("BPFile");
#if !defined(MPI_ENABLED)
    adios2::Engine reader = io.Open(fname, adios2::Mode::Read);
#else
    adios2::Engine reader = io.Open(fname, adios2::Mode::Read, MPI_COMM_SELF);
#endif

    reader.BeginStep();
    for (auto& ldidx : l_subdomain_indices()) {
      auto&                     domain = g_subdomains[ldidx];
      adios2::Box<adios2::Dims> range;
      for (auto d { 0u }; d < M::Dim; ++d) {
        range.first.push_back(domain.offset_ncells()[d] +
                              2 * N_GHOSTS * domain.offset_ndomains()[d]);
        range.second.push_back(domain.mesh.n_all()[d]);
      }
      range.first.push_back(0);
      range.second.push_back(6);
      checkpoint::ReadFields<M::Dim, 6>(io, reader, "em", range, domain.fields.em);
      if constexpr (S == ntt::SimEngine::GRPIC) {
        checkpoint::ReadFields<M::Dim, 6>(io,
                                          reader,
                                          "em0",
                                          range,
                                          domain.fields.em0);
        adios2::Box<adios2::Dims> range3;
        for (auto d { 0u }; d < M::Dim; ++d) {
          range3.first.push_back(domain.offset_ncells()[d] +
                                 2 * N_GHOSTS * domain.offset_ndomains()[d]);
          range3.second.push_back(domain.mesh.n_all()[d]);
        }
        range3.first.push_back(0);
        range3.second.push_back(3);
        checkpoint::ReadFields<M::Dim, 3>(io,
                                          reader,
                                          "cur0",
                                          range3,
                                          domain.fields.cur0);
      }
      for (auto s { 0u }; s < (unsigned short)(domain.species.size()); ++s) {
        const auto [loc_npart, offset_npart] =
          checkpoint::ReadParticleCount(io, reader, s, ldidx, ndomains());
        raise::ErrorIf(loc_npart > domain.species[s].maxnpart(),
                       "loc_npart > domain.species[s].maxnpart()",
                       HERE);
        if (loc_npart == 0) {
          continue;
        }
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i1",
                                            s,
                                            domain.species[s].i1,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx1",
                                                 s,
                                                 domain.species[s].dx1,
                                                 loc_npart,
                                                 offset_npart);
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i1_prev",
                                            s,
                                            domain.species[s].i1_prev,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx1_prev",
                                                 s,
                                                 domain.species[s].dx1_prev,
                                                 loc_npart,
                                                 offset_npart);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i2",
                                            s,
                                            domain.species[s].i2,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx2",
                                                 s,
                                                 domain.species[s].dx2,
                                                 loc_npart,
                                                 offset_npart);
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i2_prev",
                                            s,
                                            domain.species[s].i2_prev,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx2_prev",
                                                 s,
                                                 domain.species[s].dx2_prev,
                                                 loc_npart,
                                                 offset_npart);
        }
        if constexpr (M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i3",
                                            s,
                                            domain.species[s].i3,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx3",
                                                 s,
                                                 domain.species[s].dx3,
                                                 loc_npart,
                                                 offset_npart);
          checkpoint::ReadParticleData<int>(io,
                                            reader,
                                            "i3_prev",
                                            s,
                                            domain.species[s].i3_prev,
                                            loc_npart,
                                            offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(io,
                                                 reader,
                                                 "dx3_prev",
                                                 s,
                                                 domain.species[s].dx3_prev,
                                                 loc_npart,
                                                 offset_npart);
        }
        if constexpr (M::Dim == Dim::_2D and M::CoordType != Coord::Cart) {
          checkpoint::ReadParticleData<real_t>(io,
                                               reader,
                                               "phi",
                                               s,
                                               domain.species[s].phi,
                                               loc_npart,
                                               offset_npart);
        }
        checkpoint::ReadParticleData<real_t>(io,
                                             reader,
                                             "ux1",
                                             s,
                                             domain.species[s].ux1,
                                             loc_npart,
                                             offset_npart);
        checkpoint::ReadParticleData<real_t>(io,
                                             reader,
                                             "ux2",
                                             s,
                                             domain.species[s].ux2,
                                             loc_npart,
                                             offset_npart);
        checkpoint::ReadParticleData<real_t>(io,
                                             reader,
                                             "ux3",
                                             s,
                                             domain.species[s].ux3,
                                             loc_npart,
                                             offset_npart);
        checkpoint::ReadParticleData<short>(io,
                                            reader,
                                            "tag",
                                            s,
                                            domain.species[s].tag,
                                            loc_npart,
                                            offset_npart);
        checkpoint::ReadParticleData<real_t>(io,
                                             reader,
                                             "weight",
                                             s,
                                             domain.species[s].weight,
                                             loc_npart,
                                             offset_npart);

        const auto nplds = domain.species[s].npld();
        if (nplds > 0) {
          checkpoint::ReadParticlePayloads(io,
                                           reader,
                                           s,
                                           domain.species[s].pld,
                                           nplds,
                                           loc_npart,
                                           offset_npart);
        }
        domain.species[s].set_npart(loc_npart);
      } // species loop

    } // local subdomain loop

    reader.EndStep();
    reader.Close();
    logger::Checkpoint(
      fmt::format("Checkpoint reading done from %s", fname.c_str()),
      HERE);
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt
