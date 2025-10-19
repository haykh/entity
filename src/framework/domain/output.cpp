#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include "kernels/divergences.hpp"
#include "kernels/fields_to_phys.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/prtls_to_phys.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <algorithm>
#include <iterator>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::InitWriter(adios2::ADIOS*          ptr_adios,
                                    const SimulationParams& params) {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");

    auto glob_shape_with_ghosts = mesh().n_active();
    auto off_ncells_with_ghosts = local_domain->offset_ncells();
    auto off_ndomains           = local_domain->offset_ndomains();
    auto loc_shape_with_ghosts  = local_domain->mesh.n_active();
    if (incl_ghosts) {
      for (auto d { 0 }; d <= M::Dim; ++d) {
        glob_shape_with_ghosts[d] += 2 * N_GHOSTS * ndomains_per_dim()[d];
        off_ncells_with_ghosts[d] += 2 * N_GHOSTS * off_ndomains[d];
        loc_shape_with_ghosts[d]  += 2 * N_GHOSTS;
      }
    }

    g_writer.init(ptr_adios,
                  params.template get<std::string>("output.format"),
                  params.template get<std::string>("simulation.name"),
                  params.template get<bool>("output.separate_files"));
    g_writer.defineMeshLayout(glob_shape_with_ghosts,
                              off_ncells_with_ghosts,
                              loc_shape_with_ghosts,
                              { local_domain->index(), ndomains() },
                              params.template get<std::vector<unsigned int>>(
                                "output.fields.downsampling"),
                              incl_ghosts,
                              M::CoordType);
    const auto fields_to_write = params.template get<std::vector<std::string>>(
      "output.fields.quantities");
    const auto custom_fields_to_write = params.template get<std::vector<std::string>>(
      "output.fields.custom");
    std::vector<std::string> all_fields_to_write;
    std::merge(fields_to_write.begin(),
               fields_to_write.end(),
               custom_fields_to_write.begin(),
               custom_fields_to_write.end(),
               std::back_inserter(all_fields_to_write));
    const auto species_to_write = params.template get<std::vector<spidx_t>>(
      "output.particles.species");
    g_writer.defineFieldOutputs(S, all_fields_to_write);

    Dimension dim = M::PrtlDim;
    if constexpr (M::CoordType != Coord::Cart) {
      dim = Dim::_3D;
    }
    g_writer.defineParticleOutputs(dim, species_to_write);

    // spectra write all particle species
    std::vector<spidx_t> spectra_species {};
    for (const auto& sp : species_params()) {
      spectra_species.push_back(sp.index());
    }
    g_writer.defineSpectraOutputs(spectra_species);
    for (const auto& type : { "fields", "particles", "spectra" }) {
      g_writer.addTracker(type,
                          params.template get<timestep_t>(
                            "output." + std::string(type) + ".interval"),
                          params.template get<simtime_t>(
                            "output." + std::string(type) + ".interval_time"));
    }
    g_writer.writeAttrs(params);
  }

  template <SimEngine::type S, class M, FldsID::type F>
  void ComputeMoments(const SimulationParams& params,
                      const Mesh<M>&          mesh,
                      const std::vector<Particles<M::Dim, M::CoordType>>& prtl_species,
                      const std::vector<spidx_t>&        species,
                      const std::vector<unsigned short>& components,
                      ndfield_t<M::Dim, 6>&              buffer,
                      idx_t                              buff_idx) {
    std::vector<spidx_t> specs = species;
    if (specs.size() == 0) {
      // if no species specified, take all massive species
      for (auto& sp : prtl_species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }
    }
    for (const auto& sp : specs) {
      raise::ErrorIf((sp > prtl_species.size()) or (sp == 0),
                     "Invalid species index " + std::to_string(sp),
                     HERE);
    }
    auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);

    // some parameters
    const auto use_weights = params.template get<bool>("particles.use_weights");
    const auto ni2         = mesh.n_active(in::x2);
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto window      = params.template get<unsigned short>(
      "output.fields.mom_smooth");

    for (const auto& sp : specs) {
      auto& prtl_spec = prtl_species[sp - 1];
      // clang-format off
      Kokkos::parallel_for(
        "ComputeMoments",
        prtl_spec.rangeActiveParticles(),
        kernel::ParticleMoments_kernel<S, M, F, 6>(components, scatter_buff, buff_idx,
                                                   prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
                                                   prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
                                                   prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
                                                   prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
                                                   prtl_spec.mass(), prtl_spec.charge(),
                                                   use_weights,
                                                   mesh.metric, mesh.flds_bc(),
                                                   ni2, inv_n0, window));
      // clang-format on
    }
    Kokkos::Experimental::contribute(buffer, scatter_buff);
  }

  template <Dimension D, int N, int M>
  void DeepCopyFields(ndfield_t<D, N>&     fld_from,
                      ndfield_t<D, M>&     fld_to,
                      const range_tuple_t& from,
                      const range_tuple_t& to) {
    for (auto d { 0u }; d < D; ++d) {
      raise::ErrorIf(fld_from.extent(d) != fld_to.extent(d),
                     "Fields have different sizes " +
                       std::to_string(fld_from.extent(d)) +
                       " != " + std::to_string(fld_to.extent(d)),
                     HERE);
    }
    if constexpr (D == Dim::_1D) {
      Kokkos::deep_copy(Kokkos::subview(fld_to, Kokkos::ALL, to),
                        Kokkos::subview(fld_from, Kokkos::ALL, from));
    } else if constexpr (D == Dim::_2D) {
      Kokkos::deep_copy(Kokkos::subview(fld_to, Kokkos::ALL, Kokkos::ALL, to),
                        Kokkos::subview(fld_from, Kokkos::ALL, Kokkos::ALL, from));
    } else if constexpr (D == Dim::_3D) {
      Kokkos::deep_copy(
        Kokkos::subview(fld_to, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, to),
        Kokkos::subview(fld_from, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, from));
    }
  }

  template <SimEngine::type S, class M>
  void ComputeVectorPotential(ndfield_t<M::Dim, 6>& buffer,
                              ndfield_t<M::Dim, 6>& EM,
                              unsigned short        buff_idx,
                              const Mesh<M>         mesh) {
    if constexpr (M::Dim == Dim::_2D) {
      const auto metric = mesh.metric;
      Kokkos::parallel_for(
        "ComputeVectorPotential",
        mesh.rangeActiveCells(),
        Lambda(index_t i1, index_t i2) {
          const real_t   i1_ { COORD(i1) };
          const ncells_t k_min = 0;
          const ncells_t k_max = (i2 - (N_GHOSTS));
          real_t         A3    = ZERO;
          for (auto k { k_min }; k <= k_max; ++k) {
            const real_t k_ = static_cast<real_t>(k);
            const real_t sqrt_detH_ij1 { metric.sqrt_det_h({ i1_, k_ - HALF }) };
            const real_t sqrt_detH_ij2 { metric.sqrt_det_h({ i1_, k_ + HALF }) };
            const auto k1 { k + N_GHOSTS };
            A3 += HALF * (sqrt_detH_ij1 * EM(i1, k1 - 1, em::bx1) +
                          sqrt_detH_ij2 * EM(i1, k1, em::bx1));
          }
          buffer(i1, i2, buff_idx) = A3;
        });

      // @TODO: Implementation with team policies works on AMD, but not on NVIDIA GPUs
      //
      // using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
      // const auto nx1   = mesh.n_active(in::x1);
      // const auto nx2   = mesh.n_active(in::x2);
      //
      // TeamPolicy policy(nx1, Kokkos::AUTO);
      //
      // Kokkos::parallel_for(
      //   "ComputeVectorPotential",
      //   policy,
      //   Lambda(const TeamPolicy::member_type& team_member) {
      //     index_t i1 = team_member.league_rank();
      //     Kokkos::parallel_scan(
      //       Kokkos::TeamThreadRange(team_member, nx2),
      //       [=](index_t i2, real_t& update, const bool final_pass) {
      //         const auto i1_ { static_cast<real_t>(i1) };
      //         const auto i2_ { static_cast<real_t>(i2) };
      //         const real_t sqrt_detH_ijM { metric.sqrt_det_h({ i1_, i2_ - HALF }) };
      //         const real_t sqrt_detH_ijP { metric.sqrt_det_h({ i1_, i2_ + HALF }) };
      //         const auto input_val =
      //           HALF *
      //           (sqrt_detH_ijM * EM(i1 + N_GHOSTS, i2 + N_GHOSTS - 1, em::bx1) +
      //            sqrt_detH_ijP * EM(i1 + N_GHOSTS, i2 + N_GHOSTS, em::bx1));
      //         if (final_pass) {
      //           buffer(i1 + N_GHOSTS, i2 + N_GHOSTS, buff_idx) = update;
      //         }
      //         update += input_val;
      //       });
      //   });
    } else {
      raise::KernelError(
        HERE,
        "ComputeVectorPotential: 2D implementation called for D != 2");
    }
  }

#if defined(MPI_ENABLED) && defined(OUTPUT_ENABLED)
  template <SimEngine::type S, class M>
  void ExtractVectorPotential(ndfield_t<M::Dim, 6>& buffer,
                              array_t<real_t*>&     aphi_r,
                              unsigned short        buff_idx,
                              const Mesh<M>         mesh) {
    Kokkos::parallel_for(
      "AddVectorPotential",
      mesh.rangeActiveCells(),
      Lambda(index_t i1, index_t i2) {
        buffer(i1, i2, buff_idx) += aphi_r(i1 - N_GHOSTS);
      });
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::CommunicateVectorPotential(unsigned short buff_idx) {
    if constexpr (M::Dim == Dim::_2D) {
      auto       local_domain = subdomain_ptr(l_subdomain_indices()[0]);
      const auto nx1          = local_domain->mesh.n_active(in::x1);
      const auto nx2          = local_domain->mesh.n_active(in::x2);

      auto& buffer = local_domain->fields.bckp;

      const auto nranks_x1 = ndomains_per_dim()[0];
      const auto nranks_x2 = ndomains_per_dim()[1];

      for (auto nr2 { 1u }; nr2 < nranks_x2; ++nr2) {
        const auto rank_send_pre = (nr2 - 1u) * nranks_x1;
        const auto rank_recv_pre = nr2 * nranks_x1;
        for (auto nr1 { 0u }; nr1 < nranks_x1; ++nr1) {
          const auto rank_send = rank_send_pre + nr1;
          const auto rank_recv = rank_recv_pre + nr1;
          if (local_domain->mpi_rank() == rank_send) {
            array_t<real_t*> aphi_r { "Aphi_r", nx1 };
            Kokkos::deep_copy(
              aphi_r,
              Kokkos::subview(buffer,
                              std::make_pair(N_GHOSTS, N_GHOSTS + nx1),
                              N_GHOSTS + nx2 - 1,
                              buff_idx));
            MPI_Send(aphi_r.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_recv,
                     0,
                     MPI_COMM_WORLD);
          } else if (local_domain->mpi_rank() == rank_recv) {
            array_t<real_t*> aphi_r { "Aphi_r", nx1 };
            MPI_Recv(aphi_r.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_send,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            ExtractVectorPotential<S, M>(buffer, aphi_r, buff_idx, local_domain->mesh);
          }
        }
      }
    } else {
      raise::Error("CommunicateVectorPotential: comm vector potential only "
                   "possible for 2D",
                   HERE);
    }
  }
#endif

  template <SimEngine::type S, class M>
  auto Metadomain<S, M>::Write(
    const SimulationParams&                  params,
    timestep_t                               current_step,
    timestep_t                               finished_step,
    simtime_t                                current_time,
    simtime_t                                finished_time,
    std::function<void(const std::string&,
                       ndfield_t<M::Dim, 6>&,
                       index_t,
                       timestep_t,
                       simtime_t,
                       const Domain<S, M>&)> CustomFieldOutput) -> bool {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    const auto write_fields = params.template get<bool>(
                                "output.fields.enable") and
                              g_writer.shouldWrite("fields",
                                                   finished_step,
                                                   finished_time);
    const auto write_particles = params.template get<bool>(
                                   "output.particles.enable") and
                                 g_writer.shouldWrite("particles",
                                                      finished_step,
                                                      finished_time);
    const auto write_spectra = params.template get<bool>(
                                 "output.spectra.enable") and
                               g_writer.shouldWrite("spectra",
                                                    finished_step,
                                                    finished_time);
    const auto extension = params.template get<std::string>("output.format");
    if (not(write_fields or write_particles or write_spectra) and
        extension != "disabled") {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing output", HERE);
    if (write_fields) {
      g_writer.beginWriting(WriteMode::Fields, current_step, current_time);
      const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");
      const auto dwn         = params.template get<std::vector<unsigned int>>(
        "output.fields.downsampling");

      auto off_ncells_with_ghosts = local_domain->offset_ncells();
      auto loc_shape_with_ghosts  = local_domain->mesh.n_active();
      { // compute positions/sizes of meshblocks in cells in all dimensions
        const auto off_ndomains = local_domain->offset_ndomains();
        if (incl_ghosts) {
          for (auto d { 0 }; d <= M::Dim; ++d) {
            off_ncells_with_ghosts[d] += 2 * N_GHOSTS * off_ndomains[d];
            loc_shape_with_ghosts[d]  += 2 * N_GHOSTS;
          }
        }
      }
      for (auto dim { 0u }; dim < M::Dim; ++dim) {
        const auto l_size   = local_domain->mesh.n_active()[dim];
        const auto l_offset = local_domain->offset_ncells()[dim];
        const auto g_size   = mesh().n_active()[dim];

        const auto dwn_in_dim = dwn[dim];

        const double n = l_size;
        const double d = dwn_in_dim;
        const double l = l_offset;
        const double f = math::ceil(l / d) * d - l;

        const auto first_cell = static_cast<ncells_t>(f);
        const auto l_size_dwn = static_cast<ncells_t>(math::ceil((n - f) / d));

        const auto is_last = l_offset + l_size == g_size;

        const auto add_ghost = (incl_ghosts ? 2 * N_GHOSTS : 0);
        const auto add_last  = (is_last ? 1 : 0);

        array_t<real_t*> xc { "Xc", l_size_dwn + add_ghost };
        array_t<real_t*> xe { "Xe", l_size_dwn + add_ghost + add_last };

        const auto offset = (incl_ghosts ? N_GHOSTS : 0);
        const auto ncells = l_size_dwn;

        const auto& metric = local_domain->mesh.metric;

        Kokkos::parallel_for(
          "GenerateMesh",
          ncells,
          Lambda(index_t i_dwn) {
            const auto      i  = first_cell + i_dwn * dwn_in_dim;
            const auto      i_ = static_cast<real_t>(i);
            coord_t<M::Dim> x_Cd { ZERO }, x_Ph { ZERO };
            x_Cd[dim] = i_ + HALF;
            // TODO : change to convert by component
            metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
            xc(offset + i_dwn) = x_Ph[dim];
            x_Cd[dim]          = i_;
            metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
            xe(offset + i_dwn) = x_Ph[dim];
            if (is_last && i_dwn == ncells - 1) {
              x_Cd[dim] = i_ + ONE;
              metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
              xe(offset + i_dwn + 1) = x_Ph[dim];
            }
          });
        g_writer.writeMesh(
          dim,
          xc,
          xe,
          { off_ncells_with_ghosts[dim], loc_shape_with_ghosts[dim] });
      }
      const auto output_asis = params.template get<bool>("output.debug.as_is");
      // !TODO: this can probably be optimized to dump things at once
      for (auto& fld : g_writer.fieldWriters()) {
        Kokkos::deep_copy(local_domain->fields.bckp, ZERO);
        std::vector<std::string> names;
        std::vector<std::size_t> addresses;
        if (fld.comp.size() == 0 || fld.comp.size() == 1) { // scalar
          names.push_back(fld.name());
          addresses.push_back(0);
          if (fld.is_moment()) {
            // output a particle distribution moment (single component)
            // this includes T, Rho, Charge, N, Nppc
            const auto c = static_cast<idx_t>(addresses.back());
            if (fld.id() == FldsID::T) {
              raise::ErrorIf(fld.comp.size() != 1,
                             "Wrong # of components requested for T output",
                             HERE);
              ComputeMoments<S, M, FldsID::T>(params,
                                              local_domain->mesh,
                                              local_domain->species,
                                              fld.species,
                                              fld.comp[0],
                                              local_domain->fields.bckp,
                                              c);
            } else if (fld.id() == FldsID::Rho) {
              ComputeMoments<S, M, FldsID::Rho>(params,
                                                local_domain->mesh,
                                                local_domain->species,
                                                fld.species,
                                                {},
                                                local_domain->fields.bckp,
                                                c);
            } else if (fld.id() == FldsID::Charge) {
              ComputeMoments<S, M, FldsID::Charge>(params,
                                                   local_domain->mesh,
                                                   local_domain->species,
                                                   fld.species,
                                                   {},
                                                   local_domain->fields.bckp,
                                                   c);
            } else if (fld.id() == FldsID::N) {
              ComputeMoments<S, M, FldsID::N>(params,
                                              local_domain->mesh,
                                              local_domain->species,
                                              fld.species,
                                              {},
                                              local_domain->fields.bckp,
                                              c);
            } else if (fld.id() == FldsID::Nppc) {
              ComputeMoments<S, M, FldsID::Nppc>(params,
                                                 local_domain->mesh,
                                                 local_domain->species,
                                                 fld.species,
                                                 {},
                                                 local_domain->fields.bckp,
                                                 c);
            } else if (fld.id() == FldsID::V) {
              if constexpr (S != SimEngine::GRPIC) {
                ComputeMoments<S, M, FldsID::V>(params,
                                                local_domain->mesh,
                                                local_domain->species,
                                                fld.species,
                                                fld.comp[0],
                                                local_domain->fields.bckp,
                                                c);
              } else {
                raise::Error("Bulk velocity not supported for GRPIC", HERE);
              }
            } else {
              raise::Error("Wrong moment requested for output", HERE);
            }
          } else if (fld.is_divergence()) {
            // @TODO: is this correct for GR too? not em0?
            const auto c = static_cast<idx_t>(addresses.back());
            Kokkos::parallel_for(
              "ComputeDivergence",
              local_domain->mesh.rangeActiveCells(),
              kernel::ComputeDivergence_kernel<M, 6>(local_domain->mesh.metric,
                                                     local_domain->fields.em,
                                                     local_domain->fields.bckp,
                                                     c));
          } else if (fld.is_custom()) {
            if (CustomFieldOutput) {
              CustomFieldOutput(fld.name().substr(1),
                                local_domain->fields.bckp,
                                addresses.back(),
                                finished_step,
                                finished_time,
                                *local_domain);
            } else {
              raise::Error("Custom output requested but no function provided",
                           HERE);
            }
          } else if (fld.is_vpotential()) {
            if constexpr (S == SimEngine::GRPIC && M::Dim == Dim::_2D) {
              const auto c = static_cast<unsigned short>(addresses.back());
              ComputeVectorPotential<S, M>(local_domain->fields.bckp,
                                           local_domain->fields.em,
                                           c,
                                           local_domain->mesh);
#if defined(MPI_ENABLED)
              CommunicateVectorPotential(c);
#endif
            } else {
              raise::Error(
                "Vector potential can only be computed for GRPIC in 2D",
                HERE);
            }
          } else {
            raise::Error("Wrong # of components requested for "
                         "non-moment/non-custom output",
                         HERE);
          }
          SynchronizeFields(*local_domain,
                            Comm::Bckp,
                            { addresses.back(), addresses.back() + 1 });
        } else if (fld.comp.size() == 3) { // vector
          for (auto i = 0; i < 3; ++i) {
            names.push_back(fld.name(i));
            addresses.push_back(i + 3);
          }
          if (fld.is_moment()) {
            for (auto i = 0; i < 3; ++i) {
              const auto c = static_cast<idx_t>(addresses[i]);
              if (fld.id() == FldsID::T) {
                raise::ErrorIf(fld.comp[i].size() != 2,
                               "Wrong # of components requested for moment",
                               HERE);
                ComputeMoments<S, M, FldsID::T>(params,
                                                local_domain->mesh,
                                                local_domain->species,
                                                fld.species,
                                                fld.comp[i],
                                                local_domain->fields.bckp,
                                                c);
              } else if (fld.id() == FldsID::V) {
                raise::ErrorIf(fld.comp[i].size() != 1,
                               "Wrong # of components requested for 3vel",
                               HERE);
                if constexpr (S == SimEngine::SRPIC) {
                  ComputeMoments<S, M, FldsID::V>(params,
                                                  local_domain->mesh,
                                                  local_domain->species,
                                                  fld.species,
                                                  fld.comp[i],
                                                  local_domain->fields.bckp,
                                                  c);
                } else {
                  raise::Error("Bulk velocity not supported for GRPIC", HERE);
                }
              } else {
                raise::Error("Wrong moment requested for output", HERE);
              }
            }
            raise::ErrorIf(addresses[1] - addresses[0] !=
                             addresses[2] - addresses[1],
                           "Indices for the backup are not contiguous",
                           HERE);
            SynchronizeFields(*local_domain,
                              Comm::Bckp,
                              { addresses[0], addresses[2] + 1 });
            if constexpr (S == SimEngine::SRPIC) {
              if (fld.id() == FldsID::V) {
                // normalize 3vel * rho (combuted above) by rho
                ComputeMoments<S, M, FldsID::Rho>(params,
                                                  local_domain->mesh,
                                                  local_domain->species,
                                                  fld.species,
                                                  {},
                                                  local_domain->fields.bckp,
                                                  0u);
                SynchronizeFields(*local_domain, Comm::Bckp, { 0, 1 });
                Kokkos::parallel_for("NormalizeVectorByRho",
                                     local_domain->mesh.rangeActiveCells(),
                                     kernel::NormalizeVectorByRho_kernel<M::Dim, 6>(
                                       local_domain->fields.bckp,
                                       local_domain->fields.bckp,
                                       0,
                                       addresses[0],
                                       addresses[1],
                                       addresses[2]));
              }
            }
          } else {
            // copy fields to bckp (:, 0, 1, 2)
            // if as-is specified ==> copy directly to 3, 4, 5
            range_tuple_t copy_to = { 0, 3 };
            if (output_asis) {
              copy_to = { 3, 6 };
            }
            if (fld.is_current()) {
              DeepCopyFields<M::Dim, 3, 6>(local_domain->fields.cur,
                                           local_domain->fields.bckp,
                                           { cur::jx1, cur::jx3 + 1 },
                                           copy_to);
            } else if (fld.is_field()) {
              if (S == SimEngine::GRPIC && fld.is_gr_aux_field()) {
                if (fld.is_efield()) {
                  // GR: E
                  DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.aux,
                                               local_domain->fields.bckp,
                                               { em::ex1, em::ex3 + 1 },
                                               copy_to);
                } else {
                  // GR: H
                  DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.aux,
                                               local_domain->fields.bckp,
                                               { em::hx1, em::hx3 + 1 },
                                               copy_to);
                }
              } else {
                if (fld.is_efield()) {
                  // GR/SR: D/E
                  DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.em,
                                               local_domain->fields.bckp,
                                               { em::ex1, em::ex3 + 1 },
                                               copy_to);
                } else {
                  // GR/SR: B
                  DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.em,
                                               local_domain->fields.bckp,
                                               { em::bx1, em::bx3 + 1 },
                                               copy_to);
                }
              }
            } else {
              raise::Error("Wrong field requested for output", HERE);
            }
            if (not output_asis) {
              // copy fields from bckp(:, 0, 1, 2) -> bckp(:, 3, 4, 5)
              // converting to proper basis and properly interpolating
              list_t<idx_t, 3> comp_from = { 0, 1, 2 };
              list_t<idx_t, 3> comp_to   = { 3, 4, 5 };
              DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.bckp,
                                           local_domain->fields.bckp,
                                           { 0, 3 },
                                           { 3, 6 });
              Kokkos::parallel_for("FieldsToPhys",
                                   local_domain->mesh.rangeActiveCells(),
                                   kernel::FieldsToPhys_kernel<M, 6, 6>(
                                     local_domain->fields.bckp,
                                     local_domain->fields.bckp,
                                     comp_from,
                                     comp_to,
                                     fld.interp_flag | fld.prepare_flag,
                                     local_domain->mesh.metric));
            }
          }
        } else if (fld.comp.size() == 6) { // tensor
          raise::ErrorIf(not fld.is_moment() or fld.id() != FldsID::T,
                         "Only T tensor has 6 components",
                         HERE);
          for (auto i = 0; i < 6; ++i) {
            names.push_back(fld.name(i));
            addresses.push_back(i);
            const auto c = static_cast<idx_t>(addresses.back());
            raise::ErrorIf(fld.comp[i].size() != 2,
                           "Wrong # of components requested for moment",
                           HERE);
            ComputeMoments<S, M, FldsID::T>(params,
                                            local_domain->mesh,
                                            local_domain->species,
                                            fld.species,
                                            fld.comp[i],
                                            local_domain->fields.bckp,
                                            c);
          }
          SynchronizeFields(*local_domain,
                            Comm::Bckp,
                            { addresses[0], addresses[5] + 1 });
        } else {
          raise::Error("Wrong # of components requested for output", HERE);
        }
        g_writer.writeField<M::Dim, 6>(names, local_domain->fields.bckp, addresses);
      }
      g_writer.endWriting(WriteMode::Fields);
    } // end shouldWrite("fields", step, time)

    if (write_particles) {
      g_writer.beginWriting(WriteMode::Particles, current_step, current_time);
      const auto prtl_stride = params.template get<npart_t>(
        "output.particles.stride");
      for (const auto& prtl : g_writer.speciesWriters()) {
        auto& species = local_domain->species[prtl.species() - 1];
        if (not species.is_sorted()) {
          species.RemoveDead();
        }
        // @HACK_FOR_SASHA >:
        // const npart_t    nout = species.npart() / prtl_stride;
        // ignoring prtl_stride
        npart_t ncriterion = 0;
        auto    tag        = species.tag;
        auto    ux1        = species.ux1;
        auto    ux2        = species.ux2;
        auto    ux3        = species.ux3;
        Kokkos::parallel_reduce(
          "CountNout",
          species.rangeActiveParticles(),
          Lambda(index_t p, npart_t & l_nout) {
            if (tag(p) == ParticleTag::alive &&
                math::sqrt(ONE + ux1(p) * ux1(p) + ux2(p) * ux2(p) +
                           ux3(p) * ux3(p)) > 10.0) { // criterion
              l_nout += 1;
            }
          },
          ncriterion);
        array_t<npart_t>  prtl_ind_cntr { "prtl_ind_cntr" };
        array_t<npart_t*> prtl_indices { "prtl_indices", ncriterion };
        Kokkos::parallel_for(
          "GeneratePrtlIndices",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) == ParticleTag::alive &&
                math::sqrt(ONE + ux1(p) * ux1(p) + ux2(p) * ux2(p) +
                           ux3(p) * ux3(p)) > 10.0) { // criterion
              const auto idx    = Kokkos::atomic_fetch_add(&prtl_ind_cntr(), 1);
              prtl_indices(idx) = p;
            }
          });
        const npart_t    nout = ncriterion / prtl_stride;
        // @HACK_FOR_SASHA <:
        array_t<real_t*> buff_x1, buff_x2, buff_x3;
        array_t<real_t*> buff_ux1 { "u1", nout };
        array_t<real_t*> buff_ux2 { "ux2", nout };
        array_t<real_t*> buff_ux3 { "ux3", nout };
        array_t<real_t*> buff_wei { "w", nout };
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          buff_x1 = array_t<real_t*> { "x1", nout };
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          buff_x2 = array_t<real_t*> { "x2", nout };
        }
        if constexpr (M::Dim == Dim::_3D or
                      ((D == Dim::_2D) and (M::CoordType != Coord::Cart))) {
          buff_x3 = array_t<real_t*> { "x3", nout };
        }
        if (nout > 0) {
          // clang-format off
          Kokkos::parallel_for(
            "PrtlToPhys",
            nout,
            kernel::PrtlToPhys_kernel<S, M>(prtl_stride,
                                                    // @HACK_FOR_SASHA >:
                                                    prtl_indices,
                                                    // @HACK_FOR_SASHA <:
                                                    buff_x1, buff_x2, buff_x3,
                                                    buff_ux1, buff_ux2, buff_ux3,
                                                    buff_wei,
                                                    species.i1, species.i2, species.i3,
                                                    species.dx1, species.dx2, species.dx3,
                                                    species.ux1, species.ux2, species.ux3,
                                                    species.phi, species.weight,
                                                    local_domain->mesh.metric));
          // clang-format on
        }
        npart_t offset   = 0;
        npart_t glob_tot = nout;
#if defined(MPI_ENABLED)
        auto glob_nout = std::vector<npart_t>(g_ndomains);
        MPI_Allgather(&nout,
                      1,
                      mpi::get_type<npart_t>(),
                      glob_nout.data(),
                      1,
                      mpi::get_type<npart_t>(),
                      MPI_COMM_WORLD);
        glob_tot = 0;
        for (auto r = 0; r < g_mpi_size; ++r) {
          if (r < g_mpi_rank) {
            offset += glob_nout[r];
          }
          glob_tot += glob_nout[r];
        }
#endif // MPI_ENABLED
        g_writer.writeParticleQuantity(buff_wei, glob_tot, offset, prtl.name("W", 0));
        g_writer.writeParticleQuantity(buff_ux1, glob_tot, offset, prtl.name("U", 1));
        g_writer.writeParticleQuantity(buff_ux2, glob_tot, offset, prtl.name("U", 2));
        g_writer.writeParticleQuantity(buff_ux3, glob_tot, offset, prtl.name("U", 3));
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          g_writer.writeParticleQuantity(buff_x1, glob_tot, offset, prtl.name("X", 1));
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          g_writer.writeParticleQuantity(buff_x2, glob_tot, offset, prtl.name("X", 2));
        }
        if constexpr (M::Dim == Dim::_3D or
                      ((D == Dim::_2D) and (M::CoordType != Coord::Cart))) {
          g_writer.writeParticleQuantity(buff_x3, glob_tot, offset, prtl.name("X", 3));
        }
      }
      g_writer.endWriting(WriteMode::Particles);
    } // end shouldWrite("particles", step, time)

    if (write_spectra) {
      g_writer.beginWriting(WriteMode::Spectra, current_step, current_time);
      const auto log_bins = params.template get<bool>(
        "output.spectra.log_bins");
      const auto n_bins = params.template get<std::size_t>(
        "output.spectra.n_bins");
      auto e_min = params.template get<real_t>("output.spectra.e_min");
      auto e_max = params.template get<real_t>("output.spectra.e_max");
      if (log_bins) {
        e_min = math::log10(e_min);
        e_max = math::log10(e_max);
      }
      array_t<real_t*> energy { "energy", n_bins + 1 };
      Kokkos::parallel_for(
        "GenerateEnergyBins",
        n_bins + 1,
        Lambda(index_t e) {
          if (log_bins) {
            energy(e) = math::pow(10.0, e_min + (e_max - e_min) * e / n_bins);
          } else {
            energy(e) = e_min + (e_max - e_min) * e / n_bins;
          }
        });
      for (const auto& spec : g_writer.spectraWriters()) {
        auto&            species = local_domain->species[spec.species() - 1];
        array_t<real_t*> dn { "dn", n_bins };
        auto       dn_scatter = Kokkos::Experimental::create_scatter_view(dn);
        auto       ux1        = species.ux1;
        auto       ux2        = species.ux2;
        auto       ux3        = species.ux3;
        auto       weight     = species.weight;
        auto       tag        = species.tag;
        const auto is_massive = species.mass() > 0.0f;
        Kokkos::parallel_for(
          "ComputeSpectra",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            real_t en;
            if (is_massive) {
              en = U2GAMMA(ux1(p), ux2(p), ux3(p)) - ONE;
            } else {
              en = NORM(ux1(p), ux2(p), ux3(p));
            }
            if (log_bins) {
              en = math::log10(en);
            }
            std::size_t e_ind = 0;
            if (en <= e_min) {
              e_ind = 0;
            } else if (en >= e_max) {
              e_ind = n_bins;
            } else {
              e_ind = static_cast<std::size_t>(
                static_cast<real_t>(n_bins) * (en - e_min) / (e_max - e_min));
            }
            auto dn_acc    = dn_scatter.access();
            dn_acc(e_ind) += weight(p);
          });
        Kokkos::Experimental::contribute(dn, dn_scatter);
        g_writer.writeSpectrum(dn, spec.name());
      }
      g_writer.writeSpectrumBins(energy, "sEbn");
      g_writer.endWriting(WriteMode::Spectra);
    } // end shouldWrite("spectra", step, time)

    return true;
  }

#define METADOMAIN_OUTPUT(S, M)                                                \
  template void Metadomain<S, M>::InitWriter(adios2::ADIOS*,                   \
                                             const SimulationParams&);         \
  template auto Metadomain<S, M>::Write(                                       \
    const SimulationParams&,                                                   \
    timestep_t,                                                                \
    timestep_t,                                                                \
    simtime_t,                                                                 \
    simtime_t,                                                                 \
    std::function<void(const std::string&,                                     \
                       ndfield_t<M::Dim, 6>&,                                  \
                       index_t,                                                \
                       timestep_t,                                             \
                       simtime_t,                                              \
                       const Domain<S, M>&)>) -> bool;

  METADOMAIN_OUTPUT(SimEngine::SRPIC, metric::Minkowski<Dim::_1D>)
  METADOMAIN_OUTPUT(SimEngine::SRPIC, metric::Minkowski<Dim::_2D>)
  METADOMAIN_OUTPUT(SimEngine::SRPIC, metric::Minkowski<Dim::_3D>)
  METADOMAIN_OUTPUT(SimEngine::SRPIC, metric::Spherical<Dim::_2D>)
  METADOMAIN_OUTPUT(SimEngine::SRPIC, metric::QSpherical<Dim::_2D>)
  METADOMAIN_OUTPUT(SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>)
  METADOMAIN_OUTPUT(SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>)
  METADOMAIN_OUTPUT(SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>)

#undef METADOMAIN_OUTPUT

#if defined(MPI_ENABLED)
  #define COMMVECTORPOTENTIAL(S, M)                                            \
    template void Metadomain<S, M>::CommunicateVectorPotential(unsigned short);

  COMMVECTORPOTENTIAL(SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>)
  COMMVECTORPOTENTIAL(SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>)
  COMMVECTORPOTENTIAL(SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>)

  #undef COMMVECTORPOTENTIAL
#endif

} // namespace ntt
