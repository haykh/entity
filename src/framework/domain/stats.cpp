#include "enums.h"
#include "global.h"

#include "utils/comparators.h"
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
#include "framework/specialization_registry.h"
#include "framework/parameters.h"

#include "kernels/reduced_stats.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <filesystem>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::InitStatsWriter(const SimulationParams& params,
                                         bool                    is_resuming) {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "StatsWriter for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    const auto simname  = params.template get<std::string>("simulation.name");
    const auto filename = std::filesystem::path(simname) /
                          (simname + "_stats.csv");
    const auto enable_stats = params.template get<bool>("output.stats.enable");
    if (enable_stats and (not is_resuming)) {
      CallOnce(
        [](auto& filename) {
          if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
          }
        },
        filename);
    }
    const auto stats_to_write = params.template get<std::vector<std::string>>(
      "output.stats.quantities");
    const auto custom_stats_to_write = params.template get<std::vector<std::string>>(
      "output.stats.custom");
    g_stats_writer.init(
      params.template get<timestep_t>("output.stats.interval"),
      params.template get<simtime_t>("output.stats.interval_time"));
    g_stats_writer.defineStatsFilename(filename);
    g_stats_writer.defineStatsOutputs(stats_to_write, false);
    g_stats_writer.defineStatsOutputs(custom_stats_to_write, true);

    if (not std::filesystem::exists(filename)) {
      g_stats_writer.writeHeader();
    }
  }

  template <SimEngine::type S, class M, StatsID::type P>
  auto ComputeMoments(const SimulationParams& params,
                      const Mesh<M>&          mesh,
                      const M&                global_metric,
                      const std::vector<Particles<M::Dim, M::CoordType>>& prtl_species,
                      const std::vector<spidx_t>&        species,
                      const std::vector<unsigned short>& components) -> real_t {
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
    // some parameters
    const auto use_weights = params.template get<bool>("particles.use_weights");

    real_t buffer = static_cast<real_t>(0);
    for (const auto& sp : specs) {
      auto& prtl_spec = prtl_species[sp - 1];
      if (P == StatsID::Charge and cmp::AlmostZero_host(prtl_spec.charge())) {
        continue;
      }
      if (P == StatsID::Rho and cmp::AlmostZero_host(prtl_spec.mass())) {
        continue;
      }
      real_t temp_buff = ZERO;
      Kokkos::parallel_reduce(
        "ComputeMoments",
        prtl_spec.rangeActiveParticles(),
        // clang-format off
        kernel::ReducedParticleMoments_kernel<S, M, P>(components,
                                                       prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
                                                       prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
                                                       prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
                                                       prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
                                                       prtl_spec.mass(), prtl_spec.charge(),
                                                       use_weights, mesh.metric),
        // clang-format on
        temp_buff);
      buffer += temp_buff;
    }
    if (P != StatsID::Npart) {
      return buffer / (global_metric.totVolume() *
                       params.template get<real_t>("particles.ppc0"));
    } else {
      return buffer;
    }
  }

  template <SimEngine::type S, class M, StatsID::type F>
  auto ReduceFields(Domain<S, M>*                      domain,
                    const M&                           global_metric,
                    const std::vector<unsigned short>& components) -> real_t {
    auto buffer { ZERO };
    if constexpr (F == StatsID::JdotE) {
      if (components.size() == 0) {
        Kokkos::parallel_reduce(
          "ReduceFields",
          domain->mesh.rangeActiveCells(),
          kernel::ReducedFields_kernel<S, M, F, 0>(domain->fields.em,
                                                   domain->fields.cur,
                                                   domain->mesh.metric),
          buffer);
      } else {
        raise::Error("Components not supported for JdotE", HERE);
      }
    } else if constexpr (
      (S == SimEngine::SRPIC) and
      (F == StatsID::B2 or F == StatsID::E2 or F == StatsID::ExB)) {
      raise::ErrorIf(components.size() != 1,
                     "Components must be of size 1 for B2, E2 or ExB stats",
                     HERE);
      const auto comp = components[0];
      if (comp == 1) {
        Kokkos::parallel_reduce(
          "ReduceFields",
          domain->mesh.rangeActiveCells(),
          kernel::ReducedFields_kernel<S, M, F, 1>(domain->fields.em,
                                                   domain->fields.cur,
                                                   domain->mesh.metric),
          buffer);
      } else if (comp == 2) {
        Kokkos::parallel_reduce(
          "ReduceFields",
          domain->mesh.rangeActiveCells(),
          kernel::ReducedFields_kernel<S, M, F, 2>(domain->fields.em,
                                                   domain->fields.cur,
                                                   domain->mesh.metric),
          buffer);
      } else if (comp == 3) {
        Kokkos::parallel_reduce(
          "ReduceFields",
          domain->mesh.rangeActiveCells(),
          kernel::ReducedFields_kernel<S, M, F, 3>(domain->fields.em,
                                                   domain->fields.cur,
                                                   domain->mesh.metric),
          buffer);
      } else {
        raise::Error(
          "Invalid component for B2, E2 or ExB stats: " + std::to_string(comp),
          HERE);
      }
    } else {
      raise::Error("ReduceFields not implemented for this stats ID + SimEngine "
                   "combination",
                   HERE);
    }

    return buffer / global_metric.totVolume();
  }

  template <SimEngine::type S, class M>
  auto Metadomain<S, M>::WriteStats(
    const SimulationParams& params,
    timestep_t              current_step,
    timestep_t              finished_step,
    simtime_t               current_time,
    simtime_t               finished_time,
    std::function<real_t(const std::string&, timestep_t, simtime_t, const Domain<S, M>&)>
      CustomStat) -> bool {
    if (not(params.template get<bool>("output.stats.enable") and
            g_stats_writer.shouldWrite(finished_step, finished_time))) {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    logger::Checkpoint("Writing stats", HERE);
    g_stats_writer.write(current_step, false);
    g_stats_writer.write(current_time, false);
    for (const auto& stat : g_stats_writer.statsWriters()) {
      if (stat.id() == StatsID::Custom) {
        if (CustomStat != nullptr) {
          g_stats_writer.write(
            CustomStat(stat.name(), finished_step, finished_time, *local_domain));
        } else {
          raise::Error("Custom output requested but no function provided", HERE);
        }
      } else if (stat.id() == StatsID::N) {
        g_stats_writer.write(ComputeMoments<S, M, StatsID::N>(params,
                                                              local_domain->mesh,
                                                              g_mesh.metric,
                                                              local_domain->species,
                                                              stat.species,
                                                              {}));
      } else if (stat.id() == StatsID::Npart) {
        g_stats_writer.write(
          ComputeMoments<S, M, StatsID::Npart>(params,
                                               local_domain->mesh,
                                               g_mesh.metric,
                                               local_domain->species,
                                               stat.species,
                                               {}));
      } else if (stat.id() == StatsID::Rho) {
        g_stats_writer.write(
          ComputeMoments<S, M, StatsID::Rho>(params,
                                             local_domain->mesh,
                                             g_mesh.metric,
                                             local_domain->species,
                                             stat.species,
                                             {}));
      } else if (stat.id() == StatsID::Charge) {
        g_stats_writer.write(
          ComputeMoments<S, M, StatsID::Charge>(params,
                                                local_domain->mesh,
                                                g_mesh.metric,
                                                local_domain->species,
                                                stat.species,
                                                {}));
      } else if (stat.id() == StatsID::T) {
        for (const auto& comp : stat.comp) {
          g_stats_writer.write(
            ComputeMoments<S, M, StatsID::T>(params,
                                             local_domain->mesh,
                                             g_mesh.metric,
                                             local_domain->species,
                                             stat.species,
                                             comp));
        }
      } else if (stat.id() == StatsID::JdotE) {
        g_stats_writer.write(
          ReduceFields<S, M, StatsID::JdotE>(local_domain, g_mesh.metric, {}));
      } else if (S == SimEngine::SRPIC) {
        if (stat.id() == StatsID::E2) {
          for (const auto& comp : stat.comp) {
            g_stats_writer.write(
              ReduceFields<S, M, StatsID::E2>(local_domain, g_mesh.metric, comp));
          }
        } else if (stat.id() == StatsID::B2) {
          for (const auto& comp : stat.comp) {
            g_stats_writer.write(
              ReduceFields<S, M, StatsID::B2>(local_domain, g_mesh.metric, comp));
          }
        } else if (stat.id() == StatsID::ExB) {
          for (const auto& comp : stat.comp) {
            g_stats_writer.write(
              ReduceFields<S, M, StatsID::ExB>(local_domain, g_mesh.metric, comp));
          }
        } else {
          raise::Error("Unrecognized stats ID " + stat.name(), HERE);
        }
      } else {
        raise::Error("StatsID not implemented for particular SimEngine: " +
                       std::to_string(static_cast<int>(S)),
                     HERE);
      }
    }
    g_stats_writer.endWriting();
    return true;
  }

#define METADOMAIN_STATS(S, M, D)                                                \
  template void Metadomain<S, M<D>>::InitStatsWriter(const SimulationParams&, bool); \
  template auto Metadomain<S, M<D>>::WriteStats(                                \
    const SimulationParams&,                                                    \
    timestep_t,                                                                 \
    timestep_t,                                                                 \
    simtime_t,                                                                  \
    simtime_t,                                                                  \
    std::function<                                                              \
      real_t(const std::string&, timestep_t, simtime_t, const Domain<S, M<D>>&)>)-> bool;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_STATS)

#undef METADOMAIN_STATS

} // namespace ntt
