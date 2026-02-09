#include "framework/parameters/grid.h"

#include "defaults.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"
#include "utils/toml.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/parameters/parameters.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ntt {
  namespace params {

    template <typename M>
    auto get_dx0_V0(const std::vector<ncells_t>&         resolution,
                    const boundaries_t<real_t>&          extent,
                    const std::map<std::string, real_t>& params)
      -> std::pair<real_t, real_t> {
      const auto      metric = M(resolution, extent, params);
      const auto      dx0    = metric.dxMin();
      coord_t<M::Dim> x_corner { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x_corner[d] = HALF;
      }
      const auto V0 = metric.sqrt_det_h(x_corner);
      return { dx0, V0 };
    }

    auto GetBoundaryConditions(SimulationParams*  params,
                               const SimEngine&   engine_enum,
                               Dimension          dim,
                               const Coord&       coord_enum,
                               const toml::value& toml_data)
      -> std::tuple<boundaries_t<FldsBC>, boundaries_t<PrtlBC>> {
      auto flds_bc = toml::find<std::vector<std::vector<std::string>>>(
        toml_data,
        "grid",
        "boundaries",
        "fields");
      {
        raise::ErrorIf(flds_bc.size() < 1 || flds_bc.size() > 3,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        params->promiseToDefine("grid.boundaries.fields");
        auto atm_defined = false;
        for (const auto& bcs : flds_bc) {
          for (const auto& bc : bcs) {
            if (fmt::toLower(bc) == "match") {
              params->promiseToDefine("grid.boundaries.match.ds");
            }
            if (fmt::toLower(bc) == "atmosphere") {
              raise::ErrorIf(atm_defined,
                             "ATMOSPHERE is only allowed in one direction",
                             HERE);
              atm_defined = true;
              params->promiseToDefine("grid.boundaries.atmosphere.temperature");
              params->promiseToDefine("grid.boundaries.atmosphere.density");
              params->promiseToDefine("grid.boundaries.atmosphere.height");
              params->promiseToDefine("grid.boundaries.atmosphere.ds");
              params->promiseToDefine("grid.boundaries.atmosphere.species");
              params->promiseToDefine("grid.boundaries.atmosphere.g");
            }
          }
        }
      }

      auto prtl_bc = toml::find<std::vector<std::vector<std::string>>>(
        toml_data,
        "grid",
        "boundaries",
        "particles");
      {
        raise::ErrorIf(prtl_bc.size() < 1 || prtl_bc.size() > 3,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        params->promiseToDefine("grid.boundaries.particles");
        auto atm_defined = false;
        for (const auto& bcs : prtl_bc) {
          for (const auto& bc : bcs) {
            if (fmt::toLower(bc) == "absorb") {
              params->promiseToDefine("grid.boundaries.absorb.ds");
            }
            if (fmt::toLower(bc) == "atmosphere") {
              raise::ErrorIf(atm_defined,
                             "ATMOSPHERE is only allowed in one direction",
                             HERE);
              atm_defined = true;
              params->promiseToDefine("grid.boundaries.atmosphere.temperature");
              params->promiseToDefine("grid.boundaries.atmosphere.density");
              params->promiseToDefine("grid.boundaries.atmosphere.height");
              params->promiseToDefine("grid.boundaries.atmosphere.ds");
              params->promiseToDefine("grid.boundaries.atmosphere.species");
              params->promiseToDefine("grid.boundaries.atmosphere.g");
            }
          }
        }
      }
      std::vector<std::vector<FldsBC>> flds_bc_enum;
      std::vector<std::vector<PrtlBC>> prtl_bc_enum;
      if (coord_enum == Coord::Cart) {
        raise::ErrorIf(flds_bc.size() != (std::size_t)dim,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        raise::ErrorIf(prtl_bc.size() != (std::size_t)dim,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        for (auto d { 0u }; d < (dim_t)dim; ++d) {
          flds_bc_enum.push_back({});
          prtl_bc_enum.push_back({});
          const auto fbc = flds_bc[d];
          const auto pbc = prtl_bc[d];
          raise::ErrorIf(fbc.size() < 1 || fbc.size() > 2,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          raise::ErrorIf(pbc.size() < 1 || pbc.size() > 2,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[0]).c_str());
          auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[0]).c_str());
          if (fbc.size() == 1) {
            raise::ErrorIf(fbc_enum != FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
            flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
          } else {
            raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(fbc_enum);
            auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[1]).c_str());
            raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(fbc_enum);
          }
          if (pbc.size() == 1) {
            raise::ErrorIf(pbc_enum != PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(PrtlBC(PrtlBC::PERIODIC));
            prtl_bc_enum.back().push_back(PrtlBC(PrtlBC::PERIODIC));
          } else {
            raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(pbc_enum);
            auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[1]).c_str());
            raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(pbc_enum);
          }
        }
      } else {
        raise::ErrorIf(flds_bc.size() > 1, "invalid `grid.boundaries.fields`", HERE);
        raise::ErrorIf(prtl_bc.size() > 1,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        if (engine_enum == SimEngine::SRPIC) {
          raise::ErrorIf(flds_bc[0].size() != 2,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          flds_bc_enum.push_back(
            { FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()),
              FldsBC::pick(fmt::toLower(flds_bc[0][1]).c_str()) });
          flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
          if (dim == Dim::_3D) {
            flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
          }
          raise::ErrorIf(prtl_bc[0].size() != 2,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          prtl_bc_enum.push_back(
            { PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()),
              PrtlBC::pick(fmt::toLower(prtl_bc[0][1]).c_str()) });
          prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
          if (dim == Dim::_3D) {
            prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
          }
        } else {
          raise::ErrorIf(flds_bc[0].size() != 1,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          raise::ErrorIf(prtl_bc[0].size() != 1,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          flds_bc_enum.push_back(
            { FldsBC::HORIZON, FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()) });
          flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
          if (dim == Dim::_3D) {
            flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
          }
          prtl_bc_enum.push_back(
            { PrtlBC::HORIZON, PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()) });
          prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
          if (dim == Dim::_3D) {
            prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
          }
        }
      }

      raise::ErrorIf(flds_bc_enum.size() != (std::size_t)dim,
                     "invalid inferred `grid.boundaries.fields`",
                     HERE);
      raise::ErrorIf(prtl_bc_enum.size() != (std::size_t)dim,
                     "invalid inferred `grid.boundaries.particles`",
                     HERE);
      boundaries_t<FldsBC> flds_bc_pairwise;
      boundaries_t<PrtlBC> prtl_bc_pairwise;
      for (auto d { 0u }; d < (dim_t)dim; ++d) {
        raise::ErrorIf(
          flds_bc_enum[d].size() != 2,
          fmt::format("invalid inferred `grid.boundaries.fields[%d]`", d),
          HERE);
        raise::ErrorIf(
          prtl_bc_enum[d].size() != 2,
          fmt::format("invalid inferred `grid.boundaries.particles[%d]`", d),
          HERE);
        flds_bc_pairwise.push_back({ flds_bc_enum[d][0], flds_bc_enum[d][1] });
        prtl_bc_pairwise.push_back({ prtl_bc_enum[d][0], prtl_bc_enum[d][1] });
      }
      return { flds_bc_pairwise, prtl_bc_pairwise };
    }

    void Boundaries::read(Dimension                   dim,
                          const Coord&                coord_enum,
                          const boundaries_t<real_t>& extent_pairwise,
                          const toml::value&          toml_data) {
      if (needs_match_boundaries) {
        if (coord_enum == Coord::Cart) {
          auto min_extent = std::numeric_limits<real_t>::max();
          for (const auto& e : extent_pairwise) {
            min_extent = std::min(min_extent, e.second - e.first);
          }
          const auto default_ds = min_extent * defaults::bc::match::ds_frac;
          try {
            auto ds = toml::find<real_t>(toml_data, "grid", "boundaries", "match", "ds");
            for (auto d = 0u; d < dim; ++d) {
              match_ds_array.push_back({ ds, ds });
            }
          } catch (...) {
            try {
              const auto ds = toml::find<std::vector<std::vector<real_t>>>(
                toml_data,
                "grid",
                "boundaries",
                "match",
                "ds");
              raise::ErrorIf(ds.size() != dim,
                             "invalid # in `grid.boundaries.match.ds`",
                             HERE);
              for (auto d = 0u; d < dim; ++d) {
                if (ds[d].size() == 1) {
                  match_ds_array.push_back({ ds[d][0], ds[d][0] });
                } else if (ds[d].size() == 2) {
                  match_ds_array.push_back({ ds[d][0], ds[d][1] });
                } else if (ds[d].size() == 0) {
                  match_ds_array.push_back({});
                } else {
                  raise::Error("invalid `grid.boundaries.match.ds`", HERE);
                }
              }
            } catch (...) {
              for (auto d = 0u; d < dim; ++d) {
                match_ds_array.push_back({ default_ds, default_ds });
              }
            }
          }
        } else {
          auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
          const auto ds = toml::find_or<real_t>(
            toml_data,
            "grid",
            "boundaries",
            "match",
            "ds",
            r_extent * defaults::bc::match::ds_frac);
          match_ds_array.push_back({ ds, ds });
        }
      }

      if (needs_absorb_boundaries) {
        if (coord_enum == Coord::Cart) {
          auto min_extent = std::numeric_limits<real_t>::max();
          for (const auto& e : extent_pairwise) {
            min_extent = std::min(min_extent, e.second - e.first);
          }
          absorb_ds = toml::find_or(toml_data,
                                    "grid",
                                    "boundaries",
                                    "absorb",
                                    "ds",
                                    min_extent * defaults::bc::absorb::ds_frac);
        } else {
          auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
          absorb_ds     = toml::find_or(toml_data,
                                    "grid",
                                    "boundaries",
                                    "absorb",
                                    "ds",
                                    r_extent * defaults::bc::absorb::ds_frac);
        }
      }

      if (needs_atmosphere_boundaries) {
        atmosphere_temperature = toml::find<real_t>(toml_data,
                                                    "grid",
                                                    "boundaries",
                                                    "atmosphere",
                                                    "temperature");
        atmosphere_height      = toml::find<real_t>(toml_data,
                                               "grid",
                                               "boundaries",
                                               "atmosphere",
                                               "height");
        atmosphere_density     = toml::find<real_t>(toml_data,
                                                "grid",
                                                "boundaries",
                                                "atmosphere",
                                                "density");
        atmosphere_ds =
          toml::find_or(toml_data, "grid", "boundaries", "atmosphere", "ds", ZERO);
        atmosphere_g       = atmosphere_temperature / atmosphere_height;
        atmosphere_species = toml::find<std::pair<spidx_t, spidx_t>>(
          toml_data,
          "grid",
          "boundaries",
          "atmosphere",
          "species");
      }
    }

    void Boundaries::setParams(SimulationParams* params) const {
      if (needs_match_boundaries) {
        params->set("grid.boundaries.match.ds", match_ds_array);
      }
      if (needs_absorb_boundaries) {
        params->set("grid.boundaries.absorb.ds", absorb_ds);
      }
      if (needs_atmosphere_boundaries) {
        params->set("grid.boundaries.atmosphere.temperature",
                    atmosphere_temperature);
        params->set("grid.boundaries.atmosphere.density", atmosphere_density);
        params->set("grid.boundaries.atmosphere.height", atmosphere_height);
        params->set("grid.boundaries.atmosphere.ds", atmosphere_ds);
        params->set("grid.boundaries.atmosphere.g", atmosphere_g);
        params->set("grid.boundaries.atmosphere.species", atmosphere_species);
      }
    }

    void Grid::read(const SimEngine& engine_enum, const toml::value& toml_data) {
      /* domain decomposition ------------------------------------------------ */
      int default_ndomains = 1;
#if defined(MPI_ENABLED)
      raise::ErrorIf(MPI_Comm_size(MPI_COMM_WORLD, &default_ndomains) != MPI_SUCCESS,
                     "MPI_Comm_size failed",
                     HERE);
#endif
      number_of_domains = toml::find_or(toml_data,
                                        "simulation",
                                        "domain",
                                        "number",
                                        (unsigned int)default_ndomains);

      domain_decomposition = toml::find_or<std::vector<int>>(
        toml_data,
        "simulation",
        "domain",
        "decomposition",
        std::vector<int> { -1, -1, -1 });

      /* resolution and dimension ------------------------------------------- */
      resolution = toml::find<std::vector<ncells_t>>(toml_data, "grid", "resolution");
      raise::ErrorIf(resolution.size() < 1 || resolution.size() > 3,
                     "invalid `grid.resolution`",
                     HERE);
      dim = static_cast<Dimension>(resolution.size());

      if (domain_decomposition.size() > dim) {
        domain_decomposition.erase(domain_decomposition.begin() + (std::size_t)(dim),
                                   domain_decomposition.end());
      }
      raise::ErrorIf(domain_decomposition.size() != dim,
                     "invalid `simulation.domain.decomposition`",
                     HERE);

      /* metric and coordinates -------------------------------------------- */
      metric_enum = Metric::pick(
        fmt::toLower(toml::find<std::string>(toml_data, "grid", "metric", "metric"))
          .c_str());
      std::string coord;
      if (metric_enum == Metric::Minkowski) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "minkowski metric is only supported for SRPIC",
                       HERE);
        coord = "cart";
      } else if (metric_enum == Metric::QKerr_Schild or
                 metric_enum == Metric::QSpherical) {
        // quasi-spherical geometry
        raise::ErrorIf(dim == Dim::_1D,
                       "not enough dimensions for qspherical geometry",
                       HERE);
        raise::ErrorIf(dim == Dim::_3D,
                       "3D not implemented for qspherical geometry",
                       HERE);
        coord                    = "qsph";
        metric_params["qsph_r0"] = toml::find_or(toml_data,
                                                 "grid",
                                                 "metric",
                                                 "qsph_r0",
                                                 defaults::qsph::r0);
        metric_params["qsph_h"]  = toml::find_or(toml_data,
                                                "grid",
                                                "metric",
                                                "qsph_h",
                                                defaults::qsph::h);
      } else {
        // spherical geometry
        raise::ErrorIf(dim == Dim::_1D,
                       "not enough dimensions for spherical geometry",
                       HERE);
        raise::ErrorIf(dim == Dim::_3D,
                       "3D not implemented for spherical geometry",
                       HERE);
        coord = "sph";
      }
      if ((engine_enum == SimEngine::GRPIC) &&
          (metric_enum != Metric::Kerr_Schild_0)) {
        const auto ks_a        = toml::find_or(toml_data,
                                        "grid",
                                        "metric",
                                        "ks_a",
                                        defaults::ks::a);
        metric_params["ks_a"]  = ks_a;
        metric_params["ks_rh"] = ONE + math::sqrt(ONE - SQR(ks_a));
      }
      coord_enum = Coord::pick(coord.c_str());

      /* extent ------------------------------------------------------------- */
      extent = toml::find<std::vector<std::vector<real_t>>>(toml_data,
                                                            "grid",
                                                            "extent");

      if (extent.size() > dim) {
        extent.erase(extent.begin() + (std::size_t)(dim), extent.end());
      }
      raise::ErrorIf(extent[0].size() != 2, "invalid `grid.extent[0]`", HERE);
      if (coord_enum != Coord::Cart) {
        raise::ErrorIf(extent.size() > 1,
                       "invalid `grid.extent` for non-cartesian geometry",
                       HERE);
        extent.push_back({ ZERO, constant::PI });
        if (dim == Dim::_3D) {
          extent.push_back({ ZERO, TWO * constant::PI });
        }
      }
      raise::ErrorIf(extent.size() != dim, "invalid inferred `grid.extent`", HERE);
      for (auto d { 0u }; d < (dim_t)dim; ++d) {
        raise::ErrorIf(extent[d].size() != 2,
                       fmt::format("invalid inferred `grid.extent[%d]`", d),
                       HERE);
        extent_pairwise_.push_back({ extent[d][0], extent[d][1] });
      }

      /* metric parameters ------------------------------------------------------ */
      if (coord_enum == Coord::Qsph) {
        metric_params_short_["r0"] = metric_params["qsph_r0"];
        metric_params_short_["h"]  = metric_params["qsph_h"];
      }
      if ((engine_enum == SimEngine::GRPIC) &&
          (metric_enum != Metric::Kerr_Schild_0)) {
        metric_params_short_["a"] = metric_params["ks_a"];
      }
      // set("grid.metric.params", params);

      std::pair<real_t, real_t> dx0_V0;
      if (metric_enum == Metric::Minkowski) {
        if (dim == Dim::_1D) {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_1D>>(resolution,
                                                           extent_pairwise_,
                                                           metric_params_short_);
        } else if (dim == Dim::_2D) {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_2D>>(resolution,
                                                           extent_pairwise_,
                                                           metric_params_short_);
        } else {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_3D>>(resolution,
                                                           extent_pairwise_,
                                                           metric_params_short_);
        }
      } else if (metric_enum == Metric::Spherical) {
        dx0_V0 = get_dx0_V0<metric::Spherical<Dim::_2D>>(resolution,
                                                         extent_pairwise_,
                                                         metric_params_short_);
      } else if (metric_enum == Metric::QSpherical) {
        dx0_V0 = get_dx0_V0<metric::QSpherical<Dim::_2D>>(resolution,
                                                          extent_pairwise_,
                                                          metric_params_short_);
      } else if (metric_enum == Metric::Kerr_Schild) {
        dx0_V0 = get_dx0_V0<metric::KerrSchild<Dim::_2D>>(resolution,
                                                          extent_pairwise_,
                                                          metric_params_short_);
      } else if (metric_enum == Metric::Kerr_Schild_0) {
        dx0_V0 = get_dx0_V0<metric::KerrSchild0<Dim::_2D>>(resolution,
                                                           extent_pairwise_,
                                                           metric_params_short_);
      } else if (metric_enum == Metric::QKerr_Schild) {
        dx0_V0 = get_dx0_V0<metric::QKerrSchild<Dim::_2D>>(resolution,
                                                           extent_pairwise_,
                                                           metric_params_short_);
      }
      auto [dx0, V0] = dx0_V0;
      scale_dx0      = dx0;
      scale_V0       = V0;
    }

    void Grid::setParams(SimulationParams* params) const {
      params->set("simulation.domain.number", number_of_domains);
      params->set("simulation.domain.decomposition", domain_decomposition);

      params->set("grid.resolution", resolution);
      params->set("grid.dim", dim);
      params->set("grid.metric.metric", metric_enum);
      params->set("grid.metric.coord", coord_enum);
      for (const auto& [key, value] : metric_params) {
        params->set("grid.metric." + key, value);
      }
      params->set("grid.metric.params", metric_params_short_);
      params->set("grid.extent", extent_pairwise_);

      params->set("scales.dx0", scale_dx0);
      params->set("scales.V0", scale_V0);
    }

  } // namespace params
} // namespace ntt
