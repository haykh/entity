#include "framework/parameters.h"

#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/toml.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/containers/species.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <typename M>
  auto get_dx0_V0(
    const std::vector<std::size_t>&      resolution,
    const boundaries_t<real_t>&          extent,
    const std::map<std::string, real_t>& params) -> std::pair<real_t, real_t> {
    const auto      metric = M(resolution, extent, params);
    const auto      dx0    = metric.dxMin();
    coord_t<M::Dim> x_corner { ZERO };
    for (unsigned short d { 0 }; d < (unsigned short)(M::Dim); ++d) {
      x_corner[d] = HALF;
    }
    const auto V0 = metric.sqrt_det_h(x_corner);
    return { dx0, V0 };
  }

  /*
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   * Parameters that must not be changed during the checkpoint restart
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   */
  void SimulationParams::setImmutableParams(const toml::value& toml_data) {
    /* [simulation] --------------------------------------------------------- */
    const auto engine_enum = SimEngine::pick(
      fmt::toLower(toml::find<std::string>(toml_data, "simulation", "engine")).c_str());
    set("simulation.engine", engine_enum);

    int default_ndomains = 1;
#if defined(MPI_ENABLED)
    raise::ErrorIf(MPI_Comm_size(MPI_COMM_WORLD, &default_ndomains) != MPI_SUCCESS,
                   "MPI_Comm_size failed",
                   HERE);
#endif
    const auto ndoms = toml::find_or(toml_data,
                                     "simulation",
                                     "domain",
                                     "number",
                                     default_ndomains);
    set("simulation.domain.number", (unsigned int)ndoms);

    auto decomposition = toml::find_or<std::vector<int>>(
      toml_data,
      "simulation",
      "domain",
      "decomposition",
      std::vector<int> { -1, -1, -1 });
    promiseToDefine("simulation.domain.decomposition");

    /* [grid] --------------------------------------------------------------- */
    const auto res = toml::find<std::vector<std::size_t>>(toml_data,
                                                          "grid",
                                                          "resolution");
    raise::ErrorIf(res.size() < 1 || res.size() > 3,
                   "invalid `grid.resolution`",
                   HERE);
    set("grid.resolution", res);
    const auto dim = static_cast<Dimension>(res.size());
    set("grid.dim", dim);

    if (decomposition.size() > dim) {
      decomposition.erase(decomposition.begin() + (std::size_t)(dim),
                          decomposition.end());
    }
    raise::ErrorIf(decomposition.size() != dim,
                   "invalid `simulation.domain.decomposition`",
                   HERE);
    set("simulation.domain.decomposition", decomposition);

    auto extent = toml::find<std::vector<std::vector<real_t>>>(toml_data,
                                                               "grid",
                                                               "extent");
    raise::ErrorIf(extent.size() < 1 || extent.size() > 3,
                   "invalid `grid.extent`",
                   HERE);
    promiseToDefine("grid.extent");

    /* [grid.metric] -------------------------------------------------------- */
    const auto metric_enum = Metric::pick(
      fmt::toLower(toml::find<std::string>(toml_data, "grid", "metric", "metric"))
        .c_str());
    promiseToDefine("grid.metric.metric");
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
      coord = "qsph";
      set("grid.metric.qsph_r0",
          toml::find_or(toml_data, "grid", "metric", "qsph_r0", defaults::qsph::r0));
      set("grid.metric.qsph_h",
          toml::find_or(toml_data, "grid", "metric", "qsph_h", defaults::qsph::h));
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
      const auto ks_a = toml::find_or(toml_data,
                                      "grid",
                                      "metric",
                                      "ks_a",
                                      defaults::ks::a);
      set("grid.metric.ks_a", ks_a);
      set("grid.metric.ks_rh", ONE + math::sqrt(ONE - SQR(ks_a)));
    }
    const auto coord_enum = Coord::pick(coord.c_str());
    set("grid.metric.coord", coord_enum);

    /* [scales] ------------------------------------------------------------- */
    const auto larmor0 = toml::find<real_t>(toml_data, "scales", "larmor0");
    const auto skindepth0 = toml::find<real_t>(toml_data, "scales", "skindepth0");
    raise::ErrorIf(larmor0 <= ZERO || skindepth0 <= ZERO,
                   "larmor0 and skindepth0 must be positive",
                   HERE);
    set("scales.larmor0", larmor0);
    set("scales.skindepth0", skindepth0);
    promiseToDefine("scales.dx0");
    promiseToDefine("scales.V0");
    promiseToDefine("scales.n0");
    promiseToDefine("scales.q0");
    set("scales.sigma0", SQR(skindepth0 / larmor0));
    set("scales.B0", ONE / larmor0);
    set("scales.omegaB0", ONE / larmor0);

    /* [particles] ---------------------------------------------------------- */
    const auto ppc0 = toml::find<real_t>(toml_data, "particles", "ppc0");
    set("particles.ppc0", ppc0);
    raise::ErrorIf(ppc0 <= 0.0, "ppc0 must be positive", HERE);
    set("particles.use_weights",
        toml::find_or(toml_data, "particles", "use_weights", false));

    /* [particles.species] -------------------------------------------------- */
    std::vector<ParticleSpecies> species;
    const auto species_tab = toml::find_or<toml::array>(toml_data,
                                                        "particles",
                                                        "species",
                                                        toml::array {});
    set("particles.nspec", species_tab.size());

    unsigned short idx = 1;
    for (const auto& sp : species_tab) {
      const auto label  = toml::find_or<std::string>(sp,
                                                    "label",
                                                    "s" + std::to_string(idx));
      const auto mass   = toml::find<float>(sp, "mass");
      const auto charge = toml::find<float>(sp, "charge");
      raise::ErrorIf((charge != 0.0f) && (mass == 0.0f),
                     "mass of the charged species must be non-zero",
                     HERE);
      const auto is_massless   = (mass == 0.0f) && (charge == 0.0f);
      const auto def_pusher    = (is_massless ? defaults::ph_pusher
                                              : defaults::em_pusher);
      const auto maxnpart_real = toml::find<double>(sp, "maxnpart");
      const auto maxnpart      = static_cast<std::size_t>(maxnpart_real);
      auto       pusher = toml::find_or(sp, "pusher", std::string(def_pusher));
      const auto npayloads = toml::find_or(sp,
                                           "n_payloads",
                                           static_cast<unsigned short>(0));
      const auto cooling   = toml::find_or(sp, "cooling", std::string("None"));
      raise::ErrorIf((fmt::toLower(cooling) != "none") && is_massless,
                     "cooling is only applicable to massive particles",
                     HERE);
      raise::ErrorIf((fmt::toLower(pusher) == "photon") && !is_massless,
                     "photon pusher is only applicable to massless particles",
                     HERE);
      bool use_gca = false;
      if (pusher.find(',') != std::string::npos) {
        raise::ErrorIf(fmt::toLower(pusher.substr(pusher.find(',') + 1,
                                                  pusher.size())) != "gca",
                       "invalid pusher syntax",
                       HERE);
        use_gca = true;
        pusher  = pusher.substr(0, pusher.find(','));
      }
      const auto pusher_enum  = PrtlPusher::pick(pusher.c_str());
      const auto cooling_enum = Cooling::pick(cooling.c_str());
      if (use_gca) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "GCA pushers are only supported for SRPIC",
                       HERE);
        promiseToDefine("algorithms.gca.e_ovr_b_max");
        promiseToDefine("algorithms.gca.larmor_max");
      }
      if (cooling_enum == Cooling::SYNCHROTRON) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "Synchrotron cooling is only supported for SRPIC",
                       HERE);
        promiseToDefine("algorithms.synchrotron.gamma_rad");
      }

      species.emplace_back(ParticleSpecies(idx,
                                           label,
                                           mass,
                                           charge,
                                           maxnpart,
                                           pusher_enum,
                                           use_gca,
                                           cooling_enum,
                                           npayloads));
      idx += 1;
    }
    set("particles.species", species);

    /* inferred variables --------------------------------------------------- */
    // extent
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
    boundaries_t<real_t> extent_pairwise;
    for (unsigned short d = 0; d < (unsigned short)dim; ++d) {
      raise::ErrorIf(extent[d].size() != 2,
                     fmt::format("invalid inferred `grid.extent[%d]`", d),
                     HERE);
      extent_pairwise.push_back({ extent[d][0], extent[d][1] });
    }
    set("grid.extent", extent_pairwise);

    // metric, dx0, V0, n0, q0
    {
      boundaries_t<real_t> ext;
      for (const auto& e : extent) {
        ext.push_back({ e[0], e[1] });
      }
      std::map<std::string, real_t> params;
      if (coord_enum == Coord::Qsph) {
        params["r0"] = get<real_t>("grid.metric.qsph_r0");
        params["h"]  = get<real_t>("grid.metric.qsph_h");
      }
      if ((engine_enum == SimEngine::GRPIC) &&
          (metric_enum != Metric::Kerr_Schild_0)) {
        params["a"] = get<real_t>("grid.metric.ks_a");
      }
      set("grid.metric.params", params);

      std::pair<real_t, real_t> dx0_V0;
      if (metric_enum == Metric::Minkowski) {
        if (dim == Dim::_1D) {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_1D>>(res, ext, params);
        } else if (dim == Dim::_2D) {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_2D>>(res, ext, params);
        } else {
          dx0_V0 = get_dx0_V0<metric::Minkowski<Dim::_3D>>(res, ext, params);
        }
      } else if (metric_enum == Metric::Spherical) {
        dx0_V0 = get_dx0_V0<metric::Spherical<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::QSpherical) {
        dx0_V0 = get_dx0_V0<metric::QSpherical<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::Kerr_Schild) {
        dx0_V0 = get_dx0_V0<metric::KerrSchild<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::Kerr_Schild_0) {
        dx0_V0 = get_dx0_V0<metric::KerrSchild0<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::QKerr_Schild) {
        dx0_V0 = get_dx0_V0<metric::QKerrSchild<Dim::_2D>>(res, ext, params);
      }
      auto [dx0, V0] = dx0_V0;
      set("scales.dx0", dx0);
      set("scales.V0", V0);
      set("scales.n0", ppc0 / V0);
      set("scales.q0", V0 / (ppc0 * SQR(skindepth0)));

      set("grid.metric.metric", metric_enum);
    }
  }

  /*
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   * Parameters that may be changed during the checkpoint restart
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   */
  void SimulationParams::setMutableParams(const toml::value& toml_data) {
    const auto engine_enum     = get<SimEngine>("simulation.engine");
    const auto coord_enum      = get<Coord>("grid.metric.coord");
    const auto dim             = get<Dimension>("grid.dim");
    const auto extent_pairwise = get<boundaries_t<real_t>>("grid.extent");

    /* [simulation] --------------------------------------------------------- */
    set("simulation.name",
        toml::find<std::string>(toml_data, "simulation", "name"));
    set("simulation.runtime",
        toml::find<long double>(toml_data, "simulation", "runtime"));

    /* [grid.boundaraies] --------------------------------------------------- */
    auto flds_bc = toml::find<std::vector<std::vector<std::string>>>(
      toml_data,
      "grid",
      "boundaries",
      "fields");
    {
      raise::ErrorIf(flds_bc.size() < 1 || flds_bc.size() > 3,
                     "invalid `grid.boundaries.fields`",
                     HERE);
      promiseToDefine("grid.boundaries.fields");
      auto atm_defined = false;
      for (const auto& bcs : flds_bc) {
        for (const auto& bc : bcs) {
          if (fmt::toLower(bc) == "match") {
            promiseToDefine("grid.boundaries.match.ds");
            promiseToDefine("grid.boundaries.match.coeff");
          }
          if (fmt::toLower(bc) == "atmosphere") {
            raise::ErrorIf(atm_defined,
                           "ATMOSPHERE is only allowed in one direction",
                           HERE);
            atm_defined = true;
            promiseToDefine("grid.boundaries.atmosphere.temperature");
            promiseToDefine("grid.boundaries.atmosphere.density");
            promiseToDefine("grid.boundaries.atmosphere.height");
            promiseToDefine("grid.boundaries.atmosphere.ds");
            promiseToDefine("grid.boundaries.atmosphere.species");
            promiseToDefine("grid.boundaries.atmosphere.g");
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
      promiseToDefine("grid.boundaries.particles");
      auto atm_defined = false;
      for (const auto& bcs : prtl_bc) {
        for (const auto& bc : bcs) {
          if (fmt::toLower(bc) == "absorb") {
            promiseToDefine("grid.boundaries.absorb.ds");
          }
          if (fmt::toLower(bc) == "atmosphere") {
            raise::ErrorIf(atm_defined,
                           "ATMOSPHERE is only allowed in one direction",
                           HERE);
            atm_defined = true;
            promiseToDefine("grid.boundaries.atmosphere.temperature");
            promiseToDefine("grid.boundaries.atmosphere.density");
            promiseToDefine("grid.boundaries.atmosphere.height");
            promiseToDefine("grid.boundaries.atmosphere.ds");
            promiseToDefine("grid.boundaries.atmosphere.species");
            promiseToDefine("grid.boundaries.atmosphere.g");
          }
        }
      }
    }

    /* [algorithms] --------------------------------------------------------- */
    set("algorithms.current_filters",
        toml::find_or(toml_data,
                      "algorithms",
                      "current_filters",
                      defaults::current_filters));

    /* [algorithms.toggles] ------------------------------------------------- */
    set("algorithms.toggles.fieldsolver",
        toml::find_or(toml_data, "algorithms", "toggles", "fieldsolver", true));
    set("algorithms.toggles.deposit",
        toml::find_or(toml_data, "algorithms", "toggles", "deposit", true));

    /* [algorithms.timestep] ------------------------------------------------ */
    set("algorithms.timestep.CFL",
        toml::find_or(toml_data, "algorithms", "timestep", "CFL", defaults::cfl));
    set("algorithms.timestep.dt",
        get<real_t>("algorithms.timestep.CFL") * get<real_t>("scales.dx0"));
    set("algorithms.timestep.correction",
        toml::find_or(toml_data,
                      "algorithms",
                      "timestep",
                      "correction",
                      defaults::correction));

    /* [algorithms.gr] ------------------------------------------------------ */
    if (engine_enum == SimEngine::GRPIC) {
      set("algorithms.gr.pusher_eps",
          toml::find_or(toml_data,
                        "algorithms",
                        "gr",
                        "pusher_eps",
                        defaults::gr::pusher_eps));
      set("algorithms.gr.pusher_niter",
          toml::find_or(toml_data,
                        "algorithms",
                        "gr",
                        "pusher_niter",
                        defaults::gr::pusher_niter));
    }
    /* [particles] ---------------------------------------------------------- */
    set("particles.clear_interval",
        toml::find_or(toml_data, "particles", "clear_interval", defaults::clear_interval));

    /* [output] ------------------------------------------------------------- */
    // fields
    set("output.format",
        toml::find_or(toml_data, "output", "format", defaults::output::format));
    set("output.interval",
        toml::find_or(toml_data, "output", "interval", defaults::output::interval));
    set("output.interval_time",
        toml::find_or<long double>(toml_data, "output", "interval_time", -1.0));
    set("output.separate_files",
        toml::find_or<bool>(toml_data, "output", "separate_files", true));

    promiseToDefine("output.fields.interval");
    promiseToDefine("output.fields.interval_time");
    promiseToDefine("output.fields.enable");
    promiseToDefine("output.particles.interval");
    promiseToDefine("output.particles.interval_time");
    promiseToDefine("output.particles.enable");
    promiseToDefine("output.spectra.interval");
    promiseToDefine("output.spectra.interval_time");
    promiseToDefine("output.spectra.enable");

    const auto flds_out        = toml::find_or(toml_data,
                                        "output",
                                        "fields",
                                        "quantities",
                                        std::vector<std::string> {});
    const auto custom_flds_out = toml::find_or(toml_data,
                                               "output",
                                               "fields",
                                               "custom",
                                               std::vector<std::string> {});
    if (flds_out.size() == 0) {
      raise::Warning("No fields output specified", HERE);
    }
    set("output.fields.quantities", flds_out);
    set("output.fields.custom", custom_flds_out);
    set("output.fields.mom_smooth",
        toml::find_or(toml_data,
                      "output",
                      "fields",
                      "mom_smooth",
                      defaults::output::mom_smooth));
    auto field_dwn = toml::find_or(toml_data,
                                   "output",
                                   "fields",
                                   "downsampling",
                                   std::vector<unsigned int> { 1, 1, 1 });
    raise::ErrorIf(field_dwn.size() > 3, "invalid `output.fields.downsampling`", HERE);
    if (field_dwn.size() > dim) {
      field_dwn.erase(field_dwn.begin() + (std::size_t)(dim), field_dwn.end());
    }
    for (const auto& dwn : field_dwn) {
      raise::ErrorIf(dwn == 0, "downsampling factor must be nonzero", HERE);
    }
    set("output.fields.downsampling", field_dwn);

    // particles
    auto       all_specs = std::vector<unsigned short> {};
    const auto nspec     = get<std::size_t>("particles.nspec");
    for (auto i = 0u; i < nspec; ++i) {
      all_specs.push_back(static_cast<unsigned short>(i + 1));
    }
    const auto prtl_out = toml::find_or(toml_data,
                                        "output",
                                        "particles",
                                        "species",
                                        all_specs);
    set("output.particles.species", prtl_out);
    set("output.particles.stride",
        toml::find_or(toml_data,
                      "output",
                      "particles",
                      "stride",
                      defaults::output::prtl_stride));

    // spectra
    set("output.spectra.e_min",
        toml::find_or(toml_data, "output", "spectra", "e_min", defaults::output::spec_emin));
    set("output.spectra.e_max",
        toml::find_or(toml_data, "output", "spectra", "e_max", defaults::output::spec_emax));
    set("output.spectra.log_bins",
        toml::find_or(toml_data,
                      "output",
                      "spectra",
                      "log_bins",
                      defaults::output::spec_log));
    set("output.spectra.n_bins",
        toml::find_or(toml_data,
                      "output",
                      "spectra",
                      "n_bins",
                      defaults::output::spec_nbins));

    // intervals
    for (const auto& type : { "fields", "particles", "spectra" }) {
      const auto q_int      = toml::find_or<std::size_t>(toml_data,
                                                    "output",
                                                    std::string(type),
                                                    "interval",
                                                    0);
      const auto q_int_time = toml::find_or<long double>(toml_data,
                                                         "output",
                                                         std::string(type),
                                                         "interval_time",
                                                         -1.0);
      set("output." + std::string(type) + ".enable",
          toml::find_or(toml_data, "output", std::string(type), "enable", true));
      if (q_int == 0 && q_int_time == -1.0) {
        set("output." + std::string(type) + ".interval",
            get<std::size_t>("output.interval"));
        set("output." + std::string(type) + ".interval_time",
            get<long double>("output.interval_time"));
      } else {
        set("output." + std::string(type) + ".interval", q_int);
        set("output." + std::string(type) + ".interval_time", q_int_time);
      }
    }

    /* [output.debug] ------------------------------------------------------- */
    set("output.debug.as_is",
        toml::find_or(toml_data, "output", "debug", "as_is", false));
    const auto output_ghosts = toml::find_or(toml_data,
                                             "output",
                                             "debug",
                                             "ghosts",
                                             false);
    set("output.debug.ghosts", output_ghosts);
    if (output_ghosts) {
      for (const auto& dwn : field_dwn) {
        raise::ErrorIf(
          dwn != 1,
          "full resolution required when outputting with ghost cells",
          HERE);
      }
    }

    /* [checkpoint] --------------------------------------------------------- */
    set("checkpoint.interval",
        toml::find_or(toml_data,
                      "checkpoint",
                      "interval",
                      defaults::checkpoint::interval));
    set("checkpoint.interval_time",
        toml::find_or<long double>(toml_data, "checkpoint", "interval_time", -1.0));
    set("checkpoint.keep",
        toml::find_or(toml_data, "checkpoint", "keep", defaults::checkpoint::keep));

    /* [diagnostics] -------------------------------------------------------- */
    set("diagnostics.interval",
        toml::find_or(toml_data, "diagnostics", "interval", defaults::diag::interval));
    set("diagnostics.blocking_timers",
        toml::find_or(toml_data, "diagnostics", "blocking_timers", false));
    set("diagnostics.colored_stdout",
        toml::find_or(toml_data, "diagnostics", "colored_stdout", false));

    /* inferred variables --------------------------------------------------- */
    // fields/particle boundaries
    std::vector<std::vector<FldsBC>> flds_bc_enum;
    std::vector<std::vector<PrtlBC>> prtl_bc_enum;
    if (coord_enum == Coord::Cart) {
      raise::ErrorIf(flds_bc.size() != (std::size_t)dim,
                     "invalid `grid.boundaries.fields`",
                     HERE);
      raise::ErrorIf(prtl_bc.size() != (std::size_t)dim,
                     "invalid `grid.boundaries.particles`",
                     HERE);
      for (unsigned short d = 0; d < (unsigned short)dim; ++d) {
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
      raise::ErrorIf(prtl_bc.size() > 1, "invalid `grid.boundaries.particles`", HERE);
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
    for (unsigned short d = 0; d < (unsigned short)dim; ++d) {
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
    set("grid.boundaries.fields", flds_bc_pairwise);
    set("grid.boundaries.particles", prtl_bc_pairwise);

    if (isPromised("grid.boundaries.match.ds")) {
      if (coord_enum == Coord::Cart) {
        auto min_extent = std::numeric_limits<real_t>::max();
        for (const auto& e : extent_pairwise) {
          min_extent = std::min(min_extent, e.second - e.first);
        }
        set("grid.boundaries.match.ds",
            toml::find_or(toml_data,
                          "grid",
                          "boundaries",
                          "match",
                          "ds",
                          min_extent * defaults::bc::match::ds_frac));
      } else {
        auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
        set("grid.boundaries.match.ds",
            toml::find_or(toml_data,
                          "grid",
                          "boundaries",
                          "match",
                          "ds",
                          r_extent * defaults::bc::match::ds_frac));
      }
      set("grid.boundaries.match.coeff",
          toml::find_or(toml_data,
                        "grid",
                        "boundaries",
                        "match",
                        "coeff",
                        defaults::bc::match::coeff));
    }

    if (isPromised("grid.boundaries.absorb.ds")) {
      if (coord_enum == Coord::Cart) {
        auto min_extent = std::numeric_limits<real_t>::max();
        for (const auto& e : extent_pairwise) {
          min_extent = std::min(min_extent, e.second - e.first);
        }
        set("grid.boundaries.absorb.ds",
            toml::find_or(toml_data,
                          "grid",
                          "boundaries",
                          "absorb",
                          "ds",
                          min_extent * defaults::bc::absorb::ds_frac));
      } else {
        auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
        set("grid.boundaries.absorb.ds",
            toml::find_or(toml_data,
                          "grid",
                          "boundaries",
                          "absorb",
                          "ds",
                          r_extent * defaults::bc::absorb::ds_frac));
      }
    }

    if (isPromised("grid.boundaries.atmosphere.temperature")) {
      const auto atm_T = toml::find<real_t>(toml_data,
                                            "grid",
                                            "boundaries",
                                            "atmosphere",
                                            "temperature");
      const auto atm_h = toml::find<real_t>(toml_data,
                                            "grid",
                                            "boundaries",
                                            "atmosphere",
                                            "height");
      set("grid.boundaries.atmosphere.temperature", atm_T);
      set("grid.boundaries.atmosphere.density",
          toml::find<real_t>(toml_data, "grid", "boundaries", "atmosphere", "density"));
      set("grid.boundaries.atmosphere.ds",
          toml::find_or(toml_data, "grid", "boundaries", "atmosphere", "ds", ZERO));
      set("grid.boundaries.atmosphere.height", atm_h);
      set("grid.boundaries.atmosphere.g", atm_T / atm_h);
      const auto atm_species = toml::find<std::pair<unsigned short, unsigned short>>(
        toml_data,
        "grid",
        "boundaries",
        "atmosphere",
        "species");
      set("grid.boundaries.atmosphere.species", atm_species);
    }

    // gca
    if (isPromised("algorithms.gca.e_ovr_b_max")) {
      set("algorithms.gca.e_ovr_b_max",
          toml::find_or(toml_data,
                        "algorithms",
                        "gca",
                        "e_ovr_b_max",
                        defaults::gca::EovrB_max));
      set("algorithms.gca.larmor_max",
          toml::find_or(toml_data, "algorithms", "gca", "larmor_max", ZERO));
    }

    // cooling
    if (isPromised("algorithms.synchrotron.gamma_rad")) {
      set("algorithms.synchrotron.gamma_rad",
          toml::find_or(toml_data,
                        "algorithms",
                        "synchrotron",
                        "gamma_rad",
                        defaults::synchrotron::gamma_rad));
    }
  }

  void SimulationParams::setSetupParams(const toml::value& toml_data) {
    /* [setup] -------------------------------------------------------------- */
    const auto& setup = toml::find_or(toml_data, "setup", toml::table {});
    for (const auto& [key, val] : setup) {
      if (val.is_boolean()) {
        set("setup." + key, (bool)(val.as_boolean()));
      } else if (val.is_integer()) {
        set("setup." + key, (int)(val.as_integer()));
      } else if (val.is_floating()) {
        set("setup." + key, (real_t)(val.as_floating()));
      } else if (val.is_string()) {
        set("setup." + key, (std::string)(val.as_string()));
      } else if (val.is_array()) {
        const auto val_arr = val.as_array();
        if (val_arr.size() == 0) {
          continue;
        } else {
          if (val_arr[0].is_integer()) {
            std::vector<int> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_integer());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_floating()) {
            std::vector<real_t> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_floating());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_boolean()) {
            std::vector<bool> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_boolean());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_string()) {
            std::vector<std::string> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_string());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_array()) {
            raise::Error("only 1D arrays allowed in [setup]", HERE);
          } else {
            raise::Error("invalid setup variable type", HERE);
          }
        }
      }
    }
  }

  void SimulationParams::setCheckpointParams(bool        is_resuming,
                                             std::size_t start_step,
                                             long double start_time) {
    set("checkpoint.is_resuming", is_resuming);
    set("checkpoint.start_step", start_step);
    set("checkpoint.start_time", start_time);
  }

  void SimulationParams::checkPromises() const {
    raise::ErrorIf(!promisesFulfilled(),
                   "Have not defined all the necessary variables",
                   HERE);
  }
} // namespace ntt
