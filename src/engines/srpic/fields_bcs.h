#ifndef ENGINES_SRPIC_FIELDS_BCS_H
#define ENGINES_SRPIC_FIELDS_BCS_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include "archetypes/traits.h"
#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

#include "kernels/fields_bcs.hpp"

namespace ntt {
  namespace srpic {

    template <class M, class T, in O>
      requires metric::traits::HasD<M>
    void CallMatchFields(ndfield_t<M::Dim, 6>&       fields,
                         const boundaries_t<FldsBC>& boundaries,
                         const T&                    match_fields,
                         const M&                    metric,
                         real_t                      xg_edge,
                         real_t                      ds,
                         BCTags                      tags,
                         tuple_t<ncells_t, M::Dim>&  range_min,
                         tuple_t<ncells_t, M::Dim>&  range_max) {
      Kokkos::parallel_for(
        "MatchFields",
        CreateRangePolicy<M::Dim>(range_min, range_max),
        kernel::bc::MatchBoundaries_kernel<SimEngine::SRPIC, T, M, O>(fields,
                                                                      match_fields,
                                                                      metric,
                                                                      xg_edge,
                                                                      ds,
                                                                      tags,
                                                                      boundaries));
    }

    template <class M, class PG>
      requires metric::traits::HasD<M>
    void MatchFieldsIn(dir::direction_t<M::Dim>     direction,
                       Domain<SimEngine::SRPIC, M>& domain,
                       const Grid<M::Dim>&          global_grid,
                       const PG&                    pgen,
                       const prm::Parameters&       engine_params,
                       const SimulationParams&      params,
                       BCTags                       tags) {
      const auto time     = engine_params.get<simtime_t>("time");
      /**
       * matching boundaries
       */
      const auto ds_array = params.template get<boundaries_t<real_t>>(
        "grid.boundaries.match.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      auto       sign = direction.get_sign();
      real_t     ds;
      if (sign > 0) { // + direction
        ds      = ds_array[(short)dim].second;
        xg_max  = global_grid.extent(dim).second;
        xg_min  = xg_max - ds;
        xg_edge = xg_max;
      } else { // - direction
        ds      = ds_array[(short)dim].first;
        xg_min  = global_grid.extent(dim).first;
        xg_max  = xg_min + ds;
        xg_edge = xg_min;
      }
      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (dim_t d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<dim_t>(dim)) {
          box.push_back({ xg_min, xg_max });
          if (sign > 0) {
            incl_ghosts.push_back({ false, true });
          } else {
            incl_ghosts.push_back({ true, false });
          }
        } else {
          box.push_back(Range::All);
          incl_ghosts.push_back({ true, true });
        }
      }
      if (not domain.mesh.Intersects(box)) {
        return;
      }
      const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);
      tuple_t<ncells_t, M::Dim> range_min { 0 };
      tuple_t<ncells_t, M::Dim> range_max { 0 };

      for (auto d { 0u }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }

      if (dim == in::x1) {
        if constexpr (arch::traits::pgen::HasMatchFields<PG>) {
          auto match_fields = pgen.MatchFields(time);
          CallMatchFields<M, decltype(match_fields), in::x1>(domain.fields.em,
                                                             domain.mesh.flds_bc(),
                                                             match_fields,
                                                             domain.mesh.metric,
                                                             xg_edge,
                                                             ds,
                                                             tags,
                                                             range_min,
                                                             range_max);
        } else if constexpr (arch::traits::pgen::HasMatchFieldsInX1<PG>) {
          auto match_fields = pgen.MatchFieldsInX1(time);
          CallMatchFields<M, decltype(match_fields), in::x1>(domain.fields.em,
                                                             domain.mesh.flds_bc(),
                                                             match_fields,
                                                             domain.mesh.metric,
                                                             xg_edge,
                                                             ds,
                                                             tags,
                                                             range_min,
                                                             range_max);
        }
      } else if (dim == in::x2) {
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          if constexpr (arch::traits::pgen::HasMatchFields<PG>) {
            auto match_fields = pgen.MatchFields(time);
            CallMatchFields<M, decltype(match_fields), in::x2>(
              domain.fields.em,
              domain.mesh.flds_bc(),
              match_fields,
              domain.mesh.metric,
              xg_edge,
              ds,
              tags,
              range_min,
              range_max);
          } else if constexpr (arch::traits::pgen::HasMatchFieldsInX2<PG>) {
            auto match_fields = pgen.MatchFieldsInX2(time);
            CallMatchFields<M, decltype(match_fields), in::x2>(
              domain.fields.em,
              domain.mesh.flds_bc(),
              match_fields,
              domain.mesh.metric,
              xg_edge,
              ds,
              tags,
              range_min,
              range_max);
          }
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else if (dim == in::x3) {
        if constexpr (M::Dim == Dim::_3D) {
          if constexpr (arch::traits::pgen::HasMatchFields<PG>) {
            auto match_fields = pgen.MatchFields(time);
            CallMatchFields<M, decltype(match_fields), in::x3>(
              domain.fields.em,
              domain.mesh.flds_bc(),
              match_fields,
              domain.mesh.metric,
              xg_edge,
              ds,
              tags,
              range_min,
              range_max);
          } else if constexpr (arch::traits::pgen::HasMatchFieldsInX3<PG>) {
            auto match_fields = pgen.MatchFieldsInX3(time);
            CallMatchFields<M, decltype(match_fields), in::x3>(
              domain.fields.em,
              domain.mesh.flds_bc(),
              match_fields,
              domain.mesh.metric,
              xg_edge,
              ds,
              tags,
              range_min,
              range_max);
          }
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    }

    template <class M>
      requires metric::traits::HasD<M>
    void AxisFieldsIn(dir::direction_t<M::Dim>     direction,
                      Domain<SimEngine::SRPIC, M>& domain,
                      BCTags                       tags) {
      /**
       * axis boundaries
       */
      if constexpr (M::CoordType != Coord::Cart) {
        raise::ErrorIf(direction.get_dim() != in::x2,
                       "Invalid axis direction, should be x2",
                       HERE);
        const auto i2_min = domain.mesh.i_min(in::x2);
        const auto i2_max = domain.mesh.i_max(in::x2);
        if (direction.get_sign() < 0) {
          Kokkos::parallel_for(
            "AxisBCFields",
            domain.mesh.n_all(in::x1),
            kernel::bc::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em,
                                                             i2_min,
                                                             tags));
        } else {
          Kokkos::parallel_for(
            "AxisBCFields",
            domain.mesh.n_all(in::x1),
            kernel::bc::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em,
                                                            i2_max,
                                                            tags));
        }
      } else {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Invalid coordinate type for axis BCs", HERE);
      }
    }

    template <class M, class PG>
      requires metric::traits::HasD<M>
    void FixedFieldsIn(dir::direction_t<M::Dim>     direction,
                       Domain<SimEngine::SRPIC, M>& domain,
                       const PG&                    pgen,
                       BCTags                       tags) {
      /**
       * fixed field boundaries
       */
      const auto sign = direction.get_sign();
      const auto dim  = direction.get_dim();
      raise::ErrorIf(dim != in::x1 and M::CoordType != Coord::Cart,
                     "Fixed BCs only implemented for x1 in "
                     "non-cartesian coordinates",
                     HERE);
      em normal_b_comp, tang_e_comp1, tang_e_comp2;
      if (dim == in::x1) {
        normal_b_comp = em::bx1;
        tang_e_comp1  = em::ex2;
        tang_e_comp2  = em::ex3;
      } else if (dim == in::x2) {
        normal_b_comp = em::bx2;
        tang_e_comp1  = em::ex1;
        tang_e_comp2  = em::ex3;
      } else if (dim == in::x3) {
        normal_b_comp = em::bx3;
        tang_e_comp1  = em::ex1;
        tang_e_comp2  = em::ex2;
      } else {
        raise::Error("Invalid dimension", HERE);
      }
      std::vector<ncells_t> xi_min, xi_max;
      const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
      for (dim_t d { 0u }; d < M::Dim; ++d) {
        const auto dd = all_dirs[d];
        if (dim == dd) {
          if (sign > 0) { // + direction
            xi_min.push_back(domain.mesh.n_all(dd) - N_GHOSTS);
            xi_max.push_back(domain.mesh.n_all(dd));
          } else { // - direction
            xi_min.push_back(0);
            xi_max.push_back(N_GHOSTS);
          }
        } else {
          xi_min.push_back(0);
          xi_max.push_back(domain.mesh.n_all(dd));
        }
      }
      raise::ErrorIf(xi_min.size() != xi_max.size() or
                       xi_min.size() != static_cast<std::size_t>(M::Dim),
                     "Invalid range size",
                     HERE);
      std::vector<unsigned short> comps;
      if (tags & BC::E) {
        comps.push_back(tang_e_comp1);
        comps.push_back(tang_e_comp2);
      }
      if (tags & BC::B) {
        comps.push_back(normal_b_comp);
      }
      if constexpr (arch::traits::pgen::HasFixFieldsConst<PG>) {
        for (const auto& comp : comps) {
          auto value     = ZERO;
          bool shouldset = false;
          // if fix field function present, read from it
          const auto newset = pgen.FixFieldsConst((bc_in)(sign * ((short)dim + 1)),
                                                  (em)comp);
          value     = newset.first;
          shouldset = newset.second;
          if (shouldset) {
            if constexpr (M::Dim == Dim::_1D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                comp),
                value);
            } else if constexpr (M::Dim == Dim::_2D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                std::make_pair(xi_min[1], xi_max[1]),
                                comp),
                value);
            } else if constexpr (M::Dim == Dim::_3D) {
              Kokkos::deep_copy(
                Kokkos::subview(domain.fields.em,
                                std::make_pair(xi_min[0], xi_max[0]),
                                std::make_pair(xi_min[1], xi_max[1]),
                                std::make_pair(xi_min[2], xi_max[2]),
                                comp),
                value);
            } else {
              raise::Error("Invalid dimension", HERE);
            }
          }
        }
      } else {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Fixed fields not present (both const and non-const)", HERE);
      }
    }

    template <class M>
      requires metric::traits::HasD<M>
    void PerfectConductorFieldsIn(dir::direction_t<M::Dim>     direction,
                                  Domain<SimEngine::SRPIC, M>& domain,
                                  BCTags                       tags) {
      /**
       * perfect conductor field boundaries
       */
      if constexpr (M::CoordType != Coord::Cart) {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error(
          "Perfect conductor BCs only applicable to cartesian coordinates",
          HERE);
      } else {
        const auto sign = direction.get_sign();
        const auto dim  = direction.get_dim();

        std::vector<std::size_t> xi_min, xi_max;

        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };

        for (auto d { 0u }; d < M::Dim; ++d) {
          const auto dd = all_dirs[d];
          if (dim == dd) {
            xi_min.push_back(0);
            xi_max.push_back((sign < 0) ? (N_GHOSTS + 1) : N_GHOSTS);
          } else {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd));
          }
        }
        raise::ErrorIf(xi_min.size() != xi_max.size() or
                         xi_min.size() != static_cast<std::size_t>(M::Dim),
                       "Invalid range size",
                       HERE);

        range_t<M::Dim> range;
        if constexpr (M::Dim == Dim::_1D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1] },
                                            { xi_max[0], xi_max[1] });
        } else if constexpr (M::Dim == Dim::_3D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1], xi_min[2] },
                                            { xi_max[0], xi_max[1], xi_max[2] });
        } else {
          raise::Error("Invalid dimension", HERE);
        }
        std::size_t i_edge;
        if (sign > 0) {
          i_edge = domain.mesh.i_max(dim);
        } else {
          i_edge = domain.mesh.i_min(dim);
        }

        if (dim == in::x1) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x1, true>(
                domain.fields.em,
                i_edge,
                tags));
          } else {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x1, false>(
                domain.fields.em,
                i_edge,
                tags));
          }
        } else if (dim == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x2, true>(
                  domain.fields.em,
                  i_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x2, false>(
                  domain.fields.em,
                  i_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          if constexpr (M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x3, true>(
                  domain.fields.em,
                  i_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "ConductorFields",
                range,
                kernel::bc::ConductorBoundaries_kernel<M::Dim, in::x3, false>(
                  domain.fields.em,
                  i_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        }
      }
    }

    template <class M, class PG>
    void AtmosphereFieldsIn(dir::direction_t<M::Dim>     direction,
                            Domain<SimEngine::SRPIC, M>& domain,
                            const M&                     global_metric,
                            const Grid<M::Dim>&          global_grid,
                            const PG&                    pgen,
                            const SimulationParams&      params,
                            const prm::Parameters&       engine_params,
                            BCTags                       tags) {
      const auto time = engine_params.get<simtime_t>("time");
      /**
       * atmosphere field boundaries
       */
      if constexpr (arch::traits::pgen::HasAtmFields<PG>) {
        const auto [sign, dim, xg_min, xg_max] = GetAtmosphereExtent(direction,
                                                                     global_metric,
                                                                     global_grid,
                                                                     params);
        // get_atm_extent(direction);
        const auto           dd                = static_cast<dim_t>(dim);
        boundaries_t<real_t> box;
        boundaries_t<bool>   incl_ghosts;
        for (auto d { 0u }; d < M::Dim; ++d) {
          if (d == dd) {
            box.push_back({ xg_min, xg_max });
            if (sign > 0) {
              incl_ghosts.push_back({ false, true });
            } else {
              incl_ghosts.push_back({ true, false });
            }
          } else {
            box.push_back(Range::All);
            incl_ghosts.push_back({ true, true });
          }
        }
        if (not domain.mesh.Intersects(box)) {
          return;
        }
        const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);
        tuple_t<std::size_t, M::Dim> range_min { 0 };
        tuple_t<std::size_t, M::Dim> range_max { 0 };

        for (auto d { 0u }; d < M::Dim; ++d) {
          range_min[d] = intersect_range[d].first;
          range_max[d] = intersect_range[d].second;
        }
        auto        atm_fields = pgen.AtmFields(time);
        std::size_t il_edge;
        if (sign > 0) {
          il_edge = range_min[dd] - N_GHOSTS;
        } else {
          il_edge = range_max[dd] - 1 - N_GHOSTS;
        }
        const auto range = CreateRangePolicy<M::Dim>(range_min, range_max);
        if (dim == in::x1) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "AtmosphereBCFields",
              range,
              kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x1>(
                domain.fields.em,
                atm_fields,
                domain.mesh.metric,
                il_edge,
                tags));
          } else {
            Kokkos::parallel_for(
              "AtmosphereBCFields",
              range,
              kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x1>(
                domain.fields.em,
                atm_fields,
                domain.mesh.metric,
                il_edge,
                tags));
          }
        } else if (dim == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x2>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x2>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else if (dim == in::x3) {
          if constexpr (M::Dim == Dim::_3D) {
            if (sign > 0) {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, true, in::x3>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            } else {
              Kokkos::parallel_for(
                "AtmosphereBCFields",
                range,
                kernel::bc::EnforcedBoundaries_kernel<decltype(atm_fields), M, false, in::x3>(
                  domain.fields.em,
                  atm_fields,
                  domain.mesh.metric,
                  il_edge,
                  tags));
            }
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error("Atm fields not implemented in PGEN for atmosphere BCs", HERE);
      }
    }

    template <class M>
      requires metric::traits::HasD<M>
    void CustomFieldsIn(dir::direction_t<M::Dim>     direction,
                        Domain<SimEngine::SRPIC, M>& domain,
                        BCTags                       tags) {
      (void)direction;
      (void)domain;
      (void)tags;
      raise::Error("Custom boundaries not implemented", HERE);
      // if constexpr (
      //   traits::has_member<traits::pgen::custom_fields_t, pgen_t>::value) {
      //   const auto [box, custom_fields] = m_pgen.CustomFields(time);
      //   if (domain.mesh.Intersects(box)) {
      //   }
      //
      // } else {
      //   raise::Error("Custom boundaries not implemented", HERE);
      // }
    }

    template <class M, class PG>
      requires metric::traits::HasD<M>
    void FieldBoundaries(Domain<SimEngine::SRPIC, M>& domain,
                         const M&                     global_metric,
                         const Grid<M::Dim>&          global_grid,
                         const PG&                    pgen,
                         const prm::Parameters&       engine_params,
                         const SimulationParams&      params,
                         BCTags                       tags) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (global_grid.flds_bc_in(direction) == FldsBC::MATCH) {
          MatchFieldsIn<M, PG>(direction,
                               domain,
                               global_grid,
                               pgen,
                               engine_params,
                               params,
                               tags);
        } else if (global_grid.flds_bc_in(direction) == FldsBC::AXIS) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::AXIS) {
            AxisFieldsIn<M>(direction, domain, tags);
          }
        } else if (global_grid.flds_bc_in(direction) == FldsBC::ATMOSPHERE) {
          AtmosphereFieldsIn<M, PG>(direction,
                                    domain,
                                    global_metric,
                                    global_grid,
                                    pgen,
                                    params,
                                    engine_params,
                                    tags);
        } else if (global_grid.flds_bc_in(direction) == FldsBC::FIXED) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::FIXED) {
            FixedFieldsIn<M, PG>(direction, domain, pgen, tags);
          }
        } else if (global_grid.flds_bc_in(direction) == FldsBC::CONDUCTOR) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CONDUCTOR) {
            PerfectConductorFieldsIn<M>(direction, domain, tags);
          }
        } else if (global_grid.flds_bc_in(direction) == FldsBC::CUSTOM) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CUSTOM) {
            CustomFieldsIn<M>(direction, domain, tags);
          }
        } else if (global_grid.flds_bc_in(direction) == FldsBC::HORIZON) {
          raise::Error("HORIZON BCs only applicable for GR", HERE);
        }
      } // loop over directions
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_FIELDS_BCS_H
