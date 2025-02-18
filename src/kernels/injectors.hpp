/**
 * @file kernels/injectors.hpp
 * @brief Kernels for injecting particles in different ways
 * @implements
 *   - kernel::UniformInjector_kernel<>
 *   - kernel::GlobalInjector_kernel<>
 *   - kernel::NonUniformInjector_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_INJECTORS_HPP
#define KERNELS_INJECTORS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"

namespace kernel {
  using namespace ntt;
  using spidx_t = unsigned short;

  template <SimEngine::type S, class M, class ED>
  struct UniformInjector_kernel {
    static_assert(ED::is_energy_dist, "ED must be an energy distribution class");
    static_assert(M::is_metric, "M must be a metric class");

    const spidx_t spidx1, spidx2;

    array_t<int*>      i1s_1, i2s_1, i3s_1;
    array_t<prtldx_t*> dx1s_1, dx2s_1, dx3s_1;
    array_t<real_t*>   ux1s_1, ux2s_1, ux3s_1;
    array_t<real_t*>   phis_1;
    array_t<real_t*>   weights_1;
    array_t<short*>    tags_1;

    array_t<int*>      i1s_2, i2s_2, i3s_2;
    array_t<prtldx_t*> dx1s_2, dx2s_2, dx3s_2;
    array_t<real_t*>   ux1s_2, ux2s_2, ux3s_2;
    array_t<real_t*>   phis_2;
    array_t<real_t*>   weights_2;
    array_t<short*>    tags_2;

    std::size_t            offset1, offset2;
    const M                metric;
    const array_t<real_t*> ni;
    const ED               energy_dist;
    const real_t           inv_V0;
    random_number_pool_t   random_pool;

    UniformInjector_kernel(spidx_t                          spidx1,
                           spidx_t                          spidx2,
                           Particles<M::Dim, M::CoordType>& species1,
                           Particles<M::Dim, M::CoordType>& species2,
                           std::size_t                      offset1,
                           std::size_t                      offset2,
                           const M&                         metric,
                           const array_t<real_t*>&          ni,
                           const ED&                        energy_dist,
                           real_t                           inv_V0,
                           random_number_pool_t&            random_pool)
      : spidx1 { spidx1 }
      , spidx2 { spidx2 }
      , i1s_1 { species1.i1 }
      , i2s_1 { species1.i2 }
      , i3s_1 { species1.i3 }
      , dx1s_1 { species1.dx1 }
      , dx2s_1 { species1.dx2 }
      , dx3s_1 { species1.dx3 }
      , ux1s_1 { species1.ux1 }
      , ux2s_1 { species1.ux2 }
      , ux3s_1 { species1.ux3 }
      , phis_1 { species1.phi }
      , weights_1 { species1.weight }
      , tags_1 { species1.tag }
      , i1s_2 { species2.i1 }
      , i2s_2 { species2.i2 }
      , i3s_2 { species2.i3 }
      , dx1s_2 { species2.dx1 }
      , dx2s_2 { species2.dx2 }
      , dx3s_2 { species2.dx3 }
      , ux1s_2 { species2.ux1 }
      , ux2s_2 { species2.ux2 }
      , ux3s_2 { species2.ux3 }
      , phis_2 { species2.phi }
      , weights_2 { species2.weight }
      , tags_2 { species2.tag }
      , offset1 { offset1 }
      , offset2 { offset2 }
      , metric { metric }
      , ni { ni }
      , energy_dist { energy_dist }
      , inv_V0 { inv_V0 }
      , random_pool { random_pool } {}

    Inline void operator()(index_t p) const {
      coord_t<M::Dim> x_Cd { ZERO };
      vec_t<Dim::_3D> v1 { ZERO }, v2 { ZERO };
      { // generate a random coordinate
        auto rand_gen = random_pool.get_state();
        x_Cd[0]       = Random<real_t>(rand_gen) * ni(0);
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          x_Cd[1] = Random<real_t>(rand_gen) * ni(1);
        }
        if constexpr (M::Dim == Dim::_3D) {
          x_Cd[2] = Random<real_t>(rand_gen) * ni(2);
        }
        random_pool.free_state(rand_gen);
      }
      { // generate the velocity
        coord_t<M::Dim> x_Ph { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        if constexpr (M::CoordType == Coord::Cart) {
          vec_t<Dim::_3D> v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph, v_Ph, v2);
        } else if constexpr (S == SimEngine::SRPIC) {
          coord_t<M::PrtlDim> x_Ph_ { ZERO };
          x_Ph_[0] = x_Ph[0];
          x_Ph_[1] = x_Ph[1];
          x_Ph_[2] = ZERO; // phi = 0
          vec_t<Dim::_3D> v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph_, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph_, v_Ph, v2);
        } else if constexpr (S == SimEngine::GRPIC) {
          vec_t<Dim::_3D> v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform<Idx::T, Idx::D>(x_Ph, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform<Idx::T, Idx::D>(x_Ph, v_Ph, v2);
        } else {
          raise::KernelError(HERE, "Unknown simulation engine");
        }
      }
      // inject
      i1s_1(p + offset1)  = static_cast<int>(x_Cd[0]);
      dx1s_1(p + offset1) = static_cast<prtldx_t>(
        x_Cd[0] - static_cast<real_t>(i1s_1(p + offset1)));
      i1s_2(p + offset2)  = i1s_1(p + offset1);
      dx1s_2(p + offset2) = dx1s_1(p + offset1);
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i2s_1(p + offset1)  = static_cast<int>(x_Cd[1]);
        dx2s_1(p + offset1) = static_cast<prtldx_t>(
          x_Cd[1] - static_cast<real_t>(i2s_1(p + offset1)));
        i2s_2(p + offset2)  = i2s_1(p + offset1);
        dx2s_2(p + offset2) = dx2s_1(p + offset1);
        if constexpr (S == SimEngine::SRPIC && M::CoordType != Coord::Cart) {
          phis_1(p + offset1) = ZERO;
          phis_2(p + offset2) = ZERO;
        }
      }
      if constexpr (M::Dim == Dim::_3D) {
        i3s_1(p + offset1)  = static_cast<int>(x_Cd[2]);
        dx3s_1(p + offset1) = static_cast<prtldx_t>(
          x_Cd[2] - static_cast<real_t>(i3s_1(p + offset1)));
        i3s_2(p + offset2)  = i3s_1(p + offset1);
        dx3s_2(p + offset2) = dx3s_1(p + offset1);
      }
      ux1s_1(p + offset1) = v1[0];
      ux2s_1(p + offset1) = v1[1];
      ux3s_1(p + offset1) = v1[2];
      ux1s_2(p + offset2) = v2[0];
      ux2s_2(p + offset2) = v2[1];
      ux3s_2(p + offset2) = v2[2];
      tags_1(p + offset1) = ParticleTag::alive;
      tags_2(p + offset2) = ParticleTag::alive;
      if constexpr (M::CoordType == Coord::Cart) {
        weights_1(p + offset1) = ONE;
        weights_2(p + offset2) = ONE;
      } else {
        const auto sqrt_det_h  = metric.sqrt_det_h(x_Cd);
        weights_1(p + offset1) = sqrt_det_h * inv_V0;
        weights_2(p + offset2) = sqrt_det_h * inv_V0;
      }
    }
  }; // struct UniformInjector_kernel

  template <SimEngine::type S, class M>
  struct GlobalInjector_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const bool use_weights;

    array_t<real_t*> in_ux1;
    array_t<real_t*> in_ux2;
    array_t<real_t*> in_ux3;
    array_t<real_t*> in_x1;
    array_t<real_t*> in_x2;
    array_t<real_t*> in_x3;
    array_t<real_t*> in_phi;
    array_t<real_t*> in_wei;

    array_t<std::size_t> idx { "idx" };
    array_t<int*>        i1s, i2s, i3s;
    array_t<prtldx_t*>   dx1s, dx2s, dx3s;
    array_t<real_t*>     ux1s, ux2s, ux3s;
    array_t<real_t*>     phis;
    array_t<real_t*>     weights;
    array_t<short*>      tags;

    const std::size_t offset;

    M global_metric;

    real_t      x1_min, x1_max, x2_min, x2_max, x3_min, x3_max;
    std::size_t i1_offset, i2_offset, i3_offset;

    GlobalInjector_kernel(Particles<M::Dim, M::CoordType>& species,
                          const M&                         global_metric,
                          const Domain<S, M>&              local_domain,
                          const std::map<std::string, std::vector<real_t>>& data,
                          bool use_weights)
      : use_weights { use_weights }
      , i1s { species.i1 }
      , i2s { species.i2 }
      , i3s { species.i3 }
      , dx1s { species.dx1 }
      , dx2s { species.dx2 }
      , dx3s { species.dx3 }
      , ux1s { species.ux1 }
      , ux2s { species.ux2 }
      , ux3s { species.ux3 }
      , phis { species.phi }
      , weights { species.weight }
      , tags { species.tag }
      , offset { species.npart() }
      , global_metric { global_metric } {
      const auto n_inject = data.at("x1").size();

      x1_min    = local_domain.mesh.extent(in::x1).first;
      x1_max    = local_domain.mesh.extent(in::x1).second;
      i1_offset = local_domain.offset_ncells()[0];

      copy_from_vector("x1", in_x1, data, n_inject);
      copy_from_vector("ux1", in_ux1, data, n_inject);
      copy_from_vector("ux2", in_ux2, data, n_inject);
      copy_from_vector("ux3", in_ux3, data, n_inject);
      if (use_weights) {
        copy_from_vector("weights", in_wei, data, n_inject);
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        x2_min    = local_domain.mesh.extent(in::x2).first;
        x2_max    = local_domain.mesh.extent(in::x2).second;
        i2_offset = local_domain.offset_ncells()[1];
        copy_from_vector("x2", in_x2, data, n_inject);
      }
      if constexpr (D == Dim::_2D and M::CoordType != Coord::Cart) {
        copy_from_vector("phi", in_phi, data, n_inject);
      }
      if constexpr (D == Dim::_3D) {
        x3_min    = local_domain.mesh.extent(in::x3).first;
        x3_max    = local_domain.mesh.extent(in::x3).second;
        i3_offset = local_domain.offset_ncells()[2];
        copy_from_vector("x3", in_x3, data, n_inject);
      }
    }

    void copy_from_vector(const std::string& name,
                          array_t<real_t*>&  arr,
                          const std::map<std::string, std::vector<real_t>>& data,
                          std::size_t n_inject) {
      raise::ErrorIf(data.find(name) == data.end(), name + " not found in data", HERE);
      raise::ErrorIf(data.at(name).size() != n_inject, "Inconsistent data size", HERE);
      arr        = array_t<real_t*> { name, n_inject };
      auto arr_h = Kokkos::create_mirror_view(arr);
      for (std::size_t i = 0; i < data.at(name).size(); ++i) {
        arr_h(i) = data.at(name)[i];
      }
      Kokkos::deep_copy(arr, arr_h);
    }

    auto number_injected() const -> std::size_t {
      auto idx_h = Kokkos::create_mirror_view(idx);
      Kokkos::deep_copy(idx_h, idx);
      return idx_h();
    }

    Inline void operator()(index_t p) const {
      if constexpr (D == Dim::_1D) {
        if (in_x1(p) >= x1_min and in_x1(p) < x1_max) {
          coord_t<Dim::_1D>     x_Cd { ZERO };
          vec_t<Dim::_3D>       u_XYZ { ZERO };
          const vec_t<Dim::_3D> u_Ph { in_ux1(p), in_ux2(p), in_ux3(p) };

          auto index { offset + Kokkos::atomic_fetch_add(&idx(), 1) };
          global_metric.template convert<Crd::Ph, Crd::Cd>({ in_x1(p) }, x_Cd);
          global_metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, u_Ph, u_XYZ);

          const auto i1 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[0]) - i1_offset);
          const auto dx1 = static_cast<prtldx_t>(
            x_Cd[0] - static_cast<real_t>(i1 + i1_offset));

          i1s(index)  = i1;
          dx1s(index) = dx1;
          ux1s(index) = u_XYZ[0];
          ux2s(index) = u_XYZ[1];
          ux3s(index) = u_XYZ[2];
          tags(index) = ParticleTag::alive;
          if (use_weights) {
            weights(index) = weights(p);
          } else {
            weights(index) = ONE;
          }
        }
      } else if constexpr (D == Dim::_2D) {
        if ((in_x1(p) >= x1_min and in_x1(p) < x1_max) and
            (in_x2(p) >= x2_min and in_x2(p) < x2_max)) {
          coord_t<Dim::_2D>   x_Cd { ZERO };
          vec_t<Dim::_3D>     u_Cd { ZERO };
          vec_t<Dim::_3D>     u_Ph { in_ux1(p), in_ux2(p), in_ux3(p) };
          coord_t<M::PrtlDim> x_Cd_ { ZERO };

          auto index {
            offset + Kokkos::atomic_fetch_add(&idx(), static_cast<std::size_t>(1))
          };
          global_metric.template convert<Crd::Ph, Crd::Cd>({ in_x1(p), in_x2(p) },
                                                           x_Cd);
          x_Cd_[0] = x_Cd[0];
          x_Cd_[1] = x_Cd[1];
          if constexpr (S == SimEngine::SRPIC and M::CoordType != Coord::Cart) {
            x_Cd_[2] = in_phi(p);
          }
          if constexpr (S == SimEngine::SRPIC) {
            global_metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd_, u_Ph, u_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            global_metric.template transform<Idx::PD, Idx::D>(x_Cd, u_Ph, u_Cd);
          } else {
            raise::KernelError(HERE, "Unknown simulation engine");
          }
          const auto i1 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[0]) - i1_offset);
          const auto dx1 = static_cast<prtldx_t>(
            x_Cd[0] - static_cast<real_t>(i1 + i1_offset));
          const auto i2 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[1]) - i2_offset);
          const auto dx2 = static_cast<prtldx_t>(
            x_Cd[1] - static_cast<real_t>(i2 + i2_offset));

          i1s(index)  = i1;
          dx1s(index) = dx1;
          i2s(index)  = i2;
          dx2s(index) = dx2;
          ux1s(index) = u_Cd[0];
          ux2s(index) = u_Cd[1];
          ux3s(index) = u_Cd[2];
          if (M::CoordType != Coord::Cart) {
            phis(index) = in_phi(p);
          }
          tags(index) = ParticleTag::alive;
          if (use_weights) {
            weights(index) = weights(p);
          } else {
            weights(index) = ONE;
          }
        }
      } else {
        if ((in_x1(p) >= x1_min and in_x1(p) < x1_max) and
            (in_x2(p) >= x2_min and in_x2(p) < x2_max) and
            (in_x3(p) >= x3_min and in_x3(p) < x3_max)) {
          coord_t<Dim::_3D> x_Cd { ZERO };
          vec_t<Dim::_3D>   u_Cd { ZERO };
          vec_t<Dim::_3D>   u_Ph { in_ux1(p), in_ux2(p), in_ux3(p) };

          auto index { offset + Kokkos::atomic_fetch_add(&idx(), 1) };
          global_metric.template convert<Crd::Ph, Crd::Cd>(
            { in_x1(p), in_x2(p), in_x3(p) },
            x_Cd);
          if constexpr (S == SimEngine::SRPIC) {
            global_metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, u_Ph, u_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            global_metric.template transform<Idx::PD, Idx::D>(x_Cd, u_Ph, u_Cd);
          } else {
            raise::KernelError(HERE, "Unknown simulation engine");
          }
          const auto i1 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[0]) - i1_offset);
          const auto dx1 = static_cast<prtldx_t>(
            x_Cd[0] - static_cast<real_t>(i1 + i1_offset));
          const auto i2 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[1]) - i2_offset);
          const auto dx2 = static_cast<prtldx_t>(
            x_Cd[1] - static_cast<real_t>(i2 + i2_offset));
          const auto i3 = static_cast<int>(
            static_cast<std::size_t>(x_Cd[2]) - i3_offset);
          const auto dx3 = static_cast<prtldx_t>(
            x_Cd[2] - static_cast<real_t>(i3 + i3_offset));

          i1s(index)  = i1;
          dx1s(index) = dx1;
          i2s(index)  = i2;
          dx2s(index) = dx2;
          i3s(index)  = i3;
          dx3s(index) = dx3;
          ux1s(index) = u_Cd[0];
          ux2s(index) = u_Cd[1];
          ux3s(index) = u_Cd[2];
          tags(index) = ParticleTag::alive;
          if (use_weights) {
            weights(index) = weights(p);
          } else {
            weights(index) = ONE;
          }
        }
      }
    }
  }; // struct GlobalInjector_kernel

  template <SimEngine::type S, class M, class ED, class SD>
  struct NonUniformInjector_kernel {
    static_assert(ED::is_energy_dist, "ED must be an energy distribution class");
    static_assert(SD::is_spatial_dist, "SD must be a spatial distribution class");
    static_assert(M::is_metric, "M must be a metric class");

    const real_t  ppc0;
    const spidx_t spidx1, spidx2;

    array_t<int*>      i1s_1, i2s_1, i3s_1;
    array_t<prtldx_t*> dx1s_1, dx2s_1, dx3s_1;
    array_t<real_t*>   ux1s_1, ux2s_1, ux3s_1;
    array_t<real_t*>   phis_1;
    array_t<real_t*>   weights_1;
    array_t<short*>    tags_1;

    array_t<int*>      i1s_2, i2s_2, i3s_2;
    array_t<prtldx_t*> dx1s_2, dx2s_2, dx3s_2;
    array_t<real_t*>   ux1s_2, ux2s_2, ux3s_2;
    array_t<real_t*>   phis_2;
    array_t<real_t*>   weights_2;
    array_t<short*>    tags_2;

    array_t<std::size_t> idx { "idx" };

    std::size_t          offset1, offset2;
    M                    metric;
    const ED             energy_dist;
    const SD             spatial_dist;
    const real_t         inv_V0;
    random_number_pool_t random_pool;

    NonUniformInjector_kernel(real_t                           ppc0,
                              spidx_t                          spidx1,
                              spidx_t                          spidx2,
                              Particles<M::Dim, M::CoordType>& species1,
                              Particles<M::Dim, M::CoordType>& species2,
                              std::size_t                      offset1,
                              std::size_t                      offset2,
                              const M&                         metric,
                              const ED&                        energy_dist,
                              const SD&                        spatial_dist,
                              real_t                           inv_V0,
                              random_number_pool_t&            random_pool)
      : ppc0 { ppc0 }
      , spidx1 { spidx1 }
      , spidx2 { spidx2 }
      , i1s_1 { species1.i1 }
      , i2s_1 { species1.i2 }
      , i3s_1 { species1.i3 }
      , dx1s_1 { species1.dx1 }
      , dx2s_1 { species1.dx2 }
      , dx3s_1 { species1.dx3 }
      , ux1s_1 { species1.ux1 }
      , ux2s_1 { species1.ux2 }
      , ux3s_1 { species1.ux3 }
      , phis_1 { species1.phi }
      , weights_1 { species1.weight }
      , tags_1 { species1.tag }
      , i1s_2 { species2.i1 }
      , i2s_2 { species2.i2 }
      , i3s_2 { species2.i3 }
      , dx1s_2 { species2.dx1 }
      , dx2s_2 { species2.dx2 }
      , dx3s_2 { species2.dx3 }
      , ux1s_2 { species2.ux1 }
      , ux2s_2 { species2.ux2 }
      , ux3s_2 { species2.ux3 }
      , phis_2 { species2.phi }
      , weights_2 { species2.weight }
      , tags_2 { species2.tag }
      , offset1 { offset1 }
      , offset2 { offset2 }
      , metric { metric }
      , energy_dist { energy_dist }
      , spatial_dist { spatial_dist }
      , inv_V0 { inv_V0 }
      , random_pool { random_pool } {}

    auto number_injected() const -> std::size_t {
      auto idx_h = Kokkos::create_mirror_view(idx);
      Kokkos::deep_copy(idx_h, idx);
      return idx_h();
    }

    Inline void operator()(index_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Cd { i1_ + HALF };
        coord_t<Dim::_1D> x_Ph { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        const auto ppc = static_cast<std::size_t>(ppc0 * spatial_dist(x_Ph));
        if (ppc == 0) {
          return;
        }
        auto rand_gen = random_pool.get_state();
        for (auto p { 0u }; p < ppc; ++p) {
          const auto index = Kokkos::atomic_fetch_add(&idx(), 1);
          const auto dx1   = Random<prtldx_t>(rand_gen);

          i1s_1(index + offset1)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_1(index + offset1) = dx1;
          i1s_2(index + offset2)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_2(index + offset2) = dx1;

          vec_t<Dim::_3D> v_T { ZERO }, v_XYZ { ZERO };
          energy_dist(x_Ph, v_T, spidx1);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, v_T, v_XYZ);
          ux1s_1(index + offset1) = v_XYZ[0];
          ux2s_1(index + offset1) = v_XYZ[1];
          ux3s_1(index + offset1) = v_XYZ[2];
          energy_dist(x_Ph, v_T, spidx2);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, v_T, v_XYZ);
          ux1s_2(index + offset2) = v_XYZ[0];
          ux2s_2(index + offset2) = v_XYZ[1];
          ux3s_2(index + offset2) = v_XYZ[2];

          tags_1(index + offset1) = ParticleTag::alive;
          tags_2(index + offset2) = ParticleTag::alive;
          if (M::CoordType == Coord::Cart) {
            weights_1(index + offset1) = ONE;
            weights_2(index + offset2) = ONE;
          } else {
            const auto wei = metric.sqrt_det_h({ i1_ + HALF }) * inv_V0;
            weights_1(index + offset1) = wei;
            weights_2(index + offset2) = wei;
          }
        }
        random_pool.free_state(rand_gen);
      } else {
        raise::KernelError(HERE, "NonUniformInjector_kernel 1D called for 2D/3D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        const auto          i1_ = COORD(i1);
        const auto          i2_ = COORD(i2);
        coord_t<Dim::_2D>   x_Cd { i1_ + HALF, i2_ + HALF };
        coord_t<Dim::_2D>   x_Ph { ZERO };
        coord_t<M::PrtlDim> x_Cd_ { ZERO };
        x_Cd_[0] = x_Cd[0];
        x_Cd_[1] = x_Cd[1];
        if constexpr (S == SimEngine::SRPIC and M::CoordType != Coord::Cart) {
          x_Cd_[2] = ZERO;
        }
        metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        const auto ppc = static_cast<std::size_t>(ppc0 * spatial_dist(x_Ph));
        if (ppc == 0) {
          return;
        }
        auto rand_gen = random_pool.get_state();
        for (auto p { 0u }; p < ppc; ++p) {
          const auto index = Kokkos::atomic_fetch_add(&idx(), 1);
          const auto dx1   = Random<prtldx_t>(rand_gen);
          const auto dx2   = Random<prtldx_t>(rand_gen);

          i1s_1(index + offset1)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_1(index + offset1) = dx1;
          i1s_2(index + offset2)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_2(index + offset2) = dx1;

          i2s_1(index + offset1)  = static_cast<int>(i2) - N_GHOSTS;
          dx2s_1(index + offset1) = dx2;
          i2s_2(index + offset2)  = static_cast<int>(i2) - N_GHOSTS;
          dx2s_2(index + offset2) = dx2;

          vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
          energy_dist(x_Ph, v_T, spidx1);
          if constexpr (S == SimEngine::SRPIC) {
            metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd_, v_T, v_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            metric.template transform<Idx::T, Idx::D>(x_Cd_, v_T, v_Cd);
          }
          ux1s_1(index + offset1) = v_Cd[0];
          ux2s_1(index + offset1) = v_Cd[1];
          ux3s_1(index + offset1) = v_Cd[2];
          energy_dist(x_Ph, v_T, spidx2);
          if constexpr (S == SimEngine::SRPIC) {
            metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd_, v_T, v_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            metric.template transform<Idx::T, Idx::D>(x_Cd_, v_T, v_Cd);
          }
          ux1s_2(index + offset2) = v_Cd[0];
          ux2s_2(index + offset2) = v_Cd[1];
          ux3s_2(index + offset2) = v_Cd[2];

          tags_1(index + offset1) = ParticleTag::alive;
          tags_2(index + offset2) = ParticleTag::alive;
          if (M::CoordType == Coord::Cart) {
            weights_1(index + offset1) = ONE;
            weights_2(index + offset2) = ONE;
          } else {
            const auto wei = metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF }) * inv_V0;
            weights_1(index + offset1) = wei;
            weights_2(index + offset2) = wei;
          }
        }
        random_pool.free_state(rand_gen);
      }

      else {
        raise::KernelError(HERE, "NonUniformInjector_kernel 2D called for 1D/3D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        const auto        i3_ = COORD(i3);
        coord_t<Dim::_3D> x_Cd { i1_ + HALF, i2_ + HALF, i3_ + HALF };
        coord_t<Dim::_3D> x_Ph { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        const auto ppc = static_cast<std::size_t>(ppc0 * spatial_dist(x_Ph));
        if (ppc == 0) {
          return;
        }
        auto rand_gen = random_pool.get_state();
        for (auto p { 0u }; p < ppc; ++p) {
          const auto index = Kokkos::atomic_fetch_add(&idx(), 1);
          const auto dx1   = Random<prtldx_t>(rand_gen);
          const auto dx2   = Random<prtldx_t>(rand_gen);
          const auto dx3   = Random<prtldx_t>(rand_gen);

          i1s_1(index + offset1)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_1(index + offset1) = dx1;
          i1s_2(index + offset2)  = static_cast<int>(i1) - N_GHOSTS;
          dx1s_2(index + offset2) = dx1;

          i2s_1(index + offset1)  = static_cast<int>(i2) - N_GHOSTS;
          dx2s_1(index + offset1) = dx2;
          i2s_2(index + offset2)  = static_cast<int>(i2) - N_GHOSTS;
          dx2s_2(index + offset2) = dx2;

          i3s_1(index + offset1)  = static_cast<int>(i3) - N_GHOSTS;
          dx3s_1(index + offset1) = dx3;
          i3s_2(index + offset2)  = static_cast<int>(i3) - N_GHOSTS;
          dx3s_2(index + offset2) = dx3;

          vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
          energy_dist(x_Ph, v_T, spidx1);
          if constexpr (S == SimEngine::SRPIC) {
            metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, v_T, v_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            metric.template transform<Idx::T, Idx::D>(x_Cd, v_T, v_Cd);
          }
          ux1s_1(index + offset1) = v_Cd[0];
          ux2s_1(index + offset1) = v_Cd[1];
          ux3s_1(index + offset1) = v_Cd[2];
          energy_dist(x_Ph, v_T, spidx2);
          if constexpr (S == SimEngine::SRPIC) {
            metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, v_T, v_Cd);
          } else if constexpr (S == SimEngine::GRPIC) {
            metric.template transform<Idx::T, Idx::D>(x_Cd, v_T, v_Cd);
          }
          ux1s_2(index + offset2) = v_Cd[0];
          ux2s_2(index + offset2) = v_Cd[1];
          ux3s_2(index + offset2) = v_Cd[2];

          tags_1(index + offset1) = ParticleTag::alive;
          tags_2(index + offset2) = ParticleTag::alive;
          if (M::CoordType == Coord::Cart) {
            weights_1(index + offset1) = ONE;
            weights_2(index + offset2) = ONE;
          } else {
            const auto wei = metric.sqrt_det_h(
                               { i1_ + HALF, i2_ + HALF, i3_ + HALF }) *
                             inv_V0;
            weights_1(index + offset1) = wei;
            weights_2(index + offset2) = wei;
          }
        }
        random_pool.free_state(rand_gen);
      } else {
        raise::KernelError(HERE, "NonUniformInjector_kernel 3D called for 1D/2D");
      }
    }
  }; // struct NonUniformInjector_kernel

} // namespace kernel

#endif // KERNELS_INJECTORS_HPP
