/**
 * @file arch/kokkos_aliases.h
 * @brief Kokkos type & function aliases
 * @implements
 *   - ClassLambda, Lambda, Function, Inline macros
 *   - array_t, array_mirror_t, scatter_array_t
 *   - ndarray_t, ndfield_t
 *   - ndfield_mirror_t, scatter_ndfield_t
 *   - range_t, range_h_t
 *   - CreateRangePolicy, CreateRangePolicyOnHost
 *   - random_number_pool_t, random_generator_t
 *   - Random function
 * @cpp:
 *   - arch/kokkos_aliases.cpp
 * @namespaces:
 *   - math:: (aliased to Kokkos::)
 */

#ifndef GLOBAL_ARCH_KOKKOS_ALIASES_H
#define GLOBAL_ARCH_KOKKOS_ALIASES_H

#include "global.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>

#define ClassLambda KOKKOS_CLASS_LAMBDA
#define Lambda      KOKKOS_LAMBDA
#define Function    KOKKOS_FUNCTION
#define Inline      KOKKOS_INLINE_FUNCTION

namespace math = Kokkos;

template <typename T>
using array_t = Kokkos::View<T>;

// Array mirror alias of arbitrary type
template <typename T>
using array_mirror_t = typename array_t<T>::HostMirror;

// Scatter view alias of arbitrary type
template <typename T>
using scatter_array_t = Kokkos::Experimental::ScatterView<T>;

// Array aliases of arbitrary type and dimensions (up to 3)
namespace kokkos_aliases_hidden {
  // c++ magic
  template <unsigned short D>
  struct ndarray_impl {
    using type = void;
  };

  template <>
  struct ndarray_impl<1> {
    using type = array_t<real_t*>;
  };

  template <>
  struct ndarray_impl<2> {
    using type = array_t<real_t**>;
  };

  template <>
  struct ndarray_impl<3> {
    using type = array_t<real_t***>;
  };

  template <>
  struct ndarray_impl<4> {
    using type = array_t<real_t****>;
  };
} // namespace kokkos_aliases_hidden

template <unsigned short D>
using ndarray_t = typename kokkos_aliases_hidden::ndarray_impl<D>::type;

namespace kokkos_aliases_hidden {
  // c++ magic
  template <Dimension D, unsigned short N>
  struct ndfield_impl {
    using type = void;
  };

  template <unsigned short N>
  struct ndfield_impl<Dim::_1D, N> {
    using type = array_t<real_t* [N]>;
  };

  template <unsigned short N>
  struct ndfield_impl<Dim::_2D, N> {
    using type = array_t<real_t** [N]>;
  };

  template <unsigned short N>
  struct ndfield_impl<Dim::_3D, N> {
    using type = array_t<real_t*** [N]>;
  };
} // namespace kokkos_aliases_hidden

template <Dimension D, unsigned short N>
using ndfield_t = typename kokkos_aliases_hidden::ndfield_impl<D, N>::type;

// D x N dimensional array (host memspace) for storing fields on ND hypercube
template <Dimension D, unsigned short N>
using ndfield_mirror_t = typename ndfield_t<D, N>::HostMirror;

// D x N dimensional scatter array for storing fields on ND hypercubes
namespace kokkos_aliases_hidden {
  // c++ magic
  template <Dimension D, unsigned short N>
  struct scatter_ndfield_impl {
    using type = void;
  };

  template <unsigned short N>
  struct scatter_ndfield_impl<Dim::_1D, N> {
    using type = scatter_array_t<real_t* [N]>;
  };

  template <unsigned short N>
  struct scatter_ndfield_impl<Dim::_2D, N> {
    using type = scatter_array_t<real_t** [N]>;
  };

  template <unsigned short N>
  struct scatter_ndfield_impl<Dim::_3D, N> {
    using type = scatter_array_t<real_t*** [N]>;
  };
} // namespace kokkos_aliases_hidden

template <Dimension D, unsigned short N>
using scatter_ndfield_t =
  typename kokkos_aliases_hidden::scatter_ndfield_impl<D, N>::type;

namespace kokkos_aliases_hidden {
  // c++ magic
  template <Dimension D, unsigned short N>
  struct randacc_ndfield_impl {
    using type = void;
  };

  template <unsigned short N>
  struct randacc_ndfield_impl<Dim::_1D, N> {
    using type =
      Kokkos::View<const real_t* [N], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  };

  template <unsigned short N>
  struct randacc_ndfield_impl<Dim::_2D, N> {
    using type =
      Kokkos::View<const real_t** [N], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  };

  template <unsigned short N>
  struct randacc_ndfield_impl<Dim::_3D, N> {
    using type =
      Kokkos::View<const real_t*** [N], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  };
} // namespace kokkos_aliases_hidden

template <Dimension D, unsigned short N>
using randacc_ndfield_t =
  typename kokkos_aliases_hidden::randacc_ndfield_impl<D, N>::type;

// Defining aliases for `RangePolicy` and `MDRangePolicy` for the device space
namespace kokkos_aliases_hidden {
  // c++ magic
  template <Dimension D>
  struct range_impl {
    using type = void;
  };

  template <>
  struct range_impl<Dim::_1D> {
    using type = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>;
  };

  template <>
  struct range_impl<Dim::_2D> {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>;
  };

  template <>
  struct range_impl<Dim::_3D> {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>;
  };
} // namespace kokkos_aliases_hidden

template <Dimension D>
using range_t = typename kokkos_aliases_hidden::range_impl<D>::type;

// Defining aliases for `RangePolicy` and `MDRangePolicy` for the host space
namespace kokkos_aliases_hidden {
  // c++ magic
  template <Dimension D>
  struct range_h_impl {
    using type = void;
  };

  template <>
  struct range_h_impl<Dim::_1D> {
    using type = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
  };

  template <>
  struct range_h_impl<Dim::_2D> {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultHostExecutionSpace>;
  };

  template <>
  struct range_h_impl<Dim::_3D> {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultHostExecutionSpace>;
  };

} // namespace kokkos_aliases_hidden

template <Dimension D>
using range_h_t = typename kokkos_aliases_hidden::range_h_impl<D>::type;

/**
 * @brief Function template for generating ND Kokkos range policy.
 * @tparam D Dimension
 * @param i1 array of size D `std::size_t`: { min }.
 * @param i2 array of size D `std::size_t`: { max }.
 * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the accelerator execution space.
 */
template <Dimension D>
auto CreateRangePolicy(const tuple_t<std::size_t, D>&,
                       const tuple_t<std::size_t, D>&) -> range_t<D>;

/**
 * @brief Function template for generating ND Kokkos range policy on the host.
 * @tparam D Dimension
 * @param i1 array of size D `std::size_t`: { min }.
 * @param i2 array of size D `std::size_t`: { max }.
 * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the host execution space.
 */
template <Dimension D>
auto CreateRangePolicyOnHost(const tuple_t<std::size_t, D>&,
                             const tuple_t<std::size_t, D>&) -> range_h_t<D>;

// Random number pool/generator type alias
using random_number_pool_t = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
using random_generator_t = typename random_number_pool_t::generator_type;

// Random number generator functions
template <typename T>
Inline auto Random(random_generator_t&) -> T;

template <>
Inline auto Random<float>(random_generator_t& gen) -> float {
  return gen.frand();
}

template <>
Inline auto Random<double>(random_generator_t& gen) -> double {
  return gen.drand();
}

#endif // GLOBAL_ARCH_KOKKOS_ALIASES_H
