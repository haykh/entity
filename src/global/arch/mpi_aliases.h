/**
 * @file arch/mpi_aliases.h
 * @brief Aliases for easy access to MPI variables & thread-agnostic function calls
 * @implements
 *   - CallOnce
 *   - mpi::get_type<>
 *   - MPI_ROOT_RANK [if MPI_ENABLED]
 * @namespaces:
 *   - mpi::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_ARCH_MPI_ALIASES_H
#define GLOBAL_ARCH_MPI_ALIASES_H

#include <cstdint>
#include <utility>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#if !defined(MPI_ENABLED)

template <typename Func, typename... Args>
void CallOnce(Func func, Args&&... args) {
  func(std::forward<Args>(args)...);
}

#else // defined MPI_ENABLED

  #define MPI_ROOT_RANK 0

template <typename Func, typename... Args>
void CallOnce(Func func, Args&&... args) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == MPI_ROOT_RANK) {
    func(std::forward<Args>(args)...);
  }
}

namespace mpi {

  template <typename T>
  constexpr MPI_Datatype get_type() noexcept {
    if constexpr (std::is_same<T, char>::value) {
      return MPI_CHAR;
    } else if constexpr (std::is_same<T, signed char>::value) {
      return MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same<T, unsigned char>::value) {
      return MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same<T, wchar_t>::value) {
      return MPI_WCHAR;
    } else if constexpr (std::is_same<T, signed short>::value) {
      return MPI_SHORT;
    } else if constexpr (std::is_same<T, unsigned short>::value) {
      return MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same<T, signed int>::value) {
      return MPI_INT;
    } else if constexpr (std::is_same<T, unsigned int>::value) {
      return MPI_UNSIGNED;
    } else if constexpr (std::is_same<T, signed long int>::value) {
      return MPI_LONG;
    } else if constexpr (std::is_same<T, unsigned long int>::value) {
      return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same<T, signed long long int>::value) {
      return MPI_LONG_LONG;
    } else if constexpr (std::is_same<T, unsigned long long int>::value) {
      return MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same<T, std::size_t>::value) {
      return MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same<T, float>::value) {
      return MPI_FLOAT;
    } else if constexpr (std::is_same<T, double>::value) {
      return MPI_DOUBLE;
    } else if constexpr (std::is_same<T, long double>::value) {
      return MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same<T, std::int8_t>::value) {
      return MPI_INT8_T;
    } else if constexpr (std::is_same<T, std::int16_t>::value) {
      return MPI_INT16_T;
    } else if constexpr (std::is_same<T, std::int32_t>::value) {
      return MPI_INT32_T;
    } else if constexpr (std::is_same<T, std::int64_t>::value) {
      return MPI_INT64_T;
    } else if constexpr (std::is_same<T, std::uint8_t>::value) {
      return MPI_UINT8_T;
    } else if constexpr (std::is_same<T, std::uint16_t>::value) {
      return MPI_UINT16_T;
    } else if constexpr (std::is_same<T, std::uint32_t>::value) {
      return MPI_UINT32_T;
    } else if constexpr (std::is_same<T, std::uint64_t>::value) {
      return MPI_UINT64_T;
    } else if constexpr (std::is_same<T, bool>::value) {
      return MPI_C_BOOL;
    } else {
      return MPI_BYTE;
    }
  }

} // namespace mpi

#endif // MPI_ENABLED

#endif // GLOBAL_ARCH_MPI_ALIASES_H
