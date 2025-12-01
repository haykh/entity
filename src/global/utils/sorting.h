/**
 * @file utils/sorting.h
 * @brief Comparator structs for Kokkos bin-sorting
 * @implements
 *   - sort::BinBool<>
 *   - sort::BinTag<>
 * @namespaces:
 *   - sort::
 * @note BinBool sorts by boolean values "true" then "false"
 * @note BinTag sorts by tag values "1" then "0" then "2" ... "n"
 */

#ifndef GLOBAL_UTILS_SORTING_H
#define GLOBAL_UTILS_SORTING_H

#include "arch/kokkos_aliases.h"

namespace sort {

  template <class KeyViewType>
  struct BinBool {
    BinBool() = default;

    template <class ViewType>
    Inline auto bin(ViewType& keys, int i) const -> int {
      return keys(i) ? 1 : 0;
    }

    Inline auto max_bins() const -> int {
      return 2;
    }

    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType&, iT1&, iT2&) const -> bool {
      return false;
    }
  };

  template <class KeyViewType>
  struct BinTag {
    BinTag(int max_bins) : m_max_bins { max_bins } {}

    template <class ViewType>
    Inline auto bin(ViewType& keys, int i) const -> int {
      return (keys(i) == 0) ? 1 : ((keys(i) == 1) ? 0 : keys(i));
    }

    Inline auto max_bins() const -> int {
      return m_max_bins;
    }

    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType&, iT1&, iT2&) const -> bool {
      return false;
    }

  private:
    const int m_max_bins;
  };

} // namespace sort

#endif // GLOBAL_UTILS_SORTING_H
