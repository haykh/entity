/**
 * @file arch/directions.h
 * @brief Defines the functionality to work with logically cartesian directions
 * @implements
 *   - dir::Directions
 *   - dir::direction_t
 *   - dir::map_t
 *   - dir::dirs_t
 * @namespaces:
 *   - dir::Directions
 * @note
 * dir::Directions<D>::all contains all possible directions in D
 * dimensions with increments in each direction
 * @note
 * dir::Directions<D>::orth contains only orthogonal directions
 * @note
 * dir::Directions<D>::unique is similar to ::all,
 * but contains only unique directions (i.e., no difference between +1/-1)
 */

#ifndef GLOBAL_ARCH_DIRECTIONS_H
#define GLOBAL_ARCH_DIRECTIONS_H

#include "global.h"

#include "utils/error.h"

#include <initializer_list>
#include <iomanip>
#include <map>
#include <ostream>
#include <vector>

namespace dir {

  template <Dimension D>
  struct Directions {};

  template <Dimension D>
  struct direction_t : public std::vector<short> {
    direction_t() : std::vector<short>(D, 0) {}

    direction_t(std::initializer_list<short> list) : std::vector<short>(list) {
      raise::ErrorIf(list.size() != D,
                     "Wrong number of elements in direction_t initializer list",
                     HERE);
    }

    auto hash() const -> short;

    auto operator-() const -> direction_t<D> {
      auto result = direction_t<D> {};
      for (std::size_t i = 0; i < (short)D; ++i) {
        result[i] = -(*this)[i];
      }
      return result;
    }

    auto operator==(const direction_t<D>& other) const -> bool {
      for (std::size_t i = 0; i < (short)D; ++i) {
        if ((*this)[i] != other[i]) {
          return false;
        }
      }
      return true;
    }

    auto to_string() const -> std::string {
      std::stringstream ss;
      ss << *this;
      return ss.str();
    }

    /**
     * @brief get the associated orthogonal directions
     * @example {-1, 1} -> [{-1, 0}, {0, 1}]
     * @example {1, 1, 0} -> [{1, 0, 0}, {0, 1, 0}]
     * @example {-1, 1, -1} -> [{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}]
     */
    auto get_assoc_orth() const -> std::vector<direction_t> {
      auto result = std::vector<direction_t> {};
      for (std::size_t i = 0; i < this->size(); ++i) {
        if ((*this)[i] != 0) {
          direction_t<D> dir;
          dir[i] = (*this)[i];
          result.push_back(dir);
        }
      }
      return result;
    }

    auto get_sign() const -> short {
      short sign = 0;
      for (std::size_t i = 0; i < this->size(); ++i) {
        if ((*this)[i] != 0) {
          raise::ErrorIf(sign != 0,
                         "Undefined signature for non-orth direction",
                         HERE);
          sign = (*this)[i];
        }
      }
      raise::ErrorIf(sign == 0, "Undefined signature", HERE);
      return sign;
    }

    auto get_dim() const -> in {
      short dir = -1;
      for (std::size_t i = 0; i < this->size(); ++i) {
        if ((*this)[i] != 0) {
          raise::ErrorIf(dir > 0, "Undefined dim for non-orth direction", HERE);
          dir = i;
        }
      }
      raise::ErrorIf(dir == -1, "Undefined dim", HERE);
      if (dir == 0) {
        return in::x1;
      } else if (dir == 1) {
        return in::x2;
      } else if (dir == 2) {
        return in::x3;
      } else {
        raise::Error("Undefined dim", HERE);
        throw;
      }
    }
  };

  template <Dimension D, typename T>
  using map_t = std::map<direction_t<D>, T>;

  template <Dimension D>
  using dirs_t = std::vector<direction_t<D>>;

  template <Dimension D>
  inline auto operator<<(std::ostream&         os,
                         const direction_t<D>& dir) -> std::ostream& {
    for (auto& d : dir) {
      os << std::setw(2) << std::left;
      if (d > 0) {
        os << "+";
      } else if (d < 0) {
        os << "-";
      } else {
        os << "0";
      }
    }
    return os;
  }

  template <>
  inline auto direction_t<Dim::_1D>::hash() const -> short {
    return (*this)[0];
  }

  template <>
  inline auto direction_t<Dim::_2D>::hash() const -> short {
    return (2 + (*this)[0] + (*this)[1]) * (3 + (*this)[0] + (*this)[1]) / 2 +
           (*this)[1];
  }

  template <>
  inline auto direction_t<Dim::_3D>::hash() const -> short {
    short k1 = (2 + (*this)[0] + (*this)[1]) * (3 + (*this)[0] + (*this)[1]) / 2 +
               (*this)[1];
    return (2 + k1 + (*this)[2]) * (3 + k1 + (*this)[2]) / 2 + (*this)[2];
  }

  template <>
  struct Directions<Dim::_1D> {
    inline static const dirs_t<Dim::_1D> all    = { { -1 }, { 1 } };
    inline static const dirs_t<Dim::_1D> orth   = { { -1 }, { 1 } };
    inline static const dirs_t<Dim::_1D> unique = { { 1 } };
  };

  template <>
  struct Directions<Dim::_2D> {
    inline static const dirs_t<Dim::_2D> all = {
      { -1, -1 },
      { -1,  0 },
      { -1,  1 },
      {  0, -1 },
      {  0,  1 },
      {  1, -1 },
      {  1,  0 },
      {  1,  1 }
    };
    inline static const dirs_t<Dim::_2D> orth = {
      { -1,  0 },
      {  0, -1 },
      {  0,  1 },
      {  1,  0 }
    };
    inline static const dirs_t<Dim::_2D> unique = {
      {  0, 1 },
      {  1, 1 },
      {  1, 0 },
      { -1, 1 }
    };
  };

  template <>
  struct Directions<Dim::_3D> {
    inline static const dirs_t<Dim::_3D> all = {
      { -1, -1, -1 },
      { -1, -1,  0 },
      { -1, -1,  1 },
      { -1,  0, -1 },
      { -1,  0,  0 },
      { -1,  0,  1 },
      { -1,  1, -1 },
      { -1,  1,  0 },
      { -1,  1,  1 },
      {  0, -1, -1 },
      {  0, -1,  0 },
      {  0, -1,  1 },
      {  0,  0, -1 },
      {  0,  0,  1 },
      {  0,  1, -1 },
      {  0,  1,  0 },
      {  0,  1,  1 },
      {  1, -1, -1 },
      {  1, -1,  0 },
      {  1, -1,  1 },
      {  1,  0, -1 },
      {  1,  0,  0 },
      {  1,  0,  1 },
      {  1,  1, -1 },
      {  1,  1,  0 },
      {  1,  1,  1 }
    };
    inline static const dirs_t<Dim::_3D> orth = {
      { -1,  0,  0 },
      {  0, -1,  0 },
      {  0,  0, -1 },
      {  0,  0,  1 },
      {  0,  1,  0 },
      {  1,  0,  0 }
    };
    inline static const dirs_t<Dim::_3D> unique = {
      {  0,  0,  1 },
      {  0,  1,  0 },
      {  1,  0,  0 },
      {  1,  1,  0 },
      { -1,  1,  0 },
      {  0,  1,  1 },
      {  0, -1,  1 },
      {  1,  0,  1 },
      { -1,  0,  1 },
      {  1,  1,  1 },
      { -1,  1,  1 },
      {  1, -1,  1 },
      {  1,  1, -1 }
    };
  };

} // namespace dir

#endif // GLOBAL_ARCH_DIRECTIONS_H
