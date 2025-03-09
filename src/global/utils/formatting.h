/**
 * @file utils/formatting.h
 * @brief String formatting utilities
 * @implements
 *   - fmt::format<> -> std::string
 *   - fmt::pad -> std::string
 *   - fmt::toLower -> std::string
 *   - fmt::splitString -> std::vector<std::string>
 *   - fmt::repeat -> std::string
 *   - fmt::formatVector -> std::string
 *   - fmt::strlen_utf8 -> std::size_t
 *   - fmt::alignedTable -> std::string
 * @namespaces:
 *   - fmt::
 */

#ifndef GLOBAL_UTILS_FORMATTING_H
#define GLOBAL_UTILS_FORMATTING_H

#include "arch/traits.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fmt {

  /**
   * @brief Format a string with arguments
   * @note Implements minimal C-style formatting
   */
  template <typename... Args>
  inline auto format(const char* format, Args... args) -> std::string {
    auto size_s = std::snprintf(nullptr, 0, format, args...) + 1;
    if (size_s <= 0) {
      throw std::runtime_error("Error during formatting.");
    }
    auto                    size { static_cast<std::size_t>(size_s) };
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format, args...);
    return std::string(buf.get(), buf.get() + size - 1);
  }

  /**
   * @brief pads a string with a given character
   * @param str String to pad
   * @param n Number of characters in total (incl. padding)
   * @param c Character to pad with
   * @param right Pad on the right
   */
  inline auto pad(const std::string& str, std::size_t n, char c, bool right = false)
    -> std::string {
    if (n <= str.size()) {
      return str;
    }
    if (right) {
      return str + std::string(n - str.size(), c);
    }
    return std::string(n - str.size(), c) + str;
  }

  /**
   * @brief formats a vector of arbitrary type: [a, b, c, ...]
   */
  template <typename T>
  auto formatVector(const std::vector<T>& vec) -> std::string {
    std::ostringstream oss;
    oss << "[";
    if (!vec.empty()) {
      if constexpr (traits::is_pair<T>::value) {
        if constexpr (
          traits::has_method<traits::to_string_t, typename T::first_type>::value) {
          oss << "{" << vec[0].first.to_string() << ", "
              << vec[0].second.to_string() << "}";
          for (size_t i = 1; i < vec.size(); ++i) {
            oss << ", {" << vec[i].first.to_string() << ", "
                << vec[i].second.to_string() << "}";
          }
        } else {
          oss << "{" << vec[0].first << ", " << vec[0].second << "}";
          for (size_t i = 1; i < vec.size(); ++i) {
            oss << ", {" << vec[i].first << ", " << vec[i].second << "}";
          }
        }
      } else {
        oss << vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
          oss << ", " << vec[i];
        }
      }
    }
    oss << "]";
    return oss.str();
  }

  /**
   * @brief Convert a string to lowercase
   */
  inline auto toLower(const std::string& str) -> std::string {
    std::string result { str };
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  }

  /**
   * @brief Split a string into a vector of strings
   * @param str String to split
   * @param delim Delimiter
   * @return Vector of strings
   */
  inline auto splitString(const std::string& str, const std::string& delim)
    -> std::vector<std::string> {
    std::regex regexz(delim);
    return { std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
             std::sregex_token_iterator() };
  }

  /**
   * @brief Repeat a string n number of times
   * @param s String to repeat
   * @param n Number of times to repeat
   */
  inline auto repeat(const std::string& s, std::size_t n) -> std::string {
    std::string result;
    result.reserve(n * s.size());
    for (std::size_t i = 0; i < n; ++i) {
      result += s;
    }
    return result;
  }

  inline auto repeat(char s, std::size_t n) -> std::string {
    return repeat(std::string(1, s), n);
  }

  /**
   * @brief Calculate the length of a UTF-8 string
   * @param str UTF-8 string
   */
  inline auto strlenUTF8(const std::string& str) -> std::size_t {
    std::size_t length = 0;
    for (char c : str) {
      if ((c & 0xC0) != 0x80) {
        ++length;
      }
    }
    return length;
  }

  /**
   * @brief Create a table with aligned columns and custom colors & separators
   * @param columns Vector of column strings
   * @param colors Vector of colors
   * @param anchors Vector of column anchors (position of edge, negative means left-align)
   * @param fillers Vector of separators
   * @param c_bblack Black color
   * @param c_reset Reset color
   */
  inline auto alignedTable(const std::vector<std::string>& columns,
                           const std::vector<std::string>& colors,
                           const std::vector<int>&         anchors,
                           const std::vector<char>&        fillers,
                           const std::string&              c_bblack,
                           const std::string& c_reset) -> std::string {
    std::string result { c_reset };
    std::size_t cntr { 0 };
    for (auto i { 0u }; i < columns.size(); ++i) {
      const auto  anch { static_cast<std::size_t>(anchors[i] < 0 ? -anchors[i]
                                                                 : anchors[i]) };
      const auto  leftalign { anchors[i] <= 0 };
      const auto  cmn { columns[i] };
      const auto  cmn_len { strlenUTF8(cmn) };
      std::string left { c_bblack };
      if (leftalign) {
        if (fillers[i] == ':') {
          left += " :";
          left += repeat(' ', anch - cntr - 2);
        } else {
          left += repeat(fillers[i], anch - cntr);
        }
        cntr += anch - cntr;
      } else {
        left += repeat(fillers[i], anch - cntr - cmn_len);
        cntr += anch - cntr - cmn_len;
      }
      result += left + colors[i] + cmn + c_reset;
      cntr   += cmn_len;
    }
    return result + c_reset + "\n";
  }

} // namespace fmt

#endif // GLOBAL_UTILS_FORMATTING_H
