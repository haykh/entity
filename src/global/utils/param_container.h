/**
 * @file utils/param_container.h
 * @brief Smart container for safely storing and retrieving parameters with stringize support
 * @implements
 *   - prm::Parameters
 * @namespaces:
 *   - prm::
 */

#ifndef GLOBAL_UTILS_PARAM_CONTAINER_H
#define GLOBAL_UTILS_PARAM_CONTAINER_H

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif

#include <any>
#include <ios>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace prm {

  class Parameters {
    using key_t = const std::string&;

  protected:
    std::map<std::string, std::any> vars;
    std::vector<std::string>        promises;

  public:
    auto allVars() const -> const std::map<std::string, std::any>& {
      return vars;
    }

    auto contains(key_t key) const -> bool {
      return vars.find(key) != vars.end();
    }

    auto promiseToDefine(key_t key) -> void {
      if (std::find(promises.begin(), promises.end(), key) == promises.end()) {
        promises.push_back(key);
      }
    }

    auto isPromised(key_t key) const -> bool {
      return std::find(promises.begin(), promises.end(), key) != promises.end();
    }

    auto promisesFulfilled() const -> bool {
      return std::all_of(promises.begin(), promises.end(), [this](key_t key) {
        return contains(key);
      });
    }

    void set(key_t key, const std::any& value) {
      vars[key] = value;
    }

    template <typename T>
    [[nodiscard]]
    auto get(key_t key, const std::optional<T>& def = std::nullopt) const -> T {
      try {
        return std::any_cast<T>(vars.at(key));
      } catch (const std::out_of_range&) {
        if (def.has_value()) {
          raise::Warning(
            fmt::format("Key %s not found, falling back to default", key.c_str()),
            HERE);
          return def.value();
        }
        raise::Error(fmt::format("Key %s not found", key.c_str()), HERE);
        throw;
      } catch (const std::bad_any_cast&) {
        raise::Error(fmt::format("Bad any_cast for %s", key.c_str()), HERE);
        throw;
      } catch (const std::exception& e) {
        raise::Error(
          fmt::format("Unknown error for %s : %s", key.c_str(), e.what()),
          HERE);
        throw;
      }
    }

    template <typename T>
    [[nodiscard]]
    auto stringize(key_t key) const -> std::string {
      std::ostringstream result;
      auto remove_last_n = [](std::ostringstream& ss, std::size_t n) {
        auto temp = ss.str();
        temp      = temp.substr(0, temp.size() - n);
        ss.str("");
        ss << temp;
      };
      auto stringize_key = [](std::ostringstream& ss, std::any v) {
        if constexpr (std::is_class_v<T> &&
                      not std::is_same_v<std::string, std::decay_t<T>>) {
          static_assert(std::is_member_function_pointer_v<decltype(&T::to_string)>);
          ss << std::any_cast<T>(v).to_string();
        } else {
          ss << std::any_cast<T>(v);
        }
      };
      result << std::boolalpha;
      try {
        using vecT     = std::vector<T>;
        using vecvecT  = std::vector<vecT>;
        using pairT    = std::pair<T, T>;
        using vecpairT = std::vector<std::pair<T, T>>;
        if (vars.at(key).type() == typeid(vecvecT)) {
          const auto vecvec = get<vecvecT>(key);
          result << "[";
          for (const auto& v : vecvec) {
            result << "{";
            for (const auto& i : v) {
              stringize_key(result, i);
              result << ", ";
            }
            remove_last_n(result, 2);
            result << "}, ";
          }
          remove_last_n(result, 2);
          result << "]";
        } else if (vars.at(key).type() == typeid(vecT)) {
          const auto vec = get<vecT>(key);
          result << "[";
          for (const auto& i : vec) {
            stringize_key(result, i);
            result << ", ";
          }
          remove_last_n(result, 2);
          result << "]";
        } else if (vars.at(key).type() == typeid(pairT)) {
          const auto pair = get<pairT>(key);
          result << "[";
          stringize_key(result, pair.first);
          result << ", ";
          stringize_key(result, pair.second);
          result << "]";
        } else if (vars.at(key).type() == typeid(vecpairT)) {
          const auto vecpair = get<vecpairT>(key);
          result << "[";
          for (const auto& p : vecpair) {
            result << "{";
            stringize_key(result, p.first);
            result << ", ";
            stringize_key(result, p.second);
            result << "}, ";
          }
          remove_last_n(result, 2);
          result << "]";
        } else if (vars.at(key).type() == typeid(T)) {
          const auto v = get<T>(key);
          stringize_key(result, v);
        } else {
          raise::Error("Unsupported type for stringize", HERE);
        }
      } catch (const std::out_of_range&) {
        raise::Error(fmt::format("Key %s not found", key.c_str()), HERE);
      } catch (const std::bad_any_cast&) {
        raise::Error(fmt::format("Bad any_cast for %s", key.c_str()), HERE);
      } catch (const std::exception& e) {
        raise::Error(
          fmt::format("Unknown error for %s : %s", key.c_str(), e.what()),
          HERE);
      }
      return result.str();
    }

#if defined(OUTPUT_ENABLED)
    void write(adios2::IO& io) const;
#endif
  };

} // namespace prm

#endif // GLOBAL_UTILS_PARAM_CONTAINER_H
