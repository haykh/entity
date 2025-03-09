/**
 * @file output/fields.h
 * @brief Class defining the metadata necessary to prepare the field for output
 * @implements
 *   - out::OutputField
 * @cpp:
 *   - fields.cpp
 * @namespaces:
 *   - out::
 */

#ifndef OUTPUT_FIELDS_H
#define OUTPUT_FIELDS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include <string>
#include <vector>

using namespace ntt;

namespace out {

  class OutputField {

    const std::string m_name;
    FldsID            m_id { FldsID::INVALID };

  public:
    PrepareOutputFlags prepare_flag { PrepareOutput::None };
    PrepareOutputFlags interp_flag { PrepareOutput::None };

    std::vector<std::vector<unsigned short>> comp {};
    std::vector<unsigned short>              species {};

    OutputField(const SimEngine& S, const std::string&);

    ~OutputField() = default;

    [[nodiscard]]
    auto is_moment() const -> bool {
      return (id() == FldsID::T || id() == FldsID::Rho || id() == FldsID::Nppc ||
              id() == FldsID::N || id() == FldsID::Charge || id() == FldsID::V);
    }

    [[nodiscard]]
    auto is_field() const -> bool {
      return (id() == FldsID::E || id() == FldsID::B || id() == FldsID::D ||
              id() == FldsID::H);
    }

    [[nodiscard]]
    auto is_divergence() const -> bool {
      return (id() == FldsID::divE || id() == FldsID::divD);
    }

    [[nodiscard]]
    auto is_current() const -> bool {
      return (id() == FldsID::J);
    }

    [[nodiscard]]
    auto is_efield() const -> bool {
      return (id() == FldsID::E || id() == FldsID::D);
    }

    [[nodiscard]]
    auto is_gr_aux_field() const -> bool {
      return (id() == FldsID::E || id() == FldsID::H);
    }

    [[nodiscard]]
    auto is_vpotential() const -> bool {
      return (id() == FldsID::A);
    }

    [[nodiscard]]
    auto is_custom() const -> bool {
      return (id() == FldsID::Custom);
    }

    [[nodiscard]]
    inline auto name() const -> std::string {
      // generate the name
      std::string tmp;
      if (id() == FldsID::Custom) {
        tmp = m_name;
      } else {
        tmp = std::string(id().to_string());
        if (id() == FldsID::T) {
          tmp += m_name.substr(1, 2);
        } else if (id() == FldsID::A) {
          tmp += "3";
        } else if (is_field() || id() == FldsID::V) {
          tmp += "i";
        }
        if (species.size() > 0) {
          tmp += "_";
          for (auto& s : species) {
            tmp += std::to_string(s);
            tmp += "_";
          }
          tmp.pop_back();
        }
        // capitalize the first letter
        tmp[0] = std::toupper(tmp[0]);
      }
      return "f" + tmp;
    }

    [[nodiscard]]
    inline auto name(const std::size_t& ci) const -> std::string {
      raise::ErrorIf(
        comp.size() == 0,
        "OutputField::name(ci) called but no components were available",
        HERE);
      raise::ErrorIf(
        ci >= comp.size(),
        "OutputField::name(ci) called with an invalid component index",
        HERE);
      raise::ErrorIf(
        comp[ci].size() == 0,
        "OutputField::name(ci) called but no components were available",
        HERE);
      // generate the name
      auto tmp = std::string(id().to_string());
      for (auto& c : comp[ci]) {
        tmp += std::to_string(c);
      }
      if (species.size() > 0) {
        tmp += "_";
        for (auto& s : species) {
          tmp += std::to_string(s);
          tmp += "_";
        }
        tmp.pop_back();
      }
      // capitalize the first letter
      tmp[0] = std::toupper(tmp[0]);
      return "f" + tmp;
    }

    [[nodiscard]]
    auto id() const -> FldsID {
      return m_id;
    }
  };

} // namespace out

#endif // OUTPUT_FIELDS_H
