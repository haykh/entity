#ifndef GLOBAL_UTILS_REPORTER_H
#define GLOBAL_UTILS_REPORTER_H

#include "utils/colors.h"
#include "utils/formatting.h"

#include <string>
#include <utility>
#include <vector>

namespace reporter {
  void AddHeader(std::string&,
                 const std::vector<std::string>&,
                 const std::vector<const char*>&);

  void AddCategory(std::string&, unsigned short, const char*);

  void AddSubcategory(std::string&, unsigned short, const char*);

  void AddLabel(std::string&, unsigned short, const char*);

  template <typename... Args>
  void AddParam(std::string&   report,
                unsigned short indent,
                const char*    name,
                const char*    format,
                Args... args) {
    report += fmt::format("%s%s-%s %s: %s%s%s\n",
                          std::string(indent, ' ').c_str(),
                          color::BRIGHT_BLACK,
                          color::RESET,
                          name,
                          color::BRIGHT_YELLOW,
                          fmt::format(format, args...).c_str(),
                          color::RESET);
  }

  template <typename... Args>
  void AddUnlabeledParam(std::string&   report,
                         unsigned short indent,
                         const char*    name,
                         const char*    format,
                         Args... args) {
    report += fmt::format("%s%s: %s%s%s\n",
                          std::string(indent, ' ').c_str(),
                          name,
                          color::BRIGHT_YELLOW,
                          fmt::format(format, args...).c_str(),
                          color::RESET);
  }

  auto Bytes2HumanReadable(std::size_t) -> std::pair<double, std::string>;

  auto Backend() -> std::string;

} // namespace reporter

#endif // GLOBAL_UTILS_REPORTER_H
