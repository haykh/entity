/**
 * @file utils/colors.h
 * @brief simple header for getting 8-bit color codes
 * @implements
 *   - color:: instances
 *   - color::strip -> std::string
 *   - color::get_colors -> std::map<std::string, std::string>
 * @namespaces:
 *   - color::
 */

#ifndef GLOBAL_UTILS_COLORS_H
#define GLOBAL_UTILS_COLORS_H

#include <cstring>
#include <map>
#include <string>

namespace color {
  static constexpr const char* RESET { "\033[0m" };
  static constexpr const char* BLACK { "\033[30m" };         /* Black */
  static constexpr const char* RED { "\033[31m" };           /* Red */
  static constexpr const char* GREEN { "\033[32m" };         /* Green */
  static constexpr const char* YELLOW { "\033[33m" };        /* Yellow */
  static constexpr const char* BLUE { "\033[34m" };          /* Blue */
  static constexpr const char* MAGENTA { "\033[35m" };       /* Magenta */
  static constexpr const char* CYAN { "\033[36m" };          /* Cyan */
  static constexpr const char* WHITE { "\033[37m" };         /* White */
  static constexpr const char* BRIGHT_BLACK { "\033[90m" };  /* Bright Black */
  static constexpr const char* BRIGHT_RED { "\033[91m" };    /* Bright Red */
  static constexpr const char* BRIGHT_GREEN { "\033[92m" };  /* Bright Green */
  static constexpr const char* BRIGHT_YELLOW { "\033[93m" }; /* Bright Yellow */
  static constexpr const char* BRIGHT_BLUE { "\033[94m" };   /* Bright Blue */
  static constexpr const char* BRIGHT_MAGENTA { "\033[95m" }; /* Bright Magenta */
  static constexpr const char* BRIGHT_CYAN { "\033[96m" };    /* Bright Cyan */
  static constexpr const char* BRIGHT_WHITE { "\033[97m" };   /* Bright White */
  static constexpr const char* all[] = {
    RESET,       BLACK,        RED,           GREEN,       YELLOW,
    BLUE,        MAGENTA,      CYAN,          WHITE,       BRIGHT_BLACK,
    BRIGHT_RED,  BRIGHT_GREEN, BRIGHT_YELLOW, BRIGHT_BLUE, BRIGHT_MAGENTA,
    BRIGHT_CYAN, BRIGHT_WHITE
  };

  inline auto strip(const std::string& msg) -> std::string {
    auto msg_nocol = msg;
    for (const auto c : all) {
      std::size_t pos = 0;
      while ((pos = msg_nocol.find(c, pos)) != std::string::npos) {
        msg_nocol.replace(pos, strlen(c), "");
        pos += strlen("");
      }
    }
    return msg_nocol;
  }

  inline auto get_color(const std::string& s, bool eight_bit = true)
    -> std::string {
    if (not eight_bit) {
      return "";
    } else {
      if (s == "reset") {
        return RESET;
      } else if (s == "black") {
        return BLACK;
      } else if (s == "red") {
        return RED;
      } else if (s == "green") {
        return GREEN;
      } else if (s == "yellow") {
        return YELLOW;
      } else if (s == "blue") {
        return BLUE;
      } else if (s == "magenta") {
        return MAGENTA;
      } else if (s == "cyan") {
        return CYAN;
      } else if (s == "white") {
        return WHITE;
      } else if (s == "bblack") {
        return BRIGHT_BLACK;
      } else if (s == "bred") {
        return BRIGHT_RED;
      } else if (s == "bgreen") {
        return BRIGHT_GREEN;
      } else if (s == "byellow") {
        return BRIGHT_YELLOW;
      } else if (s == "bblue") {
        return BRIGHT_BLUE;
      } else if (s == "bmagenta") {
        return BRIGHT_MAGENTA;
      } else if (s == "bcyan") {
        return BRIGHT_CYAN;
      } else if (s == "bwhite") {
        return BRIGHT_WHITE;
      }
      return "";
    }
  }

  inline auto get_colors(bool eight_bit) -> std::map<std::string, std::string> {
    std::string c_reset, c_black, c_red, c_green, c_yellow, c_blue, c_magenta,
      c_cyan, c_white, c_bblack, c_bred, c_bgreen, c_byellow, c_bblue,
      c_bmagenta, c_bcyan, c_bwhite;
    if (eight_bit) {
      c_reset    = RESET;
      c_black    = BLACK;
      c_red      = RED;
      c_green    = GREEN;
      c_yellow   = YELLOW;
      c_blue     = BLUE;
      c_magenta  = MAGENTA;
      c_cyan     = CYAN;
      c_white    = WHITE;
      c_bblack   = BRIGHT_BLACK;
      c_bred     = BRIGHT_RED;
      c_bgreen   = BRIGHT_GREEN;
      c_byellow  = BRIGHT_YELLOW;
      c_bblue    = BRIGHT_BLUE;
      c_bmagenta = BRIGHT_MAGENTA;
      c_bcyan    = BRIGHT_CYAN;
      c_bwhite   = BRIGHT_WHITE;
    } else {
      c_reset = c_black = c_red = c_green = c_yellow = c_blue = c_magenta =
        c_cyan = c_white = c_bblack = c_bred = c_bgreen = c_byellow = c_bblue =
          c_bmagenta = c_bcyan = c_bwhite = "";
    }
    return {
      {   "reset",    c_reset},
      {   "black",    c_black},
      {     "red",      c_red},
      {   "green",    c_green},
      {  "yellow",   c_yellow},
      {    "blue",     c_blue},
      { "magenta",  c_magenta},
      {    "cyan",     c_cyan},
      {   "white",    c_white},
      {  "bblack",   c_bblack},
      {    "bred",     c_bred},
      {  "bgreen",   c_bgreen},
      { "byellow",  c_byellow},
      {   "bblue",    c_bblue},
      {"bmagenta", c_bmagenta},
      {   "bcyan",    c_bcyan},
      {  "bwhite",   c_bwhite}
    };
  }
} // namespace color

#endif // GLOBAL_UTILS_COLORS_H
