/**
 * @file utils/plog.h
 * @brief Plog formatters for the Entity
 * @implements
 *   - plog::NttConsoleFormatter
 *   - plog::NttInfoFormatter
 *   - logger::initPlog -> void
 * @namespaces:
 *   - plog::
 *   - logger::
 */

#ifndef GLOBAL_UTILS_PLOG_H
#define GLOBAL_UTILS_PLOG_H

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

#include <filesystem>
#include <iomanip>
#include <string>

namespace plog {

  class NttConsoleFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      ss << std::setw(6) << std::left << severityToString(record.getSeverity())
         << PLOG_NSTR(": ");
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };

  class NttInfoFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };
} // namespace plog

namespace logger {

  template <int log_tag, int info_tag, int err_tag>
  inline void initPlog(const std::string& fname) {
    // setup logging
    const auto logfile_name  = fname + ".log";
    const auto infofile_name = fname + ".info";
    const auto errfile_name  = fname + ".err";

    namespace fs = std::filesystem;
    fs::path logfile_path { logfile_name };
    fs::path infofile_path { infofile_name };
    fs::path errfile_path { errfile_name };
    fs::remove(logfile_path);
    fs::remove(infofile_path);
    fs::remove(errfile_path);

    static plog::RollingFileAppender<plog::TxtFormatter> logfileAppender(
      logfile_name.c_str());
    static plog::RollingFileAppender<plog::NttInfoFormatter> infofileAppender(
      infofile_name.c_str());
    static plog::RollingFileAppender<plog::NttInfoFormatter> errfileAppender(
      errfile_name.c_str());
    plog::init<log_tag>(plog::verbose, &logfileAppender);
    plog::init<info_tag>(plog::verbose, &infofileAppender);
    plog::init<err_tag>(plog::verbose, &errfileAppender);

#if defined(DEBUG)
    const auto severity = plog::verbose;
#else
    const auto severity = plog::info;
#endif

    static plog::ColorConsoleAppender<plog::NttConsoleFormatter> consoleAppender;
    plog::init(severity, &consoleAppender);
  }

} // namespace logger

#endif // GLOBAL_UTILS_PLOG_H
