#include "global.h"
#include "pgen.h"

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>

ProblemGenerator ntt_pgen;

void initLogger(plog::ColorConsoleAppender<plog::TxtFormatter> *console_appender);

auto main(int argc, char *argv[]) -> int {
  plog::ColorConsoleAppender<plog::TxtFormatter> console_appender;
  initLogger(&console_appender);

  try {
    ntt_pgen.start(argc, argv);
  } catch (std::exception &err) {
    std::cerr << err.what() << std::endl;
    return -1;
  }

  return 0;
}

void initLogger(plog::ColorConsoleAppender<plog::TxtFormatter> *console_appender) {
  plog::Severity max_severity;
#ifdef VERBOSE
  max_severity = plog::verbose;
#elif DEBUG
  max_severity = plog::debug;
#else
  max_severity = plog::warning;
#endif
  plog::init(max_severity, console_appender);
}
