#include "global.h"

#include <iostream>
#include <stdexcept>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    // ...
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
}
