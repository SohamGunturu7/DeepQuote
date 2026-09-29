cmake_minimum_required(VERSION 3.16)
project(DeepQuote VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
# Let CMake find a pip-installed pybind11 (pip install pybind11)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_PIP_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_PIP_CMAKE_DIR})

include_directories(include)

# Source files for the core library
set(CORE_SOURCES
    src/core/order.cpp
    src/core/types.cpp
    src/market/trader.cpp
    src/market/rl_trader.cpp
    src/market/market_simulator.cpp
    src/market/order_book.cpp
    src/market/matching_engine.cpp
    src/market/market_maker.cpp
    src/market/market_events.cpp
    src/strategies/trading_strategy.cpp
    src/strategies/market_making_strategy.cpp
    src/strategies/mean_reversion_strategy.cpp
    src/strategies/pairs_trading_strategy.cpp
)

add_library(deepquote_core STATIC ${CORE_SOURCES})

set_target_properties(deepquote_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Python bindings
pybind11_add_module(deepquote_simulator src/bindings/python_bindings.cpp)

target_link_libraries(deepquote_simulator PRIVATE deepquote_core)

set_target_properties(deepquote_simulator PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "deepquote_simulator"
)

# Installation
install(TARGETS deepquote_simulator
    LIBRARY DESTINATION ${Python_SITEARCH}
)

message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}") 