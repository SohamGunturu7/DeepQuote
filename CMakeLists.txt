cmake_minimum_required(VERSION 3.16)
project(DeepQuote VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

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
)

add_library(deepquote_core STATIC ${CORE_SOURCES})

set_target_properties(deepquote_core PROPERTIES
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

# Test executables
add_executable(test_order_book test/test_order_book.cpp)
target_link_libraries(test_order_book deepquote_core)

add_executable(test_matching_engine test/test_matching_engine.cpp)
target_link_libraries(test_matching_engine deepquote_core)

add_executable(test_trader test/test_trader.cpp)
target_link_libraries(test_trader deepquote_core)

add_executable(test_rl_trader test/test_rl_trader.cpp)
target_link_libraries(test_rl_trader deepquote_core)

add_executable(test_market_simulator test/test_market_simulator.cpp)
target_link_libraries(test_market_simulator deepquote_core)

add_executable(test_market_simulator_traders test/test_market_simulator_traders.cpp)
target_link_libraries(test_market_simulator_traders deepquote_core)

add_executable(test_market_simulator_rl_traders test/test_market_simulator_rl_traders.cpp)
target_link_libraries(test_market_simulator_rl_traders deepquote_core)

# Installation
install(TARGETS deepquote_simulator
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS deepquote_simulator
    LIBRARY DESTINATION ${Python_SITEARCH}
)

message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}") 