cmake_minimum_required(VERSION 3.15)
project(The-Game-of-Stars)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
    message(STATUS "Enabled additional Release flags: -march=native -ffast-math")
endif()

find_package(OpenMP REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(move_ver1 move_ver1.cpp)
set_target_properties(move_ver1 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
if (OpenMP_CXX_FOUND)
    target_link_libraries(move_ver1 PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for move_ver1")
endif()

pybind11_add_module(move_ver2 move_ver2.cpp)
set_target_properties(move_ver2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
if (OpenMP_CXX_FOUND)
    target_link_libraries(move_ver2 PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for move_ver2")
endif()

pybind11_add_module(game_simulator_cpp game_simulator.cpp)
set_target_properties(game_simulator_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})