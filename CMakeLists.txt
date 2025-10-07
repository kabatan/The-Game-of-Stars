cmake_minimum_required(VERSION 3.15)
project(The-Game-of-Stars)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(move_ver1 move_ver1.cpp)
set_target_properties(move_ver1 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

pybind11_add_module(move_ver2 move_ver2.cpp)
set_target_properties(move_ver2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})