cmake_minimum_required(VERSION 3.4...3.18)
project(feature_extractor_cpp)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(feature_extractor_cpp feature_extractor.cpp)