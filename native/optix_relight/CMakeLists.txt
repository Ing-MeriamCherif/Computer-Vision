cmake_minimum_required(VERSION 3.24)
project(p123_optix LANGUAGES CXX CUDA)

if(NOT DEFINED OPTIX_ROOT)
  message(FATAL_ERROR "Set OPTIX_ROOT to the NVIDIA OptiX SDK before configuring the RTX backend")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_p123_optix MODULE src/optix_module.cpp)
target_compile_features(_p123_optix PRIVATE cxx_std_17)
target_include_directories(_p123_optix PRIVATE "${OPTIX_ROOT}/include")
target_link_libraries(_p123_optix PRIVATE CUDA::cudart)
set_target_properties(_p123_optix PROPERTIES PREFIX "")
