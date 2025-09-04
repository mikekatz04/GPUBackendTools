# ======================================
# == GPUBACKENDTOOLS project base configuration ==
# ======================================

# TODO:
#  - check cmake_minimum_required range
#  - check if project-specific CMake function are needed

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)

# ---- Main project definition ----
project(gpubackendtools VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define project specific options ----

# GPUBACKENDTOOLS_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(
  GPUBACKENDTOOLS_WITH_GPU "AUTO" CACHE
  STRING "Whether to compile GPU backend"
)
set_property(
  CACHE GPUBACKENDTOOLS_WITH_GPU PROPERTY
  STRINGS "AUTO" "ON" "OFF" "ONLY" "BARE"
)

# GPUBACKENDTOOLS_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(
  GPUBACKENDTOOLS_CUDA_ARCH "native"
  CACHE STRING "CUDA Architecture targetted for GPUBACKENDTOOLS compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES)."
)

set(
  GPUBACKENDTOOLS_MARCH "native"
  CACHE STRING "Value of the -march compiler option if supported by compiler"
)

# ---- Phony target for project specific properties ----
add_library(gpubackendtools INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(gpubackendtools PROPERTIES WITH_CPU ON)

# ---- Test whether the GPUBACKENDTOOLS_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(GPUBACKENDTOOLS_MARCH_CXX_OPT "-march=${GPUBACKENDTOOLS_MARCH}")
check_cxx_compiler_flag("${GPUBACKENDTOOLS_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_GPUBACKENDTOOLS_MARCH)
if(CXX_COMPILER_SUPPORTS_GPUBACKENDTOOLS_MARCH)
  set_property(TARGET gpubackendtools PROPERTY CXX_MARCH
    "${GPUBACKENDTOOLS_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${GPUBACKENDTOOLS_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${GPUBACKENDTOOLS_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(GPUBACKENDTOOLS_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
      "GPUBACKENDTOOLS GPU support was set to AUTO and will be turned ON as CUDA and \
CUDA Toolkit are available with CUDA version \
${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(gpubackendtools PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
      "GPUBACKENDTOOLS GPU support was set to AUTO and will be turned OFF as CUDA and \
CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(gpubackendtools PROPERTIES WITH_GPU OFF)
  endif()
elseif(GPUBACKENDTOOLS_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "GPUBACKENDTOOLS GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU ON)
  set_target_properties(gpubackendtools PROPERTIES WITH_CPU OFF)
elseif(GPUBACKENDTOOLS_WITH_GPU STREQUAL "BARE")
  message(STATUS "GPUBACKENDTOOLS GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU OFF)
  set_target_properties(gpubackendtools PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "GPUBACKENDTOOLS GPU support is set to ${GPUBACKENDTOOLS_WITH_GPU}.")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU ${GPUBACKENDTOOLS_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_GPUBACKENDTOOLS_WITH_GPU gpubackendtools WITH_GPU)
if(_GPUBACKENDTOOLS_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- Include sources ----
add_subdirectory(src)
