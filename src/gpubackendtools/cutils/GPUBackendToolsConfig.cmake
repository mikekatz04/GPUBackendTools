# GPUBackendToolsConfig.cmake
#
# Forward-compatibility export so downstream sprint packages
# (LISAanalysistools, GBGPU, BBHx, FastEMRIWaveforms) can pick up GBT's
# public C++/CUDA headers via find_package(...) instead of the Python
# shell-out get_include() path. Both mechanisms point at the same
# directory and are equivalent.
#
# Usage (downstream CMakeLists.txt):
#
#   execute_process(
#     COMMAND ${Python_EXECUTABLE} -c
#     "import gpubackendtools; print(gpubackendtools.get_cmake_module_path())"
#     OUTPUT_VARIABLE GBT_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
#   find_package(GPUBackendTools CONFIG REQUIRED PATHS ${GBT_CMAKE_DIR})
#   target_link_libraries(my_target PRIVATE GPUBackendTools::headers)
#
# This config file does NOT export any compiled targets — see the per-
# wheel-self-contained build rule in the sprint plan. Downstreams
# recompile their own translation units against these headers.

get_filename_component(_GPUBackendTools_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(GPUBackendTools_INCLUDE_DIR "${_GPUBackendTools_INCLUDE_DIR}")
set(GPUBackendTools_INCLUDE_DIRS "${_GPUBackendTools_INCLUDE_DIR}")

if(EXISTS "${_GPUBackendTools_INCLUDE_DIR}/cmake_functions.cmake")
  include("${_GPUBackendTools_INCLUDE_DIR}/cmake_functions.cmake" OPTIONAL)
endif()

if(NOT TARGET GPUBackendTools::headers)
  add_library(GPUBackendTools::headers INTERFACE IMPORTED)
  set_target_properties(GPUBackendTools::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_GPUBackendTools_INCLUDE_DIR}")
endif()

set(GPUBackendTools_FOUND TRUE)

unset(_GPUBackendTools_INCLUDE_DIR)
