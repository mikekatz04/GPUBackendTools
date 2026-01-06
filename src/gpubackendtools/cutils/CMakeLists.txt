# =============================
# ==== BACKEND COMPILATION ====
# =============================

include(cmake_functions.cmake)
set(HERE_CUDA_ARCH ${GBT_CUDA_ARCH})

# * * * * * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to define backend option * *
# * * * * * * * * * * * * * * * * * * * * * * * * *

# In the project root CMakeLists.txt, we defined a "gbt" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(GBT_WITH_CPU gpubackendtools WITH_CPU)
get_target_property(GBT_WITH_GPU gpubackendtools WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${gpubackendtools_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()


# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *


# ----------------
# --- interp ---
# ----------------

# I. Process interp.pyx into a C++ file
add_custom_command(
  OUTPUT "interp.cxx"
  COMMENT "Cythonize interp.pyx into interp.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/interp.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/interp.cxx" -3 -+ --module-name
    "interp" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "interp.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(GBT_WITH_CPU)
  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND
      ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Interpolate.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "Interpolate.cu"
    VERBATIM)

  python_add_library(gbt_cpu_interp MODULE WITH_SOABI interp.cxx
                     Interpolate.cxx)
  apply_cpu_backend_common_options(interp gbt gpubackendtools false)
  target_sources(
    gbt_cpu_interp PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp gbt_global.h
                  Interpolate.hh)

  # interp needs to link against LAPACKE. USe the following method to detect
  # it and obtain ${GBT_LAPACKE_LIBS}. In case of failure, it will stop the
  # processing execution.
  enable_language(Fortran)
  gpubackendtools_get_lapacke()
  # Now, link against ${GBT_LAPACKE_LIBS} to link against liblapacke.so and
  # include lapacke.h
  target_link_libraries(gbt_cpu_interp LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})

endif()

# III. Declare the GPU backend
if(GBT_WITH_GPU)
  python_add_library(gbt_gpu_interp MODULE WITH_SOABI interp.cxx
                     Interpolate.cu)
  apply_gpu_backend_common_options(interp gbt gpubackendtools false)
  target_sources(
    gbt_gpu_interp PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp gbt_global.h
                            Interpolate.hh)
endif()
