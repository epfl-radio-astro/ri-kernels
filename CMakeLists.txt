cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Parse version from pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_TOML)
string(REGEX MATCH "\nversion = \"([^\"]+)\"" _ "${PYPROJECT_TOML}")
set(RI_KERNELS_VERSION "${CMAKE_MATCH_1}")

project(ri_kernels LANGUAGES CXX VERSION "${RI_KERNELS_VERSION}")
set(RI_KERNELS_SO_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})


# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
	)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 20)
set(CUDA_STANDARD 20)
set(CUDA_STANDARD_REQUIRED ON)

include(CMakeDependentOption)
include(FetchContent)
include(CheckLinkerFlag)

# --exclude-libs is a GNU ld / lld feature; Apple's linker rejects it.
check_linker_flag(CXX "-Wl,--exclude-libs,ALL" RI_KERNELS_HAVE_EXCLUDE_LIBS)

# Export only the FFI entry points annotated with RI_KERNELS_API (see
# src/visibility.h). Everything else, including symbols pulled in from static
# archives such as Highway or a statically linked CUDA runtime, stays out of
# the dynamic symbol table. Without this the CUDA runtime symbols would be
# exported and could be interposed by the runtime XLA already loaded into the
# process, since the dynamic loader searches the global scope before a dlopen'd
# object's own definitions.
function(ri_kernels_restrict_exports target)
	set_target_properties(${target} PROPERTIES
		C_VISIBILITY_PRESET hidden
		CXX_VISIBILITY_PRESET hidden
		CUDA_VISIBILITY_PRESET hidden
		HIP_VISIBILITY_PRESET hidden
		VISIBILITY_INLINES_HIDDEN ON
	)
	if(RI_KERNELS_HAVE_EXCLUDE_LIBS)
		target_link_options(${target} PRIVATE "-Wl,--exclude-libs,ALL")
	endif()
endfunction()

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

# Options
option(RI_KERNELS_CPU "Build CPU extension" ON)
option(RI_KERNELS_CUDA "Build CUDA extension" OFF)
option(RI_KERNELS_ROCM "Build ROCm extension" OFF)

option(RI_KERNELS_BUNDLED_LIBS "Use all bundled libraries" ON)
cmake_dependent_option(RI_KERNELS_BUNDLED_HIGHWAY "Use bundled highway lib" ON "RI_KERNELS_BUNDLED_LIBS" OFF)
option(RI_KERNELS_MULTI_ARCH "Build kernels for multiple CPU architectues with dynamic dispatch. When disabled, arch flags should be set through CMAKE_CXX_FLAGS." ON)

# Directory inside the wheel the GPU libraries are installed into. The CPU
# library always ships in the ri_kernels package, but the GPU libraries are
# published as separate add-on distributions (ri_kernels_cuda12,
# ri_kernels_cuda13) that own a top-level package directory of their own.
set(RI_KERNELS_GPU_INSTALL_DIR "ri_kernels" CACHE STRING
	"Directory inside the wheel that the GPU libraries are installed into")

set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;90-real;90-virtual" CACHE STRING "CUDA Architectures")
set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1030;gfx1100" CACHE STRING "HIP Architectures")

set(Python_EXECUTABLE "python3" CACHE STRING "The python interpreter")

# Detect jaxlib include directory (override with -DJAXLIB_HOME=...)
if(NOT DEFINED JAXLIB_HOME)
	execute_process(
		COMMAND ${Python_EXECUTABLE} -c
			"import jaxlib, os; print(os.path.dirname(jaxlib.__file__))"
		OUTPUT_VARIABLE JAXLIB_HOME
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE _jaxlib_result
	)
	if(NOT _jaxlib_result EQUAL 0)
		message(FATAL_ERROR "Failed to locate jaxlib via python3. "
			"Install jaxlib or pass -DJAXLIB_HOME=<path>.")
	endif()
endif()
message(STATUS "Using JAXLIB_HOME: ${JAXLIB_HOME}")


set(RI_KERNELS_CPU_SOURCES
  ./src/rfi_kernel.cpp
  ./src/rfi_jvp_kernel.cpp
  ./src/rfi_transpose_kernel.cpp
)

set(RI_KERNELS_GPU_SOURCES
  ./src/rfi_kernel_gpu.cu
  ./src/rfi_jvp_kernel_gpu.cu
  ./src/rfi_transpose_kernel_gpu.cu
  ./src/util_gpu.cu
)

if(RI_KERNELS_CPU)
  add_library(ri_kernels SHARED ${RI_KERNELS_CPU_SOURCES})
  target_include_directories(ri_kernels PRIVATE ${JAXLIB_HOME}/include ./src)

  target_compile_options(ri_kernels PRIVATE
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-return-type>
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-attributes>
  )

  if(RI_KERNELS_BUNDLED_HIGHWAY)
	# add google highway
	set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
	set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
	set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
	set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
	set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "")
	set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
	FetchContent_Declare(
	  hwy
	  URL https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz
	  URL_MD5 9d335797777e17f827c7980b8313a34b
	)
	FetchContent_MakeAvailable(hwy)
	if(NOT TARGET hwy::hwy)
	  add_library(hwy::hwy ALIAS hwy)
	endif()
  else()
	find_package(hwy CONFIG REQUIRED)
  endif()
  target_link_libraries(ri_kernels PRIVATE hwy::hwy)

  if(RI_KERNELS_MULTI_ARCH)
	target_compile_definitions(ri_kernels PUBLIC -DRI_KERNELS_MULTI_ARCH)
  endif()
endif()

if(RI_KERNELS_CUDA)
  enable_language(CUDA)
  # find toolkit after language is enabled to ensure version matching
  find_package(CUDAToolkit REQUIRED)

  set_source_files_properties(${RI_KERNELS_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_library(ri_kernels_cuda SHARED ${RI_KERNELS_GPU_SOURCES})
  # set_target_properties(ri_kernels_cuda PROPERTIES OUTPUT_NAME "ri_kernels_cuda_${CUDAToolkit_VERSION_MAJOR}")
  target_include_directories(ri_kernels_cuda PRIVATE ${JAXLIB_HOME}/include ./src)
  target_link_libraries(ri_kernels_cuda PRIVATE CUDA::cudart_static)
  target_compile_options(ri_kernels_cuda PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=940>
	$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2473>
  )
endif()

if(RI_KERNELS_ROCM)
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)

  set_source_files_properties(${RI_KERNELS_GPU_SOURCES} PROPERTIES LANGUAGE HIP)

  add_library(ri_kernels_hip SHARED ${RI_KERNELS_GPU_SOURCES})
  target_include_directories(ri_kernels_hip PRIVATE ${JAXLIB_HOME}/include ./src)
  target_link_libraries(ri_kernels_hip PRIVATE hip::host)
endif()

# Normalize shared-library suffix across platforms so Python loader finds
# libri_kernels*.so on macOS as well as Linux, and restrict the exported
# symbols to the annotated FFI entry points.
if(TARGET ri_kernels)
  set_target_properties(ri_kernels PROPERTIES SUFFIX ".so")
  ri_kernels_restrict_exports(ri_kernels)
endif()
if(TARGET ri_kernels_cuda)
  set_target_properties(ri_kernels_cuda PROPERTIES SUFFIX ".so")
  ri_kernels_restrict_exports(ri_kernels_cuda)
endif()
if(TARGET ri_kernels_hip)
  set_target_properties(ri_kernels_hip PROPERTIES SUFFIX ".so")
  ri_kernels_restrict_exports(ri_kernels_hip)
endif()

# Install rules so scikit-build-core stages libraries into the wheel at the
# location where ri_kernels/rfi_vis_op.py loads them from.
if(TARGET ri_kernels)
  install(TARGETS ri_kernels
	LIBRARY DESTINATION ri_kernels
	RUNTIME DESTINATION ri_kernels)
endif()
if(TARGET ri_kernels_cuda)
  install(TARGETS ri_kernels_cuda
	LIBRARY DESTINATION ${RI_KERNELS_GPU_INSTALL_DIR}
	RUNTIME DESTINATION ${RI_KERNELS_GPU_INSTALL_DIR})
endif()
if(TARGET ri_kernels_hip)
  install(TARGETS ri_kernels_hip
	LIBRARY DESTINATION ${RI_KERNELS_GPU_INSTALL_DIR}
	RUNTIME DESTINATION ${RI_KERNELS_GPU_INSTALL_DIR})
endif()
