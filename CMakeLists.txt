cmake_minimum_required (VERSION 2.8)
project (DPLASMA C Fortran)

#
# Do NOT remove this. Make sure it is set to OFF to
# avoid the spurious output from make, but please don't
# remove it.
#

# should we use our own math functions
option(DPLASMA_COLLECTIVE
  "Use optimized asynchronous operations where collective communication pattern is detected" OFF)
option(DPLASMA_MPI
  "Build DPLASMA for distributed memory with MPI backend" ON)
option(DPLASMA_CACHE_AWARE "Activate the cache awareness support (only if hardware
  locality is enabled and discovered" OFF)
option(DPLASMA_PROFILING
  "Keep a sumarized history of critical events in memory that can be dumped in gdb when deadlock occur" OFF)
option(DPLASMA_PAPI
  "Enable PAPI performance hardware counters" OFF)
option(DPLASMA_STATS
  "Enable the generation of statistics informations during execution
  (microbenchmarking)" OFF)
option(DPLASMA_DEBUG_HISTORY
  "Enable the generation of the profiling information during
  execution" OFF)
option(DPLASMA_GRAPHER
  "Enable the generation of the dot output during execution" OFF)
option(DPLASMA_CALL_TRACE
  "Enable the output of the BLAS kernels call trace during execution" OFF)
option(DPLASMA_DRY_RUN
  "Disable call to actual BLAS kernels; unfold the dependencies only" OFF)
option(BUILD_SHARED_LIBS
  "Build shared libraries" OFF)
option(BUILD_64bits
  "Build 64 bits mode" ON)
option(DPLASMA_CUDA
  "Enable dupport for CUDA" ON)
# The current version number
set (DPLASMA_VERSION_MAJOR 0)
set (DPLASMA_VERSION_MINOR 1)

set( PROJECT_INCLUDE_DIR ./)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/")

#
# check for the CPU we build for
#
MESSAGE(STATUS "Building for target ${CMAKE_SYSTEM_PROCESSOR}")

STRING(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86)
    MESSAGE(STATUS "Found target for X86")
    SET(ARCH_X86 1)
ENDIF (_mach_x86)

STRING(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86_64)
    MESSAGE(STATUS "Found target X86_64")
    SET(ARCH_X86_64 1)
ENDIF (_mach_x86_64)

STRING(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_ppc)
    MESSAGE(STATUS "Found target for PPC")
    SET(ARCH_PPC 1)
ENDIF (_mach_ppc)

ADD_DEFINITIONS(-DHAVE_CONFIG_H)

#
# Fix the building system for 32 or 64 bits.
#
# On MAC OS X there is a easy solution, by setting the 
# CMAKE_OSX_ARCHITECTURES to a subset of the following values:
# ppc;ppc64;i386;x86_64.
# On Linux this is a little bit tricky. We have to check that the
# compiler supports the -m32/-m64 flags as well as the linker.
# Once this issue resolved the CMAKE_C_FLAGS and CMAKE_C_LDFLAGS
# have to be updated accordingly.
#
# TODO: Same trick for the Fortran compiler...
#       no idea how to correctly detect if the required/optional
#          libraries are in the correct format.
#
set(LOCAL_FORTRAN_LINK_FLAGS "")

include(${CMAKE_ROOT}/Modules/CheckCCompilerFlag.cmake)
set(SAVE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
if (BUILD_64bits)
  set( ARCH_BUILD "-m64" )
else (BUILD_64bits)
  set( ARCH_BUILD "-m32" )
endif (BUILD_64bits)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${ARCH_BUILD}")
check_c_compiler_flag(${ARCH_BUILD} C_M32or64)

if (C_M32or64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_BUILD}")
  set(CMAKE_C_LDFLAGS "${CMAKE_C_LDFLAGS} ${ARCH_BUILD}")
  set(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} ${ARCH_BUILD}")
else (C_M32or64)
  set(CMAKE_REQUIRED_FLAGS "${SAVE_CMAKE_REQUIRED_FLAGS}")
endif (C_M32or64)
unset( SAVE_CMAKE_REQUIRED_FLAGS )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
string( TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
if( ${CMAKE_BUILD_TYPE} STREQUAL "debug" )
  set(DPLASMA_DEBUG 1)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif( ${CMAKE_BUILD_TYPE} STREQUAL "debug" )

# Check for available functions
include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
include (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)
include (${CMAKE_ROOT}/Modules/FindFLEX.cmake)
include (${CMAKE_ROOT}/Modules/FindBISON.cmake)
include (${CMAKE_ROOT}/Modules/CheckCCompilerFlag.cmake)
include (${CMAKE_ROOT}/Modules/CMakeDetermineSystem.cmake)
include (${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)

find_package(BISON)
find_package(FLEX)

CHECK_C_COMPILER_FLAG( "-std=c99" HAVE_STD_C99)
IF( HAVE_STD_C99 )
  SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" )
ENDIF( HAVE_STD_C99 )

BISON_TARGET(dplasma_yacc dplasma.y ${CMAKE_CURRENT_BINARY_DIR}/dplasma.tab.c)
FLEX_TARGET(dplasma_flex dplasma.l  ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.c)
ADD_FLEX_BISON_DEPENDENCY(dplasma_flex dplasma_yacc)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${PROJECT_INCLUDE_DIR}/Buf2Cache")

include(cmake_modules/CheckAtomicIntrinsic.cmake)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set( MAC_OS_X 1 CACHE INTERNAL "Compile on MAC OS X")
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")

find_package(Threads)
if(Threads_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CMAKE_THREAD_LIBS_INIT}")
  check_function_exists(pthread_create HAVE_PTHREAD)
  if(HAVE_PTHREAD)
    set(EXTRA_LIBS "${EXTRA_LIBS};${CMAKE_THREAD_LIBS_INIT}")
  endif(HAVE_PTHREAD)
endif(Threads_FOUND)

set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};rt")
check_function_exists(sched_setaffinity HAVE_SCHED_SETAFFINITY)

include(CheckStructHasMember)
check_struct_has_member("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
if( NOT HAVE_TIMESPEC_TV_NSEC )
  add_definitions(-D_GNU_SOURCE)
  #  set(CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS} -D_GNU_SOURCE")
  check_struct_has_member("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
endif( NOT HAVE_TIMESPEC_TV_NSEC )
check_library_exists(rt clock_gettime "" HAVE_CLOCK_GETTIME)
if( HAVE_CLOCK_GETTIME )
  set(EXTRA_LIBS "${EXTRA_LIBS};rt")
endif( HAVE_CLOCK_GETTIME )

STRING(REGEX MATCH ".*ifort$" _match_ifort ${CMAKE_Fortran_COMPILER})
IF (_match_ifort)
    MESSAGE(STATUS "Add -nofor_main to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -nofor_main")
ENDIF (_match_ifort)

STRING(REGEX MATCH ".*ftn$" _match_ftn ${CMAKE_Fortran_COMPILER})
IF (_match_ftn)
    MESSAGE(STATUS "Add -Mnomain to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -Mnomain -Bstatic")
ENDIF (_match_ftn)

find_package(PLASMA REQUIRED)

#
# Find optional packages
#

find_package(HWLOC)
if( HAVE_HWLOC )
  list(APPEND EXTRA_SOURCES dplasma_hwloc.c)
  list(APPEND EXTRA_LIBS ${HWLOC_LIB})
  check_struct_has_member( "struct hwloc_obj" parent hwloc.h HAVE_HWLOC_PARENT_MEMBER )
  check_struct_has_member( "struct hwloc_cache_attr_s" size hwloc.h HAVE_HWLOC_CACHE_ATTR )
  check_c_source_compiles( "#include <hwloc.h>
 int main(void) { hwloc_obj_t o; o->type = HWLOC_OBJ_PU; return 0;}" HAVE_HWLOC_OBJ_PU)
endif( HAVE_HWLOC )
if(DPLASMA_CACHE_AWARE)
  if( !HAVE_HWLOC )
    message( error "Cache awarness requires hwloc" )
  endif( !HAVE_HWLOC )
endif(DPLASMA_CACHE_AWARE)

if (DPLASMA_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    include_directories( ${MPI_INCLUDE_PATH} ) 
  else (MPI_FOUND)
    message("MPI support is required in order to build the distributed version of DPLASMA."
      "Unfortunately, MPI does not seem to be properly installed on this system, at least"
      "not on the default path.")
  endif (MPI_FOUND)
endif (DPLASMA_MPI)

if (DPLASMA_PAPI)
  find_package(PAPI REQUIRED)
  if (PAPI_FOUND)
    list(APPEND EXTRA_LIBS ${PAPI_LIBRARY})
    include_directories( ${PAPI_INCLUDE_DIR} )
  endif (PAPI_FOUND)
endif (DPLASMA_PAPI)

if( DPLASMA_CUDA )
  include (cmake_modules/FindCUDA.cmake)
  set(CUDA_HOST_COMPILATION_CPP OFF)
  find_package(CUDA)
  if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    set( DPLASMA_CUDA_SUPPORT 1)
  else (CUDA_FOUND)
    set( DPLASMA_CUDA_SUPPORT 0)
  endif (CUDA_FOUND)
endif( DPLASMA_CUDA )

check_include_files(stdarg.h HAVE_STDARG_H)
check_function_exists(va_copy HAVE_VA_COPY)
if (NOT HAVE_VA_COPY)
  check_function_exists(__va_copy HAVE_UNDERSCORE_VA_COPY)
endif (NOT HAVE_VA_COPY)
check_function_exists(asprintf HAVE_ASPRINTF)
check_function_exists(vasprintf HAVE_VASPRINTF)
check_include_files(getopt.h HAVE_GETOPT_H)
check_function_exists(getopt_long HAVE_GETOPT_LONG)
check_include_files(errno.h HAVE_ERRNO_H)


include_directories(data_dist data_dist/matrix data_dist/matrix/two_dim_rectangle_cyclic data_dist/matrix/sym_two_dim_rectangle_cyclic)


#
# Setup targets
#
add_library(dplasma STATIC dplasma.c symbol.c assignment.c expr.c params.c dep.c
  scheduling.c profiling.c stats.c remote_dep.c barrier.c precompile.c Buf2Cache/buf2cache.c
  freelist.c debug.c bindthread.c ${EXTRA_SOURCES}
  ${BISON_dplasma_yacc_OUTPUTS} ${FLEX_dplasma_flex_OUTPUTS} data_dist/matrix/matrix.c 
  data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c
  data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c)
set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
#target_link_libraries(dplasma ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})

if (DPLASMA_MPI AND MPI_FOUND)
  add_library(dplasma-mpi STATIC dplasma.c symbol.c assignment.c expr.c params.c dep.c
    scheduling.c profiling.c stats.c remote_dep.c barrier.c precompile.c Buf2Cache/buf2cache.c
    freelist.c debug.c bindthread.c ${EXTRA_SOURCES}
    ${BISON_dplasma_yacc_OUTPUTS} ${FLEX_dplasma_flex_OUTPUTS} data_dist/matrix/matrix.c 
  data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c
  data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c)
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DUSE_MPI")
#  target_link_libraries(dplasma-mpi ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
endif (DPLASMA_MPI AND MPI_FOUND)

#
# Now continue with compiling the tests.
#
#add_subdirectory(data_dist)
add_subdirectory(tests)
add_subdirectory(tools)

# Configuration header
configure_file (
  "${PROJECT_INCLUDE_DIR}/dplasma_config.h.in"
  "${PROJECT_INCLUDE_DIR}/dplasma_config.h")

