cmake_minimum_required (VERSION 2.8)
project (DAGUE C Fortran)
# The current version number
set (DAGUE_VERSION_MAJOR 0)
set (DAGUE_VERSION_MINOR 1)

# ccmake tunable parameters
option(DAGUE_COLLECTIVE
  "Use optimized asynchronous operations where collective communication pattern is detected" OFF)
option(DAGUE_MPI
  "Build DAGuE for distributed memory with MPI backend" ON)
option(DAGUE_CACHE_AWARE "Activate the cache awareness support (only if hardware
  locality is enabled and discovered" OFF)
option(DAGUE_USE_COUNTER_FOR_DEPENDENCIES "Use counters to track the
  dependencies, instead of bitmasks -- reduces the debugging features,
  but might be necessary if using more than 30 input dependencies" OFF)
option(DAGUE_PROFILING
  "Enable the generation of the profiling information during
  execution" OFF)
option(DAGUE_PAPI
  "Enable PAPI performance hardware counters" OFF)
option(DAGUE_STATS
  "Enable the generation of statistics informations during execution
  (microbenchmarking)" OFF)
option(DAGUE_LIFO_USE_LOCKS
  "Use a spinlock-based implementation of LIFOs (instead of an atomic
  operations-based LIFO implementation)" OFF)
option(DAGUE_DEBUG_HISTORY
  "Keep a sumarized history of critical events in memory that can be dumped in gdb when deadlock occur" OFF)
option(DAGUE_GRAPHER
  "Enable the generation of the dot output during execution" OFF)
option(DAGUE_CALL_TRACE
  "Enable the output of the BLAS kernels call trace during execution" OFF)
option(DAGUE_DRY_RUN
  "Disable call to actual BLAS kernels; unfold the dependencies only" OFF)
option(BUILD_SHARED_LIBS
  "Build shared libraries" OFF)
option(BUILD_64bits
  "Build 64 bits mode" ON)
option(DAGUE_CUDA
  "Enable dupport for CUDA" ON)
option(DAGUE_DEBUG
  "Enable the internal debugging of the DAGUE library" OFF)
option(DAGUE_DO_UNIT_TESTS
  "Enable compilation of units tests in tests/units, tests/syn and tests/debug directories" OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# cmake modules setup
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/")
include (CMakeDetermineSystem)
include (CheckCCompilerFlag)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckIncludeFiles)


#
# check the capabilities of the system we are building for 
#

# Check for compiler tools 
find_package(BISON)
find_package(FLEX)

# Fortran tricks
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


# check for the CPU we build for
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

#
# Check compiler flags and capabilities
#
CHECK_C_COMPILER_FLAG( "-std=c99" HAVE_STD_C99)
IF( HAVE_STD_C99 )
  SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" )
ENDIF( HAVE_STD_C99 )

# Set warnings for debug builds 
CHECK_C_COMPILER_FLAG( "-Wall" HAVE_WALL )
IF( HAVE_WALL )
    SET( C_WFLAGS "${C_WFLAGS} -Wall" )
ENDIF( HAVE_WALL )
CHECK_C_COMPILER_FLAG( "-Wextra" HAVE_WEXTRA )
IF( HAVE_WEXTRA )
    SET( C_WFLAGS "${C_WFLAGS} -Wextra" )
ENDIF( HAVE_WEXTRA )
# flags for the overly verbose icc
CHECK_C_COMPILER_FLAG( "-wd424" HAVE_WD )
IF( HAVE_WD )
    # 424: checks for duplicate ";"
    # 981: every volatile triggers a "unspecified evaluation order", obnoxious
    #      but might be useful for some debugging sessions. 
    # 1419: warning about extern functions being declared in .c
    #       files
    # 1572: cuda compares floats with 0.0f. 
    SET( C_WFLAGS "${C_WFLAGS} -wd424 -wd981 -wd1419 -wd1572" )
ENDIF( HAVE_WD )
SET( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_WFLAGS}" )
SET( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_WFLAGS}" )

# threads and atomics
include (cmake_modules/CheckAtomicIntrinsic.cmake)
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

# timeval, timespec, realtime clocks, etc
include(CheckStructHasMember)
check_struct_has_member("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
if( NOT HAVE_TIMESPEC_TV_NSEC )
  add_definitions(-D_GNU_SOURCE)
  check_struct_has_member("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
endif( NOT HAVE_TIMESPEC_TV_NSEC )
check_library_exists(rt clock_gettime "" HAVE_CLOCK_GETTIME)
if( HAVE_CLOCK_GETTIME )
  set(EXTRA_LIBS "${EXTRA_LIBS};rt")
endif( HAVE_CLOCK_GETTIME )

# stdlib, stdio, string, getopt, etc
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
check_include_files(stddef.h HAVE_STDDEF_H)
check_function_exists(getrusage HAVE_GETRUSAGE)


#
# Find required packages
#
find_package(PLASMA REQUIRED)
if( PLASMA_SRC )
    list(APPEND PLASMA_INCLUDE_DIR ${PLASMA_SRC})
endif( PLASMA_SRC )
list(APPEND CMAKE_REQUIRED_INCLUDES ${PLASMA_INCLUDE_DIR})
message(STATUS "Looking for control/common.h;context.h;allocate.h in ${PLASMA_INCLUDE_DIR}")
#check_include_files("control/common.h;control/context.h;control/allocate.h"   FOUND_PLASMA_CONTROL)
#if( !FOUND_PLASMA_CONTROL )
#  message(FATAL_ERROR "Cannot find control/common.h;context.h;allocate.h in the PLASMA source directory")
#endif( !FOUND_PLASMA_CONTROL )

#
# Find optional packages
#
find_package(HWLOC)
if( HAVE_HWLOC )
  list(APPEND EXTRA_SOURCES src/dague_hwloc.c)
  list(APPEND EXTRA_LIBS ${HWLOC_LIB})
  include_directories( ${HWLOC_INCLUDE_DIR} )
else( HAVE_HWLOC )
  if( DAGUE_CACHE_AWARE )
    message( error "Cache awarness requires hwloc" )
  endif( DAGUE_CACHE_AWARE )
endif( HAVE_HWLOC )

if (DAGUE_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    include_directories( ${MPI_INCLUDE_PATH} ) 
  else (MPI_FOUND)
    message(error "MPI support is required in order to build the distributed version of DAGuE."
      "Unfortunately, MPI does not seem to be properly installed on this system, at least"
      "not on the default path.")
  endif (MPI_FOUND)
endif (DAGUE_MPI)

if (DAGUE_PAPI)
  find_package(PAPI REQUIRED)
  if (PAPI_FOUND)
    list(APPEND EXTRA_SOURCES src/papime.c)
    list(APPEND EXTRA_LIBS ${PAPI_LIBRARY})
    include_directories( ${PAPI_INCLUDE_DIR} )
  endif (PAPI_FOUND)
endif (DAGUE_PAPI)

if( DAGUE_CUDA )
  find_package(CUDA)
  set(CUDA_HOST_COMPILATION_CPP OFF)
  if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    set( DAGUE_CUDA_SUPPORT 1)
  else (CUDA_FOUND)
    set( DAGUE_CUDA_SUPPORT 0)
  endif (CUDA_FOUND)
endif( DAGUE_CUDA )

#
##
###
# Finished detecting the system, lets do our own things now
###
##
#
set(PROJECT_INCLUDE_DIR "include")
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${PROJECT_INCLUDE_DIR}/Buf2Cache")
add_definitions(-DHAVE_CONFIG_H)

#
#  Settings for targets
# 

set(SRCS 
  src/dague.c 
  src/symbol.c 
  src/expr.c 
  src/params.c 
  src/dep.c 
  src/arena.c
  src/scheduling.c 
  src/profiling.c 
  src/stats.c 
  src/remote_dep.c 
  src/barrier.c 
  Buf2Cache/buf2cache.c
  src/freelist.c 
  src/debug.c 
  src/bindthread.c 
  src/gpu_data.c
  src/mempool.c 
  ${EXTRA_SOURCES} 
)

#
# Setup targets
#
add_library(dague STATIC ${SRCS})
set_target_properties(dague PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
#target_link_libraries(dague ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
install(TARGETS dague ARCHIVE DESTINATION lib)

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dague-mpi STATIC ${SRCS})
  set_target_properties(dague-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DUSE_MPI")
  #  target_link_libraries(dague-mpi ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  install(TARGETS dague-mpi ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)

#
# Now continue with compiling the tests.
#
add_subdirectory(data_dist)
add_subdirectory(tools)
add_subdirectory(tests)
add_subdirectory(dplasma)

# Configuration header
configure_file (
  "${PROJECT_INCLUDE_DIR}/dague_config.h.in"
  "${PROJECT_INCLUDE_DIR}/dague_config.h")
install(FILES "${PROJECT_INCLUDE_DIR}/dague_config.h" DESTINATION include)

