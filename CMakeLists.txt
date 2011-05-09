cmake_minimum_required (VERSION 2.8)
project (DAGUE C CXX Fortran)
# The current version number
set (DAGUE_VERSION_MAJOR 0)
set (DAGUE_VERSION_MINOR 1)

#####
# ccmake tunable parameters
#####

## Multicore scheduler parameters
mark_as_advanced(DAGUE_SCHED_HIERARCHICAL_QUEUES DAGUE_SCHED_CACHE_AWARE
    DAGUE_SCHED_DEPS_MASK)
option(DAGUE_SCHED_HIERARCHICAL_QUEUES
  "Use the hierarchical queues to favor locality (requires HWLOC, conflicts DAGUE_SCHED_CACHE_AWARE)" OFF)
option(DAGUE_SCHED_CACHE_AWARE 
  "Activate the cache awareness support (requires HWLOC, conflicts DAGUE_HIERARCHICAL_QUEUES)" OFF)
if(DAGUE_SCHED_CACHE_AWARE AND DAGUE_SCHED_HIERARCHICAL_QUEUES)
  message(FATAL_ERROR "DAGUE_SCHED_CACHE_AWARE and DAGUE_SCHED_HIERARCHICAL_QUEUES are mutually exclusive options, please select only one")
endif()
option(DAGUE_SCHED_DEPS_MASK 
  "Use a complete bitmask to track the dependencies, instead of a counter -- increase the debugging features, but limits to a maximum of 30 input dependencies" ON)


### Distributed engine parameters
mark_as_advanced(DAGUE_DIST_THREAD DAGUE_DIST_PRIORITIES)
option(DAGUE_DIST_WITH_MPI
  "Build DAGuE for distributed memory with MPI backend (conflicts all other backends)" ON)
if(DAGUE_DIST_WITH_MPI AND 0)
  message(FATAL_ERROR "DAGUE_DIST_WITH_MPI and DAGUE_DIST_WITH_OTHER are mutually exclusive, please select only one")
endif()
option(DAGUE_DIST_THREAD
  "Use an extra thread to progress the data movements" ON)
option(DAGUE_DIST_PRIORITIES
  "Favor the communications that unlock the most prioritary tasks" ON)
option(DAGUE_DIST_COLLECTIVES 
  "Use optimized asynchronous operations where collective communication pattern is detected" OFF)

### GPU engine parameters
mark_as_advanced(DAGUE_GPU_WITH_OPENCL) # Hide this as it is not supported yet
option(DAGUE_GPU_WITH_CUDA
  "Enable GPU support using CUDA kernels" ON)
option(DAGUE_GPU_WITH_OPENCL
  "Enable GPU support using OpenCL kernels" OFF)
if(DAGUE_GPU_WITH_OPENCL)
  message(WARNING "Open CL is not supported yet, ignored.")
endif()

### Debug options
mark_as_advanced(DAGUE_DEBUG_LIFO_USES_LOCKS)
option(DAGUE_DEBUG
  "Enable the internal debugging verbose output of the DAGUE library" OFF)
option(DAGUE_DEBUG_LIFO_USES_LOCKS
  "Use a spinlock-based implementation of LIFOs (instead of an atomic operations-based LIFO implementation)" OFF)
option(DAGUE_DEBUG_HISTORY
  "Keep a sumarized history of critical events in memory that can be dumped in gdb when deadlock occur" OFF)
option(DAGUE_DEBUG_BUILD_UNIT_TESTS
  "Enable compilation of units tests in tests/units, tests/syn and tests/debug directories" OFF)
option(DAGUE_CALL_TRACE
  "Enable the output of the kernels call trace during execution" OFF)

### Profiling options
option(DAGUE_PROF_TRACE
  "Enable the generation of the profiling information during execution" OFF)
option(DAGUE_PROF_PAPI
  "Enable PAPI performance hardware counters" OFF)
option(DAGUE_PROF_STATS
  "Enable the generation of short statistics information during execution (microbenchmarking)" OFF)
option(DAGUE_PROF_GRAPHER
  "Enable the generation of the dot graph representation during execution" OFF)
option(DAGUE_PROF_DRY_RUN
  "Disable calls to the actual bodies and do not move the data between nodes; unfold the dependencies only" OFF)
option(DAGUE_PROF_DRY_BODY
  "Disable calls to the actual bodies; no computation is performed" OFF)
option(DAGUE_PROF_DRY_DEP
  "Disable calls to the actual data transport; remote dependencies are notified, but no data movement takes place" OFF)

### Look for OMEGA
option(DAGUE_Q2J
  "Build the JDF generator" OFF)
set(FOUND_OMEGA false)
if(DAGUE_Q2J)
  message("--  Looking for DAGUE_OMEGA_DIR to enable DAGUE_Q2J")
  if(DAGUE_OMEGA_DIR)
    message("--  DAGUE_OMEGA_DIR found in ${DAGUE_OMEGA_DIR}")
    set(FOUND_OMEGA true)
  else(DAGUE_OMEGA_DIR)
    message(WARNING "DAGUE_OMEGA_DIR not set: disabling DAGUE_Q2J")
    set(DAGUE_Q2J false)
    set(FOUND_OMEGA false)
  endif(DAGUE_OMEGA_DIR)
endif(DAGUE_Q2J)

### Misc options
option(BUILD_SHARED_LIBS
  "Build shared libraries" OFF)
option(BUILD_64bits
  "Build 64 bits mode" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

### Dague PP options
set(DAGUEPP_CFLAGS "" CACHE STRING "Additional daguepp precompiling flags" )
mark_as_advanced(DAGUEPP_CFLAGS)




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

check_function_exists(sched_setaffinity HAVE_SCHED_SETAFFINITY)
if( NOT HAVE_SCHED_SETAFFINITY )
  check_library_exists(rt sched_setaffinity "" HAVE_SCHED_SETAFFINITY)
endif( NOT HAVE_SCHED_SETAFFINITY )

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
check_include_files(unistd.h HAVE_UNISTD_H)
check_function_exists(getopt_long HAVE_GETOPT_LONG)
check_include_files(errno.h HAVE_ERRNO_H)
check_include_files(stddef.h HAVE_STDDEF_H)
check_function_exists(getrusage HAVE_GETRUSAGE)


#
# Find optional packages
#
find_package(HWLOC QUIET)
set(HAVE_HWLOC ${HWLOC_FOUND})
if( HWLOC_FOUND )
  list(APPEND EXTRA_SOURCES src/dague_hwloc.c)
  list(APPEND EXTRA_LIBS ${HWLOC_LIB})
  include_directories( ${HWLOC_INCLUDE_DIR} )
else( HWLOC_FOUND )
  if( DAGUE_SCHED_CACHE_AWARE OR DAGUE_SCHED_HIERARCHICAL_QUEUES)
    message( WARNING "DAGUE_SCHED_CACHE_AWARE and DAGUE_SCHED_HIERARCHICAL_QUEUES require the HWLOC package, but it has not been found. Revert to flat queues." )
    set(DAGUE_SCHED_CACHE_AWARE 0)
    set(DAGUE_SCHED_HIERARCHICAL_QUEUES 0)
  endif()
endif (HWLOC_FOUND)

if (DAGUE_DIST_WITH_MPI)
  find_package(MPI REQUIRED)
  if (MPI_FOUND)
    set(HAVE_MPI 1)
    include_directories( ${MPI_INCLUDE_PATH} )
  else (MPI_FOUND)
    set(HAVE_MPI 0)
    message(WARNING "MPI support is required in order to build the distributed version of DAGuE."
      "Unfortunately, MPI does not seem to be properly installed on this system, at least"
      "not on the default path.")
  endif (MPI_FOUND)
endif (DAGUE_DIST_WITH_MPI)

if( DAGUE_GPU_WITH_CUDA )
  find_package(CUDA)
  if (CUDA_FOUND)
    set(CUDA_HOST_COMPILATION_CPP OFF)
    set(CUDA_BUILD_EMULATION OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY} )
    list(APPEND EXTRA_SOURCES src/gpu_data.c)
    set(HAVE_CUDA 1)
  else (CUDA_FOUND)
    set(HAVE_CUDA 0)
  endif (CUDA_FOUND)
endif( DAGUE_GPU_WITH_CUDA )

if (DAGUE_PROF_PAPI)
  find_package(PAPI REQUIRED)
  if (PAPI_FOUND)
    list(APPEND EXTRA_SOURCES src/papime.c)
    list(APPEND EXTRA_LIBS ${PAPI_LIBRARY})
    include_directories( ${PAPI_INCLUDE_DIR} )
    set(HAVE_PAPI 1)
  else (PAPI_FOUND)
    set(HAVE_PAPI 0)
  endif (PAPI_FOUND)
endif (DAGUE_PROF_PAPI)

if( DAGUE_PROF_TRACE )
  list(APPEND EXTRA_SOURCES src/profiling.c)
endif( DAGUE_PROF_TRACE )

#
##
###
# Finished detecting the system, lets do our own things now
###
##
#
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${PROJECT_INCLUDE_DIR}")
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} DAGUE_COMPILE_INPLACE)
if(NOT DAGUE_COMPILE_INPLACE)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
endif(NOT DAGUE_COMPILE_INPLACE)
#include_directories("${PROJECT_INCLUDE_DIR}/Buf2Cache")
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
  src/stats.c 
  src/remote_dep.c 
  src/barrier.c 
  Buf2Cache/buf2cache.c
  src/freelist.c 
  src/debug.c 
  src/bindthread.c 
  src/mempool.c 
  ${EXTRA_SOURCES} 
)

#
# Setup targets
#

if (MPI_FOUND)
  add_library(dague-mpi STATIC ${SRCS})
  set_target_properties(dague-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  #  target_link_libraries(dague-mpi ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  install(TARGETS dague-mpi ARCHIVE DESTINATION lib)
else (MPI_FOUND)
  add_library(dague STATIC ${SRCS})
  set_target_properties(dague PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  #  target_link_libraries(dague ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  install(TARGETS dague ARCHIVE DESTINATION lib)
endif (MPI_FOUND)

#
# Now continue with compiling the tests.
#

add_subdirectory(tools)
add_subdirectory(dplasma)
add_subdirectory(data_dist)
add_subdirectory(tests)

# Configuration header
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dague_config.h.in"
  "${PROJECT_INCLUDE_DIR}/dague_config.h")
install(FILES "${PROJECT_INCLUDE_DIR}/dague_config.h" DESTINATION include)

# pkg-config file
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dague.pc.in"
  "${PROJECT_INCLUDE_DIR}/dague.pc")
install(FILES "${PROJECT_INCLUDE_DIR}/dague.pc" DESTINATION include)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${DAGUE_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${DAGUE_VERSION_MINOR}")
include (CPack)

