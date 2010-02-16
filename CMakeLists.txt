cmake_minimum_required (VERSION 2.8)
project (DPLASMA C Fortran)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# should we use our own math functions
option(DPLASMA_PROFILING
  "Generate the profiling information during execution" OFF)
option(DPLASMA_GRAPHER
  "Generate the dot output during execution" OFF)
option(DPLASMA_WARM_UP
  "Execute a round of warmup for all tests" ON)
option(DO_THE_NASTY_VALIDATIONS
  "Do the validity checks at the end of each test" OFF)
option(BUILD_SHARED_LIBS
  "Build shared libraries by default" OFF)
option(DPLASMA_DISTRIBUTED
  "If DPLASMA support distributed memory systems (i.e. remote dependencies tracking)" ON)
option(CHECK_PAPI
  "If PAPI support is expected or not. Default OFF" OFF)
option(BUILD_64bits
  "Build 64 bits mode (default on)" ON)
option(DPLASMA_CACHE_AWARENESS
  "Enable cache awareness in the scheduling - default activated iff HWLOC is detected" ON)

# The current version number
set (DPLASMA_VERSION_MAJOR 0)
set (DPLASMA_VERSION_MINOR 1)

set( PROJECT_INCLUDE_DIR ./)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/")

#
# check for the CPU we build for
#
EXECUTE_PROCESS(
    COMMAND ${CMAKE_C_COMPILER} -dumpmachine
    OUTPUT_VARIABLE MACHINE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
MESSAGE(STATUS "Building for target ${MACHINE}")

STRING(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${MACHINE})
IF (_mach_x86)
    MESSAGE(STATUS "Found target for X86")
    SET(ARCH_X86 1)
ENDIF (_mach_x86)

STRING(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${MACHINE})
IF (_mach_x86_64)
    MESSAGE(STATUS "Found target X86_64")
    SET(ARCH_X86_64 1)
ENDIF (_mach_x86_64)

STRING(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${MACHINE})
IF (_mach_ppc)
    MESSAGE(STATUS "Found target for PPC")
    SET(ARCH_PPC 1)
ENDIF (_mach_ppc)

ADD_DEFINITIONS(-DHAVE_CONFIG_H)
#
# If we are on a 64 bits machine make sure to compile in 64 bits mode.
#
if (BUILD_64bits)
  include(${CMAKE_ROOT}/Modules/CheckCCompilerFlag.cmake)
  check_c_compiler_flag("-m64" C_M64)
  if (C_M64)
    message(STATUS "Add -m64 to the compiler flags")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_C_LDFLAGS "${CMAKE_C_LDFLAGS} -m64")
  endif (C_M64)
endif (BUILD_64bits)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
string( TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
if( ${CMAKE_BUILD_TYPE} STREQUAL "debug" )
  set(DPLASMA_DEBUG 1)
endif( ${CMAKE_BUILD_TYPE} STREQUAL "debug" )

# Check for available functions
include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
include (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)
include (${CMAKE_ROOT}/Modules/FindFLEX.cmake)
include (${CMAKE_ROOT}/Modules/FindBISON.cmake)
include (${CMAKE_ROOT}/Modules/CheckCCompilerFlag.cmake)
include (${CMAKE_ROOT}/Modules/CMakeDetermineSystem.cmake)

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

message("-- Check if the compiler provides atomic operations directives")
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set( MAC_OS_X 1 CACHE INTERNAL "Compile on MAC OS X")
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap32 HAVE_COMPARE_AND_SWAP_32)
  CHECK_FUNCTION_EXISTS(OSAtomicCompareAndSwap64 HAVE_COMPARE_AND_SWAP_64)
  if( CMAKE_SIZEOF_VOID_P MATCHES "8" AND NOT HAVE_COMPARE_AND_SWAP_64 )
    error( "64 bits pointers without 64 bits atomic compare and swap")
  endif( CMAKE_SIZEOF_VOID_P MATCHES "8" AND NOT HAVE_COMPARE_AND_SWAP_64 )
  message("--  Mac OS X detected")
else (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  if(NOT HAVE_COMPARE_AND_SWAP_32)
    check_symbol_exists( __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 "" HAVE_COMPARE_AND_SWAP_32)
    if(NOT HAVE_COMPARE_AND_SWAP_32)
      #  icc doesn't have the define but it still have the intrinsic function
      CHECK_FUNCTION_EXISTS(__sync_bool_compare_and_swap  HAVE_COMPARE_AND_SWAP_32)
    endif (NOT HAVE_COMPARE_AND_SWAP_32)

    if( CMAKE_SIZEOF_VOID_P MATCHES "8" AND HAVE_COMPARE_AND_SWAP_32 )
      check_symbol_exists(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 "" HAVE_COMPARE_AND_SWAP_64)
      if( NOT HAVE_COMPARE_AND_SWAP_64 )
        CHECK_FUNCTION_EXISTS(__sync_bool_compare_and_swap  HAVE_COMPARE_AND_SWAP_64)
      endif( NOT HAVE_COMPARE_AND_SWAP_64 )
      if( NOT HAVE_COMPARE_AND_SWAP_64 )
        message(FATAL_ERROR "64 bits pointers without 64 bits atomic compare and swap")
      endif( NOT HAVE_COMPARE_AND_SWAP_64 )
    endif( CMAKE_SIZEOF_VOID_P MATCHES "8" AND HAVE_COMPARE_AND_SWAP_32 )
  endif(NOT HAVE_COMPARE_AND_SWAP_32)
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

find_package(PLASMA REQUIRED)

#
# If we want support for MPI based DPLASMA.
#
if (DPLASMA_DISTRIBUTED)
  find_package(MPI)
endif (DPLASMA_DISTRIBUTED)

#
# If support for PAPI should be enabled.
#
if (CHECK_PAPI)
  find_package(PAPI)
  if (PAPI_FOUND)
    set(HAVE_PAPI 1 CACHE BOOL "Compile PAPI support")
  endif (PAPI_FOUND)
endif (CHECK_PAPI)

add_library(dplasma STATIC dplasma.c symbol.c assignment.c expr.c params.c dep.c
  scheduling.c profiling.c remote_dep.c barrier.c precompile.c Buf2Cache/buf2cache.c
  ${BISON_dplasma_yacc_OUTPUTS} ${FLEX_dplasma_flex_OUTPUTS})

add_executable(dpc dpc.c)
target_link_libraries(dpc dplasma ${EXTRA_LIBS})

add_executable(buildDAG tools/buildDAG.c)
target_link_libraries(buildDAG dplasma ${EXTRA_LIBS})

add_executable(grapher grapher.c)
target_link_libraries(grapher dplasma ${EXTRA_LIBS})

if (DPLASMA_DISTRIBUTED)
  if (MPI_FOUND)
    add_library(dplasma-dm STATIC dplasma.c symbol.c assignment.c expr.c params.c dep.c
      scheduling.c profiling.c remote_dep.c barrier.c precompile.c Buf2Cache/buf2cache.c
      ${BISON_dplasma_yacc_OUTPUTS} ${FLEX_dplasma_flex_OUTPUTS})
    set_target_properties(dplasma-dm PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")
    set_target_properties(dplasma-dm PROPERTIES COMPILER MPI_COMPILER)
  else (MPI_FOUND)
    message("MPI support is required in order to build the distributed version of DPLASMA."
      "Unfortunately, MPI does not seem to be properly installed on this system, at least"
      "not on the default path.")
  endif (MPI_FOUND)
endif (DPLASMA_DISTRIBUTED)

#
# Now continue with compiling the tests.
#
add_subdirectory(cholesky)
add_subdirectory(QR)

# Configuration header
configure_file (
  "${PROJECT_INCLUDE_DIR}/dplasma_config.h.in"
  "${PROJECT_INCLUDE_DIR}/dplasma_config.h")

