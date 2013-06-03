cmake_minimum_required (VERSION 2.8)
project (DAGUE C)
enable_language(CXX OPTIONAL)
enable_language(Fortran OPTIONAL)

include(CMakeDependentOption)

# The current version number
set (DAGUE_VERSION_MAJOR 1)
set (DAGUE_VERSION_MINOR 0)

# CTest system
SET(DART_TESTING_TIMEOUT 120)
enable_testing()
include(CTest)
enable_testing()
set(MPI_TEST_CMD
      "mpirun" CACHE STRING
      "A command to run distributed memory testings")
string(REPLACE " " ";" MPI_TEST_CMD_LIST ${MPI_TEST_CMD})
set(SHM_TEST_CMD
      "" CACHE STRING
      "A command to run shared memory testings")
string(REPLACE " " ";" SHM_TEST_CMD_LIST ${SHM_TEST_CMD})

#####
# ccmake tunable parameters
#####

## Selectively compile only parts of the framework
option(BUILD_TOOLS
       "Build the helper tools such as the pre-compilers, profiling manipulation and so on" ON)
mark_as_advanced(BUILD_TOOLS)
option(BUILD_DAGUE
       "Compile the DAGuE framework" ON)
mark_as_advanced(BUILD_DAGUE)
if( BUILD_DAGUE )
  if( NOT BUILD_TOOLS AND NOT CMAKE_CROSSCOMPILING )
    message(FATAL_ERROR "Building the DAGuE layer requires the DAGuE tools")
  endif( NOT BUILD_TOOLS AND NOT CMAKE_CROSSCOMPILING )
endif( BUILD_DAGUE )
option(BUILD_DPLASMA
       "Compile the DPLASMA layer" ON)
mark_as_advanced(BUILD_DPLASMA)
if( BUILD_DPLASMA AND NOT BUILD_DAGUE )
  message(FATAL_ERROR "Building the DPLASMA layer requires the DAGuE framework")
endif( BUILD_DPLASMA AND NOT BUILD_DAGUE )

### Misc options
option(BUILD_SHARED_LIBS
  "Build shared libraries" OFF)
option(BUILD_64bits
  "Build 64 bits mode" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
option(DAGUE_WANT_HOME_CONFIG_FILES
       "Should the runtime check for the parameter configuration file in the user home (\$HOME/.dague/mca-params.conf)" ON)

### Dague PP options
set(DAGUEPP_CFLAGS "" CACHE STRING "Additional daguepp precompiling flags" )
mark_as_advanced(DAGUEPP_CFLAGS)

## Multicore scheduler parameters
mark_as_advanced(DAGUE_SCHED_REPORT_STATISTICS)
option(DAGUE_SCHED_REPORT_STATISTICS
  "Display statistics on the scheduling at the end of the run")

mark_as_advanced(DAGUE_SCHED_DEPS_MASK)
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
  "Use optimized asynchronous operations where collective communication pattern is detected" ON)
set   (DAGUE_DIST_EAGER_LIMIT 0 CACHE STRING
  "Use the eager protocol (no flow control) for messages smaller than the limit in KB")
set   (DAGUE_DIST_SHORT_LIMIT 0 CACHE STRING
  "Use the short protocol (no flow control) for messages smaller than the limit in KB")

### GPU engine parameters
option(DAGUE_GPU_WITH_CUDA
  "Enable GPU support using CUDA kernels" ON)
option(DAGUE_GPU_CUDA_ALLOC_PER_TILE
  "Tile based allocation engine for GPU memory (instead of internal management
  of a complete allocation)" ON)
mark_as_advanced(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
option(DAGUE_GPU_WITH_OPENCL
  "Enable GPU support using OpenCL kernels" OFF)
mark_as_advanced(DAGUE_GPU_WITH_OPENCL) # Hide this as it is not supported yet
if(DAGUE_GPU_WITH_OPENCL)
  message(WARNING "Open CL is not supported yet, ignored.")
endif()

### Debug options
option(DAGUE_DEBUG_ENABLE
    "Enable extra paranoid checks" OFF)
set   (DAGUE_DEBUG_VERBOSE 0 CACHE STRING
    "Controls the amount of verbose debug output spit on stderr (0,1,2,3)")
option(DAGUE_DEBUG_HISTORY
  "Keep a summarized history of critical events in memory that can be dumped in gdb when deadlock occur" OFF)
option(DAGUE_CALL_TRACE
  "Enable the output of the kernels call trace during execution" OFF)
option(DAGUE_DEBUG_BUILD_UNIT_TESTS
  "Enable compilation of units tests in tests/units, tests/syn and tests/debug directories" OFF)
mark_as_advanced(DAGUE_DEBUG_LIFO_USE_ATOMICS)
option(DAGUE_DEBUG_LIFO_USE_ATOMICS
  "Use the atomic compare and swap base implementation of LIFOs (instead of the spinlock implementation)" ON)

### Simulating Options
option(DAGUE_SIM
  "Enable the computation of the critical path, through simulation" OFF)
if( DAGUE_SIM AND DAGUE_DIST_WITH_MPI )
  message(FATAL_ERROR "DAGUE_SIM cannot be enabled with DAGUE_DIST_WITH_MPI, please select only one")
endif()

### Profiling options
option(DAGUE_PROF_TRACE
  "Enable the generation of the profiling information during execution" OFF)
option(DAGUE_PROF_PAPI
  "Enable PAPI performance hardware counters" OFF)
option(DAGUE_PROF_STATS
  "Enable the generation of short statistics information during execution (microbenchmarking)" OFF)
option(DAGUE_PROF_GRAPHER
  "Enable the generation of the dot graph representation during execution" OFF)
option(DAGUE_PROF_PTR_FILE
  "Enable the generation of the file to check anti dependencies" OFF)
option(DAGUE_PROF_DRY_RUN
  "Disable calls to the actual bodies and do not move the data between nodes; unfold the dependencies only" OFF)
option(DAGUE_PROF_DRY_BODY
  "Disable calls to the actual bodies; no computation is performed" OFF)
option(DAGUE_PROF_DRY_DEP
  "Disable calls to the actual data transport; remote dependencies are notified, but no data movement takes place" OFF)

### Instrumentation Options
option(PINS_ENABLE
  "Enable the use of the PaRSEC callback instrumentation system (requires DAGUE_PROF_TRACE)" ON)
# if( PINS_ENABLE AND NOT DAGUE_PROF_TRACE )
#  message(WARNING "PINS Instrumentation System requires to enable DAGUE_PROF_TRACE. Forcing profiling system on.")
#  SET(DAGUE_PROF_TRACE ON CACHE BOOL "Enable the generation of the profiling information during execution" FORCE)
# endif( PINS_ENABLE AND NOT DAGUE_PROF_TRACE )
SET(CORES_PER_SOCKET 6 CACHE STRING
  "This is a temporary configuration parameter until dague-hwloc enables dynamic determination of this number.") # default for ig.icl
mark_as_advanced(CORES_PER_SOCKET)

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

IF( BUILD_TOOLS )
  # Check for compiler tools
  find_package(BISON)
  find_package(FLEX)
  set(HAVE_RECENT_LEX 0)
ENDIF( BUILD_TOOLS )

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
  if( _match_xlc)
    set( ARCH_BUILD "-q64" )
  else (_match_xlc)
    set( ARCH_BUILD "-m64" )
  endif(_match_xlc)
else (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q32" )
  else (_match_xlc)
    set( ARCH_BUILD "-m32" )
  endif(_match_xlc)
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
IF( NOT _match_xlc )
  CHECK_C_COMPILER_FLAG( "-std=c99" HAVE_STD_C99)
  IF( HAVE_STD_C99 )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" )
  ENDIF( HAVE_STD_C99 )
ELSE( NOT _match_xlc )
  CHECK_C_COMPILER_FLAG( "-qlanglvl=extc99" HAVE_STD_C99)
  IF( HAVE_STD_C99 )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qlanglvl=extc99" )
  ENDIF( HAVE_STD_C99 )
ENDIF( NOT _match_xlc )

# Set warnings for debug builds 
CHECK_C_COMPILER_FLAG( "-Wall" HAVE_WALL )
IF( HAVE_WALL )
    SET( C_WFLAGS "${C_WFLAGS} -Wall" )
ENDIF( HAVE_WALL )
CHECK_C_COMPILER_FLAG( "-Wextra" HAVE_WEXTRA )
IF( HAVE_WEXTRA )
    SET( C_WFLAGS "${C_WFLAGS} -Wextra" )
ENDIF( HAVE_WEXTRA )
#
# flags for the overly verbose icc
#
STRING(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
if(_match_icc)
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
endif(_match_icc)
CHECK_C_COMPILER_FLAG( "-g3" HAVE_G3 )
IF( HAVE_G3 )
    SET( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3 ${C_WFLAGS}" )
ELSE()
    SET( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g ${C_WFLAGS}" )
ENDIF( HAVE_G3)
SET( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_WFLAGS}" )
SET( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG" )

# Check if the linker has group. MacOS doesn't support it
include(CheckCSourceRuns)
set(CMAKE_REQUIRED_FLAGS_save "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Wl,--start-group -Wl,--end-group")
check_c_source_runs( "int main() { return 0; }" HAVE_LINKER_GROUP )
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_save}")

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
    list(APPEND EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})
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
  list(APPEND EXTRA_LIBS rt)
endif( HAVE_CLOCK_GETTIME )

# stdlib, stdio, string, getopt, etc
check_include_files(stdarg.h HAVE_STDARG_H)
# va_copy is special as it is not required to be a function.
if (HAVE_STDARG_H)
  check_c_source_compiles("
      #include <stdarg.h>
      int main(void) {
          va_list a, b;
          va_copy(a, b);
          return 0;
      }"
      HAVE_VA_COPY
      )
 
  if (NOT HAVE_VA_COPY)
    check_c_source_compiles("
        #include <stdarg.h>
        int main(void) {
            va_list a, b;
            __va_copy(a, b);
            return 0;
        }"
        HAVE_UNDERSCORE_VA_COPY
        )
  endif (NOT HAVE_VA_COPY)
endif (HAVE_STDARG_H)
check_function_exists(asprintf HAVE_ASPRINTF)
check_function_exists(vasprintf HAVE_VASPRINTF)
check_include_files(getopt.h HAVE_GETOPT_H)
check_include_files(unistd.h HAVE_UNISTD_H)
check_function_exists(getopt_long HAVE_GETOPT_LONG)
check_include_files(errno.h HAVE_ERRNO_H)
check_include_files(stddef.h HAVE_STDDEF_H)
check_include_files(stdbool.h HAVE_STDBOOL_H)
check_function_exists(getrusage HAVE_GETRUSAGE)
check_include_files(limits.h HAVE_LIMITS_H)
check_include_files(string.h HAVE_STRING_H)
check_include_files(complex.h HAVE_COMPLEX_H)
check_include_files(sys/param.h HAVE_SYS_PARAM_H)
check_include_files(sys/types.h HAVE_SYS_TYPES_H)
check_include_files(syslog.h HAVE_SYSLOG_H)

#
# Find optional packages
#
IF( BUILD_DAGUE )
  find_package(HWLOC)
  set(HAVE_HWLOC ${HWLOC_FOUND})
  if( HWLOC_FOUND )
    list(APPEND EXTRA_LIBS ${HWLOC_LIBRARIES})
    include_directories( ${HWLOC_INCLUDE_DIRS} )
  endif (HWLOC_FOUND)

  if (DAGUE_DIST_WITH_MPI)
    find_package(MPI REQUIRED)
    set(HAVE_MPI ${MPI_FOUND})
    if (MPI_FOUND)
      include_directories( ${MPI_INCLUDE_PATH} )
    endif (MPI_FOUND)
  endif (DAGUE_DIST_WITH_MPI)
  #
  # Check to see if support for MPI 2.0 is available
  #
  if (MPI_FOUND)
    set(saved_include "${CMAKE_REQUIRED_INCLUDES}")
    set(saved_libs "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_INCLUDES  "${CMAKE_REQUIRED_INCLUDES};${MPI_C_INCLUDE_PATH}")
    set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${MPI_C_LIBRARIES}")
    check_function_exists(MPI_Type_create_resized HAVE_MPI_20)
    set(CMAKE_REQUIRED_INCLUDES  "${saved_include}")
    set(CMAKE_REQUIRED_LIBRARIES "${saved_libs}")
  endif (MPI_FOUND)

  if( DAGUE_GPU_WITH_CUDA )
    find_package(CUDA QUIET)
    set(HAVE_CUDA ${CUDA_FOUND})
    if (CUDA_FOUND)
      if(CUDA_VERSION VERSION_LESS "3.0")
        set(CUDA_HOST_COMPILATION_CPP OFF)
      endif(CUDA_VERSION VERSION_LESS "3.0")
      set(CUDA_BUILD_EMULATION OFF)
      include_directories(${CUDA_INCLUDE_DIRS})
      list(APPEND EXTRA_LIBS ${CUDA_LIBRARIES} )
      set(saved_include "${CMAKE_REQUIRED_INCLUDES}")
      set(saved_libs "${CMAKE_REQUIRED_LIBRARIES}")
      set(CMAKE_REQUIRED_INCLUDES  "${CMAKE_REQUIRED_INCLUDES};${CUDA_INCLUDE_DIRS}")
      set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CUDA_LIBRARIES}")
      if(CUDA_VERSION VERSION_LESS "4.0")
          set(DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS 0)
      else()
          check_function_exists(cuDeviceCanAccessPeer DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS)
      endif()
      set(CMAKE_REQUIRED_INCLUDES  "${saved_include}")
      set(CMAKE_REQUIRED_LIBRARIES "${saved_libs}")
    endif (CUDA_FOUND)
  endif( DAGUE_GPU_WITH_CUDA )

  find_package(AYUDAME QUIET)
  set(HAVE_AYUDAME ${AYUDAME_FOUND})
#
# If AYUDAME support is enabled it means we need to deal with weak symbols. On
# MAC OS X we need to add a special linker flag or the applications will not
# compile correctly.
#
  if(AYUDAME_FOUND)
    include_directories( ${AYUDAME_INCLUDE_DIR} )
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
      message(STATUS "Add '-undefined dynamic_lookup' to the linking flags")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup")
      SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -undefined dynamic_lookup")
    endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  endif(AYUDAME_FOUND)
ENDIF( BUILD_DAGUE )

#
# Fortran tricks
#
IF (CMAKE_Fortran_COMPILER_WORKS)
  STRING(REGEX MATCH "Intel" _match_intel ${CMAKE_Fortran_COMPILER_ID})
  IF (_match_intel)
    MESSAGE(STATUS "Add -nofor_main to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -nofor_main")
  ENDIF (_match_intel)

  STRING(REGEX MATCH "PGI$" _match_pgi ${CMAKE_Fortran_COMPILER_ID})
  IF (_match_pgi)
    MESSAGE(STATUS "Add -Mnomain to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -Mnomain -Bstatic")
  ENDIF (_match_pgi)

  STRING(REGEX MATCH ".*xlc$" _match_xlc ${CMAKE_C_COMPILER})
  IF (_match_xlc)
    MESSAGE(ERROR "Please use the thread-safe version of the xlc compiler (xlc_r)")
  ENDIF (_match_xlc)
  STRING(REGEX MATCH "XL" _match_xlc ${CMAKE_C_COMPILER_ID})
  IF (_match_xlc AND BUILD_64bits)
    MESSAGE(STATUS "Add -q64 to the C compiler/linker.")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -q64")
  ENDIF (_match_xlc AND BUILD_64bits)

  STRING(REGEX MATCH ".*xlf$" _match_xlf ${CMAKE_Fortran_COMPILER})
  IF (_match_xlf)
    MESSAGE(ERROR "Please use the thread-safe version of the xlf compiler (xlf_r)")
  ENDIF (_match_xlf)
  STRING(REGEX MATCH "XL$" _match_xlf ${CMAKE_Fortran_COMPILER_ID})
  IF (_match_xlf)
    SET(arch_flags "-q32")
    IF(BUILD_64bits)
      SET(arch_flags "-q64")
    ENDIF(BUILD_64bits)
    MESSAGE(STATUS "Add ${arch_flags} and -nofor_main to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} ${arch_flags} -nofor_main")
  ENDIF (_match_xlf)

#
# Even more Fortran tricks.
#
# FFLAGS depend on the compiler

  if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    # gfortran
    set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g")
  elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
    # ifort (untested)
    set (CMAKE_Fortran_FLAGS_RELEASE "-f77rtl -O3")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-f77rtl -O0 -g")
  else (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
    message ("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
    message ("Fortran compiler: " ${Fortran_COMPILER_NAME})
    message ("No optimized Fortran compiler flags are known, we just try -O2...")
    set (CMAKE_Fortran_FLAGS_RELEASE "-O2")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
  endif (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
ENDIF (CMAKE_Fortran_COMPILER_WORKS)

#
##
###
# Finished detecting the system, lets do our own things now
###
##
#
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
include_directories(BEFORE "${PROJECT_INCLUDE_DIR}")
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} DAGUE_COMPILE_INPLACE)
if(NOT DAGUE_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif(NOT DAGUE_COMPILE_INPLACE)
add_definitions(-DHAVE_CONFIG_H)

#
# Check if indent is available on the system.
#
set(HAVE_INDENT 0)
FIND_PROGRAM(INDENT_EXECUTABLE indent DOC "path to the indent executable")
MARK_AS_ADVANCED(INDENT_EXECUTABLE)
# K&R (not supported on Mac), so we settle for less
# -nbad -bap -bbo -nbc -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0
# -cp33 -cs -d0 -di1 -nfc1 -nfca -hnl -i4 -ip0 -l75 -lp -npcs
# -nprs -npsl -saf -sai -saw -nsc -nsob -nss
SET(INDENT_OPTIONS "-nbad -bap -nbc -br -brs -ncdb -ce -cli0 -d0 -di1 -nfc1 -i4 -ip0 -lp -npcs -npsl -nsc -nsob -l120")
MARK_AS_ADVANCED(INDENT_OPTIONS)
if(INDENT_EXECUTABLE)
  set(HAVE_INDENT 1)
endif(INDENT_EXECUTABLE)

#
# Check if awk is available on the system.
#
set(HAVE_AWK 0)
FIND_PROGRAM(AWK_EXECUTABLE awk DOC "path to the awk executable")
MARK_AS_ADVANCED(AWK_EXECUTABLE)
if(AWK_EXECUTABLE)
  set(HAVE_AWK 1)
endif(AWK_EXECUTABLE)

#
# First go for the tools.
#
set(daguepp_EXE "NOT-DEFINED")
add_subdirectory(tools)
IF(CMAKE_CROSSCOMPILING)
  SET(IMPORT_EXECUTABLES "IMPORTFILE-NOTFOUND" CACHE FILEPATH "Point it to the export file from a native build")
  MESSAGE(STATUS "Prepare cross-compiling using ${IMPORT_EXECUTABLES}")
  INCLUDE(${IMPORT_EXECUTABLES})
  SET(daguepp_EXE native-daguepp)
ELSE(CMAKE_CROSSCOMPILING)
  SET(daguepp_EXE daguepp)
ENDIF(CMAKE_CROSSCOMPILING)

add_subdirectory(dague)
add_subdirectory(data_dist)

#
# Setup targets
#
if( BUILD_DAGUE )
#
# If we have support for F90 build the DAGuE module
#
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    include(FortranCInterface)
    FortranCInterface_HEADER(src/fortran/f2c_mangle.h
                             MACRO_NAMESPACE "DAGUE_"
                             SYMBOL_NAMESPACE "dague_"
                             SYMBOLS mysub daguef:my_sub)
    add_subdirectory(src/fortran)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
ENDIF( BUILD_DAGUE )

#
# Now continue with compiling the tests.
#

IF( BUILD_DPLASMA )
  add_subdirectory(dplasma)
ENDIF ( BUILD_DPLASMA )
IF( BUILD_TESTING AND BUILD_DAGUE )
  add_subdirectory(tests)
ENDIF( BUILD_TESTING AND BUILD_DAGUE )

# Configuration header
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dague_config.h.in"
  "${PROJECT_INCLUDE_DIR}/dague_config.h")
install(FILES "${PROJECT_INCLUDE_DIR}/dague_config.h" DESTINATION include)
install(FILES "include/dague.h" DESTINATION include)

# pkg-config file
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dague.pc.in"
  "${PROJECT_INCLUDE_DIR}/dague.pc")
install(FILES "${PROJECT_INCLUDE_DIR}/dague.pc" DESTINATION include)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TBZ2")
set (CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${DAGUE_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${DAGUE_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "gamma")
include (CPack)


