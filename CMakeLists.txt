cmake_minimum_required (VERSION 2.8.7)
project (PARSEC C)

include(CMakeDependentOption)
include(CMakePushCheckState)

# The current version number
set (PARSEC_VERSION_MAJOR 2)
set (PARSEC_VERSION_MINOR 0)

# CTest system
SET(DART_TESTING_TIMEOUT 120)
enable_testing()
include(CTest)

set(CTEST_SHM_LAUNCHER
      "" CACHE STRING
      "A command to run shared memory testings")
string(REPLACE " " ";" SHM_TEST_CMD_LIST "${CTEST_SHM_LAUNCHER}")
set(CTEST_MPI_LAUNCHER
      "mpiexec -n" CACHE STRING
      "A command to run distributed memory testings")
if( "${CTEST_MPI_LAUNCHER}" STREQUAL "" )
    MESSAGE(WARNING "MPI tests will most likely not work: 'CTEST_MPI_LAUNCHER' is not set")
endif()
string(REPLACE " " ";" MPI_TEST_CMD_LIST "${CTEST_MPI_LAUNCHER}")
set(CTEST_GPU_LAUNCHER_OPTIONS
      "" CACHE STRING
      "Options to pass to the CTEST_MPI_LAUNCHER to select GPU nodes")

#####
# ccmake tunable parameters
#####

## Check for the support of additional languages and capabilities
option(SUPPORT_FORTRAN
       "Enable support for Fortran bindings (default ON)" ON)
if( SUPPORT_FORTRAN )
  enable_language(Fortran OPTIONAL)
endif( SUPPORT_FORTRAN )
option(SUPPORT_CXX
       "Enable support for CXX bindings (default ON)" ON)
if( SUPPORT_CXX )
  enable_language(CXX OPTIONAL)
endif( SUPPORT_CXX )
option(SUPPORT_C11
       "Enable support for C11 capabilities. Might not work with OpenMP support due to an OpenMP compilers restriction (default ON)." ON)

## Selectively compile only parts of the framework
option(BUILD_TOOLS
       "Build the helper tools such as the pre-compilers, profiling manipulation and so on" ON)
mark_as_advanced(BUILD_TOOLS)
option(BUILD_PARSEC
       "Compile the PaRSEC framework" ON)
mark_as_advanced(BUILD_PARSEC)
if( BUILD_PARSEC )
  if( NOT BUILD_TOOLS AND NOT CMAKE_CROSSCOMPILING )
    message(FATAL_ERROR "Building the PaRSEC layer requires the PaRSEC tools")
  endif( NOT BUILD_TOOLS AND NOT CMAKE_CROSSCOMPILING )
endif( BUILD_PARSEC )

### Misc options
option(BUILD_SHARED_LIBS
    "Build shared libraries" ON)
option(BUILD_64bits
  "Build 64 bits mode" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
option(PARSEC_WANT_HOME_CONFIG_FILES
       "Should the runtime check for the parameter configuration file in the user home (\$HOME/.parsec/mca-params.conf)" ON)

### PaRSEC PP options
set(PARSEC_PTGPP_CFLAGS "--noline" CACHE STRING "Additional parsec_ptgpp precompiling flags (separate flags with ';')" )
mark_as_advanced(PARSEC_PTGPP_CFLAGS)

option(PARSEC_WITH_DEVEL_HEADERS "Install additional headers in include/parsec allowing external compilation." OFF)

## Multicore scheduler parameters

mark_as_advanced(PARSEC_SCHED_DEPS_MASK)
option(PARSEC_SCHED_DEPS_MASK
  "Use a complete bitmask to track the dependencies, instead of a counter -- increase the debugging features, but limits to a maximum of 30 input dependencies" ON)

### Distributed engine parameters
mark_as_advanced(PARSEC_DIST_THREAD PARSEC_DIST_PRIORITIES)
option(PARSEC_DIST_WITH_MPI
  "Build PaRSEC for distributed memory with MPI backend (conflicts all other backends)" ON)
if(PARSEC_DIST_WITH_MPI AND 0)
  message(FATAL_ERROR "PARSEC_DIST_WITH_MPI and PARSEC_DIST_WITH_OTHER are mutually exclusive, please select only one")
endif()
option(PARSEC_DIST_THREAD
  "Use an extra thread to progress the data movements" ON)
option(PARSEC_DIST_PRIORITIES
  "Favor the communications that unlock the most prioritary tasks" ON)
option(PARSEC_DIST_COLLECTIVES
  "Use optimized asynchronous operations where collective communication pattern is detected" ON)
set   (PARSEC_DIST_EAGER_LIMIT 1 CACHE STRING
  "Use the eager protocol (no flow control) for messages smaller than the limit in KB")
set   (PARSEC_DIST_SHORT_LIMIT 1 CACHE STRING
  "Use the short protocol (no flow control) for messages smaller than the limit in KB")

### GPU engine parameters
option(PARSEC_GPU_WITH_CUDA
  "Enable GPU support using CUDA kernels" ON)
option(PARSEC_GPU_CUDA_ALLOC_PER_TILE
  "Tile based allocation engine for GPU memory (instead of internal management
  of a complete allocation)" OFF)
mark_as_advanced(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
option(PARSEC_GPU_WITH_OPENCL
  "Enable GPU support using OpenCL kernels" OFF)
mark_as_advanced(PARSEC_GPU_WITH_OPENCL) # Hide this as it is not supported yet
if(PARSEC_GPU_WITH_OPENCL)
  message(WARNING "Open CL is not supported yet, ignored.")
endif()

### Debug options
if( "Debug" STREQUAL CMAKE_BUILD_TYPE )
  set(PARSEC_DEBUG ON)
else()
  set(PARSEC_DEBUG OFF)
endif()
option(PARSEC_DEBUG_PARANOID
  "Enable extra paranoid checks (may impact performance)." OFF)
option(PARSEC_DEBUG_NOISIER
  "Enable chatterbox-like verbose debugging (may impact performance)." OFF)
option(PARSEC_DEBUG_HISTORY
  "Keep a summarized history of critical events in memory that can be dumped in gdb when deadlock occur." OFF)
option(PARSEC_CALL_TRACE
  "Enable the output of the kernels call trace during execution" OFF)

### Simulating Options
option(PARSEC_SIM
  "Enable the computation of the critical path, through simulation" OFF)
if( PARSEC_SIM AND PARSEC_DIST_WITH_MPI )
  message(FATAL_ERROR "PARSEC_SIM cannot be enabled with PARSEC_DIST_WITH_MPI, please select only one")
endif()

### Profiling options
option(PARSEC_PROF_TRACE
  "Enable the generation of the profiling information during
  execution" OFF)
option(PARSEC_PROF_TRACE_ACTIVE_ARENA_SET
  "Enable the profiling of active arena set to track memory usage of parsec handles" OFF)
mark_as_advanced(PARSEC_PROF_TRACE_ACTIVE_ARENA_SET)
option(PARSEC_PROF_TAU
  "Experimental usage of TAU profiling framework" ON)
option(PARSEC_PROF_RUSAGE_EU
  "Print the rusage per execution unit for each progress" OFF)
option(PARSEC_PROF_TRACE_SCHEDULING_EVENTS
  "Enable the tracing of fine-grained scheduling details during execution" OFF)
mark_as_advanced(PARSEC_PROF_TRACE_SCHEDULING_EVENTS)
option(PARSEC_PROF_GRAPHER
  "Enable the generation of the dot graph representation during execution" OFF)
option(PARSEC_PROF_DRY_RUN
  "Disable calls to the actual bodies and do not move the data between nodes; unfold the dependencies only" OFF)
option(PARSEC_PROF_DRY_BODY
  "Disable calls to the actual bodies; no computation is performed" OFF)
option(PARSEC_PROF_DRY_DEP
  "Disable calls to the actual data transport; remote dependencies are notified, but no data movement takes place" OFF)
option(PARSEC_PROFILING_USE_MMAP
  "Use MMAP to create the profile files" ON)
mark_as_advanced(PARSEC_PROFILING_USE_MMAP)
option(PARSEC_PROFILING_USE_HELPER_THREAD
  "Use a Helper Thread to create the profile files" ON)
mark_as_advanced(PARSEC_PROFILING_USE_HELPER_THREAD)

### Instrumentation Options
option(PINS_ENABLE
  "Enable the use of the PaRSEC callback instrumentation system (requires PARSEC_PROF_TRACE)" ON)
# if( PINS_ENABLE AND NOT PARSEC_PROF_TRACE )
#  message(WARNING "PINS Instrumentation System requires to enable PARSEC_PROF_TRACE. Forcing profiling system on.")
#  SET(PARSEC_PROF_TRACE ON CACHE BOOL "Enable the generation of the profiling information during execution" FORCE)
# endif( PINS_ENABLE AND NOT PARSEC_PROF_TRACE )

# cmake modules setup
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")
include (CMakeDetermineSystem)
include (CheckCCompilerFlag)
include (CheckCSourceCompiles)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckIncludeFiles)

#
# check the capabilities of the system we are building for
#

IF( BUILD_TOOLS )
  # Check for compiler tools
  find_package(BISON)
ENDIF( BUILD_TOOLS )
find_package(FLEX)
set(PARSEC_HAVE_RECENT_LEX 0)

# check for the CPU we build for
MESSAGE(STATUS "Building for target ${CMAKE_SYSTEM_PROCESSOR}")
STRING(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86)
    MESSAGE(STATUS "Found target for X86")
    SET(PARSEC_ARCH_X86 1)
ENDIF (_mach_x86)

STRING(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86_64)
    MESSAGE(STATUS "Found target X86_64")
    SET(PARSEC_ARCH_X86_64 1)
ENDIF (_mach_x86_64)

STRING(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_ppc)
    MESSAGE(STATUS "Found target for PPC")
    SET(PARSEC_ARCH_PPC 1)
ENDIF (_mach_ppc)

#
# Fix the building system for 32 or 64 bits.
#
# On MAC OS X there is a easy solution, by setting the
# CMAKE_OSX_ARCHITECTURES to a subset of the following values:
# ppc;ppc64;i386;x86_64.
# On Linux this is a little bit tricky. We have to check that the
# compiler supports the -m32/-m64 flags as well as the linker.
# Once this issue is resolved the CMAKE_C_FLAGS and CMAKE_C_LDFLAGS
# have to be updated accordingly.
#
# TODO: Same trick for the Fortran compiler...
#       no idea how to correctly detect if the required/optional
#          libraries are in the correct format.
#
if (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q64" )
  else (_match_xlc)
    if( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "sparc64fx" )
      set ( ARCH_BUILD " " )
    else()
      set( ARCH_BUILD "-m64" )
    endif()
  endif(_match_xlc)
else (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q32" )
  else (_match_xlc)
    set( ARCH_BUILD "-m32" )
  endif(_match_xlc)
endif (BUILD_64bits)

check_c_compiler_flag( ${ARCH_BUILD} C_M32or64 )
if( C_M32or64 )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_BUILD}" )
  set( CMAKE_C_LDFLAGS "${CMAKE_C_LDFLAGS} ${ARCH_BUILD}" )
  set( LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} ${ARCH_BUILD}" )
endif( C_M32or64 )

#
# Check compiler flags and capabilities
#
IF( NOT _match_xlc )
  IF( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "sparc64fx" )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} " )
  ELSE()
    IF(SUPPORT_C11)
      CHECK_C_COMPILER_FLAG( "-std=c1x" PARSEC_HAVE_STD_C1x)
    ELSE(SUPPORT_C11)
      MESSAGE(STATUS "Performing Test PARSEC_HAVE_STD_C1x - C1x support disabled")
    ENDIF(SUPPORT_C11)
    IF( NOT PARSEC_HAVE_STD_C1x )
      CHECK_C_COMPILER_FLAG( "-std=c99" PARSEC_HAVE_STD_C99)
      IF( PARSEC_HAVE_STD_C99 )
        SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" )
      ENDIF( PARSEC_HAVE_STD_C99 )
    ELSE( NOT PARSEC_HAVE_STD_C1x )
      SET( PARSEC_HAVE_STD_C99 1 )
      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c1x" )
    ENDIF( NOT PARSEC_HAVE_STD_C1x )
  ENDIF()
ELSE( NOT _match_xlc )
  IF(SUPPORT_C11)
    CHECK_C_COMPILER_FLAG( "-qlanglvl=extc1x" PARSEC_HAVE_STD_C1x)
  ELSE(SUPPORT_C11)
    MESSAGE(STATUS "Performing Test PARSEC_HAVE_STD_C1x - C1x support disabled")
  ENDIF(SUPPORT_C11)
  IF( PARSEC_HAVE_STD_C1x )
    CHECK_C_COMPILER_FLAG( "-qlanglvl=extc99" PARSEC_HAVE_STD_C99)
    IF( PARSEC_HAVE_STD_C99 )
      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qlanglvl=extc99" )
    ENDIF( PARSEC_HAVE_STD_C99 )
  ELSE( PARSEC_HAVE_STD_C1x )
    SET( PARSEC_HAVE_STD_C99 1 )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qlanglvl=extc1x" )
  ENDIF( PARSEC_HAVE_STD_C1x )
ENDIF(  NOT _match_xlc )

# Set warnings for debug builds
CHECK_C_COMPILER_FLAG( "-Wall" PARSEC_HAVE_WALL )
IF( PARSEC_HAVE_WALL )
    set( C_WFLAGS "${C_WFLAGS} -Wall" )
ENDIF( PARSEC_HAVE_WALL )
CHECK_C_COMPILER_FLAG( "-Wextra" PARSEC_HAVE_WEXTRA )
IF( PARSEC_HAVE_WEXTRA )
    set( C_WFLAGS "${C_WFLAGS} -Wextra" )
ENDIF( PARSEC_HAVE_WEXTRA )

#
# flags for Intel icc
#
STRING(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
if(_match_icc)
  # Silence annoying warnings
  CHECK_C_COMPILER_FLAG( "-wd424" PARSEC_HAVE_WD )
  IF( PARSEC_HAVE_WD )
    # 424: checks for duplicate ";"
    # 981: every volatile triggers a "unspecified evaluation order", obnoxious
    #      but might be useful for some debugging sessions.
    # 1419: warning about extern functions being declared in .c
    #       files
    # 1572: cuda compares floats with 0.0f.
    # 11074: obnoxious about not inlining long functions.
    set( C_WFLAGS "${C_WFLAGS} -wd424,981,1419,1572,10237,11074,11076" )
  ENDIF( PARSEC_HAVE_WD )
else(_match_icc)
  CHECK_C_COMPILER_FLAG( "-Wno-parentheses-equality" PARSEC_HAVE_PAR_EQUALITY )
  IF( PARSEC_HAVE_PAR_EQUALITY )
        set( C_WFLAGS "${C_WFLAGS} -Wno-parentheses-equality" )
  ENDIF( PARSEC_HAVE_PAR_EQUALITY )
endif(_match_icc)

# add gdb symbols in debug and relwithdebinfo
CHECK_C_COMPILER_FLAG( "-g3" PARSEC_HAVE_G3 )
IF( PARSEC_HAVE_G3 )
    set( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3" )
    set( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g3" )
ELSE()
    set( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0" )
ENDIF( PARSEC_HAVE_G3)
# verbose compilation in debug
set( CMAKE_C_FLAGS_DEBUG "${C_WFLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
# remove asserts in release
set( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG" )
SET( CMAKE_C_FLAGS_RELWITHDEBINFO "${C_WFLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}" )
SET( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG" )

# threads and atomics
include (cmake_modules/CheckAtomicIntrinsic.cmake)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set( PARSEC_OSX 1 CACHE INTERNAL "Compile on MAC OS X")
  if( NOT PARSEC_ATOMIC_USE_C11_ATOMICS )
    # Without C11 support, atomic-macosx.h is used;
    # the old API of Mac OS X does not support 128 atomic operations
    unset( PARSEC_HAVE_INT128 CACHE )
    unset( PARSEC_HAVE_INT128 )
  endif( NOT PARSEC_ATOMIC_USE_C11_ATOMICS )
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
if(PARSEC_ARCH_PPC)
  # PPC Architecture have Int128 type, but no atomics support for this type
  unset( PARSEC_HAVE_INT128 CACHE )
  unset( PARSEC_HAVE_INT128 )
endif(PARSEC_ARCH_PPC)

#
# Remove all duplicates from the CFLAGS.
#
set(TMP_LIST ${CMAKE_C_FLAGS})
separate_arguments(TMP_LIST)
list(REMOVE_DUPLICATES TMP_LIST)
set(CMAKE_C_FLAGS "")
foreach( ITEM ${TMP_LIST})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ITEM}")
endforeach()

find_package(Threads)
if(Threads_FOUND)
  set(PARSEC_HAVE_PTHREAD true)
  list(APPEND EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})
  CMAKE_PUSH_CHECK_STATE()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  check_function_exists(pthread_getspecific PARSEC_HAVE_PTHREAD_GETSPECIFIC)
  check_function_exists(pthread_barrier_init PARSEC_HAVE_PTHREAD_BARRIER)
  if(NOT PARSEC_HAVE_PTHREAD_BARRIER)
    check_symbol_exists(pthread_barrier_init "pthread-barrier.h" PARSEC_HAVE_PTHREAD_BARRIER_H)
    if(PARSEC_HAVE_PTHREAD_BARRIER_H)
      SET(CMAKE_REQUIRED_LIBRARIES "libuv.a")
      check_function_exists(pthread_barrier_init PARSEC_HAVE_PTHREAD_BARRIER)
    endif(PARSEC_HAVE_PTHREAD_BARRIER_H)
  endif(NOT PARSEC_HAVE_PTHREAD_BARRIER)
  CMAKE_POP_CHECK_STATE()
endif(Threads_FOUND)

check_function_exists(sched_setaffinity PARSEC_HAVE_SCHED_SETAFFINITY)
if(NOT PARSEC_HAVE_SCHED_SETAFFINITY)
  check_library_exists(rt sched_setaffinity "" PARSEC_HAVE_SCHED_SETAFFINITYRT)
  if(PARSEC_HAVE_SCHED_SETAFFINITYRT)
    list(APPEND EXTRA_LIBS -lrt)
    set(PARSEC_HAVE_SCHED_SETAFFINITY true)
  endif()
endif(NOT PARSEC_HAVE_SCHED_SETAFFINITY)

# timeval, timespec, realtime clocks, etc
include(CheckStructHasMember)
check_struct_has_member("struct timespec" tv_nsec time.h PARSEC_HAVE_TIMESPEC_TV_NSEC)
if( NOT PARSEC_HAVE_TIMESPEC_TV_NSEC )
  CMAKE_PUSH_CHECK_STATE()
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D_GNU_SOURCE")
  check_struct_has_member("struct timespec" tv_nsec time.h PARSEC_HAVE_TIMESPEC_TV_NSEC_GNU)
  CMAKE_POP_CHECK_STATE()
  if( PARSEC_HAVE_TIMESPEC_TV_NSEC_GNU )
    add_definitions(-D_GNU_SOURCE)
    set(PARSEC_HAVE_TIMESPEC_TV_NSEC true)
  endif( PARSEC_HAVE_TIMESPEC_TV_NSEC_GNU )
endif( NOT PARSEC_HAVE_TIMESPEC_TV_NSEC )
# clock_gettime was located in librt until glibc 2.17 when it moved
# to the libc.
check_library_exists(c clock_gettime "time.h" PARSEC_HAVE_CLOCK_GETTIME)
if( NOT PARSEC_HAVE_CLOCK_GETTIME )
  check_library_exists(rt clock_gettime "time.h" PARSEC_HAVE_CLOCK_GETTIMERT)
  if(PARSEC_HAVE_CLOCK_GETTIMERT)
    list(APPEND EXTRA_LIBS -lrt)
    set(PARSEC_HAVE_CLOCK_GETTIME true)
  endif(PARSEC_HAVE_CLOCK_GETTIMERT)
endif( NOT PARSEC_HAVE_CLOCK_GETTIME )

# stdlib, stdio, string, getopt, etc
check_include_files(stdarg.h PARSEC_HAVE_STDARG_H)
# va_copy is special as it is not required to be a function.
if (PARSEC_HAVE_STDARG_H)
  check_c_source_compiles("
      #include <stdarg.h>
      int main(void) {
          va_list a, b;
          va_copy(a, b);
          return 0;
      }"
      PARSEC_HAVE_VA_COPY
      )

  if (NOT PARSEC_HAVE_VA_COPY)
    check_c_source_compiles("
        #include <stdarg.h>
        int main(void) {
            va_list a, b;
            __va_copy(a, b);
            return 0;
        }"
        PARSEC_HAVE_UNDERSCORE_VA_COPY
        )
    endif (NOT PARSEC_HAVE_VA_COPY)

    check_c_source_compiles("#include <stdio.h>
#include <stdarg.h>
int my_printf(const char *format, ...) __attribute__ ((format (printf, 1, 2)));
int my_printf(const char *format, ...) {
  va_list ap;
  int ret;
  va_start(ap, format);
  ret = vprintf(format, ap);
  va_end(ap);
  return ret;
}
int main() {
  return my_printf(\"toto\");
}" PARSEC_HAVE_ATTRIBUTE_FORMAT_PRINTF)

endif (PARSEC_HAVE_STDARG_H)

check_c_source_compiles("
static _Thread_local int tls = 0;

int main(int argc, char *argv[]) {
  tls = 1;
  return 0;
}" PARSEC_HAVE_THREAD_LOCAL)

check_function_exists(asprintf PARSEC_HAVE_ASPRINTF)
check_function_exists(vasprintf PARSEC_HAVE_VASPRINTF)
check_include_files(unistd.h PARSEC_HAVE_UNISTD_H)
check_include_files(getopt.h PARSEC_HAVE_GETOPT_H)
check_function_exists(getopt_long PARSEC_HAVE_GETOPT_LONG)
check_include_files(errno.h PARSEC_HAVE_ERRNO_H)
check_include_files(stddef.h PARSEC_HAVE_STDDEF_H)
check_include_files(stdbool.h PARSEC_HAVE_STDBOOL_H)
check_include_files(ctype.h PARSEC_HAVE_CTYPE_H)
check_c_source_compiles("
int main(int argc, char* argv[]) {
    __builtin_cpu_init();
    __builtin_cpu_supports(\"avx2\");
    return 0;
}" PARSEC_HAVE_BUILTIN_CPU)
if(PARSEC_HAVE_BUILTIN_CPU)
    check_c_source_compiles("
int main(int argc, char* argv[]) {
    __builtin_cpu_init();
    __builtin_cpu_supports(\"avx512f\");
    return 0;
}" PARSEC_HAVE_BUILTIN_CPU512)
endif(PARSEC_HAVE_BUILTIN_CPU)
check_function_exists(getrusage PARSEC_HAVE_GETRUSAGE)
check_symbol_exists(RUSAGE_THREAD sys/resource.h PARSEC_HAVE_RUSAGE_THREAD)
if( NOT PARSEC_HAVE_RUSAGE_THREAD )
  CMAKE_PUSH_CHECK_STATE()
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D_GNU_SOURCE")
  check_symbol_exists(RUSAGE_THREAD sys/resource.h PARSEC_HAVE_RUSAGE_THREAD_GNU)
  CMAKE_POP_CHECK_STATE()
  if( PARSEC_HAVE_RUSAGE_THREAD_GNU )
    add_definitions(-D_GNU_SOURCE)
    set(PARSEC_HAVE_RUSAGE_THREAD true)
  endif( PARSEC_HAVE_RUSAGE_THREAD_GNU )
endif( NOT PARSEC_HAVE_RUSAGE_THREAD)
check_include_files(limits.h PARSEC_HAVE_LIMITS_H)
check_include_files(string.h PARSEC_HAVE_STRING_H)
check_include_files(libgen.h PARSEC_HAVE_GEN_H)
check_include_files(complex.h PARSEC_HAVE_COMPLEX_H)
check_include_files(sys/param.h PARSEC_HAVE_SYS_PARAM_H)
check_include_files(sys/types.h PARSEC_HAVE_SYS_TYPES_H)
check_include_files(syslog.h PARSEC_HAVE_SYSLOG_H)

check_c_source_compiles("
static int toto(int c) __attribute__ ((always_inline));
static int toto(int c) {
  return c;
}
int main() {
  return toto(3);
}" PARSEC_HAVE_ATTRIBUTE_ALWAYS_INLINE)

check_c_source_compiles("
int toto(int c) __attribute__ ((visibility (\"hidden\")));
int toto(int c) {
  return c;
}
int main() {
  return toto(3);
}" PARSEC_HAVE_ATTRIBUTE_VISIBILITY)

check_c_source_compiles("
      int main( int argc, char** argv) {
         int x = 0;
         if ( __builtin_expect(!!(x), 0) )
            return -1;
         return 0;
      }
      " PARSEC_HAVE_BUILTIN_EXPECT)

set(EXTRA_INCLUDES "")
#
# Find optional packages
#
IF( BUILD_PARSEC )
  add_definitions(-DBUILD_PARSEC)
  # Look for dl library to dynamically load the mca modules and cuda kernels
  check_function_exists(dlsym PARSEC_HAVE_DL)
  if( NOT PARSEC_HAVE_DL )
    find_library(DL_LIBRARY dl)
    MARK_AS_ADVANCED(DL_LIBRARY)
    if(DL_LIBRARY)
      CMAKE_PUSH_CHECK_STATE()
      list(APPEND CMAKE_REQUIRED_LIBRARIES ${DL_LIBRARY})
      check_function_exists(dlsym PARSEC_HAVE_DLDL)
      CMAKE_POP_CHECK_STATE()
      if(PARSEC_HAVE_DLDL)
        list(APPEND EXTRA_LIBS ${DL_LIBRARY} )
        set(PARSEC_HAVE_DL true)
      endif(PARSEC_HAVE_DLDL)
    endif(DL_LIBRARY)
  endif( NOT PARSEC_HAVE_DL )

  find_package(HWLOC)
  set(PARSEC_HAVE_HWLOC ${HWLOC_FOUND})
  if( HWLOC_FOUND )
    list(APPEND EXTRA_LIBS ${HWLOC_LIBRARIES})
    include_directories( ${HWLOC_INCLUDE_DIRS} )
    list(APPEND EXTRA_INCLUDES ${HWLOC_INCLUDE_DIRS})
    set (PARSEC_PKG_REQUIRE "hwloc")
   endif (HWLOC_FOUND)

  if (PARSEC_DIST_WITH_MPI)
    # Force the detection of the C library
    find_package(MPI)
    set(PARSEC_HAVE_MPI ${MPI_C_FOUND})
    if (MPI_C_FOUND)
      list(APPEND EXTRA_LIBS ${MPI_C_LIBRARIES} )
      include_directories( ${MPI_C_INCLUDE_PATH} )
      list(APPEND EXTRA_INCLUDES ${MPI_C_INCLUDE_PATH})
    else (MPI_C_FOUND)
      MESSAGE(ERROR "MPI is required but was not found. Please provide an MPI compiler")
    endif (MPI_C_FOUND)
  endif (PARSEC_DIST_WITH_MPI)
  #
  # Check to see if support for MPI 2.0 is available
  #
  if (MPI_C_FOUND)
    CMAKE_PUSH_CHECK_STATE()
    list(APPEND CMAKE_REQUIRED_INCLUDES  "${MPI_C_INCLUDE_PATH}")
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")
    check_function_exists(MPI_Type_create_resized PARSEC_HAVE_MPI_20)
    CMAKE_POP_CHECK_STATE()
  endif (MPI_C_FOUND)

  if( PARSEC_GPU_WITH_CUDA )
    STRING(REGEX MATCH "PGI$" _match_pgi ${CMAKE_C_COMPILER_ID})
    IF (_match_pgi)
      GET_FILENAME_COMPONENT(_pgcc ${CMAKE_C_COMPILER} ABSOLUTE)
      STRING(REGEX REPLACE "/bin/.*" "" _path_pgi ${_pgcc})
      file(GLOB_RECURSE _pgnvcc FOLLOW_SYMLINKS ${_path_pgi}/cuda/*/nvcc)
      list(SORT _pgnvcc)
      list(GET _pgnvcc -1 _pgnvcc)
      STRING(REGEX REPLACE "/bin/.*" "" _path_pgcuda "${_pgnvcc}")
      if(_path_pgcuda)
        set(CUDA_TOOLKIT_ROOT_DIR ${_path_pgcuda})
        MESSAGE(STATUS "Using PGI internal CUDA SDK in ${CUDA_TOOLKIT_ROOT_DIR}")
      endif(_path_pgcuda)
    ENDIF (_match_pgi)
    find_package(CUDA)
    set(PARSEC_HAVE_CUDA ${CUDA_FOUND})
    if (CUDA_FOUND)
      message(STATUS "Found CUDA ${CUDA_VERSION} in ${CUDA_TOOLKIT_ROOT_DIR}")
      if(CUDA_VERSION VERSION_LESS "3.0")
        set(CUDA_HOST_COMPILATION_CPP OFF)
      endif(CUDA_VERSION VERSION_LESS "3.0")
      set(CUDA_BUILD_EMULATION OFF)
      include_directories(${CUDA_INCLUDE_DIRS})
      list(APPEND EXTRA_INCLUDES ${CUDA_INCLUDE_DIRS})
      list(APPEND EXTRA_LIBS ${CUDA_LIBRARIES} )
      CMAKE_PUSH_CHECK_STATE()
      list(APPEND CMAKE_REQUIRED_INCLUDES  "${CUDA_INCLUDE_DIRS}")
      list(APPEND CMAKE_REQUIRED_LIBRARIES "${CUDA_LIBRARIES}")
      if(CUDA_VERSION VERSION_LESS "4.0")
          set(PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS 0)
      else()
          check_function_exists(cudaDeviceCanAccessPeer PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS)
      endif()
      CMAKE_POP_CHECK_STATE()
    endif (CUDA_FOUND)
  endif( PARSEC_GPU_WITH_CUDA )

  find_package(AYUDAME QUIET)
  set(PARSEC_HAVE_AYUDAME ${AYUDAME_FOUND})
#
# If AYUDAME support is enabled it means we need to deal with weak symbols. On
# MAC OS X we need to add a special linker flag or the applications will not
# compile correctly.
#
  if(AYUDAME_FOUND)
    include_directories( ${AYUDAME_INCLUDE_DIR} )
    list(APPEND EXTRA_INCLUDES ${AYUDAME_INCLUDE_DIR})
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
      message(STATUS "Add '-undefined dynamic_lookup' to the linking flags")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup")
      SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -undefined dynamic_lookup")
    endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  endif(AYUDAME_FOUND)
ENDIF( BUILD_PARSEC )

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
    MESSAGE(STATUS "Fortran adds libraries path ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
    FOREACH(ITEM IN ITEMS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
      list(APPEND EXTRA_LIBS "-L${ITEM}")
    ENDFOREACH(ITEM)
    MESSAGE(STATUS "Fortran adds libraries ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")
    list(APPEND EXTRA_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
    # ifort
    set (CMAKE_Fortran_FLAGS_RELEASE "-f77rtl -O3")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-f77rtl -O0 -g")
    string (REPLACE "-i_dynamic" "" CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS}")
  else (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
    message ("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
    message ("Fortran compiler: " ${Fortran_COMPILER_NAME})
    message ("No optimized Fortran compiler flags are known, we just try -O2...")
    set (CMAKE_Fortran_FLAGS_RELEASE "-O2")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
  endif (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")

# Coherce CMake to install the generated .mod files
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/parsec/mod_files)
install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION include)

ENDIF (CMAKE_Fortran_COMPILER_WORKS)

#
##
###
# Finished detecting the system, lets do our own things now
###
##
#

#
# Using @rpath allows rellocatable libraries:
# https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
#
set(CMAKE_MACOSX_RPATH 1)
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
MESSAGE(STATUS "LINK_DIRECTORIES = ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
MESSAGE(STATUS "INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}/lib")
IF("${isSystemDir}" STREQUAL "-1")
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# Prepare the list of include directories. From the source directory we need 2 the source
# itself and then the specialized headers from parsec/include. If we compile with VPATH
# we need to add the same thing for the BINARY_DIR.
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/parsec/include")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
include_directories(BEFORE "${PROJECT_INCLUDE_DIR}")
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} PARSEC_COMPILE_INPLACE)
if(NOT PARSEC_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include")
endif(NOT PARSEC_COMPILE_INPLACE)

#
# Check if indent is available on the system.
#
set(PARSEC_HAVE_INDENT 0)
FIND_PROGRAM(INDENT_EXECUTABLE indent DOC "path to the indent executable")
MARK_AS_ADVANCED(INDENT_EXECUTABLE)
# K&R (not supported on Mac), so we settle for less
# -nbad -bap -bbo -nbc -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0
# -cp33 -cs -d0 -di1 -nfc1 -nfca -hnl -i4 -ip0 -l75 -lp -npcs
# -nprs -npsl -saf -sai -saw -nsc -nsob -nss
SET(INDENT_OPTIONS "-nbad -bap -nbc -br -brs -ncdb -ce -cli0 -d0 -di1 -nfc1 -i4 -ip0 -lp -npcs -npsl -nsc -nsob -l120")
MARK_AS_ADVANCED(INDENT_OPTIONS)
if(INDENT_EXECUTABLE)
  set(PARSEC_HAVE_INDENT 1)
endif(INDENT_EXECUTABLE)

#
# Check if awk is available on the system.
#
set(PARSEC_HAVE_AWK 0)
FIND_PROGRAM(AWK_EXECUTABLE awk DOC "path to the awk executable")
MARK_AS_ADVANCED(AWK_EXECUTABLE)
if(AWK_EXECUTABLE)
  set(PARSEC_HAVE_AWK 1)
endif(AWK_EXECUTABLE)

get_property(PARSEC_C_INCLUDES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
set(PARSEC_C_INCLUDES "${PARSEC_C_INCLUDES};${MPI_C_COMPILE_FLAGS}" CACHE "List of directories used to compile the parsec_ptgpp generated code" INTERNAL)
mark_as_advanced(PARSEC_C_INCLUDES)

#
# First go for the tools.
#
set(parsec_ptgpp_EXE "NOT-DEFINED")
add_subdirectory(tools)
IF(CMAKE_CROSSCOMPILING)
  SET(IMPORT_EXECUTABLES "IMPORTFILE-NOTFOUND" CACHE FILEPATH "Point it to the export file from a native build")
  MESSAGE(STATUS "Prepare cross-compiling using ${IMPORT_EXECUTABLES}")
  INCLUDE(${IMPORT_EXECUTABLES})
  SET(parsec_ptgpp_EXE native-parsec_ptgpp)
ELSE(CMAKE_CROSSCOMPILING)
  SET(parsec_ptgpp_EXE parsec_ptgpp)
ENDIF(CMAKE_CROSSCOMPILING)

# List of files that will go through documentation generation
include(AddDocumentedFiles)
add_documented_files(${CMAKE_SOURCE_DIR})

add_subdirectory(parsec)

#
# Setup targets
#
if( BUILD_PARSEC )
#
# If we have support for F90 build the PaRSEC module
#
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    include(FortranCInterface)
    FortranCInterface_HEADER(parsec/fortran/f2c_mangle.h
                             MACRO_NAMESPACE "PARSEC_F2C_"
                             SYMBOL_NAMESPACE "parsec_f2c_"
                             SYMBOLS my_sub parsecf:my_sub)
    add_subdirectory(parsec/fortran)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
ENDIF( BUILD_PARSEC )

#
# Now continue with compiling the tests.
#
find_program(HOSTNAME_EXE hostname)
MARK_AS_ADVANCED(HOSTNAME_EXE)
# Test that the launchers work
add_test(launcher_shm   ${SHM_TEST_CMD_LIST} ${HOSTNAME_EXE})
add_test(launcher_mpi   ${MPI_TEST_CMD_LIST} 4 ${HOSTNAME_EXE})
add_test(launcher_gpu   ${MPI_TEST_CMD_LIST} 4 ${CTEST_GPU_LAUNCHER_OPTIONS} ${HOSTNAME_EXE})
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(dplasma)

# Configuration header
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_config.h.in"
  "${PROJECT_INCLUDE_DIR}/parsec/parsec_config.h")
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_options.h.in"
  "${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h")

cmake_push_check_state()
list(APPEND CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/parsec/include ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include)
list(APPEND CMAKE_REQUIRED_DEFINITIONS -DBUILD_PARSEC)
check_symbol_exists(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128 "parsec/parsec_config.h;parsec/sys/atomic.h" PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
cmake_pop_check_state()
if(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
  message(STATUS "Internal PaRSEC uses CAS 128B. Reconfiguring parsec_options.h")
  configure_file (
     "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_options.h.in"
     "${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h")
else()
  message(STATUS "Internal PaRSEC does not use CAS 128B. Keeping parsec_options.h unchanged")
endif()

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec.h
  DESTINATION include)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/ayudame.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/constants.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_config_bottom.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/data_distribution.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/datatype.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/profiling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/private_mempool.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/deprecated.h
  DESTINATION include/parsec)

install(FILES
  ${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h
  DESTINATION include/parsec
  RENAME parsec_config.h)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/output.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/debug.h
  DESTINATION include/parsec/utils)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_object.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_hash_table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/list_item.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_rwlock.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/lifo.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/barrier.h
  DESTINATION include/parsec/class)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindPARSEC.cmake
  DESTINATION share/parsec/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/scheduling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/remote_dep.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_description_structures.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/datarepo.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mempool.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/data_internal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/arena.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/execution_stream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/parsec_internal.h
  DESTINATION include/parsec)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/pins/pins.h
  DESTINATION include/parsec/mca/pins)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/mca.h
  DESTINATION include/parsec/mca)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/devices/device.h
  DESTINATION include/parsec/devices )
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/interfaces/interface.h
  DESTINATION include/parsec/interfaces/ )
if(PARSEC_PROF_GRAPHER)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/parsec_prof_grapher.h
    DESTINATION include/parsec )
endif(PARSEC_PROF_GRAPHER)
# Install header files for jdf generated code
if( PARSEC_WITH_DEVEL_HEADERS )
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/parsec_binary_profile.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/hbbuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/os-spec-timing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/maxheap.h
    DESTINATION include/parsec )

  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/pins/pins_papi_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/pins/pins_tau_utils.h
    DESTINATION include/parsec/mca/pins )

  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/argv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/cmd_line.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/parsec_environ.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/mca_param_cmd_line.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/mca_param.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/os_path.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/show_help.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/zone_malloc.h
    DESTINATION include/parsec/utils )

  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/dequeue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/fifo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/list.h
    DESTINATION include/parsec/class )

endif(PARSEC_WITH_DEVEL_HEADERS)

#
# Never install directly the parsec/include/parsec/sys/atomic.h header.
# Instead replace it with the atomic-extermnal.h during the installation.
#
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/sys/atomic-external.h
  DESTINATION include/parsec/sys
  RENAME atomic.h)

# Due to the lack of standardization detecting libraries under CMake leads
# to unconsistents lists of libraries, where some libraries are using full path,
# other a combination of "-L* and -l*" and others just the library name. Try
# to get a consistent string that can be used for the generation of the
# pkg-config files, by leaving the fullpath libraries as is, and converting
# everything else to -l.
SET(EXTRA_LIBS_EXPANDED)
FOREACH(LIB IN ITEMS ${EXTRA_LIBS})
  if( IS_ABSOLUTE ${LIB} )
    LIST(APPEND EXTRA_LIBS_EXPANDED ${LIB})
  else( IS_ABSOLUTE ${LIB} )
    if( LIB MATCHES "(^|;)-l.*" )
      LIST(APPEND EXTRA_LIBS_EXPANDED "${LIB}")
    else( LIB MATCHES "(^|;)-l.*" )
      if( LIB MATCHES "(^|;)-L.*" )
        LIST(APPEND EXTRA_LIBS_EXPANDED "${LIB}")
      else( LIB MATCHES "(^|;)-L.*" )
        LIST(APPEND EXTRA_LIBS_EXPANDED "-l${LIB}")
      endif( LIB MATCHES "(^|;)-L.*" )
    endif( LIB MATCHES "(^|;)-l.*" )
  endif( IS_ABSOLUTE ${LIB} )
ENDFOREACH(LIB)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TBZ2")
set (CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${PARSEC_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${PARSEC_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "gamma")
include (CPack)

# build doxygen documentation
add_subdirectory(docs)

MESSAGE("\n\nConfiguration flags:")
MESSAGE("  CMAKE_C_FLAGS          = ${CMAKE_C_FLAGS}")
MESSAGE("  CMAKE_C_LDFLAGS        = ${CMAKE_C_LDFLAGS}")
MESSAGE("  CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")
MESSAGE("  EXTRA_LIBS             = ${EXTRA_LIBS}")
MESSAGE("\n\n")

# Generate configure files to help other software detect and integrate with
# PaRSEC. This generates the pkg-config files to slurp in all needed dependencies
# but also generates the CMake package files.

# pkg-config file
string (REPLACE ";" " " EXTRA_LIBS_EXPANDED "${EXTRA_LIBS_EXPANDED}")
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec.pc.in"
  "${PROJECT_INCLUDE_DIR}/parsec.pc" @ONLY)

install(FILES "${PROJECT_INCLUDE_DIR}/parsec.pc"   DESTINATION lib/pkgconfig)

# cmake config files
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake_modules/ParsecConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/cmake_modules/ParsecConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake
                              PATH_VARS CMAKE_INSTALL_PREFIX)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ParsecConfigVersion.cmake
                                 VERSION ${PARSEC_VERSION_MAJOR}.${PARSEC_VERSION_MINOR}.0
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake_modules/ParsecConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/ParsecConfigVersion.cmake cmake_modules/RulesJDF.cmake cmake_modules/FindPARSEC.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake)

# Prepare the contrib/build_with_parsec directory
configure_file (
  contrib/build_with_parsec/Makefile.in
  ${CMAKE_CURRENT_BINARY_DIR}/contrib/build_with_parsec/Makefile @ONLY)

file(GLOB_RECURSE pattern_files RELATIVE
     "${CMAKE_CURRENT_SOURCE_DIR}/" "contrib/build_with_parsec/*.[c|h]"
                                    "contrib/build_with_parsec/*.jdf")
foreach( pattern_file ${pattern_files} )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
    COMMAND cmake -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
            "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
  )
  list( APPEND pattern_files_dest "${pattern_file}" )
endforeach( pattern_file )
add_custom_target(build_with_parsec ALL DEPENDS ${pattern_files_dest})
