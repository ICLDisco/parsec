#
# Copyright (c) 2010-2024 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
cmake_minimum_required (VERSION 3.21)
project (PARSEC C)

include(CMakeDependentOption)
include(CMakePushCheckState)
include(GNUInstallDirs)

# The current version number
#   This uses the numbering scheme from libtool -version_number c:r:a
#      see http://www.sourceware.org/autobook/autobook/autobook_61.html
#   with PARSEC_VERSION_MAJOR=c-a
#        PARSEC_VERSION_MINOR=a
#        PARSEC_VERSION_RELEASE=YYMM
#   When making a backward compatible addition to the API
#     PARSEC_VERSION_MAJOR does not change
#     PARSEC_VERSION_MINOR increases by 1
#   When making a backward incompabilte change to an API (or exposed structure)
#     PARSEC_VERSION_MAJOR increases by 1
#     PARSEC_VERSION_MINOR resets to 0
# Unlike strict libtool numbering, PARSEC_VERSION_RELEASE is an monotonous 
# increasing number that corresponds to the software version release number
# and never resets to 0, where
#   YY is the last 2 digits of the release year,
#   MM is the 2 digits of the release month,
# (e.g., for a release tagged v20.02 made in the second month
# of february 2020, PARSEC_VERSION_RELEASE=2002)
set (PARSEC_VERSION_MAJOR   4)
set (PARSEC_VERSION_MINOR   1)
set (PARSEC_VERSION_RELEASE 0) # Devel master have no release number
#set (PARSEC_VERSION_RELEASE 2411) # Set and uncomment for a release

# Set the following values for a release
#set(GIT_COMMIT_TAG v24.11)

# Configure the installation paths
set(PARSEC_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(PARSEC_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(PARSEC_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(PARSEC_INSTALL_LIBEXECDIR ${CMAKE_INSTALL_LIBEXECDIR})
set(PARSEC_INSTALL_DATADIR ${CMAKE_INSTALL_DATADIR})
set(PARSEC_INSTALL_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/cmake/parsec)

# CMake Policies Tuning
if(POLICY CMP0074)
  # CMP0074: Starting with CMake 3.12, all FIND_<something> use <something>_ROOT in the search path
  #          in addition to the specified paths
  cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)
if(POLICY CMP0088)
  # CMP0088: New in version 3.14: FindBISON runs bison in CMAKE_CURRENT_BINARY_DIR when executing.
  cmake_policy(SET CMP0088 NEW)
endif(POLICY CMP0088)
if(POLICY CMP0094)
  # CMP0094: Starting with CMake 3.16, all FIND_Python will use the first matching version instead of
  #          of searching for the largest available version number (which defeats our selection logic)
  cmake_policy(SET CMP0094 NEW)
endif(POLICY CMP0094)
if(POLICY CMP0098)
  # CMP0098: New in version 3.17, FindFLEX runs flex in directory CMAKE_CURRENT_BINARY_DIR when executing.
  cmake_policy(SET CMP0098 NEW)
endif(POLICY CMP0098)
if(POLICY CMP0144)
  # CMP0144: find_package uses upper-case <PACKAGENAME>_ROOT variables in addition to <PackageName>_ROOT
  cmake_policy(SET CMP0144 NEW)
endif(POLICY CMP0144)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED True)
# On OSX only find the Apple frameworks is nothing else is available.
# Without this option the default is to provide the XCode installed version
# completely disregarding the versions installed by Mac Ports or Brew.
set(CMAKE_FIND_FRAMEWORK LAST)
# Find the latest version first if multiple are available
SET(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

# CTest system
set(DART_TESTING_TIMEOUT 120)
enable_testing()
include(CTest)

#####
# ccmake tunable parameters
#####

## Check for the support of additional languages and capabilities
option(SUPPORT_FORTRAN
       "Enable support for Fortran bindings (default ON)" ON)
# In CMake 3.12, enable_language OPTIONAL has no effect and will still
# enable the language for a compiler that does not work/is not present
# causing all sort of problems downstream. For now use CheckLanguage before.
include(CheckLanguage)
if( SUPPORT_FORTRAN )
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran OPTIONAL)
  endif()
endif( SUPPORT_FORTRAN )
option(SUPPORT_CXX
       "Enable support for CXX bindings (default ON)" ON)
if( SUPPORT_CXX )
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX OPTIONAL)
  endif()
endif( SUPPORT_CXX )
option(SUPPORT_C11
       "Enable support for C11 capabilities. Might not work with OpenMP support due to an OpenMP compilers restriction (default ON)." ON)
if(SUPPORT_C11)
  set(SUPPORT_C11 OFF)  # disable until we conirm the compiler supports it
  foreach( item IN LISTS CMAKE_C_COMPILE_FEATURES )
    if( item STREQUAL "c_std_11")
      message(STATUS "Compiler support for C11 detected and enabled")
      set(CMAKE_C_STANDARD 11)
      set(SUPPORT_C11 ON)
      break()
    endif()
  endforeach()
else(SUPPORT_C11)
  set(CMAKE_C_STANDARD 99)
endif(SUPPORT_C11)
set(CMAKE_C_STANDARD_REQUIRED ON)

## Selectively compile only parts of the framework
option(BUILD_TOOLS
       "Build the helper tools such as the pre-compilers, profiling manipulation and so on" ON)
mark_as_advanced(BUILD_TOOLS)
option(BUILD_PARSEC
       "Compile the PaRSEC framework" ON)
mark_as_advanced(BUILD_PARSEC)
### Misc options
option(BUILD_SHARED_LIBS
    "Build shared libraries" ON)
option(BUILD_64bits
  "Build 64 bits mode" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
option(PARSEC_WANT_HOME_CONFIG_FILES
       "Should the runtime check for the parameter configuration file in the user home (\$HOME/.parsec/mca-params.conf)" ON)
option(MPI_HWLOC_COMPAT_CHECK
       "Run the MPI + HWLOC compatibility check. If OFF assume the check succeeds (default ON)." ON)
mark_as_advanced(MPI_HWLOC_COMPAT_CHECK)

### PaRSEC runtime configuration parameters
set(PARSEC_MAX_LOCAL_COUNT 20 CACHE STRING "Number of local variables for tasks (default 20)")
set(PARSEC_MAX_PARAM_COUNT 20 CACHE STRING "Number of parameters for tasks (default 20)")
set(PARSEC_MAX_DEP_IN_COUNT 10 CACHE STRING "Number of input flows for each task (default 10)")
set(PARSEC_MAX_DEP_OUT_COUNT 10 CACHE STRING "Number of output flows for each task (default 10)")

### PaRSEC PP options
set(PARSEC_PTGPP_FLAGS "--noline" CACHE STRING "Additional parsec-ptgpp precompiling flags (separate flags with ';')" )
mark_as_advanced(PARSEC_PTGPP_FLAGS)

option(PARSEC_WITH_DEVEL_HEADERS "Install additional headers in include/parsec allowing external compilation." ON)

## Multicore scheduler parameters
option(PARSEC_SCHED_DEPS_MASK
  "Use a complete bitmask to track the dependencies, instead of a counter -- increase the debugging features, but limits to a maximum of 30 input dependencies" ON)
mark_as_advanced(PARSEC_SCHED_DEPS_MASK)

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
set   (PARSEC_DIST_SHORT_LIMIT 1 CACHE STRING
  "Use the short protocol (no flow control) for messages smaller than the limit in KB. Performs better if smaller than the MTU.")

### GPU engine parameters
option(PARSEC_GPU_ALLOC_PER_TILE
  "Tile based allocation engine for GPU memory (instead of internal management
  of a complete allocation)" OFF)
mark_as_advanced(PARSEC_GPU_ALLOC_PER_TILE)
option(PARSEC_GPU_WITH_CUDA
  "Enable GPU support using CUDA kernels" ON)
option(PARSEC_GPU_WITH_HIP
    "Enable GPU support using HIP kernels" ON)
option(PARSEC_GPU_WITH_LEVEL_ZERO
    "Enable GPU support using LEVEL_ZERO kernels" ON)
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
option(PARSEC_DEBUG_MEM_ADDR
  "Enable the memory access checker, if the compiler supports it." OFF)
option(PARSEC_DEBUG_MEM_LEAK
    "Enable only the memory leak checker, if the compiler supports it." OFF)
option(PARSEC_DEBUG_MEM_RACE
  "Enable the memory thread-race checker, if the compiler supports it. This option is incompatible with the memory address and leak checker in many compilers." OFF)

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
  set(PARSEC_PROF_TRACE_SYSTEM "PaRSEC Binary Tracing Format" CACHE STRING "Select the profiling system, if PARSEC_PROF_TRACE is ON")
#This enforces that only valid values are available in ccmake or cmake-gui
set_property(CACHE PARSEC_PROF_TRACE_SYSTEM PROPERTY STRINGS "OTF2" "PaRSEC Binary Tracing Format")
option(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT
  "Generate Profiling traces for the internal_init tasks in the PTG interface (expert mode only)" OFF)
mark_as_advanced(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) #This is available in export mode only

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
option(PARSEC_PROF_PINS
  "Enable the use of the PaRSEC callback instrumentation system (requires PARSEC_PROF_TRACE)" ON)
# if( PARSEC_PROF_PINS AND NOT PARSEC_PROF_TRACE )
#  message(WARNING "PINS Instrumentation System requires to enable PARSEC_PROF_TRACE. Forcing profiling system on.")
#  SET(PARSEC_PROF_TRACE ON CACHE BOOL "Enable the generation of the profiling information during execution" FORCE)
# endif( PARSEC_PROF_PINS AND NOT PARSEC_PROF_TRACE )

# cmake modules setup
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
include (CMakeDetermineSystem)
include (CheckCCompilerFlag)
include (CheckCSourceCompiles)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckIncludeFiles)
include (CheckFortranCompilerFlag)
include (CheckCXXCompilerFlag)

#
# check the capabilities of the system we are building for
#

# Check for PAPI::SDE
if( PARSEC_PROF_TRACE )
  if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR
     CMAKE_SYSTEM_NAME MATCHES "MSYS" OR
     CMAKE_SYSTEM_NAME MATCHES "MINGW64")
    set(PARSEC_PAPI_SDE OFF)
  else()
    find_package(SDE)
    if( TARGET PAPI::SDE )
      set(PARSEC_PAPI_SDE ON)
    else( TARGET PAPI::SDE )
      set(PARSEC_PAPI_SDE OFF)
    endif( TARGET PAPI::SDE )
  endif()
else( PARSEC_PROF_TRACE )
  unset(PARSEC_PAPI_SDE)
  unset(PARSEC_PAPI_SDE CACHE)
endif( PARSEC_PROF_TRACE )

# Check for compiler tools
find_package(BISON 3)
find_package(FLEX 2.6)
set(PARSEC_HAVE_RECENT_LEX 0)
set(PARSEC_PREGEN_FLEX_BISON_ARCHIVE ${PROJECT_SOURCE_DIR}/contrib/pregen_flex_bison.tar CACHE PATH "Where to store version-controlled archive of pregenerated .l.c and .y.[ch] files")
if(FLEX_FOUND AND BISON_FOUND)
  # add inactive rule to generate in the pregen_flex_bison dir
  set(PARSEC_PREGEN_FLEX_BISON_DIR ${PROJECT_BINARY_DIR}/contrib/pregen_flex_bison CACHE PATH "Where to stage the pre-generated .l.c and .y.[ch] files")
  add_custom_target(parsec_pregen_flex_bison
    WORKING_DIRECTORY ${PARSEC_PREGEN_FLEX_BISON_DIR}
    COMMAND ${CMAKE_COMMAND} -Dsrcdir=${PROJECT_SOURCE_DIR} -Dbuilddir=${PROJECT_BINARY_DIR} -Darchive=${PARSEC_PREGEN_FLEX_BISON_ARCHIVE} -P ${PROJECT_SOURCE_DIR}/cmake_modules/pregen_flex_bison.cmake)
endif()

# check for the CPU we build for
message(STATUS "Building for target ${CMAKE_SYSTEM_PROCESSOR}")
string(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${CMAKE_SYSTEM_PROCESSOR})
if (_mach_x86)
    message(STATUS "Found target for X86")
    set(PARSEC_ARCH_X86 1)
endif (_mach_x86)

string(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${CMAKE_SYSTEM_PROCESSOR})
if (_mach_x86_64)
    message(STATUS "Found target X86_64")
    set(PARSEC_ARCH_X86_64 1)
endif (_mach_x86_64)

string(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${CMAKE_SYSTEM_PROCESSOR})
if (_mach_ppc)
    message(STATUS "Found target for PPC")
    set(PARSEC_ARCH_PPC 1)
endif (_mach_ppc)

include (ParsecCompilerFlags)

# threads and atomics
include (cmake_modules/CheckAtomicIntrinsic.cmake)

if(PARSEC_DEBUG_MEM_RACE OR PARSEC_DEBUG_MEM_ADDR OR PARSEC_DEBUG_MEM_LEAK)
  # Don't redetect cached values
  if(NOT PARSEC_SANITIZE_COMPILE_OPTIONS OR NOT PARSEC_SANITIZE_LINK_OPTIONS)
    if(PARSEC_DEBUG_MEM_LEAK)
      cmake_push_check_state()
      list(APPEND CMAKE_REQUIRED_LIBRARIES "-fsanitize=leak")
      check_c_compiler_flag("-fsanitize=leak" PARSEC_HAVE_DEBUG_MEM_LEAK)
      if(PARSEC_HAVE_DEBUG_MEM_LEAK)
        list(APPEND PARSEC_SANITIZE_COMPILE_OPTIONS "-fsanitize=leak")
        list(APPEND PARSEC_SANITIZE_LINK_OPTIONS "-fsanitize=leak")
      else()
          message(WARNING "PARSEC_DEBUG_MEM_LEAK requested, but not supported by the compiler. Memory leaks will not be reported.")
      endif()
      cmake_pop_check_state()
    endif()
    if(PARSEC_DEBUG_MEM_ADDR)
      cmake_push_check_state()
      list(APPEND CMAKE_REQUIRED_LIBRARIES "-fsanitize=address")
      check_c_compiler_flag("-fsanitize=address" PARSEC_HAVE_DEBUG_MEM_ADDR)
      if(PARSEC_HAVE_DEBUG_MEM_ADDR)
        list(APPEND PARSEC_SANITIZE_COMPILE_OPTIONS "-fsanitize=address")
        list(APPEND PARSEC_SANITIZE_LINK_OPTIONS "-fsanitize=address")
      else()
          message(WARNING "PARSEC_DEBUG_MEM_ADDR requested, but not supported by the compiler. Out of bound accesses will not be reported.")
      endif()
      cmake_pop_check_state()
    endif()
    if(PARSEC_DEBUG_MEM_RACE)
      cmake_push_check_state()
      list(APPEND CMAKE_REQUIRED_LIBRARIES "-fsanitize=thread")
      check_c_compiler_flag("-fsanitize=thread" PARSEC_HAVE_DEBUG_MEM_RACE)
      if(PARSEC_HAVE_DEBUG_MEM_RACE)
        list(APPEND PARSEC_SANITIZE_COMPILE_OPTIONS "-fsanitize=thread")
        list(APPEND PARSEC_SANITIZE_LINK_OPTIONS "-fsanitize=thread")
      else()
          message(WARNING "PARSEC_DEBUG_MEM_RACE requested, but not supported by the compiler. Data races will not be reported.")
      endif()
      cmake_pop_check_state()
    endif()
    set(PARSEC_SANITIZE_COMPILE_OPTIONS "${PARSEC_SANITIZE_COMPILE_OPTIONS}" CACHE STRING "Memory sanitizer compile options")
    set(PARSEC_SANITIZE_LINK_OPTIONS "${PARSEC_SANITIZE_LINK_OPTIONS}" CACHE STRING "Memory sanitizer link options")
  endif(NOT PARSEC_SANITIZE_COMPILE_OPTIONS OR NOT PARSEC_SANITIZE_LINK_OPTIONS)
endif(PARSEC_DEBUG_MEM_RACE OR PARSEC_DEBUG_MEM_ADDR OR PARSEC_DEBUG_MEM_LEAK)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
if(Threads_FOUND)
  set(PARSEC_HAVE_PTHREAD true)
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  check_symbol_exists(pthread_getspecific "pthread.h" PARSEC_HAVE_PTHREAD_GETSPECIFIC)
  check_symbol_exists(pthread_barrier_init "pthread.h" PARSEC_HAVE_PTHREAD_BARRIER)
  if(NOT PARSEC_HAVE_PTHREAD_BARRIER)
    check_symbol_exists(pthread_barrier_init "pthread-barrier.h" PARSEC_HAVE_PTHREAD_BARRIER_H)
    if(PARSEC_HAVE_PTHREAD_BARRIER_H)
      set(CMAKE_REQUIRED_LIBRARIES "libuv.a")
      check_function_exists(pthread_barrier_init PARSEC_HAVE_PTHREAD_BARRIER)
    endif(PARSEC_HAVE_PTHREAD_BARRIER_H)
  endif(NOT PARSEC_HAVE_PTHREAD_BARRIER)
  cmake_pop_check_state()
endif(Threads_FOUND)

check_function_exists(sched_setaffinity PARSEC_HAVE_SCHED_SETAFFINITY)
if(NOT PARSEC_HAVE_SCHED_SETAFFINITY)
  check_library_exists(rt sched_setaffinity "" PARSEC_HAVE_SCHED_SETAFFINITYRT)
  if(PARSEC_HAVE_SCHED_SETAFFINITYRT)
    set(PARSEC_HAVE_SCHED_SETAFFINITY true)
    list(APPEND EXTRA_LIBS rt)
  endif()
endif(NOT PARSEC_HAVE_SCHED_SETAFFINITY)

# timeval, timespec, realtime clocks, etc
include(CheckStructHasMember)
check_struct_has_member("struct timespec" tv_nsec time.h PARSEC_HAVE_TIMESPEC_TV_NSEC)
if( NOT PARSEC_HAVE_TIMESPEC_TV_NSEC )
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D_GNU_SOURCE")
  check_struct_has_member("struct timespec" tv_nsec time.h PARSEC_HAVE_TIMESPEC_TV_NSEC_GNU)
  cmake_pop_check_state()
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
    set(PARSEC_HAVE_CLOCK_GETTIME true)
    list(APPEND EXTRA_LIBS rt)
  endif(PARSEC_HAVE_CLOCK_GETTIMERT)
endif( NOT PARSEC_HAVE_CLOCK_GETTIME )

if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR
   CMAKE_SYSTEM_NAME MATCHES "MSYS" OR
   CMAKE_SYSTEM_NAME MATCHES "MINGW64")
  check_library_exists(ws2_32 WSAStartup "" PARSEC_HAVE_WS2_32)
  list(APPEND EXTRA_LIBS ws2_32)
  list(APPEND EXTRA_LIBS wsock32)
endif()

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

check_include_files(unistd.h PARSEC_HAVE_UNISTD_H)
check_include_files(getopt.h PARSEC_HAVE_GETOPT_H)
check_include_files(errno.h PARSEC_HAVE_ERRNO_H)
check_include_files(stddef.h PARSEC_HAVE_STDDEF_H)
check_include_files(stdbool.h PARSEC_HAVE_STDBOOL_H)
check_include_files(ctype.h PARSEC_HAVE_CTYPE_H)
check_include_files(execinfo.h PARSEC_HAVE_EXECINFO_H)
check_include_files(sys/mman.h PARSEC_HAVE_SYS_MMAN_H)
check_include_files(dlfcn.h PARSEC_HAVE_DLFCN_H)

check_function_exists(asprintf PARSEC_HAVE_ASPRINTF)
check_function_exists(vasprintf PARSEC_HAVE_VASPRINTF)
check_function_exists(getopt_long PARSEC_HAVE_GETOPT_LONG)
check_function_exists(rand_r PARSEC_HAVE_RAND_R)
check_function_exists(getline PARSEC_HAVE_GETLINE)
check_function_exists(setenv PARSEC_HAVE_SETENV)
check_function_exists(sysconf PARSEC_HAVE_SYSCONF)

if( NOT PARSEC_HAVE_SYS_MMAN_H AND PARSEC_PROFILING_USE_MMAP )
  message(STATUS "sys/mman.h not found: PARSEC_PROFILING_USE_MMAP is disabled -- Profiling will allocate and free each tracing block")
  set(PARSEC_PROFILING_USE_MMAP OFF CACHE BOOL "Use a Helper Thread to create the profile files" FORCE)
endif( NOT PARSEC_HAVE_SYS_MMAN_H AND PARSEC_PROFILING_USE_MMAP )

check_c_source_compiles("
int main(int argc, char* argv[]) {
    __builtin_cpu_init();
    return 0;
}" PARSEC_HAVE_BUILTIN_CPU)
check_function_exists(getrusage PARSEC_HAVE_GETRUSAGE)
check_symbol_exists(RUSAGE_THREAD sys/resource.h PARSEC_HAVE_RUSAGE_THREAD)
if( NOT PARSEC_HAVE_RUSAGE_THREAD )
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D_GNU_SOURCE")
  check_symbol_exists(RUSAGE_THREAD sys/resource.h PARSEC_HAVE_RUSAGE_THREAD_GNU)
  cmake_pop_check_state()
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

check_include_files(valgrind/valgrind.h PARSEC_HAVE_VALGRIND_API)

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

check_c_source_compiles("
int toto(int c) __attribute__ ((deprecated(\"deprecated\")));
int main() {
  return 0;
}" PARSEC_HAVE_ATTRIBUTE_DEPRECATED)

check_c_source_compiles("
enum matrix_type {
   matrix_Byte __parsec_attribute_deprecated__(\"Use PARSEC_MATRIX_BYTE instead\") =  1, /**< unsigned char  */
   matrix_Integer __parsec_attribute_deprecated__(\"Use PARSEC_MATRIX_INTEGER instead\") = 2, /**< signed int */
};
int main() {
  return 0;
  }" PARSEC_HAVE_ENUM_ATTRIBUTE_DEPRECATED)

check_type_size("size_t" PARSEC_SIZEOF_SIZE_T)

set(EXTRA_INCLUDES "")
#
# Find optional packages
#
if( BUILD_PARSEC )
  if (PARSEC_DIST_WITH_MPI)
    # Otherwise try to find MPI in the normal way
    find_package(MPI)
    set(PARSEC_HAVE_MPI ${MPI_C_FOUND})
    if (NOT MPI_C_FOUND)
      # Force MPI: if no compiler, do not proceed further
      message(FATAL_ERROR "MPI is required but was not found. Please provide an MPI compiler")
    endif (NOT MPI_C_FOUND)
    list(APPEND EXTRA_LIBS ${MPI_C_LIBRARIES})
  endif (PARSEC_DIST_WITH_MPI)
  #
  # Check to see if
  #     support for MPI 2.0 is available
  #     known bugged version of mpi_assert_no_ordering
  #
  if (MPI_C_FOUND)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_INCLUDES  "${MPI_C_INCLUDE_PATH}")
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")
    check_function_exists(MPI_Type_create_resized PARSEC_HAVE_MPI_20)
    check_function_exists(MPI_Comm_set_info PARSEC_HAVE_MPI_30)
    if(PARSEC_HAVE_MPI_30)
      # MPI Info mpi_assert_enable_overtaking is buggy in Open MPI 3.1.x, x<=4;
      #   and Open MPI 4.0.x, x <= 1
      check_c_source_compiles("#include <mpi.h>
                               int main(int argc, char* argv[]) {
                               #if (3 == OMPI_MAJOR_VERSION && 1 >= OMPI_MINOR_VERSION && 4 >= OMPI_RELEASE_VERSION) || (4 == OMPI_MAJOR_VERSION && 0 == OMPI_MINOR_VERSION && 1 >= OMPI_RELEASE_VERSION)
                               #warning \"malfunctionning mpi_assert_enable_overtaking found\"
                               #endif
                               return 0;
                               }"
                               PARSEC_HAVE_MPI_OVERTAKE
                               FAIL_REGEX "malfunctionning mpi_assert_enable_overtaking")
    endif(PARSEC_HAVE_MPI_30)
    cmake_pop_check_state()
  endif (MPI_C_FOUND)

  # If MPI enabled and the user did not requested a specific HWLOC
  # installation try to use one that is compatible with MPI, by finding if MPI
  # provide HWLOC and use it.
  if(MPI_C_FOUND AND NOT HWLOC_ROOT)
    find_path(PARSEC_MPI_HWLOC_INCLUDE_DIR "hwloc.h"
      HINTS ${MPI_C_INCLUDE_PATH}
      NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
    if(PARSEC_MPI_HWLOC_INCLUDE_DIR)
      get_filename_component(PARSEC_MPI_HWLOC_ROOT ${PARSEC_MPI_HWLOC_INCLUDE_DIR} DIRECTORY)
      message(STATUS "Force HWLOC to match collocated MPI ${PARSEC_MPI_HWLOC_ROOT}; you can override this by setting HWLOC_ROOT")
      set(HWLOC_ROOT ${PARSEC_MPI_HWLOC_ROOT})
    endif(PARSEC_MPI_HWLOC_INCLUDE_DIR)
  endif(MPI_C_FOUND AND NOT HWLOC_ROOT)

  find_package(HWLOC)
  set(PARSEC_HAVE_HWLOC ${HWLOC_FOUND})
  if( HWLOC_FOUND )
    list(APPEND EXTRA_LIBS ${HWLOC_LIBRARIES})
    list(APPEND EXTRA_INCLUDES ${HWLOC_INCLUDE_DIRS})
    set (PARSEC_PKG_REQUIRE "hwloc")
    if(MPI_C_FOUND AND MPI_HWLOC_COMPAT_CHECK AND NOT CMAKE_CROSSCOMPILING)
        include(CheckCSourceRuns)
        cmake_push_check_state()
        # Some MPI libraries link with an external version of hwloc, and
        # compiling/linking our own with a different, incompatible ABI hwloc
        # will cause breakage. We try to capture this case ASAP during configure,
        # except if we are cross-compiling: in this case the error will be
        # mopped-up when runing the user program, late but safe.
        #
        # Note that we test the same link-order as in PaRSEC
        # Dependent upon which one links first, either PaRSEC, or MPI will break.
        #
        # Do not use use targets here becuase CMake consider all imported
        # interfaces as system headers, and instead of using -I (which provide
        # priority during header search) will use -isystem (which does not and
        # might allow to find the header in the standard path).
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${HWLOC_LIBRARIES} ${MPI_C_LIBRARIES})
        list(PREPEND CMAKE_REQUIRED_INCLUDES ${HWLOC_INCLUDE_DIRS} ${MPI_C_INCLUDE_DIRS})
        message(TRACE "CMAKE_REQUIRED_LIBRARIES = ${CMAKE_REQUIRED_LIBRARIES}")
        message(TRACE "CMAKE_REQUIRED_INCLUDES = ${CMAKE_REQUIRED_INCLUDES}")
        check_c_source_runs("
#include <mpi.h>
#include <hwloc.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
    printf(\"HWLOC headers version: %x.\\\\nCalling `hwloc_get_api_version()`...\\\\n\", HWLOC_API_VERSION); fflush(stdout);
    unsigned int lib_api_version = hwloc_get_api_version();
    printf(\"HWLOC library version: %x.\\\\n\", lib_api_version); fflush(stdout);
#if HWLOC_API_VERSION >= 0x00020000
    if (lib_api_version < 0x00020000) {
#else
    if (lib_api_version >= 0x00020000) {
#endif
        printf(\"Header and library versions are incompatible.\\\\n\"); fflush(stdout);
        return 1;
    }
    printf(\"Calling `MPI_Init()`...\\\\n\"); fflush(stdout);
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    printf(\"HWLOC+MPI worked.\\\\n\");
    return 0;
    }" MPI_AND_HWLOC_COMPATIBLE)
        if(NOT MPI_AND_HWLOC_COMPATIBLE)
            message(FATAL_ERROR "The hwloc CMake found and the one used by MPI are incompatible!\n  Make sure you use the same hwloc used to compile your MPI library\n  The test failed with the following outputs:\n${RUN_OUTPUT}")
        endif(NOT MPI_AND_HWLOC_COMPATIBLE)
        cmake_pop_check_state()
    else(MPI_C_FOUND AND MPI_HWLOC_COMPAT_CHECK AND NOT CMAKE_CROSSCOMPILING)
      message(STATUS "MPI and HWLOC assumed compatible (check explicitly disabled)")
      set(MPI_AND_HWLOC_COMPATIBLE 1 CACHE BOOL "MPI and HWLOC assumed compatible (check explicitly disabled)")
    endif(MPI_C_FOUND AND MPI_HWLOC_COMPAT_CHECK AND NOT CMAKE_CROSSCOMPILING)
  endif( HWLOC_FOUND )

  if( PARSEC_GPU_WITH_CUDA )
    string(REGEX MATCH "PGI$" _match_pgi ${CMAKE_C_COMPILER_ID})
    if (_match_pgi)
      get_filename_component(_pgcc ${CMAKE_C_COMPILER} ABSOLUTE)
      string(REGEX REPLACE "/bin/.*" "" _path_pgi ${_pgcc})
      file(GLOB_RECURSE _pgnvcc FOLLOW_SYMLINKS ${_path_pgi}/cuda/*/nvcc)
      list(SORT _pgnvcc)
      list(GET _pgnvcc -1 _pgnvcc)
      string(REGEX REPLACE "/bin/.*" "" _path_pgcuda "${_pgnvcc}")
      if(_path_pgcuda)
        set(CUDAToolkit_ROOT ${_path_pgcuda})
        message(STATUS "Using PGI internal CUDA SDK in ${CUDAToolkit_ROOT}")
      endif(_path_pgcuda)
    endif (_match_pgi)
    find_package(CUDAToolkit 4)
    set(PARSEC_HAVE_CUDA ${CUDAToolkit_FOUND} CACHE BOOL "True if PaRSEC provide support for CUDA")
    if (CUDAToolkit_FOUND)
      message(STATUS "Found CUDA ${CUDAToolkit_VERSION} in ${CUDAToolkit_LIBRARY_DIR}")
      get_target_property(extra_cuda_libs CUDA::cudart INTERFACE_LINK_LIBRARIES)
      if(extra_cuda_libs)
          list(APPEND EXTRA_LIBS ${extra_cuda_libs})
      endif(extra_cuda_libs)
      # Check if PaRSEC can provide automatic compilation of .cu files.
      check_language(CUDA)
      if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
      endif(CMAKE_CUDA_COMPILER)
    endif (CUDAToolkit_FOUND)
    set(PARSEC_HAVE_CU_COMPILER ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if PaRSEC provide support for compiling .cu files")
  endif( PARSEC_GPU_WITH_CUDA )

  if( PARSEC_GPU_WITH_HIP )
    if( NOT DEFINED ROCM_ROOT_DIR AND IS_DIRECTORY /opt/rocm )
      set(ROCM_ROOT_DIR /opt/rocm)
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT_DIR}/hip ${ROCM_ROOT_DIR})
    find_package(HIP QUIET) #quiet because hip-config.cmake is not part of core-cmake and will spam a loud warning when hip/rocm is not installed
    if(HIP_FOUND AND HIP_VERSION VERSION_LESS 5) # find_package(HIP 5...6) does not work for some reason
        message(FATAL_ERROR "Found HIP version ${HIP_VERSION} in ${HIP_LIB_INSTALL_DIR} is too old, use HIP_ROOT to select another version")
    endif()
    if(HIP_FOUND AND PARSEC_HAVE_CUDA)
      # the underlying reason is that the generated ptg code cannot include at the same time
      # cuda_runtime.h and hip_runtime.h, so we need to modify the dev_cuda.h to not expose any
      # structure that depends on these headers.
      message(WARNING "For the time being HIP and CUDA devices cannot be compiled at the same time. Choose one by setting explicitely the options PARSEC_GPU_WITH_{CUDA|HIP}")
      set(HIP_NOT_CUDA_FOUND FALSE)
    elseif(HIP_FOUND)
      message(STATUS "Found HIP ${HIP_VERSION} in ${HIP_LIB_INSTALL_DIR}")
      get_target_property(extra_hip_libs hip::host INTERFACE_LINK_LIBRARIES)
      list(APPEND EXTRA_LIBS ${extra_hip_libs})
      set(HIP_NOT_CUDA_FOUND TRUE)
    else()
      set(HIP_NOT_CUDA_FOUND FALSE)
    endif()
    set(PARSEC_HAVE_HIP ${HIP_NOT_CUDA_FOUND} CACHE BOOL "True if PaRSEC provide support for HIP")
  endif( PARSEC_GPU_WITH_HIP )

  if( PARSEC_GPU_WITH_LEVEL_ZERO )
    find_package(level-zero)
    set(PARSEC_HAVE_LEVEL_ZERO ${LEVEL_ZERO_FOUND} CACHE BOOL "True if PaRSEC provide support for Intel Level Zero")
    if(PARSEC_HAVE_LEVEL_ZERO)
      include_directories("${LEVEL_ZERO_INCLUDE_DIR}/level_zero/")
      find_package(DPCPP)
      if(DPCPP_EXECUTABLE)
        set(PARSEC_HAVE_DPCPP TRUE CACHE BOOL "True if PaRSEC knows how to compile DPCPP code")
        message(STATUS "Found Intel level-zero ${LEVEL_ZERO_VERSION} in -I${LEVEL_ZERO_INCLUDE_DIR} / -L${LEVEL_ZERO_LIBRARY_DIR}")
        message(STATUS "Found dpcpp in ${DPCPP_EXECUTABLE}")
      else(DPCPP_EXECUTABLE)
        set(PARSEC_HAVE_DPCPP FALSE CACHE BOOL "True if PaRSEC knows how to compile DPCPP code")
      endif(DPCPP_EXECUTABLE)
    endif(PARSEC_HAVE_LEVEL_ZERO)
  endif( PARSEC_GPU_WITH_LEVEL_ZERO )

  find_package(AYUDAME QUIET)
  set(PARSEC_HAVE_AYUDAME ${AYUDAME_FOUND})
#
# If AYUDAME support is enabled it means we need to deal with weak symbols. On
# MAC OS X we need to add a special linker flag or the applications will not
# compile correctly.
#
  if(AYUDAME_FOUND)
    target_include_directories(parsec
        PRIVATE
        ${AYUDAME_INCLUDE_DIRS})
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
      message(STATUS "Add '-undefined dynamic_lookup' to the linking flags")
      target_link_libraries(parsec
        PUBLIC
        "-undefined dynamic_lookup")
    endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  endif(AYUDAME_FOUND)

  # Coerce CMake to install the generated .mod files
  if(CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/parsec/include/fortran)
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION ${PARSEC_INSTALL_INCLUDEDIR})
  endif(CMAKE_Fortran_COMPILER_WORKS)
endif( BUILD_PARSEC )

# Find how to link shm_open
check_function_exists(shm_open PARSEC_HAVE_SHM_OPEN)
if(NOT PARSEC_HAVE_SHM_OPEN)
  check_library_exists(rt shm_open "" PARSEC_SHM_OPEN_IN_LIBRT)
  set(PARSEC_HAVE_SHM_OPEN ${PARSEC_SHM_OPEN_IN_LIBRT} CACHE INTERNAL "Have function shm_open")
endif(NOT PARSEC_HAVE_SHM_OPEN)

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
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBDIR}" isSystemDir)
#message(STATUS "LINK_DIRECTORIES = ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
#message(STATUS "INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBDIR}")
if(isSystemDir STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBDIR}")
endif(isSystemDir STREQUAL "-1")

# Prepare the list of include directories. From the source directory we need 2 the source
# itself and then the specialized headers from parsec/include. If we compile with VPATH
# we need to add the same thing for the BINARY_DIR.
string(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} PARSEC_BUILD_INPLACE)
if(NOT PARSEC_BUILD_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include")
endif(NOT PARSEC_BUILD_INPLACE)
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/parsec/include")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
include_directories(BEFORE "${PROJECT_INCLUDE_DIR}")

#
# Check if indent is available on the system.
#
set(PARSEC_HAVE_INDENT 0)
find_program(INDENT_EXECUTABLE indent DOC "path to the indent executable")
mark_as_advanced(INDENT_EXECUTABLE)
# K&R (not supported on Mac), so we settle for less
# -nbad -bap -bbo -nbc -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0
# -cp33 -cs -d0 -di1 -nfc1 -nfca -hnl -i4 -ip0 -l75 -lp -npcs
# -nprs -npsl -saf -sai -saw -nsc -nsob -nss
set(INDENT_OPTIONS "-nbad -bap -nbc -br -brs -ncdb -ce -cli0 -d0 -di1 -nfc1 -i4 -ip0 -lp -npcs -npsl -nsc -nsob -l120")
mark_as_advanced(INDENT_OPTIONS)
if(INDENT_EXECUTABLE)
  set(PARSEC_HAVE_INDENT 1)
endif(INDENT_EXECUTABLE)

#
# Check if awk is available on the system.
#
set(PARSEC_HAVE_AWK 0)
find_program(AWK_EXECUTABLE awk DOC "path to the awk executable")
mark_as_advanced(AWK_EXECUTABLE)
if(AWK_EXECUTABLE)
  set(PARSEC_HAVE_AWK 1)
endif(AWK_EXECUTABLE)

# Check for OTF2 support.
set(PARSEC_HAVE_OTF2 FALSE CACHE INTERNAL "Has support for the OTF2 format")
# OTF2 integration currently depends on MPI being available
if(PARSEC_PROF_TRACE AND PARSEC_DIST_WITH_MPI)
  if(PARSEC_PROF_TRACE_SYSTEM STREQUAL "OTF2" OR PARSEC_PROF_TRACE_SYSTEM STREQUAL "Auto")
    find_package(OTF2 2.1.1)
    if(OTF2_FOUND AND MPI_C_FOUND)
      set(PARSEC_HAVE_OTF2 ${OTF2_FOUND})
      set(PARSEC_PROF_TRACE_SYSTEM "OTF2")
    else(OTF2_FOUND AND MPI_C_FOUND)
      if(PARSEC_PROF_TRACE_SYSTEM STREQUAL "OTF2")
        message(FATAL_ERROR "Requested OTF2 tracing system is not available because OTF2 was not found or MPI was not found")
      endif(PARSEC_PROF_TRACE_SYSTEM STREQUAL "OTF2")
    endif(OTF2_FOUND AND MPI_C_FOUND)
  endif(PARSEC_PROF_TRACE_SYSTEM STREQUAL "OTF2" OR PARSEC_PROF_TRACE_SYSTEM STREQUAL "Auto")
endif(PARSEC_PROF_TRACE AND PARSEC_DIST_WITH_MPI)

#
# First go for the tools.
#
add_subdirectory(tools)
if(CMAKE_CROSSCOMPILING)
  #
  # Import the EXPORT file from external native ptgpp.
  #
  set(IMPORT_EXECUTABLES "IMPORTFILE-NOTFOUND" CACHE FILEPATH "Point it to the export file from a native build")
  message(STATUS "Prepare cross-compiling using ${IMPORT_EXECUTABLES}")
  include(${IMPORT_EXECUTABLES})
  set_target_properties(parsec-ptgpp PROPERTIES IMPORTED_GLOBAL ON)
  get_target_property(PARSEC_PTGPP_EXECUTABLE parsec-ptgpp LOCATION)
  # Build it, now
  get_filename_component(NATIVE_BINARY_DIR "${IMPORT_EXECUTABLES}" DIRECTORY BASE_DIR ${PROJECT_BINARY_DIR} CACHE)
  include(ExternalProject)
  ExternalProject_add(native
    PREFIX ${NATIVE_BINARY_DIR}/CMakeFiles
    SOURCE_DIR ${PROJECT_SOURCE_DIR}
    BINARY_DIR ${NATIVE_BINARY_DIR}
    CONFIGURE_COMMAND "" # Do not reconfigure, this must have been done before to produce $IMPORT_EXECUTABLES
    # Do not do BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>/parsec/interfaces/ptg
    #   because that breaks parallel make
    BUILD_COMMAND "$(MAKE)" -C parsec/interfaces/ptg
          COMMAND "$(MAKE)" -C tools
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND "$(MAKE)" -C parsec/interfaces/ptg install
            COMMAND "$(MAKE)" -C tools install
    EXCLUDE_FROM_ALL TRUE
    STEP_TARGETS build install test clean
    BYPRODUCTS ${PARSEC_PTGPP_EXECUTABLE}
  )
  ExternalProject_Add_Step(native test
    WORKING_DIRECTORY <BINARY_DIR>
    ALWAYS TRUE
    COMMAND "$(MAKE)" -C parsec/interfaces/ptg test
    COMMAND "$(MAKE)" -C tools test
  )
  ExternalProject_Add_Step(native clean
    WORKING_DIRECTORY <BINARY_DIR>
    ALWAYS TRUE
    COMMAND "$(MAKE)" -C parsec/interfaces/ptg clean
    COMMAND "$(MAKE)" -C tools clean
  )
  add_dependencies(parsec-ptgpp native-build)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target native-install)")
  add_test(NAME native-test COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target native-test)
else()
  set(PARSEC_PTGPP_EXECUTABLE parsec-ptgpp)
endif(CMAKE_CROSSCOMPILING)
set(PARSEC_PTGPP_EXECUTABLE ${PARSEC_PTGPP_EXECUTABLE} CACHE STRING "Point to the parsec-ptgpp executable")

# List of files that will go through documentation generation
include(AddDocumentedFiles)

add_subdirectory(parsec)

#
# Add dependency to Level-Zero if it is enabled
#
if(PARSEC_HAVE_LEVEL_ZERO)
    target_link_libraries(parsec PRIVATE level_zero::ze_loader)
endif(PARSEC_HAVE_LEVEL_ZERO)

#
# Now continue with compiling the tests.
#
set(CTEST_SHM_LAUNCHER
      "" CACHE STRING
      "A command to run shared memory testings")
string(REPLACE " " ";" SHM_TEST_CMD_LIST "${CTEST_SHM_LAUNCHER}")
set(CTEST_MPI_LAUNCHER
    "${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}" CACHE STRING
      "A command to run distributed memory testings")
if( CTEST_MPI_LAUNCHER STREQUAL "" )
    message(WARNING "MPI tests will most likely not work: 'CTEST_MPI_LAUNCHER' is not set")
endif()
string(REPLACE " " ";" MPI_TEST_CMD_LIST "${CTEST_MPI_LAUNCHER}")
set(CTEST_CUDA_LAUNCHER_OPTIONS
      "" CACHE STRING
      "Options to pass to the CTEST_MPI_LAUNCHER to select CUDA nodes")
set(CTEST_HIP_LAUNCHER_OPTIONS
      "" CACHE STRING
      "Options to pass to the CTEST_MPI_LAUNCHER to select RoCM/HIP nodes")

if( BUILD_TESTING )
  find_program(HOSTNAME_EXE hostname)
  mark_as_advanced(HOSTNAME_EXE)
  # Test that the launchers work
  add_test(launcher:shm   ${SHM_TEST_CMD_LIST} ${HOSTNAME_EXE})
  if(PARSEC_HAVE_MPI)
    add_test(launcher:mpi   ${MPI_TEST_CMD_LIST} 4 ${HOSTNAME_EXE})
  endif(PARSEC_HAVE_MPI)
  if(PARSEC_HAVE_CUDA)
    add_test(launcher:cuda   ${MPI_TEST_CMD_LIST} 4 ${CTEST_CUDA_LAUNCHER_OPTIONS} ${HOSTNAME_EXE})
  endif(PARSEC_HAVE_CUDA)
  if(PARSEC_HAVE_HIP)
      add_test(launcher:hip   ${MPI_TEST_CMD_LIST} 4 ${CTEST_HIP_LAUNCHER_OPTIONS} ${HOSTNAME_EXE})
  endif(PARSEC_HAVE_HIP)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif( BUILD_TESTING )

add_subdirectory(docs)

string(TIMESTAMP PARSEC_COMPILE_DATE)

# Fetch GIT info for this build, if not preset for a release
# First, check if `git archive` has substituted the special values
set(GIT_COMMIT_HASH "$Format:%h$")
if(GIT_COMMIT_HASH MATCHES "Format:")
  unset(GIT_COMMIT_HASH)
endif()
set(GIT_COMMIT_DATE "$Format:%cI$")
if(GIT_COMMIT_DATE MATCHES "Format:")
  unset(GIT_COMMIT_DATE)
endif()
# Get the latest abbreviated commit hash of the working branch
if(NOT GIT_COMMIT_HASH)
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT GIT_COMMIT_HASH)
  set(GIT_COMMIT_HASH "unknown")
  set(GIT_COMMIT_BRANCH "")
  set(GIT_COMMIT_DIRTY "")
  set(GIT_COMMIT_DATE "")
endif()
endif(NOT GIT_COMMIT_HASH)
# Get the date of the commit hash
if(NOT GIT_COMMIT_DATE)
execute_process(
  COMMAND git log -1 --format=%cI
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_DATE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
endif()
# Get the tag, if any
if(NOT GIT_COMMIT_TAG)
execute_process(
  COMMAND git describe --tags  --exact-match
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
endif()
set(GIT_COMMIT_BRANCH ${GIT_COMMIT_TAG})
# Get the current working branch if no tag
if(NOT GIT_COMMIT_BRANCH)
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
endif()
# Get the status of the clone
if(NOT GIT_COMMIT_DIRTY)
execute_process(
  COMMAND git diff HEAD --shortstat
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_DIRTY
  ERROR_QUIET)
string(STRIP "${GIT_COMMIT_DIRTY}" GIT_COMMIT_DIRTY)
if(NOT GIT_COMMIT_DIRTY)
  set(GIT_COMMIT_DIRTY "no")
endif()
endif()

#
# No external detection below this point !
#
# All detections have now been completed, we can start generating code for building and
# installing PaRSEC.
#

# Libraries that are added to the public parsec export and have special include
#flags (imported targets) have to be there set as well.
foreach(extra_flag IN ITEMS ${CUDAToolkit_INCLUDE_DIRS} ${MPI_C_INCLUDE_DIRS})
  if(NOT extra_flag)
    continue()
  endif(NOT extra_flag)
  if(extra_flag MATCHES "COMPILE_LANGUAGE:C>:([^>]*)" )
    set(extra_flag "${CMAKE_MATCH_1}")
  elseif(extra_flag MATCHES "COMPILE_LANGUAGE")
    continue()
  endif()
  list(APPEND EXTRA_INCLUDES ${extra_flag})
endforeach()

# We need this CMake variable to be defined for the generation of parsec_option.h
# even if it is not the final value (we will not use it until the cmake completion.
# We will reset the variable to the correct value later before the final generation
# of the parsec_options.h header.
set(PARSEC_C_INCLUDES "${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_INCLUDEDIR};${EXTRA_INCLUDES}"
    CACHE STRING "List of directories used to compile the parsec_ptgpp generated code")
mark_as_advanced(PARSEC_C_INCLUDES)

# Configuration header
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_config.h.in"
  "${PROJECT_INCLUDE_DIR}/parsec/parsec_config.h")
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_options.h.in"
  "${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h")

# If we use CAS on int128_t, some structures change and we need to adapt parsec_options.h
# so that .jdf code built externally uses the same variant of parsec_lifo_t structs and such.
cmake_push_check_state()
if(NOT PARSEC_BUILD_INPLACE)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/parsec/include )
endif(NOT PARSEC_BUILD_INPLACE)
list(APPEND CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include)
list(APPEND CMAKE_REQUIRED_DEFINITIONS -DBUILDING_PARSEC)
check_symbol_exists(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128 "parsec/parsec_config.h;parsec/sys/atomic.h" PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
list(APPEND CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_EXTRA_INCLUDE_FILES "parsec/parsec_config.h;parsec/sys/atomic.h;parsec/class/lifo.h")

# We remove these definitions from the cache, as using the cache version
# causes problems if switching options between two configure...
unset(HAVE_PARSEC_OPAQUE_LIFO_SIZEOF CACHE)
unset(HAVE_PARSEC_LIFO_HEAD_OFFSET CACHE)
check_type_size("parsec_lifo_t" PARSEC_LIFO_OPAQUE_SIZEOF)
include(CheckStructureFieldOffset)
# NB: the structure below *needs* to be kept in sync with the structure
# in lifo-external.h
if(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
    check_structure_field_offset("lifo_private" "struct{parsec_object_t super;uint8_t alignment; union{__int128_t i128; char lifo_private;};}" PARSEC_LIFO_HEAD_OFFSET)
else()
    check_structure_field_offset("lifo_private" "struct{parsec_object_t super;uint8_t alignment; union{int64_t i64; char lifo_private;};}" PARSEC_LIFO_HEAD_OFFSET)
endif()
cmake_pop_check_state()

if( PARSEC_LIFO_HEAD_OFFSET )
  math(EXPR PARSEC_LIFO_REMAINING_BYTES "${PARSEC_LIFO_OPAQUE_SIZEOF} - ${PARSEC_LIFO_HEAD_OFFSET}")
else()
  message(FATAL_ERROR "Cannot determine the offset of the lifo_private field in PaRSEC LIFO \
   structure. Assume something is wrong with the compilation and need to be fixed before continuing.")
endif( PARSEC_LIFO_HEAD_OFFSET )

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/lifo-external.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/parsec/class/lifo-external.h")

# It might be possible that we need to reconfigure the parsec_option.h header
# to account for the change in int128_t atomic support.
if(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
  message(STATUS "Internal PaRSEC uses CAS on int128_t. Reconfiguring parsec_options.h")
  configure_file (
     "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_options.h.in"
     "${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h")
else()
  message(STATUS "Internal PaRSEC does not use CAS on int128_t. Keeping parsec_options.h unchanged")
endif()

install(FILES
  ${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec
  RENAME parsec_config.h)
#
# Never install directly the parsec/class/lifo.h header.
# Instead replace it with the lifo-external.h during the installation.
#
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/parsec/class/lifo-external.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/class
  RENAME lifo.h)
#
# Never install directly the parsec/include/parsec/sys/atomic.h header.
# Instead replace it with the atomic-external.h during the installation.
#
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/sys/atomic-external.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/sys
  RENAME atomic.h)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
if(NOT CPACK_GENERATOR)
  set(CPACK_GENERATOR "TBZ2")
endif()
set (CPACK_PACKAGE_NAME "PaRSEC")
set (CPACK_PACKAGE_VENDOR "Innovative Computing Laboratory, The University of Tennessee")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "PaRSEC: the Parallel Runtime Scheduler and Execution Controller for micro-tasks on distributed heterogeneous systems")
set (CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/icldisco/parsec/")
set (CPACK_RESOURCE_FILE_README
     "${CMAKE_CURRENT_SOURCE_DIR}/README.rst")
set (CPACK_RESOURCE_FILE_LICENSE
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${PARSEC_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${PARSEC_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${PARSEC_VERSION_RELEASE}")
include (CPack)

# Prepare the cflags/ldflags for the PkgConfig parsec.pc file
set(EXTRA_CFLAGS ${CMAKE_C_FLAGS})
set(EXTRA_LDFLAGS ${CMAKE_EXE_LINKER_FLAGS})

# Libraries that are added to the public parsec export but not to EXTRA_LIBS have to be there set as well.
foreach(extra_lib IN ITEMS ${PARSEC_ATOMIC_SUPPORT_LIBS} ${PAPI_LIBRARIES})
  if(NOT extra_lib)
    continue()
  endif(NOT extra_lib)
  list(APPEND EXTRA_LIBS ${extra_lib})
endforeach()

# Due to the lack of standardization detecting libraries under CMake leads
# to unconsistents lists of libraries, where some libraries are using full path,
# other a combination of "-L* and -l*" and others just the library name.
# Force a coherent EXTRA_LIBS_EXPANDED that can be used for the generation of the
# pkg-config files: leave all options starting with - as they are, and for
# everything else add "-l" if they are not an abolute path.
set(EXTRA_LIBS_EXPANDED)
foreach(LIB IN ITEMS ${EXTRA_LIBS})
  if( NOT LIB MATCHES "(^|;)-.*" )
    if( NOT IS_ABSOLUTE ${LIB} )
      set(LIB "-l${LIB}")
    endif( NOT IS_ABSOLUTE ${LIB} )
  endif( NOT LIB MATCHES "(^|;)-.*" )
  list(APPEND EXTRA_LIBS_EXPANDED "${LIB}")
endforeach(LIB)

if(EXTRA_LIBS)
  # For printing only: we need the duplicates in EXPANDED for static builds
  list(REMOVE_DUPLICATES EXTRA_LIBS)
endif()

message("\n\nPkgConfig Configuration Flags:")
message("  EXTRA_CFLAGS   = ${EXTRA_CFLAGS}")
message("  EXTRA_LDFLAGS  = ${EXTRA_LDFLAGS}")
message("  EXTRA_INCLUDES = ${EXTRA_INCLUDES}")
message("  EXTRA_LIBS     = ${EXTRA_LIBS}")
message("  FEATURES       = ${MCA_ENABLED_FEATURES}")
message("\n\n")
# Generate configure files to help other software detect and integrate with
# PaRSEC. This generates the pkg-config files to slurp in all needed dependencies
# but also generates the CMake package files.

# pkg-config file
string (REPLACE ";" " " EXTRA_LIBS_EXPANDED "${EXTRA_LIBS_EXPANDED}")

set(EXTRA_INCLUDES_EXPANDED "")
if (EXTRA_INCLUDES)
  foreach(ITEM IN ITEMS ${EXTRA_INCLUDES})
    if( NOT IS_ABSOLUTE ${ITEM} )
      message(STATUS "Item ${ITEM} in EXTRA_INCLUDES might be malformed")
    endif( NOT IS_ABSOLUTE ${ITEM} )
    set(EXTRA_INCLUDES_EXPANDED "${EXTRA_INCLUDES_EXPANDED} -I${ITEM}")
  endforeach(ITEM IN ITEMS ${EXTRA_INCLUDES})
endif (EXTRA_INCLUDES)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec.pc.in"
  "${PROJECT_INCLUDE_DIR}/parsec.pc" @ONLY)
install(FILES "${PROJECT_INCLUDE_DIR}/parsec.pc"
        DESTINATION ${PARSEC_INSTALL_LIBDIR}/pkgconfig)

# Export Targets
install(EXPORT parsec-targets # rules to export the targets built here
  FILE PaRSECTargets.cmake
  NAMESPACE PaRSEC::
  DESTINATION ${PARSEC_INSTALL_CMAKEDIR})

# cmake config files
include(CMakePackageConfigHelpers)

configure_package_config_file(cmake_modules/PaRSECConfig.cmake.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PaRSECConfig.cmake
  INSTALL_DESTINATION ${PARSEC_INSTALL_CMAKEDIR}
  PATH_VARS PARSEC_INSTALL_LIBDIR
  PARSEC_INSTALL_INCLUDEDIR PARSEC_INSTALL_BINDIR PARSEC_INSTALL_LIBEXECDIR PARSEC_INSTALL_CMAKEDIR)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParsecConfigVersion.cmake
  VERSION ${PARSEC_VERSION_MAJOR}.${PARSEC_VERSION_MINOR}.${PARSEC_VERSION_RELEASE}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PaRSECConfig.cmake
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParsecConfigVersion.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindPAPI.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindOTF2.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindHWLOC.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/ParsecCompilePTG.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/Findlevel-zero.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindDPCPP.cmake
  DESTINATION ${PARSEC_INSTALL_CMAKEDIR})

# Prepare exports for submodule inclusion
set(PARSEC_INSTALL_LIBDIR ${PROJECT_BINARY_DIR}/parsec)
set(PARSEC_INSTALL_INCLUDEDIR ${PROJECT_SOURCE_DIR})
set(PARSEC_INSTALL_BINDIR ${PROJECT_BINARY_DIR})
set(PARSEC_INSTALL_LIBEXECDIR ${PROJECT_BINARY_DIR})
set(PARSEC_INSTALL_CMAKEDIR ${PROJECT_SOURCE_DIR}/cmake_modules)

configure_package_config_file(cmake_modules/PaRSECConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/parsec-submodule-config.cmake
  INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  PATH_VARS PARSEC_INSTALL_LIBDIR
  PARSEC_INSTALL_INCLUDEDIR PARSEC_INSTALL_BINDIR PARSEC_INSTALL_LIBEXECDIR PARSEC_INSTALL_CMAKEDIR)

# Prepare the contrib/build_with_parsec directory
configure_file (
  contrib/build_with_parsec/Makefile.in
  ${CMAKE_CURRENT_BINARY_DIR}/contrib/build_with_parsec/Makefile @ONLY)
configure_file (
  contrib/build_with_parsec/CMakeLists.txt.in
  ${CMAKE_CURRENT_BINARY_DIR}/contrib/build_with_parsec/CMakeLists.txt @ONLY)

file(GLOB_RECURSE pattern_files RELATIVE
     "${CMAKE_CURRENT_SOURCE_DIR}/" "contrib/build_with_parsec/*.[c|h]"
                                    "contrib/build_with_parsec/*.jdf"
                                    "contrib/build_with_parsec/*.cu")
foreach( pattern_file ${pattern_files} )
  if(NOT PARSEC_BUILD_INPLACE)
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
      COMMAND ${CMAKE_COMMAND} -E copy
              "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
              "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
    )
  endif(NOT PARSEC_BUILD_INPLACE)
  list( APPEND pattern_files_dest "${pattern_file}" )
endforeach( pattern_file )
add_custom_target(build_with_parsec ALL DEPENDS ${pattern_files_dest})
