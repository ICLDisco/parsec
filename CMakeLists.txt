cmake_minimum_required (VERSION 3.12)
project (PARSEC C)

include(CMakeDependentOption)
include(CMakePushCheckState)

# The current version number
set (PARSEC_VERSION_MAJOR 2)
set (PARSEC_VERSION_MINOR 0)
set (PARSEC_VERSION_PATCH 0)

# Set the following values for a release
#set(GIT_COMMIT_HASH deafbeed)
#set(GIT_COMMIT_TAG vX.Y.Z)
#set(GIT_COMMIT_DATE yyyy-mm-dd)

# Configure the installation paths
set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(RUNTIME_INSTALL_DIR bin)
set(SYSCONFIG_INSTALL_DIR share)
set(CONFIG_INSTALL_DIR share/cmake/parsec)

# CMake Policies Tuning
if(POLICY CMP0074)
  # CMP0074: Starting with CMake 3.12, all FIND_<something> use <something>_ROOT in the search path
  #          in addition to the specified paths
  cmake_policy(SET CMP0074 NEW)
ENDIF(POLICY CMP0074)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED True)

# CTest system
SET(DART_TESTING_TIMEOUT 120)
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
IF(SUPPORT_C11)
  set(CMAKE_C_STANDARD 11)
ELSE(SUPPORT_C11)
  set(CMAKE_C_STANDARD 99)
ENDIF(SUPPORT_C11)
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
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
option(PARSEC_WANT_HOME_CONFIG_FILES
       "Should the runtime check for the parameter configuration file in the user home (\$HOME/.parsec/mca-params.conf)" ON)

### PaRSEC PP options
set(PARSEC_PTGFLAGS "--noline" CACHE STRING "Additional parsec_ptgpp precompiling flags (separate flags with ';')" )
mark_as_advanced(PARSEC_PTGFLAGS)

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
set   (PARSEC_DIST_SHORT_LIMIT 1 CACHE STRING
  "Use the short protocol (no flow control) for messages smaller than the limit in KB")
set   (PARSEC_DIST_EAGER_LIMIT 12 CACHE STRING
  "Use the eager protocol (no flow control) for messages smaller than the limit in KB")

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
set(PARSEC_PROF_TRACE_SYSTEM "Auto" CACHE STRING "Select the profiling system,
 if PARSEC_PROF_TRACE is ON")
#This enforces that only the 3 values are available in ccmake or cmake-gui
set_property(CACHE PARSEC_PROF_TRACE_SYSTEM PROPERTY STRINGS "Auto" "OTF2" "PaRSEC Binary Tracing Format")
option(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT
  "Generate Profiling traces for the internal_init tasks in the PTG interface (expert mode only)" OFF)
mark_as_advanced(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) #This is available in export mode only
if( PARSEC_PROF_TRACE )
  SET(PARSEC_PAPI_SDE ON)
else( PARSEC_PROF_TRACE )
  UNSET(PARSEC_PAPI_SDE)
  UNSET(PARSEC_PAPI_SDE CACHE)
endif( PARSEC_PROF_TRACE )

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

# Check for compiler tools
find_package(BISON)
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

include (ParsecCompilerFlags)

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

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
if(Threads_FOUND)
  set(PARSEC_HAVE_PTHREAD true)
  CMAKE_PUSH_CHECK_STATE()
  LIST(APPEND CMAKE_REQUIRED_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  check_symbol_exists(pthread_getspecific "pthread.h" PARSEC_HAVE_PTHREAD_GETSPECIFIC)
  check_symbol_exists(pthread_barrier_init "pthread.h" PARSEC_HAVE_PTHREAD_BARRIER)
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
    set(PARSEC_HAVE_SCHED_SETAFFINITY true)
    list(APPEND EXTRA_LIBS rt)
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
    set(PARSEC_HAVE_CLOCK_GETTIME true)
    list(APPEND EXTRA_LIBS rt)
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
  set(PARSEC_HAVE_PAPI 0)
  find_package(HWLOC)
  set(PARSEC_HAVE_HWLOC ${HWLOC_FOUND})
  IF( HWLOC_FOUND )
     list(APPEND EXTRA_LIBS ${HWLOC_LIBRARIES})
     include_directories( ${HWLOC_INCLUDE_DIRS} )
     list(APPEND EXTRA_INCLUDES ${HWLOC_INCLUDE_DIRS})
     set (PARSEC_PKG_REQUIRE "hwloc")
  ENDIF (HWLOC_FOUND)

  if (PARSEC_DIST_WITH_MPI)
    # Test first if we are using a wrapper as the compiler
    # In that case, we don't need to change the include path and
    # dependent libraries.
    CHECK_C_SOURCE_COMPILES("#include <mpi.h>
                             int main(int argc, char *argv[]) {
                               MPI_Init(&argc, &argv);
                               MPI_Barrier(MPI_COMM_WORLD);
                               MPI_Finalize();
                               return 0;
                             }" CC_CONTAINS_MPI)
    if( CC_CONTAINS_MPI )
      # Create fake MPI:: targets
      set(MPI_C_FOUND TRUE)
      add_library(MPI::MPI_C INTERFACE IMPORTED)
      if(CMAKE_Fortran_COMPILER_WORKS)
        set(MPI_Fortran_FOUND TRUE)
        add_library(MPI::MPI_Fortran INTERFACE IMPORTED)
      endif()
      if(CMAKE_CXX_COMPILER_WORKS)
        set(MPI_CXX_FOUND TRUE)
        add_library(MPI::MPI_CXX INTERFACE IMPORTED)
      endif()
    else( CC_CONTAINS_MPI )
      # Otherwise try to find MPI in the normal way
      find_package(MPI)
    endif( CC_CONTAINS_MPI )
    set(PARSEC_HAVE_MPI ${MPI_C_FOUND})
    if (NOT MPI_C_FOUND)
      # Force MPI: if no compiler, do not proceed further
      MESSAGE(ERROR "MPI is required but was not found. Please provide an MPI compiler")
    endif (NOT MPI_C_FOUND)
  endif (PARSEC_DIST_WITH_MPI)
  #
  # Check to see if
  #     support for MPI 2.0 is available
  #     known bugged version of mpi_assert_no_ordering
  #
  if (MPI_C_FOUND)
    CMAKE_PUSH_CHECK_STATE()
    list(APPEND CMAKE_REQUIRED_INCLUDES  "${MPI_C_INCLUDE_PATH}")
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")
    check_function_exists(MPI_Type_create_resized PARSEC_HAVE_MPI_20)
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
    set(PARSEC_HAVE_CUDA ${CUDA_FOUND} CACHE BOOL "True if PaRSEC provide support for CUDA")
    if (CUDA_FOUND)
      message(STATUS "Found CUDA ${CUDA_VERSION} in ${CUDA_TOOLKIT_ROOT_DIR}")
      if(CUDA_VERSION VERSION_LESS "3.0")
        set(CUDA_HOST_COMPILATION_CPP OFF)
      endif(CUDA_VERSION VERSION_LESS "3.0")
      set(CUDA_BUILD_EMULATION OFF)
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
    target_include_directories(parsec
        PRIVATE
        ${AYUDAME_INCLUDE_DIR})
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
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION include)
  endif(CMAKE_Fortran_COMPILER_WORKS)
ENDIF( BUILD_PARSEC )

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
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#MESSAGE(STATUS "LINK_DIRECTORIES = ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
#MESSAGE(STATUS "INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}/lib")
IF("${isSystemDir}" STREQUAL "-1")
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# Prepare the list of include directories. From the source directory we need 2 the source
# itself and then the specialized headers from parsec/include. If we compile with VPATH
# we need to add the same thing for the BINARY_DIR.
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/parsec/include")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
include_directories(BEFORE "${PROJECT_INCLUDE_DIR}")
STRING(COMPARE EQUAL ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} PARSEC_BUILD_INPLACE)
if(NOT PARSEC_BUILD_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include")
endif(NOT PARSEC_BUILD_INPLACE)

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
set(PARSEC_C_INCLUDES "${PARSEC_C_INCLUDES};${MPI_C_COMPILE_FLAGS}" CACHE STRING "List of directories used to compile the parsec_ptgpp generated code" INTERNAL)
mark_as_advanced(PARSEC_C_INCLUDES)

#
# First go for the tools.
#
add_subdirectory(tools)
IF(CMAKE_CROSSCOMPILING)
  #
  # Import the EXPORT file from external native ptgpp.
  #
  SET(IMPORT_EXECUTABLES "IMPORTFILE-NOTFOUND" CACHE FILEPATH "Point it to the export file from a native build")
  MESSAGE(STATUS "Prepare cross-compiling using ${IMPORT_EXECUTABLES}")
  INCLUDE(${IMPORT_EXECUTABLES})
  set_target_properties(parsec_ptgpp PROPERTIES IMPORTED_GLOBAL ON)
  get_target_property(parsec_ptgpp_EXE parsec_ptgpp LOCATION)
  # Build it, now
  get_filename_component(NATIVE_BINARY_DIR "${IMPORT_EXECUTABLES}" DIRECTORY BASE_DIR ${CMAKE_BINARY_DIR} CACHE)
  include(ExternalProject)
  ExternalProject_add(native
    PREFIX ${NATIVE_BINARY_DIR}/CMakeFiles
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
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
    BYPRODUCTS ${parsec_ptgpp_EXE}
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
  add_dependencies(parsec_ptgpp native-build)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target native-install)")
  add_test(NAME native-test COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target native-test)
else()
  set(parsec_ptgpp_EXE PaRSEC::parsec_ptgpp)
ENDIF(CMAKE_CROSSCOMPILING)
set(parsec_ptgpp_EXE ${parsec_ptgpp_EXE} CACHE STRING "Point to the parsec_ptgpp executable")

# List of files that will go through documentation generation
include(AddDocumentedFiles)

add_subdirectory(parsec)

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
if( "${CTEST_MPI_LAUNCHER}" STREQUAL "" )
    MESSAGE(WARNING "MPI tests will most likely not work: 'CTEST_MPI_LAUNCHER' is not set")
endif()
string(REPLACE " " ";" MPI_TEST_CMD_LIST "${CTEST_MPI_LAUNCHER}")
set(CTEST_GPU_LAUNCHER_OPTIONS
      "" CACHE STRING
      "Options to pass to the CTEST_MPI_LAUNCHER to select GPU nodes")

if( BUILD_TESTING )
  find_program(HOSTNAME_EXE hostname)
  MARK_AS_ADVANCED(HOSTNAME_EXE)
  # Test that the launchers work
  add_test(launcher_shm   ${SHM_TEST_CMD_LIST} ${HOSTNAME_EXE})
  if(PARSEC_HAVE_MPI)
    add_test(launcher_mpi   ${MPI_TEST_CMD_LIST} 4 ${HOSTNAME_EXE})
  endif(PARSEC_HAVE_MPI)
  if(PARSEC_HAVE_CUDA)
    add_test(launcher_gpu   ${MPI_TEST_CMD_LIST} 4 ${CTEST_GPU_LAUNCHER_OPTIONS} ${HOSTNAME_EXE})
  endif(PARSEC_HAVE_CUDA)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif( BUILD_TESTING )

add_subdirectory(docs)

string(TIMESTAMP PARSEC_COMPILE_DATE)

# Fetch GIT info for this build, if not preset for a release
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
  COMMAND git log -1 --format=%cd --date=short
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_DATE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
# Get the current working branch
if(NOT GIT_COMMIT_BRANCH)
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
# Get the status of the clone
if(NOT GIT_COMMIT_DIRTY)
execute_process(
  COMMAND git diff HEAD --shortstat
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_DIRTY)
string(STRIP "${GIT_COMMIT_DIRTY}" GIT_COMMIT_DIRTY)
if(NOT GIT_COMMIT_DIRTY)
  set(GIT_COMMIT_DIRTY "no")
endif()
endif()

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

check_type_size("parsec_lifo_t" PARSEC_LIFO_OPAQUE_SIZEOF)
include(CheckStructureFieldOffset)
# NB: the structure below *needs* to be kept in sync with the structure
# in lifo-external.h
check_structure_field_offset("lifo_private" "struct{parsec_object_t super;uint8_t alignment;parsec_list_item_t*lifo_ghost;char lifo_private;}" PARSEC_LIFO_HEAD_OFFSET)
cmake_pop_check_state()

math(EXPR PARSEC_LIFO_REMAINING_BYTES "${PARSEC_LIFO_OPAQUE_SIZEOF}-${PARSEC_LIFO_HEAD_OFFSET}")

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/lifo-external.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/parsec/class/lifo-external.h")
if(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
  message(STATUS "Internal PaRSEC uses CAS on int128_t. Reconfiguring parsec_options.h")
  configure_file (
     "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_options.h.in"
     "${PROJECT_INCLUDE_DIR}/parsec/parsec_options.h")
else()
  message(STATUS "Internal PaRSEC does not use CAS on int128_t. Keeping parsec_options.h unchanged")
endif()

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec.h
  DESTINATION include)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/runtime.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/ayudame.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/constants.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/parsec_config_bottom.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/data_distribution.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/datatype.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/profiling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/dictionary.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/mca_param.h
  DESTINATION include/parsec/utils)
#
# Never install directly the parsec/class/lifo.h header.
# Instead replace it with the lifo-external.h during the installation.
#
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/parsec/class/lifo-external.h
  DESTINATION include/parsec/class
  RENAME lifo.h)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_object.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_hash_table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/list_item.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/parsec_rwlock.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/fifo.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/barrier.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/list.h
  DESTINATION include/parsec/class)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindPARSEC.cmake
  DESTINATION ${CONFIG_INSTALL_DIR})
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
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/vpmap.h
  DESTINATION include/parsec)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/pins/pins.h
  DESTINATION include/parsec/mca/pins)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/mca/mca.h
  DESTINATION include/parsec/mca)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/os_path.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/show_help.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/utils/zone_malloc.h
    DESTINATION include/parsec/utils )

  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/dequeue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parsec/class/fifo.h
    DESTINATION include/parsec/class )

endif(PARSEC_WITH_DEVEL_HEADERS)

#
# Never install directly the parsec/include/parsec/sys/atomic.h header.
# Instead replace it with the atomic-external.h during the installation.
#
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec/sys/atomic-external.h
  DESTINATION include/parsec/sys
  RENAME atomic.h)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TBZ2")
set (CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${PARSEC_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${PARSEC_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${PARSEC_VERSION_PATCH}")
include (CPack)


set(EXTRA_CFLAGS ${CMAKE_C_FLAGS})
set(EXTRA_LDFLAGS ${CMAKE_EXE_LINKER_FLAGS})

foreach(extra_flag IN ITEMS ${PARSEC_ATOMIC_SUPPORT_OPTIONS} ${PARSEC_ARCH_OPTIONS})
  if(extra_flag MATCHES "COMPILE_LANGUAGE:C>:([^>]*)" )
    set(extra_flag "${CMAKE_MATCH_1}")
  elseif(extra_flag MATCHES "COMPILE_LANGUAGE")
    continue()
  endif()
  string(APPEND EXTRA_CFLAGS " ${extra_flag}")
endforeach()

foreach(extra_lib IN ITEMS ${PARSEC_ATOMIC_SUPPORT_LIBS} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${PAPI_LIBRARIES})
  list(APPEND EXTRA_LIBS ${extra_lib})
endforeach()

# Due to the lack of standardization detecting libraries under CMake leads
# to unconsistents lists of libraries, where some libraries are using full path,
# other a combination of "-L* and -l*" and others just the library name.
# Force a coherent EXTRA_LIBS_EXPANDED that can be used for the generation of the
# pkg-config files: leave all options starting with - as they are, and for
# everything else add "-l" if they are not an abolute path.
SET(EXTRA_LIBS_EXPANDED)
FOREACH(LIB IN ITEMS ${EXTRA_LIBS})
  if( NOT LIB MATCHES "(^|;)-.*" )
    if( NOT IS_ABSOLUTE ${LIB} )
      SET(LIB "-l${LIB}")
    endif( NOT IS_ABSOLUTE ${LIB} )
  endif( NOT LIB MATCHES "(^|;)-.*" )
  LIST(APPEND EXTRA_LIBS_EXPANDED "${LIB}")
ENDFOREACH(LIB)

if(EXTRA_LIBS)
  # For printing only: we need the duplicates in EXPANDED for static builds
  list(REMOVE_DUPLICATES EXTRA_LIBS)
endif()

MESSAGE("\n\nPkgConfig Configuration Flags:")
MESSAGE("  EXTRA_CFLAGS           = ${EXTRA_CFLAGS}")
MESSAGE("  EXTRA_LDFLAGS          = ${EXTRA_LDFLAGS}")
MESSAGE("  EXTRA_INCLUDES         = ${EXTRA_INCLUDES}")
MESSAGE("  EXTRA_LIBS             = ${EXTRA_LIBS}")
MESSAGE("\n\n")

# Generate configure files to help other software detect and integrate with
# PaRSEC. This generates the pkg-config files to slurp in all needed dependencies
# but also generates the CMake package files.

# pkg-config file
string (REPLACE ";" " " EXTRA_LIBS_EXPANDED "${EXTRA_LIBS_EXPANDED}")
if (EXTRA_INCLUDES)
  string (REPLACE ";" " -I" EXTRA_INCLUDES_STRING "-I${EXTRA_INCLUDES}")
  # Remove the last -I if any
  string (REPLACE "-I$" "" EXTRA_INCLUDES_STRING ${EXTRA_INCLUDES_STRING})
endif (EXTRA_INCLUDES)
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/parsec/include/parsec.pc.in"
  "${PROJECT_INCLUDE_DIR}/parsec.pc" @ONLY)
install(FILES "${PROJECT_INCLUDE_DIR}/parsec.pc"
        DESTINATION lib/pkgconfig)

# Export Targets
install(EXPORT parsec-targets # rules to export the targets built here
  FILE PaRSECTargets.cmake
  NAMESPACE PaRSEC::
  DESTINATION ${CONFIG_INSTALL_DIR})

# cmake config files
include(CMakePackageConfigHelpers)

configure_package_config_file(cmake_modules/PaRSECConfig.cmake.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PaRSECConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/${CONFIG_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR SYSCONFIG_INSTALL_DIR
  INCLUDE_INSTALL_DIR RUNTIME_INSTALL_DIR CONFIG_INSTALL_DIR)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParsecConfigVersion.cmake
    VERSION ${PARSEC_VERSION_MAJOR}.${PARSEC_VERSION_MINOR}.${PARSEC_VERSION_PATCH}
                                 COMPATIBILITY SameMajorVersion)

install(FILES
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PaRSECConfig.cmake
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParsecConfigVersion.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindPARSEC.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindPAPI.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindOTF2.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/FindHWLOC.cmake
  ${PROJECT_SOURCE_DIR}/cmake_modules/ParsecCompilePTG.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CONFIG_INSTALL_DIR})

# Prepare exports for submodule inclusion
set(LIB_INSTALL_DIR ${PROJECT_BINARY_DIR}/parsec)
set(SYSCONFIG_INSTALL_DIR ${PROJECT_BINARY_DIR})
set(INCLUDE_INSTALL_DIR ${PROJECT_SOURCE_DIR})
set(RUNTIME_INSTALL_DIR ${PROJECT_BINARY_DIR})
set(CONFIG_INSTALL_DIR ${PROJECT_SOURCE_DIR}/cmake_modules)

configure_package_config_file(cmake_modules/PaRSECConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/parsec-submodule-config.cmake
  INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  PATH_VARS LIB_INSTALL_DIR SYSCONFIG_INSTALL_DIR
  INCLUDE_INSTALL_DIR RUNTIME_INSTALL_DIR CONFIG_INSTALL_DIR)

# Prepare the contrib/build_with_parsec directory
configure_file (
  contrib/build_with_parsec/Makefile.in
  ${CMAKE_CURRENT_BINARY_DIR}/contrib/build_with_parsec/Makefile @ONLY)

file(GLOB_RECURSE pattern_files RELATIVE
     "${CMAKE_CURRENT_SOURCE_DIR}/" "contrib/build_with_parsec/*.[c|h]"
                                    "contrib/build_with_parsec/*.jdf")
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
