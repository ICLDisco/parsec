# - Try to find LibTAU
# Once done this will define
#  TAU_FOUND - System has TAU
#  TAU_INCLUDE_DIRS - The TAU include directories
#  TAU_LIBRARIES - The libraries needed to use TAU
#  TAU_DEFINITIONS - Compiler switches required for using TAU

find_package(PkgConfig)

# This if statement is specific to TAU, and should not be copied into other
# Find cmake scripts.
set(TAU_ROOT /home/pgaultne/sw/ig.icl.utk.edu/tau/tau-2.22)
# message(WARNING "TAU ROOT is ${TAU_ROOT}")
if(NOT TAU_ROOT AND NOT $ENV{HOME_TAU} STREQUAL "")
  set(TAU_ROOT $ENV{HOME_TAU})
endif()
if(NOT TAU_ROOT AND NOT $ENV{TAU_ROOT} STREQUAL "")
  set(TAU_ROOT $ENV{TAU_ROOT})
endif()

pkg_check_modules(PC_TAU QUIET TAU)
set(TAU_DEFINITIONS ${PC_TAU_CFLAGS_OTHER})

find_path(TAU_INCLUDE_DIR TAU.h
          HINTS ${TAU_ROOT}/include
          PATH_SUFFIXES TAU )

if (${APPLE})
  find_library(TAU_LIBRARY NAMES TAU
             HINTS ${TAU_ROOT}/x86_64/lib/shared-icpc-papi-pthread-pdt )
  find_path(TAU_LIBRARY_DIR NAMES libTAU.dylib
             HINTS ${TAU_ROOT}/x86_64/lib/shared-icpc-papi-pthread-pdt )
else()
  find_library(TAU_LIBRARY NAMES TAU
             HINTS ${TAU_ROOT}/x86_64/lib/shared-icpc-papi-pthread-pdt )
  # find_library(TAU_ICPC_LIBRARY NAMES TAUsh-icpc-papi-pthread
  #            HINTS ${TAU_ROOT}/src/Profile)
  find_path(TAU_LIBRARY_DIR NAMES libTAU.so libTAU.a libTAU.dylib
             HINTS ${TAU_ROOT}/x86_64/lib/shared-icpc-papi-pthread-pdt )
#   find_path(TAU_ICPC_LIBRARY_DIR NAMES libTAUsh-icpc-papi-pthread.so
#              HINTS ${TAU_ROOT}/src/Profile)
# #             HINTS ${TAU_ROOT}/${CMAKE_SYSTEM_PROCESSOR}/lib  ${TAU_ROOT}/*/lib )
endif()

find_path(TAU_LIBRARY_DIR_2 NAMES libTauPthreadWrap.a
             HINTS ${TAU_LIBRARY_DIR}/static-papi-pthread)

if (TAU_LIBRARY_DIR_2_FOUND)
   message(ERROR "I don't think this is the one we want, because it's static...")
    set(TAU_LIBRARIES ${TAU_LIBRARY} -lpthread -L${TAU_LIBRARY_DIR_2} -Wl,-wrap,pthread_create -Wl,-wrap,pthread_join -Wl,-wrap,pthread_exit -Wl,-wrap,pthread_barrier_wait -lTauPthreadWrap )
else()
    set(TAU_LIBRARIES ${TAU_LIBRARY} ${TAU_ICPC_LIBRARY} -lpthread -ldl -lpapi)
    message(STATUS "The TAU libraries are ${TAU_LIBRARIES}")
endif()

set(TAU_INCLUDE_DIRS ${TAU_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TAU_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(TAU  DEFAULT_MSG
                                  TAU_LIBRARY TAU_INCLUDE_DIR TAU_LIBRARIES )

mark_as_advanced(TAU_INCLUDE_DIR TAU_LIBRARY TAU_LIBRARIES )
set(TAU_DIR ${TAU_ROOT})
