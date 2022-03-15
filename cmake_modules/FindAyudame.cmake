# - Find the Ayudame library
# This module finds an installed  lirary that implements the 
# Ayudame http://www.hlrs.de/organization/av/spmt/research/temanejo/
#
# This module sets the following variables:
#  AYUDAME_FOUND - set to true if a library implementing the Ayudame interface
#    is found
#  AYUDAME_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use Ayudame
#  AYUDAME_INCLUDE_DIRS - directory where the Ayudame header files are
#
##########

find_library(AYUDAME_LIBRARY ayudame
  DOC "Library path for Ayudame")
find_path(AYUDAME_INCLUDE_DIR "Ayudame.h"
  DOC "Include path for Ayudame")

if(AYUDAME_LIBRARY)
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_INCLUDES ${AYUDAME_INCLUDE_DIR})
  check_include_files("Ayudame.h" AYUDAME_INCLUDE_WORKS)
  check_library_exists("ayudame" AYU_event ${AYUDAME_LIBRARY} AYUDAME_LIB_WORKS)
  cmake_pop_check_state()
endif(AYUDAME_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AYUDAME DEFAULT_MSG
  AYUDAME_LIBRARY AYUDAME_INCLUDE_DIR AYUDAME_INCLUDE_WORKS AYUDAME_LIB_WORKS)
mark_as_advanced(FORCE AYUDAME_INCLUDE_DIR AYUDAME_LIBRARY AYUDAME_INCLUDE_WORKS AYUDAME_LIB_WORKS)

if(AYUDAME_FOUND)
  set(AYUDAME_INCLUDE_DIRS ${AYUDAME_INCLUDE_DIR})
  set(AYUDAME_LIBRARIES ${AYUDAME_LIBRARY})
endif(AYUDAME_FOUND)

