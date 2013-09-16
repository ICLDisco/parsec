# - Find the Ayudame library
# This module finds an installed  lirary that implements the 
# Ayudame http://www.hlrs.de/organization/av/spmt/research/temanejo/
#
# This module sets the following variables:
#  AYUDAME_FOUND - set to true if a library implementing the Ayudame interface
#    is found
#  AYUDAME_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use Ayudame
#  AYUDAME_INCLUDE_DIR - directory where the Ayudame header files are
#
##########
mark_as_advanced(FORCE AYUDAME_DIR AYUDAME_INCLUDE_DIR AYUDAME_LIBRARY)
set(AYUDAME_DIR "" CACHE PATH "Root directory containing the Ayudame library")

if( AYUDAME_DIR )
  if( NOT AYUDAME_INCLUDE_DIR )
    set(AYUDAME_INCLUDE_DIR "${AYUDAME_DIR}/include")
  endif( NOT AYUDAME_INCLUDE_DIR )
  if( NOT AYUDAME_LIBRARIES )
    set(AYUDAME_LIBRARIES "${AYUDAME_DIR}/lib")
  endif( NOT AYUDAME_LIBRARIES )
endif( AYUDAME_DIR )

if( NOT AYUDAME_INCLUDE_DIR )
  set(AYUDAME_INCLUDE_DIR)
endif( NOT AYUDAME_INCLUDE_DIR )
if( NOT AYUDAME_LIBRARIES )
  set(AYUDAME_LIBRARIES)
endif( NOT AYUDAME_LIBRARIES )

list(APPEND CMAKE_REQUIRED_INCLUDES ${AYUDAME_INCLUDE_DIR})
check_include_files("Ayudame.h" FOUND_AYUDAME_INCLUDE)
if(FOUND_AYUDAME_INCLUDE)
  check_library_exists("ayudame" AYU_event ${AYUDAME_LIBRARIES} FOUND_AYUDAME_LIB)
  if( FOUND_AYUDAME_LIB )
    set(AYUDAME_LIBRARIES "-L${AYUDAME_LIBRARIES} -layudame")
  endif( FOUND_AYUDAME_LIB )
endif(FOUND_AYUDAME_INCLUDE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AYUDAME DEFAULT_MSG 
                                  AYUDAME_LIBRARIES AYUDAME_INCLUDE_DIR )


