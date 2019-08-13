# - Find DL library
# This module finds an installed  library that implements the DL
#
# This module sets the following variables:
#  DL_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  DL_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  DL_INCLUDES - paths to include files
#
#  DL::DL cmake interface library target for linking
##########

include(CMakePushCheckState)
include(CheckFunctionExists)

find_path (DL_INCLUDES dlfcn.h PATHS /usr/local/include /usr/include)

## -----------------------------------------------------------------------------
## Check for the library

find_library (DL_LIBRARIES dl PATHS /usr/local/lib /usr/lib /lib)

## -----------------------------------------------------------------------------
## Actions taken when all components have been found

if (DL_INCLUDES AND DL_LIBRARIES)
  set (HAVE_DL TRUE)
else (DL_INCLUDES AND DL_LIBRARIES)
  if (NOT DL_FIND_QUIETLY)
    if (NOT DL_INCLUDES)
      message (STATUS "Unable to find DL header files!")
    endif (NOT DL_INCLUDES)
    if (NOT DL_LIBRARIES)
      message (STATUS "Unable to find DL library files!")
    endif (NOT DL_LIBRARIES)
  endif (NOT DL_FIND_QUIETLY)
endif (DL_INCLUDES AND DL_LIBRARIES)

if (HAVE_DL)
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${DL_LIBRARIES})
  check_function_exists(dlsym DL_HAVE_DLSYM)
  cmake_pop_check_state()
endif(HAVE_DL)

if (HAVE_DL)
  #===============================================================================
  # Import Target ================================================================
  if(NOT TARGET DL::DL)
    add_library(DL::DL INTERFACE IMPORTED)
  endif(NOT TARGET DL::DL)

  set_property(TARGET DL::DL PROPERTY INTERFACE_LINK_LIBRARIES "${DL_LIBRARIES}")
  set_property(TARGET DL::DL PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${DL_INCLUDES}")
  #===============================================================================

else (HAVE_DL)
  if (DL_FIND_REQUIRED)
    message (FATAL_ERROR "Could not find DL!")
  endif (DL_FIND_REQUIRED)
endif (HAVE_DL)

mark_as_advanced(
  HAVE_DL
  DL_LIBRARIES
  DL_INCLUDES
)
