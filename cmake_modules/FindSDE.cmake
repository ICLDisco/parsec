# - Find the SDE library (part of PAPI)
# This module finds an installed  lirary that implements the
# Softward Defined Events interface, part of the PAPI Project
# (see http://icl.cs.utk.edu/papi/).
#
# This module defines the PAPI::SDE target.
##########

find_path(SDE_INCLUDE_DIR sde_lib.h
          PATHS ${PAPI_ROOT}/include ENV PAPI_INCLUDE_DIR
          DOC "Include path for PAPI SDE")

find_library(SDE_LIBRARY NAMES sde
             HINTS ${PAPI_ROOT}/lib ENV PAPI_LIBRARY_DIR
             DOC "Library path for PAPI SDE")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SDE DEFAULT_MSG
                                  SDE_LIBRARY SDE_INCLUDE_DIR )
if( SDE_FOUND )
    message(STATUS "SDE Library found at ${SDE_INCLUDE_DIR} ${SDE_LIBRARY}")

    #===============================================================================
    # Importing PAPI SDE as a cmake target
    if(NOT TARGET PAPI::SDE)
      add_library(PAPI::SDE INTERFACE IMPORTED)
    endif()

    set_property(TARGET PAPI::SDE APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${SDE_LIBRARY}")
    set_property(TARGET PAPI::SDE PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SDE_INCLUDE_DIR}")
  #===============================================================================

  set(SDE_LIBRARIES ${SDE_LIBRARY} )
  set(SDE_INCLUDE_DIRS ${SDE_INCLUDE_DIR} )
  mark_as_advanced(FORCE SDE_INCLUDE_DIR SDE_LIBRARY)

endif( SDE_FOUND )
