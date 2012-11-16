# - Find the GTG library
# This module finds an installed  lirary that implements the 
# Generic Trace Generator (GTG) (see https://gforge.inria.fr/projects/gtg/).
#
# This module sets the following variables:
#  GTG_FOUND - set to true if a library implementing the GTG interface
#    is found
#  GTG_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  GTG_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use GTG
#  GTG_INCLUDE_DIR - directory where the GTG header files are
#
##########

find_package(PkgConfig QUIET)
if( GTG_DIR )
  set(ENV{PKG_CONFIG_PATH} "${GTG_DIR}/lib/pkgconfig" $ENV{PKG_CONFIG_PATH})
endif()
pkg_check_modules(PC_GTG QUIET gtg)
set(GTG_DEFINITIONS ${PC_GTG_CFLAGS_OTHER} )

find_path(GTG_INCLUDE_DIR GTG.h
          PATH ${GTG_DIR}/include
          HINTS ${PC_GTG_INCLUDEDIR} ${PC_GTG_INCLUDE_DIRS} )

find_library(GTG_LIBRARY NAMES gtg
             PATH ${GTG_DIR}/lib
             HINTS ${PC_GTG_LIBDIR} ${PC_GTG_LIBRARY_DIRS} )

set(GTG_LIBRARIES ${GTG_LIBRARY} )
set(GTG_INCLUDE_DIRS ${GTG_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTG DEFAULT_MSG 
                                  GTG_LIBRARY GTG_INCLUDE_DIR )
mark_as_advanced(GTG_INCLUDE_DIR GTG_LIBRARY )

