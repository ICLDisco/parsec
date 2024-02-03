# Find Graphviz (http://www.graphviz.org)
# Defines
# GRAPHVIZ_INCLUDE_DIRS
# GRAPHVIZ_LIBRARIES
# GRAPHVIZ_FOUND

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the GRAPHVIZ_DIR or GRAPHVIZ_PKG_DIR
if(GRAPHVIZ_DIR)
  if(NOT G_PKG_DIR)
    set(GRAPHVIZ_PKG_DIR "${GRAPHVIZ_DIR}/lib/pkgconfig")
  endif(NOT GRAPHVIZ_PKG_DIR)
endif(GRAPHVIZ_DIR)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  set(ENV{PKG_CONFIG_PATH} "${GRAPHVIZ_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(GRAPHVIZ libgvc)
endif(PKG_CONFIG_FOUND)

find_path(Graphviz_INCLUDE_DIR NAMES graphviz/gvc.h HINTS Graphviz_INCLUDE_DIRS)

find_library(GRAPHVIZ_gvc_LIBRARY NAMES gvc HINTS GRAPHVIZ_LIBRARY_DIRS)
find_library(GRAPHVIZ_graph_LIBRARY NAMES cgraph HINTS GRAPHVIZ_LIBRARY_DIRS)
if( NOT GRAPHVIZ_graph_LIBRARY_FOUND )
    find_library(GRAPHVIZ_graph_LIBRARY NAMES graph HINTS GRAPHVIZ_LIBRARY_DIRS)
endif( NOT GRAPHVIZ_graph_LIBRARY_FOUND )
find_library(GRAPHVIZ_cdt_LIBRARY NAMES cdt HINTS GRAPHVIZ_LIBRARY_DIRS)
find_library(GRAPHVIZ_pathplan_LIBRARY NAMES pathplan HINTS GRAPHVIZ_LIBRARY_DIRS)

set(Graphviz_LIBRARY ${GRAPHVIZ_gvc_LIBRARY} ${GRAPHVIZ_graph_LIBRARY} ${GRAPHVIZ_cdt_LIBRARY} ${GRAPHVIZ_pathplan_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Graphviz DEFAULT_MSG Graphviz_LIBRARY Graphviz_INCLUDE_DIR)

if(Graphviz_FOUND)
    set(Graphviz_LIBRARIES ${Graphviz_LIBRARY})
    set(Graphviz_INCLUDE_DIRS ${Graphviz_INCLUDE_DIR} ${Graphviz_INCLUDE_DIR}/graphviz)
else(Graphviz_FOUND)
    set(Graphviz_LIBRARIES)
    set(Graphviz_INCLUDE_DIRS)
endif(Graphviz_FOUND)

mark_as_advanced(Graphviz_INCLUDE_DIRS Graphviz_LIBRARIES)
