# DPLASMA specific option.
if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
endif()

#
# Find required packages
#
find_package(PLASMA REQUIRED)
if( PLASMA_SRC )
   list(APPEND PLASMA_INCLUDE_DIR ${PLASMA_SRC})
endif( PLASMA_SRC )

include_directories(lib)
include_directories(${PLASMA_INCLUDE_DIRS})
Add_Subdirectory(lib)
Add_Subdirectory(testing)
