#
# Find required packages
#
find_package(PLASMA REQUIRED)
if( PLASMA_SRC )
   list(APPEND PLASMA_INCLUDE_DIR ${PLASMA_SRC})
endif( PLASMA_SRC )
list(APPEND CMAKE_REQUIRED_INCLUDES ${PLASMA_INCLUDE_DIR})
message(STATUS "Looking for control/common.h;context.h;allocate.h in ${PLASMA_INCLUDE_DIR}")
#check_include_files("control/common.h;control/context.h;control/allocate.h" FOUND_PLASMA_CONTROL)
#if( !FOUND_PLASMA_CONTROL ) 
#  message(FATAL_ERROR "Cannot find control/common.h;context.h;allocate.h
#  in the PLASMA source directory")
#endif( !FOUND_PLASMA_CONTROL )

include_directories(lib)
include_directories(${PLASMA_INCLUDE_DIR})
Add_Subdirectory(lib)
Add_Subdirectory(testing)
