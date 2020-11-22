#include(RulesJDF)
include(ParsecCompilePTG)

include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
if(NOT PARSEC_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
endif(NOT PARSEC_COMPILE_INPLACE)

set(sources
    ${CMAKE_CURRENT_LIST_DIR}/redistribute_wrapper.c
    ${CMAKE_CURRENT_LIST_DIR}/redistribute_dtd.c)

if(NOT TARGET parsec-data-redistribute)
  add_library(parsec-data-redistribute OBJECT ${sources})
endif(NOT TARGET parsec-data-redistribute)

set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/redistribute.jdf"
                    "${CMAKE_CURRENT_SOURCE_DIR}/redistribute_reshuffle.jdf"
             APPEND PROPERTY PTGPP_COMPILE_OPTIONS "--Wremoteref")

target_ptg_sources(parsec-data-redistribute PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/redistribute.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_reshuffle.jdf")

set_target_properties(parsec-data-redistribute PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
# Some versions of the XLC compiler generate incorrect aliasing code for the PTG generated code,
# where we cast a parsec_task_t structure into more specialized task classes.
# Restricting the optimizations to avoid the pointer aliasing solves this problem.
string(REGEX MATCH "XL" _match_xlc ${CMAKE_C_COMPILER_ID})
if( _match_xlc )
  set_target_properties(parsec-data-redistribute PROPERTIES COMPILE_FLAGS -qalias=noansi)
endif( )
target_include_directories(parsec-data-redistribute PUBLIC ${MPI_C_INCLUDE_PATH})

install(FILES
  redistribute_internal.h
  DESTINATION include/parsec/data_dist/matrix/redistribute )
