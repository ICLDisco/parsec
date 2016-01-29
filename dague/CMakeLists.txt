#
#  Settings for targets
#
set(BASE_SOURCES
  class/dague_object.c
  class/dague_value_array.c
  class/dague_list.c
  class/dague_lifo.c
  class/dague_fifo.c
  class/dague_dequeue.c
  class/hash_table.c
  utils/installdirs.c
  utils/os_path.c
  utils/dague_environ.c
  utils/argv.c
  utils/output.c
  utils/show_help.c
  utils/keyval_parse.c
  utils/mca_parse_paramfile.c
  utils/cmd_line.c
  utils/mca_param_cmd_line.c
  utils/mca_param.c
  utils/zone_malloc.c
  utils/colors.c
)

FLEX_TARGET(show_help_flex utils/show_help_lex.l ${CMAKE_CURRENT_BINARY_DIR}/show_help_lex.l.c)
SET_SOURCE_FILES_PROPERTIES(${FLEX_show_help_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS}")
list(APPEND BASE_SOURCES ${FLEX_show_help_flex_OUTPUTS})

FLEX_TARGET(keyval_flex utils/keyval_lex.l ${CMAKE_CURRENT_BINARY_DIR}/keyval_lex.l.c)
SET_SOURCE_FILES_PROPERTIES(${FLEX_keyval_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS}")
list(APPEND BASE_SOURCES ${FLEX_keyval_flex_OUTPUTS})

# Read the Modular Components
#   This must be read using include and not add_subdirectory because
#   we want the mca/CMakeLists.txt to export the MCA_EXTRA_SOURCES it builds.
include(mca/CMakeLists.txt)

# Import all the available interfce
include(interfaces/superscalar/CMakeLists.txt)

# find_package(TAU REQUIRED)
# include_directories(${TAU_INCLUDE_DIRS})
# message(WARNING ${TAU_INCLUDE_DIRS})
# list(APPEND EXTRA_LIBS ${TAU_LIBRARIES})
# set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" ${TAU_LIBRARY_DIR})

set(SOURCES
  debug.c
  barrier.c
  scheduling.c
  dague.c
  data.c
  arena.c
  remote_dep.c
  bindthread.c
  mempool.c
  private_mempool.c
  vpmap.c
  devices/device.c
  mca/mca_repository.c
  debug_marks.c
  profiling.c
  ${EXTRA_SOURCES}
  ${MCA_EXTRA_SOURCES}
)

if( NOT MPI_C_FOUND )
  list(APPEND SOURCES datatype/datatype.c)
endif( NOT MPI_C_FOUND )
list(APPEND SOURCES dague_hwloc.c)

if (CUDA_FOUND)
  list(APPEND SOURCES devices/cuda/dev_cuda.c)
endif (CUDA_FOUND)

if( DAGUE_PROF_GRAPHER )
  list(APPEND SOURCES dague_prof_grapher.c)
endif( DAGUE_PROF_GRAPHER )

#
# Setup targets
#
if( BUILD_DAGUE )
  add_library(dague-base ${BASE_SOURCES})
  set_target_properties(dague-base PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  set_target_properties(dague-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if (MPI_C_FOUND)
    set_target_properties(dague-base PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
  endif (MPI_C_FOUND)
  target_link_libraries(dague-base ${EXTRA_LIBS})

  install(TARGETS dague-base
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

  add_library(dague ${SOURCES})
  set_target_properties(dague PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  if (MPI_C_FOUND)
    set_target_properties(dague PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
  endif (MPI_C_FOUND)
  target_link_libraries(dague dague-base ${EXTRA_LIBS})

  install(TARGETS dague
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

  install(FILES utils/help-mca-param.txt DESTINATION share/dague)
ENDIF( BUILD_DAGUE )

