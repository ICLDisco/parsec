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
  utils/installdirs.c
  utils/os_path.c
  utils/dague_environ.c
  utils/argv.c
  utils/output.c
  utils/show_help.c
  utils/keyval_parse.c
  utils/mca_parse_paramfile.c
  utils/mca_param.c
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

if( DAGUE_PROF_TRACE )
  list(APPEND EXTRA_SOURCES profiling.c)
endif( DAGUE_PROF_TRACE )

set(SOURCES
  barrier.c
  scheduling.c
  dague.c
  data.c
  arena.c
  remote_dep.c
  debug.c
  bindthread.c
  mempool.c
  vpmap.c
  devices/device.c
  mca/mca_repository.c
  ${EXTRA_SOURCES}
  ${MCA_EXTRA_SOURCES}
)

if( HWLOC_FOUND )
  list(APPEND SOURCES dague_hwloc.c)
endif( HWLOC_FOUND )

if (CUDA_FOUND)
  list(APPEND SOURCES devices/cuda/dev_cuda.c)
endif (CUDA_FOUND)

if( DAGUE_PROF_STATS )
  list(APPEND SOURCES stats.c)
endif( DAGUE_PROF_STATS )

if( DAGUE_PROF_GRAPHER )
  list(APPEND SOURCES dague_prof_grapher.c)
endif( DAGUE_PROF_GRAPHER )



#
# Setup targets
#
if( BUILD_DAGUE )
  add_library(dague-base STATIC ${BASE_SOURCES})
  if (MPI_FOUND)
    set_target_properties(dague-base PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  else (MPI_FOUND)
    set_target_properties(dague-base PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  endif (MPI_FOUND)
  target_link_libraries(dague-base ${EXTRA_LIBS})
  install(TARGETS dague-base ARCHIVE DESTINATION lib)

  if (MPI_FOUND)
    add_library(dague-mpi STATIC ${SOURCES})
    set_target_properties(dague-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    target_link_libraries(dague-mpi dague-base ${EXTRA_LIBS})
    install(TARGETS dague-mpi ARCHIVE DESTINATION lib)
  else (MPI_FOUND)
    add_library(dague STATIC ${SOURCES})
    set_target_properties(dague PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
    target_link_libraries(dague dague-base ${EXTRA_LIBS})
    install(TARGETS dague ARCHIVE DESTINATION lib)
  endif (MPI_FOUND)
  install(FILES utils/help-mca-param.txt DESTINATION share/dague)
ENDIF( BUILD_DAGUE )

