#
#  Settings for targets
#
set(BASE_SOURCES
  class/dague_object.c
  class/dague_value_array.c
  class/dague_list.c
  class/dague_lifo.c
  class/dague_dequeue.c
  utils/os_path.c
  utils/installdirs.c
  utils/dague_environ.c
  utils/argv.c
  utils/output.c
  utils/show_help.c
  utils/mca_param.c
)

set(SOURCES 
  barrier.c 
  scheduling.c
  schedulers.c 
  dague.c
  data.c
  arena.c
  moesi.c
  remote_dep.c 
  debug.c 
  bindthread.c 
  mempool.c 
  vpmap.c
${EXTRA_SOURCES} 
)

if( HWLOC_FOUND )
  list(APPEND SOURCES dague_hwloc.c)
endif( HWLOC_FOUND )

if (CUDA_FOUND)
  list(APPEND SOURCES gpu_data.c)
endif (CUDA_FOUND)
if (PAPI_FOUND)
  list(APPEND SOURCES papime.c)
endif (PAPI_FOUND)
 
if( DAGUE_PROF_TRACE )
  list(APPEND SOURCES profiling.c)
endif( DAGUE_PROF_TRACE )

if( DAGUE_PROF_STATS )
  list(APPEND SOURCES stats.c)
endif( DAGUE_PROF_STATS )

if( DAGUE_PROF_GRAPHER )
  list(APPEND SOURCES dague_prof_grapher.c)
endif( DAGUE_PROF_GRAPHER )

FLEX_TARGET(show_help_flex utils/show_help_lex.l ${CMAKE_CURRENT_BINARY_DIR}/show_help_lex.l.c)
SET_SOURCE_FILES_PROPERTIES(${FLEX_show_help_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS}")
list(APPEND SOURCES ${FLEX_show_help_flex_OUTPUTS})

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
ENDIF( BUILD_DAGUE )

