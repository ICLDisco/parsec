#
#  Settings for targets
#
set(BASE_SOURCES
  class/parsec_dequeue.c
  class/parsec_fifo.c
  class/parsec_lifo.c
  class/parsec_list.c
  class/parsec_object.c
  class/parsec_value_array.c
  class/parsec_hash_table.c
  class/parsec_rwlock.c
  utils/argv.c
  utils/cmd_line.c
  utils/colors.c
  utils/parsec_environ.c
  utils/installdirs.c
  utils/keyval_parse.c
  utils/mca_param.c
  utils/mca_param_cmd_line.c
  utils/mca_parse_paramfile.c
  utils/os_path.c
  utils/output.c
  utils/show_help.c
  utils/zone_malloc.c
  utils/atomic_external.c
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

# Import all the available interfaces
include(interfaces/CMakeLists.txt)

# find_package(TAU REQUIRED)
# include_directories(${TAU_INCLUDE_DIRS})
# message(WARNING ${TAU_INCLUDE_DIRS})
# list(APPEND EXTRA_LIBS ${TAU_LIBRARIES})
# set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" ${TAU_LIBRARY_DIR})

set(SOURCES
  arena.c
  barrier.c
  bindthread.c
  parsec.c
  data.c
  data_distribution.c
  debug.c
  debug_marks.c
  devices/device.c
  mca/mca_repository.c
  mempool.c
  private_mempool.c
  profiling.c
  remote_dep.c
  scheduling.c
  vpmap.c
  maxheap.c
  hbbuffer.c
  datarepo.c
  ${EXTRA_SOURCES}
  ${MCA_EXTRA_SOURCES}
)

if( NOT MPI_C_FOUND )
  list(APPEND SOURCES datatype/datatype.c)
else( NOT MPI_C_FOUND )
  list(APPEND SOURCES datatype/datatype_mpi.c)
endif( NOT MPI_C_FOUND )
list(APPEND SOURCES parsec_hwloc.c)

if (CUDA_FOUND)
  list(APPEND SOURCES devices/cuda/dev_cuda.c)
endif (CUDA_FOUND)

if( PARSEC_PROF_GRAPHER )
  list(APPEND SOURCES parsec_prof_grapher.c)
endif( PARSEC_PROF_GRAPHER )

#
# Setup targets
#
if( BUILD_PARSEC )
  add_library(parsec-base ${BASE_SOURCES})
  set_target_properties(parsec-base PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  set_target_properties(parsec-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
  #  if (MPI_C_FOUND)
  #  set_target_properties(parsec-base PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
  #endif (MPI_C_FOUND)
  #  target_link_libraries(parsec-base ${EXTRA_LIBS})

  install(TARGETS parsec-base
          DESTINATION lib)

  add_library(parsec ${SOURCES})
  set_target_properties(parsec PROPERTIES COMPILE_FLAGS "-DYYERROR_VERBOSE")
  if (MPI_C_FOUND)
    set_target_properties(parsec PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
  endif (MPI_C_FOUND)
  target_link_libraries(parsec parsec-base ${EXTRA_LIBS})

  install(TARGETS parsec
          DESTINATION lib)

  install(FILES utils/help-mca-param.txt DESTINATION share/parsec)
ENDIF( BUILD_PARSEC )

add_subdirectory(data_dist)
