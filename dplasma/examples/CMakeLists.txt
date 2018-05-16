if( COREBLAS_FOUND AND BUILD_DPLASMA)
    list (FIND DPLASMA_PRECISIONS "d" _index)
    if (${_index} GREATER -1)
        include_directories(. ${COREBLAS_INCLUDE_DIRS})
	    link_directories(${COREBLAS_LIBRARY_DIRS})

	    add_executable(dqr_driver "dqr_driver.c")

	    if(MPI_C_FOUND)
	        set_target_properties(dqr_driver PROPERTIES
                                  LINKER_LANGUAGE C
                                  COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}"
                                  LINK_FLAGS "${MPI_C_LINK_FLAGS} ${COREBLAS_LDFLAGS}")
	    else(MPI_C_FOUND)
	        set_target_properties(dqr_driver PROPERTIES
                                  LINKER_LANGUAGE C
                                  COMPILE_FLAGS ""
                                  LINK_FLAGS "${COREBLAS_LDFLAGS}")
        endif(MPI_C_FOUND)
	    target_link_libraries(dqr_driver dplasma ${COREBLAS_LIBRARIES})
    endif (${_index} GREATER -1)
endif( COREBLAS_FOUND AND BUILD_DPLASMA )
