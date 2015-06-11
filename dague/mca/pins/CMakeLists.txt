
IF(PINS_ENABLE)
  set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)
  
  IF(DAGUE_PROF_TRACE AND DAGUE_PROF_PAPI)
    find_package(PAPI)
    IF(PAPI_FOUND)
        SET(MCA_${COMPONENT}_SOURCES ${MCA_${COMPONENT}_SOURCES} mca/pins/pins_papi_utils.c)
        list(APPEND EXTRA_LIBS ${PAPI_LIBRARY})
        include_directories( ${PAPI_INCLUDE_DIR} )
    ENDIF(PAPI_FOUND)
  ENDIF(DAGUE_PROF_TRACE AND DAGUE_PROF_PAPI)
ENDIF(PINS_ENABLE)
