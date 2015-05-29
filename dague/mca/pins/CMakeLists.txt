find_package(PAPI)

IF(PINS_ENABLE)
  set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)

  IF(PAPI_FOUND AND DAGUE_PROF_TRACE)
    SET(MCA_${COMPONENT}_SOURCES ${MCA_${COMPONENT}_SOURCES} mca/pins/pins_papi_utils.c)
    list(APPEND EXTRA_LIBS ${PAPI_LIBRARY})
    include_directories( ${PAPI_INCLUDE_DIR} )
  ENDIF(PAPI_FOUND AND DAGUE_PROF_TRACE)
ENDIF(PINS_ENABLE)
