find_package(PAPI)

IF(PINS_ENABLE)
  set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)

  IF(PAPI_FOUND)
    list(APPEND EXTRA_LIBS ${PAPI_LIBRARIES})
    include_directories( ${PAPI_INCLUDE_DIRS} )
    SET(MCA_${COMPONENT}_SOURCES ${MCA_${COMPONENT}_SOURCES} mca/pins/pins_papi_utils.c)
  ENDIF(PAPI_FOUND)
ENDIF(PINS_ENABLE)
