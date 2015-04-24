find_package(PAPI)

set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)
IF(PAPI_FOUND)
  SET(MCA_${COMPONENT}_SOURCES ${MCA_${COMPONENT}_SOURCES} mca/pins/pins_papi_utils.c)
ENDIF(PAPI_FOUND)
