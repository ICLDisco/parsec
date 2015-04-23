set(PINS_SOURCES "pins.c pins_init.c")
IF(PAPI_FOUND)
  SET(APPEND PINS_SOURCES pins/pins_papi_utils.c)
endIF(PAPI_FOUND)
