
IF(PINS_ENABLE)
  set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)
ENDIF(PINS_ENABLE)
