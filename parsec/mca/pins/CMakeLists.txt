
IF(PARSEC_PROF_PINS)
  set(MCA_${COMPONENT}_SOURCES mca/pins/pins.c mca/pins/pins_init.c)
ENDIF(PARSEC_PROF_PINS)
set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/pins/pins.h)
