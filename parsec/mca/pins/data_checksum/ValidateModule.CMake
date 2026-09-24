#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#
if (PARSEC_PROF_PINS)
  set(MCA_${COMPONENT}_${MODULE} ON)
  file(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  set(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  if (NOT PARSEC_DEBUG_PARANOID)
    message(STATUS "Module ${MODULE} built inactive: requires PARSEC_DEBUG_PARANOID.")
  endif (NOT PARSEC_DEBUG_PARANOID)
else (PARSEC_PROF_PINS)
  message(STATUS "Module ${MODULE} not enabled: PINS disabled.")
  set(MCA_${COMPONENT}_${MODULE} OFF)
endif (PARSEC_PROF_PINS)
