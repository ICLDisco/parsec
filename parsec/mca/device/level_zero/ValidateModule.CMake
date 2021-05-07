# For now assume that the upper level did the CUDA search and that all
# necessary dependencies on CUDA have been correctly setup

if( PARSEC_HAVE_LEVEL_ZERO )
  SET(MCA_${COMPONENT}_${MODULE} ON)
  FILE(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/level_zero/device_level_zero.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/level_zero/device_level_zero_internal.h
          DESTINATION include/parsec/mca/device/level_zero )
else (PARSEC_HAVE_LEVEL_ZERO)
  MESSAGE(STATUS "Module ${MODULE} not selectable: does not have LEVEL_ZERO")
  SET(MCA_${COMPONENT}_${MODULE} OFF)
endif(PARSEC_HAVE_LEVEL_ZERO)
