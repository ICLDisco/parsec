# For now assume that the upper level did the CUDA search and that all
# necessary dependencies on CUDA have been correctly setup

if( CUDA_FOUND )
  SET(MCA_${COMPONENT}_${MODULE} ON)
  FILE(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/devices/cuda/dev_cuda.h
          DESTINATION include/parsec/mca/device/cuda )
else (CUDA_FOUND)
  MESSAGE(STATUS "Module ${MODULE} not selectable: does not have CUDA")
  SET(MCA_${COMPONENT}_${MODULE} OFF)
endif(CUDA_FOUND)
