# For now assume that the upper level did the CUDA search and that all
# necessary dependencies on CUDA have been correctly setup

if( PARSEC_HAVE_LEVEL_ZERO AND PARSEC_HAVE_DPCPP )
  SET(MCA_${COMPONENT}_${MODULE} ON)
  FILE(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/parsec/mca/device/level_zero)
  FILE(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  ADD_CUSTOM_COMMAND(OUTPUT ${PROJECT_BINARY_DIR}/parsec/mca/device/level_zero/device_level_zero_dpcpp_interface.o
          MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/parsec/mca/device/level_zero/device_level_zero_dpcpp_interface.cpp
          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/parsec/mca/device/level_zero
          COMMAND  ${DPCPP_EXECUTABLE}
                         -I${LEVEL_ZERO_INCLUDE_DIR}
                         -I${SYCL_INCLUDE_DIR}
                         -I${PROJECT_SOURCE_DIR}
                         -I${PROJECT_SOURCE_DIR}/parsec/include
                         -I${PROJECT_BINARY_DIR}/parsec/include
                         -I${PROJET_BINARY_DIR}
                         -Wall -fpic -g -c ${PROJECT_SOURCE_DIR}/parsec/mca/device/level_zero/device_level_zero_dpcpp_interface.cpp)
  SET_SOURCE_FILES_PROPERTIES(
          ${PROJECT_BINARY_DIR}/parsec/mca/device/level_zero/device_level_zero_dpcpp_interface.o
          PROPERTIES
          EXTERNAL_OBJECT true
          GENERATED true)
  add_library(parsec-dpcpp ${PROJECT_BINARY_DIR}/parsec/mca/device/level_zero/device_level_zero_dpcpp_interface.o)
  target_link_libraries(parsec PRIVATE parsec-dpcpp ${SYCL_LIBRARY})
  install(TARGETS parsec-dpcpp
          EXPORT parsec-targets
          DESTINATION ${PARSEC_INSTALL_LIBDIR})
  SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/level_zero/device_level_zero.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/level_zero/device_level_zero_internal.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/level_zero/device_level_zero_dpcpp.h
          DESTINATION include/parsec/mca/device/level_zero )
else (PARSEC_HAVE_LEVEL_ZERO AND PARSEC_HAVE_DPCPP)
  MESSAGE(STATUS "Module ${MODULE} not selectable: does not have LEVEL_ZERO (${PARSEC_HAVE_LEVEL_ZERO}) or DPCPP (${PARSEC_HAVE_DPCPP})")
  SET(MCA_${COMPONENT}_${MODULE} OFF)
endif(PARSEC_HAVE_LEVEL_ZERO AND PARSEC_HAVE_DPCPP)
