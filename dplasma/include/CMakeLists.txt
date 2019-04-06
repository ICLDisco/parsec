include(PrecisionGenerator)

# reset variables
set(generated_headers "")

### generate the dplasma headers for all required precisions
precisions_rules_py(generated_headers
                    "dplasma/dplasma_z.h"
                    PRECISIONS "${DPLASMA_PRECISIONS}"
                    TARGETDIR "dplasma")

set(DPLASMA_INCLUDE_STRING "")
foreach(generated_header ${generated_headers})
  set(DPLASMA_INCLUDE_STRING "${DPLASMA_INCLUDE_STRING}\n
  #include \"${generated_header}\"")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header}
          DESTINATION ${DPLASMA_INSTALL_PREFIX}/include/dplasma)
endforeach()

configure_file (
   "${CMAKE_CURRENT_SOURCE_DIR}/dplasma.h.in"
   "${CMAKE_CURRENT_BINARY_DIR}/dplasma.h" @ONLY)

add_custom_target(dplasma_includes ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dplasma.h
                  SOURCES ${generated_headers} )

### install the dplasma headers
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/dplasma.h
  DESTINATION ${DPLASMA_INSTALL_PREFIX}/include)
install(FILES
  dplasma/complex.h
  dplasma/qr_param.h
  dplasma/types.h
  DESTINATION ${DPLASMA_INSTALL_PREFIX}/include/dplasma)


