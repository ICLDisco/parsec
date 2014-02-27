include(RulesPrecisions)
include(AddDocumentedFiles)

# reset variables
set(generated_headers "")

### generate the dplasma headers for all possible precisions
precisions_rules_py(generated_headers
                 "dplasma_z.h;dplasma_zrecursive.h"
                 PRECISIONS "z;c;d;s")

add_custom_target(dplasma_includes ALL DEPENDS dplasma.h SOURCES
    ${generated_headers} )

### install the dplasma headers
install(FILES
  dplasma.h
  dplasma_qr_param.h
  DESTINATION include)

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include)
endforeach()

add_documented_files( PARSEC_ALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/" dplasma.h dplasma_qr_param.h )

