include(RulesPrecisions)
# reset variables
set(generated_headers "")

### generate the dplasma headers for all possible precisions
set(HEADERS
  dplasma_z.h
)
precisions_rules(generated_headers "+;z;c;d;s" "${HEADERS}")
add_custom_target(dplasma_includes ALL DEPENDS dplasma.h SOURCES
    ${generated_headers} )

### install the dplasma headers
install(FILES dplasma.h DESTINATION include)
foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include)
endforeach()

