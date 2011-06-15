include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

### generate the dplasma_cores headers for all possible precisions
set(HEADERS
  core_z.h
)
precisions_rules(generated_headers "+;z;c;d;s" "${HEADERS}") 

### Generate the dplasma wrappers for all required precisions 
set(SOURCES
  core_ztrdv.c
)
precisions_rules(generated_files "+;${DPLASMA_PRECISIONS}" "${SOURCES}") 

### Generate the lib 
add_library(dplasma_cores 
  ${generated_headers}
  ${generated_files})
install(TARGETS dplasma_cores ARCHIVE DESTINATION lib)

install(FILES core_z.h DESTINATION include/cores)
foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
