# For now assume that the upper level did the CUDA search and that all
# necessary dependencies on CUDA have been correctly setup

if( PARSEC_HAVE_HIP )
  SET(MCA_${COMPONENT}_${MODULE} ON)
  FILE(GLOB cuda_sources ${MCA_BASE_DIR}/${COMPONENT}/cuda/[^\\.]*.[ch])
  find_package(Perl REQUIRED)
  find_program(HIPIFY_PERL_COMMAND NAMES hipify-perl HINTS ${HIP_BIN_INSTALL_DIR} REQUIRED)
  set(module_dir "mca/${COMPONENT}/${MODULE}")
  foreach(cuda_file ${cuda_sources})
    file(RELATIVE_PATH cuda_filename ${MCA_BASE_DIR}/${COMPONENT}/cuda ${cuda_file})
    string(REPLACE cuda ${MODULE} ${MODULE}_file ${cuda_filename})
    string(PREPEND ${MODULE}_file "${CMAKE_CURRENT_BINARY_DIR}/${module_dir}/")
    add_custom_command(OUTPUT ${${MODULE}_file}
                       DEPENDS ${cuda_file} # do not use MAIN_DEPENDENCY, that overides the default .c.o rule
                       COMMAND ${CMAKE_COMMAND} -E copy "${cuda_file}" "${${MODULE}_file}.prehip"
                       COMMAND ${PERL_EXECUTABLE} ${HIPIFY_PERL_COMMAND} --inplace --print-stats "${${MODULE}_file}"
                       COMMAND ${PERL_EXECUTABLE} -i -pe "s{(cuda)}{ substr uc hip | (uc \$1 ^ \$1), 0, 3 }egi" "${${MODULE}_file}" VERBATIM) # Convert all remaining cuda/CUDA
    if(${${MODULE}_file} MATCHES [^\\.]*.c) # do not add .h to sources
      list(APPEND MCA_${COMPONENT}_${MODULE}_SOURCES ${${MODULE}_file})
    endif()
  endforeach()
  SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  install(FILES
          ${CMAKE_CURRENT_BINARY_DIR}/${module_dir}/${COMPONENT}_${MODULE}.h
          ${CMAKE_CURRENT_BINARY_DIR}/${module_dir}/${COMPONENT}_${MODULE}_internal.h
          DESTINATION include/parsec/mca/${COMPONENT}/${MODULE} )
else (PARSEC_HAVE_HIP)
  MESSAGE(STATUS "Module ${MODULE} not selectable: does not have HIP")
  SET(MCA_${COMPONENT}_${MODULE} OFF)
endif(PARSEC_HAVE_HIP)
