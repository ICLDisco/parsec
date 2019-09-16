# Setup the minimal environment to compile .JDF files.
#

#
# This function adds the .c, .h files generated from .jdf input files
# passed in ARGN as source files to the target ${target}.
# The include directory is also set so that the generated .h files 
# can be found with the visibility provided in ${mode}.
#
# If the JDF file is set with some COMPILE_OPTIONS, INCLUDE_DIRECTORIES
# COMPILE_DEFINITIONS properties, these are forwarded to the generated .c/.h files.
#
# Each jdf file can also be tagged with specific flags for the parsec_ptgpp
# binary through the PTGPP_COMPILE_OPTIONS property.
#
function(target_ptg_sources target mode)

  foreach(infile ${ARGN})
    # Remove .jdf if present
    string(REGEX REPLACE "\\.jdf" "" inname ${infile})
    string(REGEX REPLACE "^(.*/)*(.+)\\.*.*" "\\2" fnname ${inname})
    set(outname "${fnname}")
    get_property(compile_options SOURCE ${infile} PROPERTY PTGPP_COMPILE_OPTIONS)
    get_property(location SOURCE ${infile} PROPERTY LOCATION)

    #get_source_file_property(generated ${infile} GENERATED)
    #message(STATUS "Working on ${infile} which is ${generated} with location ${location}") 

    # When infile is generated, it is located in the CMAKE_CURRENT_BINARY_DIR, otherwise it is
    # in the CMAKE_CURRENT_SOURCE_DIR. We use the LOCATION property to pick the right file from
    # its cmake source_file name, yet we depend on the source_file name as it is how cmake tracks it
    add_custom_command(
        OUTPUT ${outname}.h ${outname}.c
        COMMAND $<TARGET_FILE:PaRSEC::parsec_ptgpp> ${PARSEC_PTGFLAGS} ${compile_options} -E -i ${location} -o ${outname} -f ${fnname}
        MAIN_DEPENDENCY ${infile}
        DEPENDS ${infile} PaRSEC::parsec_ptgpp)

    # Copy the properties to the generated files
    get_property(cflags     SOURCE ${infile} PROPERTY COMPILE_OPTIONS)
    get_property(includes   SOURCE ${infile} PROPERTY INCLUDE_DIRECTORIES)
    get_property(defs       SOURCE ${infile} PROPERTY COMPILE_DEFINITIONS)
    list(APPEND includes "$<$<BOOL:${PARSEC_HAVE_CUDA}>:${CUDA_INCLUDE_DIRS}>")
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${outname}.c" PROPERTIES
                                    COMPILE_OPTIONS "${cflags}"
                                    INCLUDE_DIRECTORIES "${includes}"
                                    COMPILE_DEFINITIONS "${defs}")

    # add to the target
    target_sources(${target} ${mode} "${CMAKE_CURRENT_BINARY_DIR}/${outname}.h;${CMAKE_CURRENT_BINARY_DIR}/${outname}.c")
  endforeach()
  target_include_directories(${target} ${mode} 
    ${CMAKE_CURRENT_BINARY_DIR} # set include dirs so that the target can find outname.h
    $<$<BOOL:${PARSEC_HAVE_CUDA}>:${CUDA_INCLUDE_DIRS}> # any include of outname.h will also need cuda.h atm
  )
endfunction(target_ptg_sources)
