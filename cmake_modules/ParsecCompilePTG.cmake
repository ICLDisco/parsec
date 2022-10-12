# Setup the minimal environment to compile .JDF files.
#

#
# This is the 'expert' function, giving access to all flags and features of
# the parsec_ptgpp program. Used by the 'simple' function below.
#
function(target_ptg_source_ex)
  set(options DEBUG LINE FORCE_PROFILE)
  set(oneValueArgs TARGET MODE SOURCE DESTINATION DESTINATION_C DESTINATION_H FUNCTION_NAME DEP_MANAGEMENT)
  set(multipleValueArgs WARNINGS IGNORE_PROPERTIES PTGPP_FLAGS)
  cmake_parse_arguments(PARSEC_PTGPP "${options}" "${oneValueArgs}"
          "${multiValueArgs}" ${ARGN} )

  if(NOT DEFINED PARSEC_PTGPP_TARGET)
    message(FATAL_ERROR "TARGET not defined in call to target_ptg_sources_ex")
  endif()
  if(NOT DEFINED PARSEC_PTGPP_MODE)
    message(FATAL_ERROR "MODE not defined in call to target_ptg_sources_ex (MODE is usually PRIVATE or PUBLIC. It's the second argument to target_sources)")
  endif()
  if(NOT DEFINED PARSEC_PTGPP_SOURCE)
    message(FATAL_ERROR "SOURCE not defined in call to target_ptg_sources_ex. Don't know what to compile with parsec-ptgpp")
  endif()

  set(target "${PARSEC_PTGPP_TARGET}")

  set(_ptgpp_flags "")
  if(DEFINED PARSEC_PTGPP_FLAGS) #Those are the global flags set at the top level CMakeLists.txt
    list(APPEND _ptgpp_flags ${PARSEC_PTGPP_FLAGS})
  endif()

  if(DEFINED PARSEC_PTGPP_PTGPP_FLAGS) #Those are optional flags passed to this function
    list(APPEND _ptgpp_flags ${PARSEC_PTGPP_PTGPP_FLAGS})
  endif()

  # By default, take SOURCE without 'jdf' as the destination root
  string(REGEX REPLACE "\\.jdf" "" inname ${PARSEC_PTGPP_SOURCE})
  string(REGEX REPLACE "^(.*/)*(.+)\\.*.*" "\\2" fnname ${inname})
  set(outname "${fnname}")

  if(DEFINED PARSEC_PTGPP_DESTINATION)
    set(outname "${PARSEC_PTGPP_DESTINATION}")
  endif()
  if(DEFINED PARSEC_PTGPP_DESTINATION_C)
    set(outname_c "${PARSEC_PTGPP_DESTINATION_C}")
  else()
    set(outname_c "${outname}.c")
  endif()
  if(DEFINED PARSEC_PTGPP_DESTINATION_H)
    set(outname_h "${PARSEC_PTGPP_DESTINATION_H}")
  else()
    set(outname_h "${outname}.h")
  endif()

  if(DEFINED PARSEC_PTGPP_FUNCTION_NAME)
    set(fnname "${PARSEC_PTGPP_FUNCTION_NAME}")
  endif()

  get_property(compile_options SOURCE ${PARSEC_PTGPP_SOURCE} PROPERTY PTGPP_COMPILE_OPTIONS)
  list(APPEND _ptgpp_flags "${compile_options}") #In case the user has set compile options specific to this source

  if(PARSEC_PTGPP_DEBUG)
    list(APPEND _ptgpp_flags "--debug")
  endif()
  if(PARSEC_PTGPP_LINE)
    list(APPEND _ptgpp_flags "--line")
  endif()
  if(PARSEC_PTGPP_FORCE_PROFILE)
    list(APPEND _ptgpp_flags "--force-profile")
  endif()
  if(PARSEC_PTGPP_DYNAMIC_TERMDET)
    list(APPEND -ptgpp_flags "--dynmic-termdet")
  endif()

  if(DEFINED PARSEC_PTGPP_DEP_MANAGEMENT)
    list(APPEND _ptgpp_flags "--dep-management;${PARSEC_PTGPP_DEP_MANAGEMENT}")
  endif()
  if(DEFINED PARSEC_PTGPP_WARNINGS)
    foreach(_flag "${PARSEC_PTGPP_WARNINGS}")
      list(APPEND _ptgpp_flags "-W${_flag}")
    endforeach()
  endif()
  if(DEFINED PARSEC_PTGPP_IGNORE_PROPERTIES)
    foreach(_prop "${PARSEC_PTGPP_IGNORE_PROPERTIES}")
      list(APPEND _ptgpp_flags "--ignore-property;${_prop}")
    endforeach()
  endif()

  # When infile is generated, it is located in the CMAKE_CURRENT_BINARY_DIR, otherwise it is
  # in the CMAKE_CURRENT_SOURCE_DIR. We use the LOCATION property to pick the right file from
  # its cmake source_file name, yet we depend on the source_file name as it is how cmake tracks it
  get_property(location SOURCE ${PARSEC_PTGPP_SOURCE} PROPERTY LOCATION)

  add_custom_command(
          OUTPUT ${outname_h} ${outname_c}
          COMMAND $<TARGET_FILE:PaRSEC::parsec-ptgpp> ${_ptgpp_flags} -E -i ${location} -C ${outname_c} -H ${outname_h} -f ${fnname}
          MAIN_DEPENDENCY ${PARSEC_PTGPP_SOURCE}
          DEPENDS ${PARSEC_PTGPP_SOURCE} PaRSEC::parsec-ptgpp)
  add_custom_target(ptgpp_${target}.${outname} DEPENDS ${outname_h} ${outname_c})

  # Copy the properties to the generated files
  get_property(cflags     SOURCE ${PARSEC_PTGPP_SOURCE} PROPERTY COMPILE_OPTIONS)
  get_property(includes   SOURCE ${PARSEC_PTGPP_SOURCE} PROPERTY INCLUDE_DIRECTORIES)
  get_property(defs       SOURCE ${PARSEC_PTGPP_SOURCE} PROPERTY COMPILE_DEFINITIONS)
  list(APPEND includes "$<$<BOOL:${PARSEC_HAVE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>")
  set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${outname_c}" "${CMAKE_CURRENT_BINARY_DIR}/${outname_h}"
          TARGET_DIRECTORY ${target}
          PROPERTIES
          GENERATED 1
          COMPILE_OPTIONS "${cflags}"
          INCLUDE_DIRECTORIES "${includes}"
          COMPILE_DEFINITIONS "${defs}")

  # make sure we produce .h before we build other .c in the target
  add_dependencies(${target} ptgpp_${target}.${outname})
  # add to the target
  target_sources(${target} ${PARSEC_PTGPP_MODE} "${CMAKE_CURRENT_BINARY_DIR}/${outname_h};${CMAKE_CURRENT_BINARY_DIR}/${outname_c}")

  get_target_property(_includes ${target} INCLUDE_DIRECTORIES)
  list(FIND _includes "${CMAKE_CURRENT_BINARY_DIR}" _i1)
  list(FIND _includes "$<$<BOOL:${PARSEC_HAVE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>" _i2)
  if( "${_i1}" EQUAL "-1" OR "${_i2}" EQUAL "-1" )
    target_include_directories(${target} ${PARSEC_PTGPP_MODE}
      ${CMAKE_CURRENT_BINARY_DIR} # set include dirs so that the target can find outname.h
      $<$<BOOL:${PARSEC_HAVE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}> # any include of outname.h will also need cuda.h atm
      )
  endif()
endfunction(target_ptg_source_ex)

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
  if( NOT TARGET PaRSEC::parsec-ptgpp )
    MESSAGE(FATAL_ERROR "parsec-ptgpp target was not built but it is required for target ${target}")
    return()
  endif( NOT TARGET PaRSEC::parsec-ptgpp )
  foreach(infile ${ARGN})
    target_ptg_source_ex(SOURCE ${infile} MODE ${mode} TARGET ${target})
  endforeach()
endfunction(target_ptg_sources)
