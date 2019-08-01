#
# See https://cmake.org/pipermail/cmake/2010-March/035992.html
#
function(add_documented_files)
  set(options)
  set(oneValueArgs DIR PROJECT)
  set(multiValueArgs FILES)
  cmake_parse_arguments(PARSE_ARGV 0 "ADF" "${options}" "${oneValueArgs}" "${multiValueArgs}")

  if("${ADF_DIR}" STREQUAL "")
    set(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  else()
    set(BASE_DIR "${ADF_DIR}")
  endif()
  
  if("${ADF_PROJECT}" STREQUAL "" OR "${ADF_FILES}" STREQUAL "")
    MESSAGE(FATAL_ERROR "add_documented_files: need to specify PROJECT (current: '${ADF_PROJECT}') and FILES (current: '${ADF_FILES}')")
  else()
    get_property(is_defined GLOBAL PROPERTY ${ADF_PROJECT}_DOX_SRCS DEFINED)
    if(NOT is_defined)
      define_property(GLOBAL PROPERTY ${ADF_PROJECT}_DOX_SRCS
	BRIEF_DOCS "List of source documented source files"
	FULL_DOCS "List of source files to be included into the in-code documentation")
    endif()
    # make absolute paths
    set(SRCS)
    foreach(s IN LISTS ADF_FILES)
      if(NOT IS_ABSOLUTE "${s}")
	get_filename_component(s "${s}" ABSOLUTE BASE_DIR "${BASE_DIR}")
      endif()
      list(APPEND SRCS "${s}")
    endforeach()
    # append to global list
    set_property(GLOBAL APPEND PROPERTY ${ADF_PROJECT}_DOX_SRCS "${SRCS}")
  endif()
endfunction(add_documented_files)
