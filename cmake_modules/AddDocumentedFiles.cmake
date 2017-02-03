#
# See https://cmake.org/pipermail/cmake/2010-March/035992.html
#
function(add_documented_files)
  get_property(is_defined GLOBAL PROPERTY PARSEC_DOX_SRCS DEFINED)
  if(NOT is_defined)
    define_property(GLOBAL PROPERTY PARSEC_DOX_SRCS
      BRIEF_DOCS "List of source documented source files"
      FULL_DOCS "List of source files to be included into the in-code documentation")
  endif()
  # make absolute paths
  set(SRCS)
  foreach(s IN LISTS ARGN)
    if(NOT IS_ABSOLUTE "${s}")
      get_filename_component(s "${s}" ABSOLUTE)
    endif()
    list(APPEND SRCS "${s}")
  endforeach()
  # append to global list
  set_property(GLOBAL APPEND PROPERTY PARSEC_DOX_SRCS "${SRCS}")
endfunction(add_documented_files)
