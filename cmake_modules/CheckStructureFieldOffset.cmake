#[=======================================================================[.rst:
CheckStructureFieldOffset
-------------

Check the (byte) offset of a field in a structure

.. command:: CHECK_STRUCTURE_FIELD_OFFSET

  .. code-block:: cmake

    CHECK_STRUCTURE_FIELD_OFFSET(STRUCTURE FIELD VARIABLE LANGUAGE <LANGUAGE>)

    Check if STRUCTURE is properly defined, and possesses FIELD
    as one of its fields, and determine the offset in bytes of
    FIELD in STRUCTURE. On return, ``HAVE_${VARIABLE}`` holds
    the existence of the type, and ``${VARIABLE}`` holds one of
    the following:

  ::

     <off>  = field is at byte <off> of the structure
     "-1"   = field location is arch-dependent (see below)
     ""     = the structure is not well defined or does not contain
              FIELD

  Both ``HAVE_${VARIABLE}`` and ``${VARIABLE}`` will be created as internal
  cache variables.

  Furthermore, the variable ``${VARIABLE}_CODE`` holds C preprocessor code
  to define the macro ``${VARIABLE}`` to the size of the type, or leave
  the macro undefined if the type does not exist.

  The variable ``${VARIABLE}`` may be ``-1`` when
  :variable:`CMAKE_OSX_ARCHITECTURES` has multiple architectures for building
  OS X universal binaries.  This indicates that the type size varies across
  architectures.  In this case ``${VARIABLE}_CODE`` contains C preprocessor
  tests mapping from each architecture macro to the corresponding type size.
  The list of architecture macros is stored in ``${VARIABLE}_KEYS``, and the
  value for each key is stored in ``${VARIABLE}-${KEY}``.

  If ``LANGUAGE`` is set, the specified compiler will be used to perform the
  check. Acceptable values are ``C`` and ``CXX``.

  The following variables may be set before calling this macro to modify
  the way the check is run:

::

  CMAKE_REQUIRED_FLAGS = string of compile command line flags
  CMAKE_REQUIRED_DEFINITIONS = list of macros to define (-DFOO=bar)
  CMAKE_REQUIRED_INCLUDES = list of include directories
  CMAKE_REQUIRED_LIBRARIES = list of libraries to link
  CMAKE_REQUIRED_QUIET = execute quietly without messages
  CMAKE_EXTRA_INCLUDE_FILES = list of extra headers to include
#]=======================================================================]

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

get_filename_component(__check_structure_field_offset_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)

include_guard(GLOBAL)

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)

#-----------------------------------------------------------------------------
# Helper function.  DO NOT CALL DIRECTLY.
function(__check_structure_field_offset_impl field structure var map language)
  if(NOT CMAKE_REQUIRED_QUIET)
    message(STATUS "Check offset of field ${field} in structure '${structure}'")
  endif()

  # Include header files.
  set(headers)
  foreach(h ${CMAKE_EXTRA_INCLUDE_FILES})
    string(APPEND headers "#include \"${h}\"\n")
  endforeach()

  # Perform the check.

  if(language STREQUAL "C")
    set(src ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckStructureFieldOffset/${var}.c)
  elseif(language STREQUAL "CXX")
    set(src ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckStructureFieldOffset/${var}.cpp)
  else()
    message(FATAL_ERROR "Unknown language:\n  ${language}\nSupported languages: C, CXX.\n")
  endif()
  set(bin ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckStructureFieldOffset/${var}.bin)
  configure_file(${__check_structure_field_offset_dir}/CheckStructureFieldOffset.c.in ${src} @ONLY)
  try_compile(HAVE_${var} ${CMAKE_BINARY_DIR} ${src}
    COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS
      "-DCOMPILE_DEFINITIONS:STRING=${CMAKE_REQUIRED_FLAGS}"
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    OUTPUT_VARIABLE output
    COPY_FILE ${bin}
    )

  if(HAVE_${var})
    # The check compiled.  Load information from the binary.
    file(STRINGS ${bin} strings LIMIT_COUNT 10 REGEX "INFO:offset")

    # Parse the information strings.
    set(regex_offset ".*INFO:offset\\[0*([^]]*)\\].*")
    set(regex_key " key\\[([^]]*)\\]")
    set(keys)
    set(code)
    set(mismatch)
    set(first 1)
    foreach(info ${strings})
      if("${info}" MATCHES "${regex_offset}")
        # Get the type size.
        set(offset "${CMAKE_MATCH_1}")
        if(first)
          set(${var} ${offset})
        elseif(NOT "${offset}" STREQUAL "${${var}}")
          set(mismatch 1)
        endif()
        set(first 0)

        # Get the architecture map key.
        string(REGEX MATCH   "${regex_key}"       key "${info}")
        string(REGEX REPLACE "${regex_key}" "\\1" key "${key}")
        if(key)
          string(APPEND code "\nset(${var}-${key} \"${offset}\")")
          list(APPEND keys ${key})
        endif()
      endif()
    endforeach()

    # Update the architecture-to-size map.
    if(mismatch AND keys)
      configure_file(${__check_structure_field_offset_dir}/CheckStructureFieldOffsetMap.cmake.in ${map} @ONLY)
      set(${var} -1)
    else()
      file(REMOVE ${map})
    endif()

    if(mismatch AND NOT keys)
      message(SEND_ERROR "CHECK_STRUCTURE_FIELD_OFFSET found different results, consider setting CMAKE_OSX_ARCHITECTURES or CMAKE_TRY_COMPILE_OSX_ARCHITECTURES to one or no architecture !")
    endif()

    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Check offset of field ${field} in structure ${structure} - done")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining offset of field ${field} in structure ${structure} passed with the following output:\n${output}\n\n")
    set(${var} "${${var}}" CACHE INTERNAL "CHECK_STRUCTURE_FIELD_OFFSET: offsetof(${structure}, ${field})")
  else()
    # The check failed to compile.
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Check offset of field ${field} in structure ${structure} - failed")
    endif()
    file(READ ${src} content)
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Determining offset of field ${field} in structure ${structure} failed with the following output:\n${output}\n${src}:\n${content}\n\n")
    set(${var} "" CACHE INTERNAL "CHECK_STRUCTURE_FIELD_OFFSET: ${field} unknown")
    file(REMOVE ${map})
  endif()
endfunction()

#-----------------------------------------------------------------------------
macro(CHECK_STRUCTURE_FIELD_OFFSET FIELD STRUCTURE VARIABLE)
  # parse arguments
  unset(doing)
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xLANGUAGE") # change to MATCHES for more keys
      set(doing "${arg}")
      set(_CHECK_STRUCTURE_FIELD_OFFSET_${doing} "")
    elseif("x${doing}" STREQUAL "xLANGUAGE")
      set(_CHECK_STRUCTURE_FIELD_OFFSET_${doing} "${arg}")
      unset(doing)
    else()
      message(FATAL_ERROR "Unknown argument:\n  ${arg}\n")
    endif()
  endforeach()
  if("x${doing}" MATCHES "^x(LANGUAGE)$")
    message(FATAL_ERROR "Missing argument:\n  ${doing} arguments requires a value\n")
  endif()
  if(DEFINED _CHECK_STRUCTURE_FIELD_OFFSET_LANGUAGE)
    if(NOT "x${_CHECK_STRUCTURE_FIELD_OFFSET_LANGUAGE}" MATCHES "^x(C|CXX)$")
      message(FATAL_ERROR "Unknown language:\n  ${_CHECK_STRUCTURE_FIELD_OFFSET_LANGUAGE}.\nSupported languages: C, CXX.\n")
    endif()
    set(_language ${_CHECK_STRUCTURE_FIELD_OFFSET_LANGUAGE})
  else()
    set(_language C)
  endif()

  unset(_CHECK_STRUCTURE_FIELD_OFFSET_LANGUAGE)

  # Compute or load the size or size map.
  set(${VARIABLE}_KEYS)
  set(_map_file ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CheckStructureFieldOffset/${VARIABLE}.cmake)
  if(NOT DEFINED HAVE_${VARIABLE})
    __check_structure_field_offset_impl(${FIELD} "${STRUCTURE}" ${VARIABLE} ${_map_file} ${_language})
  endif()
  include(${_map_file} OPTIONAL)
  set(_map_file)
  set(_builtin)

  # Create preprocessor code.
  if(${VARIABLE}_KEYS)
    set(${VARIABLE}_CODE)
    set(_if if)
    foreach(key ${${VARIABLE}_KEYS})
      string(APPEND ${VARIABLE}_CODE "#${_if} defined(${key})\n# define ${VARIABLE} ${${VARIABLE}-${key}}\n")
      set(_if elif)
    endforeach()
    string(APPEND ${VARIABLE}_CODE "#else\n# error ${VARIABLE} unknown\n#endif")
    set(_if)
  elseif(${VARIABLE})
    set(${VARIABLE}_CODE "#define ${VARIABLE} ${${VARIABLE}}")
  else()
    set(${VARIABLE}_CODE "/* #undef ${VARIABLE} */")
  endif()
endmacro()

#-----------------------------------------------------------------------------
cmake_policy(POP)
