include (CheckCCompilerFlag)
include (CheckCXXCompilerFlag)
include (CheckFortranCompilerFlag)

#
# This is a convenience function that will check a given option
# for a list of target languages, and add this option in the
# compile options (for that language) if it exists, optionally
# via a user-overwritable cache option
#
# Parameters:
#   OPTION:                mandatory, compile option to check
#   NAME:                  mandatory, name of the test and name of the
#                          user-overwritable cache option to be defined
#                          if a comment is provided.
#   COMMENT:               optional, if a comment is provided after this
#                          parameter, a user-overwritable CMake cache option
#                          with name NAME is defined, and COMMENT is used as
#                          a comment for this option.
#                          That CMake option is marked as masked.
#   LANGUAGES              optional, the list of languages to test.
#                          If empty, all enabled languages are tested.
#   CONFIG                 optional, restricts this flag to a specific
#                          configuration (e.g. RelWithDebInfo)
function(check_and_set_compiler_option)
  set(options "")
  set(oneValueArgs NAME COMMENT OPTION CONFIG)
  set(multipleValueArgs LANGUAGES)
  cmake_parse_arguments(parsec_cas_co "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT DEFINED parsec_cas_co_OPTION)
    foreach(v NAME COMMENT LANGUAGES OPTION)
      message(STATUS "parsec_cas_co_${v} = ${parsec_cas_co_${v}}")
    endforeach()
    message(FATAL_ERROR "OPTION not defined in call to check_and_set_compiler_option")
  endif()

  if(NOT DEFINED parsec_cas_co_NAME)
    message(FATAL_ERROR "NAME not defined in call to check_and_set_compiler_option")
  endif()

  if(NOT DEFINED parsec_cas_co_LANGUAGES)
    set(parsec_cas_co_LANGUAGES C CXX Fortran)
  endif()

  message(CHECK_START "Performing Test for flag ${parsec_cas_co_OPTION}    ")
  set(save_CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET})
  set(CMAKE_REQUIRED_QUIET true)
  set(_supported_languages "")

  foreach(parsec_cas_co_LANGUAGE ${parsec_cas_co_LANGUAGES})
    if(CMAKE_${parsec_cas_co_LANGUAGE}_COMPILER_WORKS)
      message(TRACE "Checking flag ${parsec_cas_co_OPTION} for language ${parsec_cas_co_LANGUAGE}")
      cmake_language(CALL "check_${parsec_cas_co_LANGUAGE}_compiler_flag" "${parsec_cas_co_OPTION}" "${parsec_cas_co_NAME}_${parsec_cas_co_LANGUAGE}" )
      if( "${${parsec_cas_co_NAME}_${parsec_cas_co_LANGUAGE}}" )
        set(_supported_languages "${parsec_cas_co_LANGUAGE};${_supported_languages}")
        message(TRACE "Flag ${parsec_cas_co_OPTION} supported by ${parsec_cas_co_LANGUAGE}")
        if(DEFINED parsec_cas_co_CONFIG)
          list(APPEND parsec_cas_co_defined_options "$<$<CONFIG:${parsec_cas_co_CONFIG}>:$<$<COMPILE_LANGUAGE:${parsec_cas_co_LANGUAGE}>:${parsec_cas_co_OPTION}>>")
        else()
          list(APPEND parsec_cas_co_defined_options "$<$<COMPILE_LANGUAGE:${parsec_cas_co_LANGUAGE}>:${parsec_cas_co_OPTION}>")
        endif()
      endif()
    endif()
  endforeach()
  set(CMAKE_REQUIRED_QUIET ${save_CMAKE_REQUIRED_QUIET})
  if( _supported_languages STREQUAL "" )
    message(CHECK_FAIL "[none]")
  else( _supported_languages STREQUAL "" )
    message(CHECK_PASS "[${_supported_languages}]")
  endif( _supported_languages STREQUAL "" )

  if(DEFINED parsec_cas_co_COMMENT)
    set(${parsec_cas_co_NAME} "${parsec_cas_co_defined_options}" CACHE STRING "${parsec_cas_co_COMMENT}")
    mark_as_advanced(${parsec_cas_co_NAME})
  else()
    set(${parsec_cas_co_NAME} "${parsec_cas_co_defined_options}")
  endif()
  if(DEFINED ${parsec_cas_co_NAME})
    message(TRACE "Add compile option ${${parsec_cas_co_NAME}}")
    add_compile_options("${${parsec_cas_co_NAME}}")
  endif()
endfunction(check_and_set_compiler_option)

#
# Fix the building system for 32 or 64 bits.
#
# On MAC OS X there is a easy solution, by setting the
# CMAKE_OSX_ARCHITECTURES to a subset of the following values:
# ppc;ppc64;i386;x86_64.
# On Linux this is a little bit tricky. We have to check that the
# compiler supports the -m32/-m64 flags as well as the linker.
# Once this issue is resolved the directory compile_options
# have to be updated accordingly.
# On windows you have to use the correct compiler, as there seems to
# be no specific flag for 64 bits compilations.
#
# TODO: For the Fortran compiler:
#         no idea how to correctly detect if the required/optional
#         libraries are in the correct format.
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  string(REGEX MATCH ".*xlc$" _match_xlc ${CMAKE_C_COMPILER})
  if(_match_xlc)
    message(ERROR "Please use the thread-safe version of the xlc compiler (xlc_r)")
  endif(_match_xlc)
  string(REGEX MATCH "XL" _match_xlc ${CMAKE_C_COMPILER_ID})
  if (BUILD_64bits)
    if( _match_xlc)
      set( arch_build "-q64" )
    else (_match_xlc)
      if( CMAKE_SYSTEM_PROCESSOR STREQUAL "sparc64fx" )
        set ( arch_build " " )
      else()
        set( arch_build "-m64" )
      endif()
    endif(_match_xlc)
  else (BUILD_64bits)
    if( _match_xlc)
      set( arch_build "-q32" )
    else (_match_xlc)
      set( arch_build "-m32" )
    endif(_match_xlc)
  endif (BUILD_64bits)

  check_and_set_compiler_option(OPTION ${arch_build} NAME PARSEC_ARCH_OPTIONS COMMENT "List of compile options used to select the target architecture (e.g., -m64, -mtune=haswell, etc.)")
endif(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")

#
# Check compiler debug flags and capabilities
#

# add gdb symbols in debug and relwithdebinfo, g3 for macro support when available
check_and_set_compiler_option(OPTION "-g3" NAME PARSEC_HAVE_G3 COMMENT "List of compile options used to enable highest level of debugging (e.g. -g3)" CONFIG DEBUG)

# Some compilers produce better debugging outputs with Og vs O0
# but this should only be used in RelWithDebInfo mode.
check_and_set_compiler_option(OPTION "-Og" NAME PARSEC_HAVE_Og CONFIG RELWITHDEBINFO)

# Set warnings for debug builds
check_and_set_compiler_option(OPTION "-Wall" NAME PARSEC_HAVE_WALL)
check_and_set_compiler_option(OPTION "-Wextra" NAME PARSEC_HAVE_WEXTRA)

# Flex-generated files make some compilers generate a significant
# amount of warnings. We define here warning silent options
# that are passed only to Flex-generated files if they are
# supported by the compilers.
SET(PARSEC_FLEX_GENERATED_OPTIONS)
foreach(_flag "-Wno-misleading-indentation" "-Wno-sign-compare")
  string(REPLACE "-" "_" SAFE_flag ${_flag})
  check_c_compiler_flag("${_flag}" _has${SAFE_flag})
  if( _has${_flag} )
    list(APPEND PARSEC_FLEX_GENERATED_OPTIONS "${_flag}")
  endif( _has${_flag} )
endforeach()

#
# flags for Intel icc
#
string(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
if(_match_icc)
  # Silence annoying warnings
  # 424: checks for duplicate ";"
  # 981: every volatile triggers a "unspecified evaluation order", obnoxious
  #      but might be useful for some debugging sessions.
  # 1419: warning about extern functions being declared in .c
  #       files
  # 1572: cuda compares floats with 0.0f.
  # 11074: obnoxious about not inlining long functions.
  check_and_set_compiler_option(OPTION "-wd424,981,1419,1572,10237,11074,11076" NAME PARSEC_HAVE_WD LANGUAGES C)
endif(_match_icc)

# remove asserts in release
add_compile_definitions($<$<CONFIG:RELEASE>:NDEBUG>)

if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Ninja is weird with colors. It does not present a pty to cc (hence 
  # colors get disabled by default), but if colors are forced upon it, it 
  # will do the right thing and print colors only on terminals.
  foreach(colorflag -fdiagnostics-color -fcolor-diagnostics)
    string(REPLACE "-" "_" SAFE_colorflag ${colorflag})
    check_and_set_compiler_option(OPTION ${colorflag} NAME PARSEC_CC_COLORS${SAFE_colorflag})
  endforeach()
endif()

#
# Fortran tricks: Debug/Release FFLAGS depend on the compiler
#
if(CMAKE_Fortran_COMPILER_WORKS)
  get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
  #  message(STATUS "Fortran Compiler ${Fortran_COMPILER_NAME} id is ${CMAKE_Fortran_COMPILER_ID}")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    # gfortran or g77
    if(Fortran_COMPILER_NAME MATCHES g77)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fno-f2c>")
    endif()
    # We append the implicit fortran link flags for the case where FC=/somepath/f90
    # and /somepath/lib/libf90.so is not in LD_LIBRARY_PATH. This is typical for non-system
    # installed gfortan where the implicit -lgfortran may not resolved at application link time
    # otherwise.
    foreach(item IN ITEMS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
      list(APPEND EXTRA_LIBS "-L${item}")
    endforeach()
    list(APPEND EXTRA_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    # ifort
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-f77rtl>")
    # This is a bug in CMake, which incorrectly adds this flag that does not exist on some ifort versions.
    string (REPLACE "-i_dynamic" "" CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS}")
  endif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
endif(CMAKE_Fortran_COMPILER_WORKS)

