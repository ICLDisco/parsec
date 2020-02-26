include (CheckCCompilerFlag)
include (CheckCXXCompilerFlag)
include (CheckFortranCompilerFlag)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckIncludeFiles)
include (CMakePushCheckState)

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
#
# TODO: For the Fortran compiler:
#         no idea how to correctly detect if the required/optional
#         libraries are in the correct format.
#
string(REGEX MATCH ".*xlc$" _match_xlc ${CMAKE_C_COMPILER})
if(_match_xlc)
  message(ERROR "Please use the thread-safe version of the xlc compiler (xlc_r)")
endif(_match_xlc)
string(REGEX MATCH "XL" _match_xlc ${CMAKE_C_COMPILER_ID})
if (BUILD_64bits)
  if( _match_xlc)
    set( arch_build "-q64" )
  else (_match_xlc)
    if( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "sparc64fx" )
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

check_c_compiler_flag( ${arch_build} C_M32or64 )
if( C_M32or64 )
  # Try the same for Fortran and CXX:
  # Use the same 64bit flag as the C compiler if possible
  if(CMAKE_Fortran_COMPILER_WORKS)
    check_fortran_compiler_flag( ${arch_build} F_M32or64 )
  endif()
  if(CMAKE_CXX_COMPILER_WORKS)
    check_cxx_compiler_flag( ${arch_build} CXX_M32or64 )
  endif()
  set(arch_build_lang "$<$<COMPILE_LANGUAGE:C>:${arch_build}>")
  if( F_M32or64 )
    list(APPEND arch_build_lang "$<$<COMPILE_LANGUAGE:Fortran>:${arch_build}>")
  endif( F_M32or64 )
  if( CXX_M32or64 )
    list(APPEND arch_build_lang "$<$<COMPILE_LANGUAGE:CXX>:${arch_build}>")
  endif( CXX_M32or64 )
  set(PARSEC_ARCH_OPTIONS "${arch_build_lang}" CACHE STRING "List of compile options used to select the target architecture (e.g., -m64, -mtune=haswell, etc.)")
  mark_as_advanced(PARSEC_ARCH_OPTIONS)
  add_compile_options("${PARSEC_ARCH_OPTIONS}")
endif( C_M32or64 )

#
# Check compiler debug flags and capabilities
#

# add gdb symbols in debug and relwithdebinfo, g3 for macro support when available
check_c_compiler_flag( "-g3" PARSEC_HAVE_G3 )
if( PARSEC_HAVE_G3 )
  set(wflags "-g3")
else()
  set(wflags "-g")
endif()

# Some compilers produce better debugging outputs with Og vs O0
check_c_compiler_flag( "-Og" PARSEC_HAVE_Og )
if( PARSEC_HAVE_Og )
  set(o0flag "-Og")
else()
  set(o0flag "-O0")
endif()

# Set warnings for debug builds
check_c_compiler_flag( "-Wall" PARSEC_HAVE_WALL )
if( PARSEC_HAVE_WALL )
  list(APPEND wflags "-Wall" )
endif( PARSEC_HAVE_WALL )
check_c_compiler_flag( "-Wextra" PARSEC_HAVE_WEXTRA )
if( PARSEC_HAVE_WEXTRA )
  list(APPEND wflags "-Wextra" )
endif( PARSEC_HAVE_WEXTRA )

#
# flags for Intel icc
#
string(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
if(_match_icc)
  # Silence annoying warnings
  check_c_compiler_flag( "-wd424" PARSEC_HAVE_WD )
  if( PARSEC_HAVE_WD )
    # 424: checks for duplicate ";"
    # 981: every volatile triggers a "unspecified evaluation order", obnoxious
    #      but might be useful for some debugging sessions.
    # 1419: warning about extern functions being declared in .c
    #       files
    # 1572: cuda compares floats with 0.0f.
    # 11074: obnoxious about not inlining long functions.
    list(APPEND wflags "-wd424,981,1419,1572,10237,11074,11076")
  endif( PARSEC_HAVE_WD )
else(_match_icc)
  check_c_compiler_flag( "-Wno-parentheses-equality" PARSEC_HAVE_PAR_EQUALITY )
  if( PARSEC_HAVE_PAR_EQUALITY )
    list(APPEND wflags "-Wno-parentheses-equality")
  endif( PARSEC_HAVE_PAR_EQUALITY )
endif(_match_icc)

# verbose compilation in debug
add_compile_options(
  "$<$<CONFIG:DEBUG>:${o0flag};${wflags}>"
  "$<$<CONFIG:RELWITHDEBINFO>:${wflags}>")
# remove asserts in release
add_compile_definitions(
  $<$<CONFIG:RELEASE>:NDEBUG>)

#
# Fortran tricks: Debug/Release FFLAGS depend on the compiler
#
if(CMAKE_Fortran_COMPILER_WORKS)
  get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
  #  message(STATUS "Fortran Compiler ${Fortran_COMPILER_NAME} id is ${CMAKE_Fortran_COMPILER_ID}")
  if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    # gfortran or g77
    if(${Fortran_COMPILER_NAME} MATCHES g77)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fno-f2c>")
    endif()
    #foreach(item IN ITEMS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
    # list(APPEND EXTRA_LIBS "-L${item}")
    #endforeach()
    #list(APPEND EXTRA_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
    # ifort
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-f77rtl>")
    # This is a bug in CMake, which incorrectly adds this flag that does not exist on some ifort versions.
    string (REPLACE "-i_dynamic" "" CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS}")
  endif (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
endif(CMAKE_Fortran_COMPILER_WORKS)

