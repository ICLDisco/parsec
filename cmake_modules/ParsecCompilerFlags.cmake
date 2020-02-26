
include (CMakeDetermineSystem)
include (CheckCCompilerFlag)
include (CheckCSourceCompiles)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckIncludeFiles)
include (CMakePushCheckState)
include (CheckFortranCompilerFlag)
include (CheckCXXCompilerFlag)

# check for the CPU we build for
MESSAGE(STATUS "Building for target ${CMAKE_SYSTEM_PROCESSOR}")
STRING(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86)
    MESSAGE(STATUS "Found target for X86")
    SET(PARSEC_ARCH_X86 1)
ENDIF (_mach_x86)

STRING(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86_64)
    MESSAGE(STATUS "Found target X86_64")
    SET(PARSEC_ARCH_X86_64 1)
ENDIF (_mach_x86_64)

STRING(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_ppc)
    MESSAGE(STATUS "Found target for PPC")
    SET(PARSEC_ARCH_PPC 1)
ENDIF (_mach_ppc)

#
# Fix the building system for 32 or 64 bits.
#
# On MAC OS X there is a easy solution, by setting the
# CMAKE_OSX_ARCHITECTURES to a subset of the following values:
# ppc;ppc64;i386;x86_64.
# On Linux this is a little bit tricky. We have to check that the
# compiler supports the -m32/-m64 flags as well as the linker.
# Once this issue is resolved the CMAKE_C_FLAGS and CMAKE_EXE_LINKER_FLAGS
# have to be updated accordingly.
#
# TODO: Same trick for the Fortran compiler...
#       no idea how to correctly detect if the required/optional
#          libraries are in the correct format.
#
STRING(REGEX MATCH ".*xlc$" _match_xlc ${CMAKE_C_COMPILER})
IF (_match_xlc)
  MESSAGE(ERROR "Please use the thread-safe version of the xlc compiler (xlc_r)")
ENDIF (_match_xlc)
STRING(REGEX MATCH "XL" _match_xlc ${CMAKE_C_COMPILER_ID})
if (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q64" )
  else (_match_xlc)
    if( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "sparc64fx" )
      set ( ARCH_BUILD " " )
    else()
      set( ARCH_BUILD "-m64" )
    endif()
  endif(_match_xlc)
else (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q32" )
  else (_match_xlc)
    set( ARCH_BUILD "-m32" )
  endif(_match_xlc)
endif (BUILD_64bits)

check_c_compiler_flag( ${ARCH_BUILD} C_M32or64 )
if( C_M32or64 )
  string(APPEND CMAKE_C_FLAGS " ${ARCH_BUILD}")
  # Try the same for Fortran and CXX:
  # Use the same 64bit flag as the C compiler if possible
  if(CMAKE_Fortran_COMPILER_WORKS)
    check_fortran_compiler_flag( ${ARCH_BUILD} F_M32or64 )
    if( F_M32or64 )
      string(APPEND CMAKE_Fortran_FLAGS " ${ARCH_BUILD}")
    endif( F_M32or64 )
  endif()
  if(CMAKE_CXX_COMPILER_WORKS)
    check_cxx_compiler_flag( ${ARCH_BUILD} CXX_M32or64 )
    if( CXX_M32or64 )
      string(APPEND CMAKE_CXX_FLAGS " ${ARCH_BUILD}")
    endif( CXX_M32or64 )
  endif()
endif( C_M32or64 )

#
# Check compiler debug flags and capabilities
#

# Set warnings for debug builds
CHECK_C_COMPILER_FLAG( "-Wall" PARSEC_HAVE_WALL )
IF( PARSEC_HAVE_WALL )
  STRING( APPEND C_WFLAGS " -Wall" )
ENDIF( PARSEC_HAVE_WALL )
CHECK_C_COMPILER_FLAG( "-Wextra" PARSEC_HAVE_WEXTRA )
IF( PARSEC_HAVE_WEXTRA )
  STRING( APPEND C_WFLAGS " -Wextra" )
ENDIF( PARSEC_HAVE_WEXTRA )

#
# flags for Intel icc
#
STRING(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
if(_match_icc)
  # Silence annoying warnings
  CHECK_C_COMPILER_FLAG( "-wd424" PARSEC_HAVE_WD )
  IF( PARSEC_HAVE_WD )
    # 424: checks for duplicate ";"
    # 981: every volatile triggers a "unspecified evaluation order", obnoxious
    #      but might be useful for some debugging sessions.
    # 1419: warning about extern functions being declared in .c
    #       files
    # 1572: cuda compares floats with 0.0f.
    # 11074: obnoxious about not inlining long functions.
    string(APPEND C_WFLAGS " -wd424,981,1419,1572,10237,11074,11076")
  ENDIF( PARSEC_HAVE_WD )
else(_match_icc)
  CHECK_C_COMPILER_FLAG( "-Wno-parentheses-equality" PARSEC_HAVE_PAR_EQUALITY )
  IF( PARSEC_HAVE_PAR_EQUALITY )
    string(APPEND C_WFLAGS " -Wno-parentheses-equality")
  ENDIF( PARSEC_HAVE_PAR_EQUALITY )
endif(_match_icc)

# add gdb symbols in debug and relwithdebinfo
CHECK_C_COMPILER_FLAG( "-g3" PARSEC_HAVE_G3 )
IF( PARSEC_HAVE_G3 )
  string(APPEND CMAKE_C_FLAGS_DEBUG " -O0 -g3")
  string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3")
ELSE()
  string(APPEND CMAKE_C_FLAGS_DEBUG " -O0")
ENDIF( PARSEC_HAVE_G3)
# verbose compilation in debug
string(APPEND CMAKE_C_FLAGS_DEBUG " ${C_WFLAGS}")
# remove asserts in release
string(APPEND CMAKE_C_FLAGS_RELEASE " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " ${C_WFLAGS}")

#
# Remove all duplicates from the CFLAGS.
#
if(CMAKE_C_FLAGS)
set(TMP_LIST ${CMAKE_C_FLAGS})
separate_arguments(TMP_LIST)
list(REMOVE_DUPLICATES TMP_LIST)
set(CMAKE_C_FLAGS "")
foreach( ITEM ${TMP_LIST})
  string(APPEND CMAKE_C_FLAGS " ${ITEM}")
endforeach()
endif()

#
# Fortran tricks
# Debug/Release FFLAGS depend on the compiler
#
IF (CMAKE_Fortran_COMPILER_WORKS)
  include (CheckFortranCompilerFlag)
  if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    # gfortran
    set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g")
    MESSAGE(STATUS "Fortran adds libraries path ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
    FOREACH(ITEM IN ITEMS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
      list(APPEND EXTRA_LIBS "-L${ITEM}")
    ENDFOREACH(ITEM)
    MESSAGE(STATUS "Fortran adds libraries ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")
    list(APPEND EXTRA_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
    # ifort
    set (CMAKE_Fortran_FLAGS_RELEASE "-f77rtl -O3")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-f77rtl -O0 -g")
    string (REPLACE "-i_dynamic" "" CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS}")
  else (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
    get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
    message ("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
    message ("Fortran compiler: " ${Fortran_COMPILER_NAME})
    message ("No optimized Fortran compiler flags are known, we just try -O2...")
    set (CMAKE_Fortran_FLAGS_RELEASE "-O2")
    set (CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
  endif (${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")

ENDIF (CMAKE_Fortran_COMPILER_WORKS)



