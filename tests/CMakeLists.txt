add_custom_target(parsec_build_tests)
add_test(parsec_build_tests "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target parsec_build_tests)

#! parsec_addtest_executable : this function defines how to
#    compile an executable that is used in the testings.
#
# This function works similarly to add_executable, which
# it calls with the optional source list provided.
# It defines the target provided as a the second parameter
# and assigns it the language provided as the first
# parameter for linking. It adds the necessary link
# options to link with PaRSEC.
#
# Unless the option NODEFAULTBUILD is specified, it
# adds this target in the parsec_build_tests target.
#
# Nota Bene: the target defined is added to the generic
#  'all' target only if PaRSEC is the root project, otherwise
#  the tests are defined, but not added to 'all'. Tests
#  that are not NODEFAULTBUILD are still added to the 
#  parsec_build_tests target.
#
# \argn: a list of optional arguments
# \arg:first_arg language used to link this executable
# \arg:second_arg target name for the executable
# \param:SOURCES optional list of files Source files
# \param:NODEFAULTBUILD option to not build this
#     target when invoking parsec_build_tests
#     from CTest. This is needed for files that
#     must fail compiling for example.
#
function(parsec_addtest_executable lang target)
  set(options NODEFAULTBUILD)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(PARSEC_ADDTEST "${options}" "" "${multiValueArgs}" ${ARGN})
  if("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    add_executable(${target} ${PARSEC_ADDTEST_SOURCES})
  else()
    add_executable(${target} EXCLUDE_FROM_ALL ${PARSEC_ADDTEST_SOURCES})
  endif()
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE ${lang})
  target_link_libraries(${target} PRIVATE parsec)
  if(NOT ${PARSEC_ADDTEST_NODEFAULTBUILD})
    add_dependencies(parsec_build_tests ${target})
  endif()
endfunction(parsec_addtest_executable)

#! parsec_addtest_cmd : this function defines how to
#    run a given test.
#
# This function works similarly to add_test, except
# it will introduce a dependency between the test
# defined and the parsec_build_tests_test test,
# ensuring (with caveats...) that the executable
# it depends on is compiled before the test is run.
#
# The function uses the argument name "COMMAND" to
# detect that the long form of add_test is used,
# and it will add "NAME" in front of the target
# name in that case.
#
function(parsec_addtest_cmd target)
  LIST(FIND ARGV "COMMAND" LONGFORM)
  if( NOT "${LONGFORM}" EQUAL "-1" )
    list(PREPEND ARGV "NAME")
  endif()
  add_test(${ARGV})
  SET_TESTS_PROPERTIES(${target} PROPERTIES DEPENDS parsec_build_tests_test)
endfunction(parsec_addtest_cmd)

check_function_exists(erand48 PARSEC_HAVE_ERAND48)
check_function_exists(nrand48 PARSEC_HAVE_NRAND48)
check_function_exists(lrand48 PARSEC_HAVE_LRAND48)
check_function_exists(random PARSEC_HAVE_RANDOM)

add_library(tests_common OBJECT tests_data.c)
target_link_libraries(tests_common PRIVATE parsec)

add_subdirectory(class)
add_subdirectory(api)
if( TARGET parsec-ptgpp )
  add_subdirectory(runtime)
  add_subdirectory(dsl/ptg)
  add_subdirectory(collections)
endif( TARGET parsec-ptgpp )
add_subdirectory(dsl/dtd)
add_subdirectory(apps)
if(PARSEC_PROF_TRACE)
  add_subdirectory(profiling)
  add_subdirectory(profiling-standalone)
endif(PARSEC_PROF_TRACE)

# Define Testings
include(Testings.cmake)
