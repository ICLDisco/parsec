add_custom_target(parsec_build_tests)
add_test(parsec_build_tests_test "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target parsec_build_tests)

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
# \argn: a list of optional arguments
# \arg:first_arg language used to link this executable
# \arg:second_arg target name for the executable
# \param:SOURCES optional list of files Source files
# \param:NODEFAULTBUILD option to not build this
#     target when invoking parsec_build_tests_test
#     from CTest. This is needed for files that
#     must fail compiling for example.
#
function(parsec_addtest_executable lang target)
  set(options NODEFAULTBUILD)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(PARSEC_ADDTEST "${options}" "" "${multiValueArgs}" ${ARGN})
  add_executable(${target} ${PARSEC_ADDTEST_SOURCES})
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

add_Subdirectory(unit)
add_Subdirectory(interfaces/superscalar)
add_Subdirectory(interfaces/ptg)
add_Subdirectory(haar-tree-project)
add_Subdirectory(pingpong)
add_Subdirectory(branching)
add_Subdirectory(choice)
add_Subdirectory(controlgather)
add_Subdirectory(scheduling)
add_Subdirectory(all2all)
add_Subdirectory(generalized_reduction)
if(PARSEC_HAVE_RANDOM)
add_Subdirectory(merge_sort)
endif(PARSEC_HAVE_RANDOM)
add_Subdirectory(two_dim_band)
add_Subdirectory(stencil)
add_Subdirectory(reshape)

if(PARSEC_PROF_TRACE)
  add_Subdirectory(profiling)
  add_Subdirectory(standalone-profiling)
endif(PARSEC_PROF_TRACE)
add_Subdirectory(redistribute)

parsec_addtest_executable(C operator SOURCES operator.c)
parsec_addtest_executable(C reduce SOURCES reduce.c)
parsec_addtest_executable(C dtt_bug_replicator SOURCES dtt_bug_replicator_ex.c)
target_ptg_sources(dtt_bug_replicator PRIVATE "dtt_bug_replicator.jdf")

parsec_addtest_executable(C touch_ex SOURCES touch_ex.c)
target_ptg_sources(touch_ex PRIVATE "touch.jdf")
parsec_addtest_executable(C touch_ex_inline SOURCES touch_ex.c)
target_ptg_sources(touch_ex_inline PRIVATE "touch.jdf")
target_compile_definitions(touch_ex_inline PRIVATE BUILDING_PARSEC)
target_compile_options(touch_ex_inline PRIVATE ${PARSEC_ATOMIC_SUPPORT_OPTIONS})
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest_executable(Fortran touch_exf SOURCES touch_exf.F90)
    target_link_libraries(touch_exf PRIVATE $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>)
    target_ptg_sources(touch_exf PRIVATE "touch.jdf")
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

parsec_addtest_executable(C kcyclic)
target_ptg_sources(kcyclic PRIVATE "kcyclic.jdf")
target_link_libraries(kcyclic PRIVATE m)

parsec_addtest_executable(C strange)
target_ptg_sources(strange PRIVATE "strange.jdf")

if(PARSEC_HAVE_RANDOM)
  parsec_addtest_executable(C startup)
  target_ptg_sources(startup PRIVATE "startup.jdf")
endif(PARSEC_HAVE_RANDOM)

parsec_addtest_executable(C complex_deps)
target_ptg_sources(complex_deps PRIVATE "complex_deps.jdf")

if( MPI_C_FOUND )
  parsec_addtest_executable(C multichain)
  target_ptg_sources(multichain PRIVATE "multichain.jdf")
endif( MPI_C_FOUND )

parsec_addtest_executable(C compose SOURCES compose.c)

# Define Testings
include(Testings.cmake)
