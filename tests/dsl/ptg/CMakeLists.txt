#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

add_subdirectory(ptgpp)

parsec_addtest_executable(C strange)
target_link_libraries(strange PRIVATE tests_runtime_common)
target_ptg_sources(strange PRIVATE "strange.jdf")

parsec_addtest_executable(C recursive)
target_link_libraries(recursive PRIVATE tests_runtime_common)
target_ptg_sources(recursive PRIVATE "recursive.jdf")

if(PARSEC_HAVE_RANDOM)
  parsec_addtest_executable(C startup)
  target_link_libraries(startup PRIVATE tests_runtime_common)
  target_ptg_sources(startup PRIVATE "startup.jdf")
endif(PARSEC_HAVE_RANDOM)

parsec_addtest_executable(C complex_deps)
target_link_libraries(complex_deps PRIVATE tests_runtime_common)
target_ptg_sources(complex_deps PRIVATE "complex_deps.jdf")

if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_executable(C batch_cpu)
  target_link_libraries(batch_cpu PRIVATE tests_runtime_common)
  target_ptg_sources(batch_cpu PRIVATE "batch_cpu.jdf")
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)

add_subdirectory(branching)
add_subdirectory(choice)
add_subdirectory(controlgather)
add_subdirectory(user-defined-functions)
add_subdirectory(local-indices)
add_subdirectory(multisize_bcast)
