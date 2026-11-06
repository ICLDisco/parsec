#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

parsec_addtest_executable(C reduce SOURCES reduce.c)
target_link_libraries(reduce PRIVATE tests_runtime_common)

parsec_addtest_executable(C kcyclic)
target_link_libraries(kcyclic PRIVATE tests_runtime_common)
target_ptg_sources(kcyclic PRIVATE "kcyclic.jdf")
target_link_libraries(kcyclic PRIVATE m)

add_subdirectory(two_dim_band)

if(MPI_C_FOUND)
  add_subdirectory(redistribute)
endif(MPI_C_FOUND)
add_subdirectory(reshape)
