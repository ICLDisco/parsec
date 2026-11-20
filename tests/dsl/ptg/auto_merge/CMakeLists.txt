#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#

include(ParsecCompilePTG)

# The CUDA body of am_map needs a kernel. Without a CUDA compiler the driver
# keeps every task on the CPU, and a stub stands in for the kernel.
set(AUTO_MERGE_KERNELS auto_merge_nokernels.c)
if(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
  set_source_files_properties(auto_merge_kernels.cu PROPERTIES LANGUAGE CUDA)
  set(AUTO_MERGE_KERNELS auto_merge_kernels.cu)
endif()

parsec_addtest_executable(C auto_merge SOURCES auto_merge_main.c auto_merge_ops.c ${AUTO_MERGE_KERNELS})
target_link_libraries(auto_merge PRIVATE tests_runtime_common)
target_include_directories(auto_merge PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(auto_merge PRIVATE "am_map.jdf;am_chain.jdf;am_bcast.jdf")
