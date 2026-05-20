#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C rtt SOURCES main.c rtt_wrapper.c rtt_data.c)
target_link_libraries(rtt PRIVATE tests_runtime_common)
target_ptg_sources(rtt PRIVATE "rtt.jdf")

parsec_addtest_executable(C bw_test)
target_link_libraries(bw_test PRIVATE tests_runtime_common)
set_source_files_properties("bandwidth.jdf" PROPERTIES PTGPP_COMPILE_OPTIONS "--Wremoteref")
target_ptg_sources(bw_test PRIVATE "bandwidth.jdf")
