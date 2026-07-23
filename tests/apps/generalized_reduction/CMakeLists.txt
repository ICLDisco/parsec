#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C BT_reduction SOURCES main.c BT_reduction_wrapper.c reduc_data.c)
target_link_libraries(BT_reduction PRIVATE tests_runtime_common)
target_ptg_sources(BT_reduction PRIVATE "BT_reduction.jdf")
