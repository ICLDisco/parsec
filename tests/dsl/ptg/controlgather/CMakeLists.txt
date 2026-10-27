#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C ctlgat SOURCES main.c ctlgat_wrapper.c ctlgat_data.c)
target_link_libraries(ctlgat PRIVATE tests_runtime_common)
target_ptg_sources(ctlgat PRIVATE "ctlgat.jdf")
