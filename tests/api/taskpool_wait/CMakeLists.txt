#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C taskpool_wait SOURCES main.c dtd_tp.c)
target_link_libraries(taskpool_wait PRIVATE tests_runtime_common)
target_include_directories(taskpool_wait PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(taskpool_wait PRIVATE "ptg_tp.jdf")

# Same PTG taskpool, generated under another name so both executables can build it
parsec_addtest_executable(C taskpool_wait_first SOURCES wait_first.c)
target_link_libraries(taskpool_wait_first PRIVATE tests_runtime_common)
target_ptg_source_ex(TARGET taskpool_wait_first DESTINATION ptg_tp_wait_first MODE PRIVATE SOURCE ptg_tp.jdf)
