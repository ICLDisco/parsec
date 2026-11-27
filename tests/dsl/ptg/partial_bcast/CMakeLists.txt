#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C partial_bcast
                          SOURCES main.c partial_bcast_wrapper.c)
target_link_libraries(partial_bcast PRIVATE tests_runtime_common)
target_include_directories(partial_bcast PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_ptg_sources(partial_bcast PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/partial_bcast.jdf")
