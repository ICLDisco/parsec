include(ParsecCompilePTG)

parsec_addtest_executable(C check_multisize_bcast
                          SOURCES main.c check_multisize_bcast_wrapper.c data_gen.c)
target_ptg_sources(check_multisize_bcast PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/check_multisize_bcast.jdf")

