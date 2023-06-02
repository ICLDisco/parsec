include(ParsecCompilePTG)

parsec_addtest_executable(C taskpool_wait SOURCES main.c dtd_tp.c)
target_include_directories(taskpool_wait PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(taskpool_wait PRIVATE "ptg_tp.jdf")
