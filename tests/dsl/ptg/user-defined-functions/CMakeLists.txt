include(ParsecCompilePTG)

parsec_addtest_executable(C udf SOURCES main.c udf_wrapper.c)
target_include_directories(udf PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(udf PRIVATE "udf.jdf")

parsec_addtest_executable(C utt)
target_include_directories(utt PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(utt PRIVATE "utt.jdf")
