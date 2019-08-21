include(ParsecCompilePTG)

parsec_addtest(C udf "main.c;udf_wrapper.c")
target_include_directories(udf PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(udf PRIVATE "udf.jdf")

