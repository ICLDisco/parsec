include(ParsecCompilePTG)

parsec_addtest_executable(C testing_band SOURCES main.c)
target_include_directories(testing_band PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_band PRIVATE "two_dim_band.jdf;two_dim_band_free.jdf")

