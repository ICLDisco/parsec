include(ParsecCompilePTG)

parsec_compile_ptg(udf_generated "${CMAKE_CURRENT_SOURCE_DIR}/udf.jdf")
parsec_addtest(C udf "main.c;${udf_generated};udf_wrapper.c")
target_include_directories(udf PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

