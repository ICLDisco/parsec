include(ParsecCompilePTG)

parsec_compile_ptg(udf_generated "${CMAKE_CURRENT_SOURCE_DIR}/udf.jdf")
parsec_addtest(C udf "main.c;${udf_generated};udf_wrapper.c")
if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  target_include_directories(udf PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  target_include_directories(udf PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
endif( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
