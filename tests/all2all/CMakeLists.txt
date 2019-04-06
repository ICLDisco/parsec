include(ParsecCompilePTG)

parsec_compile_ptg(a2a_generated "${CMAKE_CURRENT_SOURCE_DIR}/a2a.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C a2a "main.c;${a2a_generated};a2a_data.c")

