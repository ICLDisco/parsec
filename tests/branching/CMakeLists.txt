include(ParsecCompilePTG)

parsec_compile_ptg(branching_generated "${CMAKE_CURRENT_SOURCE_DIR}/branching.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C branching "main.c;${branching_generated};branching_wrapper.c;branching_data.c")

