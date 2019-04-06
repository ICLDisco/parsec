include(ParsecCompilePTG)

parsec_compile_ptg(BT_reduction_generated "${CMAKE_CURRENT_SOURCE_DIR}/BT_reduction.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C BT_reduction
"main.c;${BT_reduction_generated};BT_reduction_wrapper.c;reduc_data.c")
