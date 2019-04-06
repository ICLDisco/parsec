include(ParsecCompilePTG)

parsec_compile_ptg(ep_generated "${CMAKE_CURRENT_SOURCE_DIR}/ep.jdf")
string( REGEX REPLACE "ep.[hc]" "" include_dir "${ep_generated}" )
include_directories(${include_dir})

parsec_addtest(C schedmicro "main.c;${ep_generated};ep_wrapper.c;schedmicro_data.c")

