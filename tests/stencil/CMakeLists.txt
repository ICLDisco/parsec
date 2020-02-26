file(COPY loop_gen_1D DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C testing_stencil_1D "stencil_internal.c;testing_stencil_1D.c")
target_include_directories(testing_stencil_1D PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_stencil_1D PRIVATE "stencil_1D.jdf")
target_link_libraries(testing_stencil_1D PRIVATE m)

