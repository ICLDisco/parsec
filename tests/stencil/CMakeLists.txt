include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
if(NOT PARSEC_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
endif(NOT PARSEC_COMPILE_INPLACE)

file(COPY loop_gen_1D DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

parsec_compile_ptg(stencil_1D_generated "${CMAKE_CURRENT_SOURCE_DIR}/stencil_1D.jdf")

parsec_addtest(C testing_stencil_1D
               "stencil_internal.c;${stencil_1D_generated};testing_stencil_1D.c")
