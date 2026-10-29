# Julia bindings (PaRSEC4Julia)
#
# Builds the C shared libraries that Julia ccall()s into.
# Outputs land in julia/src/ so existing paths like
#   joinpath(@__DIR__, "libdtd_wrapper.so")
# keep working when this directory lives inside the PaRSEC tree.

find_package(Threads REQUIRED)

add_library(parsec_julia_dtd SHARED src/dtd_wrapper.c)
set_target_properties(parsec_julia_dtd PROPERTIES
  OUTPUT_NAME dtd_wrapper
  PREFIX "lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(parsec_julia_dtd PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(parsec_julia_dtd PRIVATE parsec m Threads::Threads)
if(TARGET MPI::MPI_C)
  target_link_libraries(parsec_julia_dtd PRIVATE MPI::MPI_C)
elseif(PARSEC_HAVE_MPI AND MPI_C_LIBRARIES)
  target_include_directories(parsec_julia_dtd PRIVATE ${MPI_C_INCLUDE_PATH})
  target_link_libraries(parsec_julia_dtd PRIVATE ${MPI_C_LIBRARIES})
endif()

# Stencil wrapper compiles PaRSEC test/app C + JDF-generated sources.
# Only available when the stencil test target is part of this build.
if(TARGET testing_stencil_1D)
  set(STENCIL_1D_C "${CMAKE_BINARY_DIR}/tests/apps/stencil/stencil_1D.c")
  set_source_files_properties("${STENCIL_1D_C}" PROPERTIES GENERATED TRUE)

  add_library(parsec_julia_stencil SHARED
    src/stencil_wrapper.c
    "${PROJECT_SOURCE_DIR}/tests/apps/stencil/stencil_internal.c"
    "${STENCIL_1D_C}"
  )
  add_dependencies(parsec_julia_stencil testing_stencil_1D)
  set_target_properties(parsec_julia_stencil PROPERTIES
    OUTPUT_NAME stencil_jl
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(parsec_julia_stencil PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/tests/apps/stencil"
    "${CMAKE_BINARY_DIR}/tests/apps/stencil"
  )
  target_link_libraries(parsec_julia_stencil PRIVATE parsec m Threads::Threads)
  if(TARGET MPI::MPI_C)
    target_link_libraries(parsec_julia_stencil PRIVATE MPI::MPI_C)
  elseif(PARSEC_HAVE_MPI AND MPI_C_LIBRARIES)
    target_include_directories(parsec_julia_stencil PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_libraries(parsec_julia_stencil PRIVATE ${MPI_C_LIBRARIES})
  endif()
endif()
