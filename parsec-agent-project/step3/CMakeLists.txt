cmake_minimum_required(VERSION 3.21)
project(parsec-agent-step3 C)

set(PaRSEC_ROOT "$ENV{HOME}/parsec/builddir/install")
list(APPEND CMAKE_PREFIX_PATH "${PaRSEC_ROOT}/share/cmake/parsec")

find_package(PaRSEC REQUIRED)
find_package(MPI REQUIRED)

add_executable(dtd_loop main.c)
target_include_directories(dtd_loop PRIVATE "${PaRSEC_ROOT}/include" ${MPI_C_INCLUDE_DIRS})
target_link_libraries(dtd_loop "${PaRSEC_ROOT}/lib/libparsec.so" ${MPI_C_LIBRARIES})
