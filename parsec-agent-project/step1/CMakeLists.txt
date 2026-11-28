cmake_minimum_required(VERSION 3.21)
project(parsec-agent-step1 C)

# Set the path to your PaRSEC installation
set(PaRSEC_ROOT "$ENV{HOME}/parsec/builddir/install")
list(APPEND CMAKE_PREFIX_PATH "${PaRSEC_ROOT}/share/cmake/parsec")

find_package(PaRSEC REQUIRED)
find_package(MPI REQUIRED)

add_executable(parsec_ok main.c)
target_include_directories(parsec_ok PRIVATE "${PaRSEC_ROOT}/include" ${MPI_C_INCLUDE_DIRS})
target_link_libraries(parsec_ok "${PaRSEC_ROOT}/lib/libparsec.so" ${MPI_C_LIBRARIES})
