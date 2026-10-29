cmake_minimum_required(VERSION 3.21)
project(parsec-agent-step5 C)

set(PaRSEC_ROOT "$ENV{HOME}/parsec/builddir/install")
list(APPEND CMAKE_PREFIX_PATH "${PaRSEC_ROOT}/share/cmake/parsec")

find_package(PaRSEC REQUIRED)
find_package(MPI REQUIRED)
find_package(CURL REQUIRED)

add_executable(dtd_agent_ollama main.c)
target_include_directories(dtd_agent_ollama PRIVATE "${PaRSEC_ROOT}/include" ${MPI_C_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS})
target_link_libraries(dtd_agent_ollama "${PaRSEC_ROOT}/lib/libparsec.so" ${MPI_C_LIBRARIES} ${CURL_LIBRARIES})
