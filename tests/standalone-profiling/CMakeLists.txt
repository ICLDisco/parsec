SET(PARSEC_PROFILING_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/parsec/")
SET(PARSEC_PROFILING_LIBRARIES "parsec;parsec-base;${CMAKE_THREAD_LIBS_INIT}")

INCLUDE_DIRECTORIES("${PARSEC_PROFILING_INCLUDE_DIR}")

add_executable(sp-demo sp-demo.c)
target_link_libraries (sp-demo "${PARSEC_PROFILING_LIBRARIES}")

if(PARSEC_HAVE_PTHREAD_BARRIER)
  add_executable(sp-perf sp-perf.c)
  target_link_libraries (sp-perf "${PARSEC_PROFILING_LIBRARIES}")
endif(PARSEC_HAVE_PTHREAD_BARRIER)

