#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

set(PARSEC_PROFILING_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/parsec/")
set(PARSEC_PROFILING_LIBRARIES "parsec;parsec-base;${CMAKE_THREAD_LIBS_INIT}")

INCLUDE_DIRECTORIES("${PARSEC_PROFILING_INCLUDE_DIR}")

if(PARSEC_HAVE_PTHREAD_BARRIER)
  add_executable(sp-demo sp-demo.c)
  target_link_libraries (sp-demo "${PARSEC_PROFILING_LIBRARIES}")
  target_link_libraries (sp-demo tests_runtime_common)

  add_executable(sp-perf sp-perf.c)
  target_link_libraries (sp-perf "${PARSEC_PROFILING_LIBRARIES}")
  target_link_libraries (sp-perf tests_runtime_common)
endif(PARSEC_HAVE_PTHREAD_BARRIER)
