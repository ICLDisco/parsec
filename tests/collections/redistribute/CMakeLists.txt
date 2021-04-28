include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

parsec_addtest_executable(C testing_redistribute SOURCES testing_redistribute.c common.c)
target_link_libraries(testing_redistribute PRIVATE m)
target_ptg_sources(testing_redistribute PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check2.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_bound.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_no_optimization.jdf")

parsec_addtest_executable(C testing_redistribute_random SOURCES testing_redistribute_random.c common.c)
target_link_libraries(testing_redistribute_random PRIVATE m)
target_ptg_sources(testing_redistribute_random PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check2.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_bound.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_no_optimization.jdf")

