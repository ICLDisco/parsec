include(RulesJDF)

jdf_rules(merge_sort_generated "${CMAKE_CURRENT_SOURCE_DIR}/merge_sort.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C merge_sort
"main.c;${merge_sort_generated};merge_sort_wrapper.c;sort_data.c")
