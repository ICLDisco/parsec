include(ParsecCompilePTG)

parsec_addtest_executable(C choice SOURCES main.c choice_wrapper.c choice_data.c)
target_link_libraries(choice PRIVATE tests_runtime_common)
target_ptg_sources(choice PRIVATE "choice.jdf")
