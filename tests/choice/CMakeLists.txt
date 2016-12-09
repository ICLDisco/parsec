include(RulesJDF)

jdf_rules(choice_generated "${CMAKE_CURRENT_SOURCE_DIR}/choice.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C choice "main.c;${choice_generated};choice_wrapper.c;choice_data.c")
