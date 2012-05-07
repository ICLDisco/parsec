include(RulesJDF)

jdf_rules(choice_generated "${CMAKE_CURRENT_SOURCE_DIR}/choice.jdf")
string( REGEX REPLACE "choice.[hc]" "" include_dir "${choice_generated}" )
include_directories(${include_dir})

dague_addtest(C choice "main.c;${choice_generated};choice_wrapper.c;choice_data.c")
