include(RulesJDF)

jdf_rules(ctlgat_generated "${CMAKE_CURRENT_SOURCE_DIR}/ctlgat.jdf")
string( REGEX REPLACE "ctlgat.[hc]" "" include_dir "${ctlgat_generated}" )
include_directories(${include_dir})

dague_addtest(C ctlgat "main.c;${ctlgat_generated};ctlgat_wrapper.c;ctlgat_data.c")

