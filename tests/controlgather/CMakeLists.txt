include(RulesJDF)

jdf_rules(ctlgat_generated "${CMAKE_CURRENT_SOURCE_DIR}/ctlgat.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C ctlgat "main.c;${ctlgat_generated};ctlgat_wrapper.c;ctlgat_data.c")

