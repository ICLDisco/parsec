include(ParsecCompilePTG)

parsec_addtest_executable(C schedmicro SOURCES schedmicro.c ep_wrapper.c rt_data.c)
target_ptg_sources(schedmicro PRIVATE "ep.jdf")
target_link_libraries(schedmicro PRIVATE m)

parsec_addtest_executable(C pcb SOURCES pcb-main.c pcb_wrapper.c rt_data.c)
target_ptg_sources(pcb PRIVATE "pcb.jdf")
target_link_libraries(pcb PRIVATE m)
