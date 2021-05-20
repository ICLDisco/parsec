parsec_addtest_executable(C local_indices)
target_ptg_sources(local_indices PRIVATE "local_indices.jdf")
target_link_libraries(local_indices PRIVATE m)
