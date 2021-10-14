include(ParsecCompilePTG)

# Default build: probably ht, but not taking a chance of missing an option
parsec_addtest_executable(C branching SOURCES main.c branching_wrapper.c branching_data.c)
target_ptg_sources(branching PRIVATE "branching.jdf")

# Force dynamic hash tables test
parsec_addtest_executable(C branching_ht SOURCES main.c branching_wrapper.c branching_data.c)
target_ptg_source_ex(TARGET branching_ht DESTINATION branching_ht MODE PRIVATE SOURCE branching.jdf DEP_MANAGEMENT dynamic-hash-table)
add_dependencies(branching_ht branching) # We need to have branching.h generated before

# Force index array test
parsec_addtest_executable(C branching_idxarr SOURCES main.c branching_wrapper.c branching_data.c)
target_ptg_source_ex(TARGET branching_idxarr DESTINATION branching_idxarr MODE PRIVATE SOURCE branching.jdf DEP_MANAGEMENT index-array)
add_dependencies(branching_idxarr branching) # We need to have branching.h generated before
