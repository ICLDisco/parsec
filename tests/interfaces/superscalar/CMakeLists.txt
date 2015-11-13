include_directories(${CMAKE_CURRENT_BINARY_DIR})

dague_addtest(C test_atomic_write "test_atomic_write.c")
dague_addtest(C test_no_overlap "test_no_overlap.c")
dague_addtest(C test_task_generation "test_task_generation.c")
dague_addtest(C test_write "test_write.c")
