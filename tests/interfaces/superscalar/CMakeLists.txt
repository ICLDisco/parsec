include_directories(${CMAKE_CURRENT_BINARY_DIR})

dague_addtest(C test_atomic_write "test_atomic_write.c")
dague_addtest(C dtd_test_WAR "dtd_test_WAR.c")
dague_addtest(C dtd_test_split_interface "dtd_test_split_interface.c")
dague_addtest(C dtd_test_overlap "dtd_test_overlap.c")
dague_addtest(C dtd_test_no_overlap "dtd_test_no_overlap.c")
dague_addtest(C test_task_generation "test_task_generation.c")
dague_addtest(C test_write "test_write.c")
