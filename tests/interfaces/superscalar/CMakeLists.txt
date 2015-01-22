include_directories(${CMAKE_CURRENT_BINARY_DIR})

dague_addtest(C insert_task_handle_not_attached "test_insert_task_interface_handle_not_attached.c")
dague_addtest(C insert_task_handle_attached "test_insert_task_interface_handle_attached.c")
dague_addtest(C test_atomic_write "test_atomic_write.c")
dague_addtest(C test_no_overlap "test_no_overlap.c")
