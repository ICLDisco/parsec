link_libraries(Threads::Threads)

parsec_addtest(C lifo lifo.c)
parsec_addtest(C list list.c)
parsec_addtest(C rwlock rwlock.c)
parsec_addtest(C hash hash.c)
parsec_addtest(C atomics atomics.c)

parsec_addtest(C inline_lifo lifo.c)
target_compile_definitions(inline_lifo PRIVATE BUILD_PARSEC)
parsec_addtest(C inline_list list.c)
target_compile_definitions(inline_list PRIVATE BUILD_PARSEC)
parsec_addtest(C inline_rwlock rwlock.c)
target_compile_definitions(inline_rwlock PRIVATE BUILD_PARSEC)
parsec_addtest(C inline_hash hash.c)
target_compile_definitions(inline_hash PRIVATE BUILD_PARSEC)
parsec_addtest(C inline_atomics atomics.c)
target_compile_definitions(inline_atomics PRIVATE BUILD_PARSEC)

