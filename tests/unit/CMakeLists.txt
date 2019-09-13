link_libraries(Threads::Threads)

parsec_addtest(C atomics atomics.c)
parsec_addtest(C rwlock rwlock.c)
parsec_addtest(C lifo lifo.c)
parsec_addtest(C list list.c)
parsec_addtest(C hash hash.c)

parsec_addtest(C atomics_inline atomics.c)
parsec_addtest(C rwlock_inline rwlock.c)
parsec_addtest(C lifo_inline lifo.c)
parsec_addtest(C list_inline list.c)
parsec_addtest(C hash_inline hash.c)
set_property(TARGET atomics_inline rwlock_inline lifo_inline list_inline hash_inline
  APPEND PROPERTY COMPILE_DEFINITIONS BUILDING_PARSEC)

