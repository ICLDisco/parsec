link_libraries(Threads::Threads)

if(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
  parsec_addtest(C atomics atomics.c)
endif(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
parsec_addtest(C rwlock rwlock.c)
parsec_addtest(C future future.c)
parsec_addtest(C future_datacopy future_datacopy.c)
parsec_addtest(C lifo lifo.c)
parsec_addtest(C list list.c)
parsec_addtest(C hash hash.c)

if(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
  parsec_addtest(C atomics_inline atomics.c)
  set_property(TARGET atomics_inline
               APPEND PROPERTY COMPILE_DEFINITIONS BUILDING_PARSEC)
  set_property(TARGET atomics_inline
               APPEND PROPERTY COMPILE_OPTIONS ${PARSEC_ATOMIC_SUPPORT_OPTIONS})
endif(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
parsec_addtest(C rwlock_inline rwlock.c)
parsec_addtest(C lifo_inline lifo.c)
parsec_addtest(C list_inline list.c)
parsec_addtest(C hash_inline hash.c)
set_property(TARGET rwlock_inline lifo_inline list_inline hash_inline
  APPEND PROPERTY COMPILE_DEFINITIONS BUILDING_PARSEC)
set_property(TARGET rwlock_inline lifo_inline list_inline hash_inline
  APPEND PROPERTY COMPILE_OPTIONS ${PARSEC_ATOMIC_SUPPORT_OPTIONS})

include(Testings.cmake)
