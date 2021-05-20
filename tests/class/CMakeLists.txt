link_libraries(Threads::Threads)

if(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
  parsec_addtest_executable(C atomics SOURCES atomics.c)
endif(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
parsec_addtest_executable(C rwlock SOURCES rwlock.c)
parsec_addtest_executable(C future SOURCES future.c)
parsec_addtest_executable(C future_datacopy SOURCES future_datacopy.c)
parsec_addtest_executable(C lifo SOURCES lifo.c)
parsec_addtest_executable(C list SOURCES list.c)
parsec_addtest_executable(C hash SOURCES hash.c)

if(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
  parsec_addtest_executable(C atomics_inline SOURCES atomics.c)
  set_property(TARGET atomics_inline
               APPEND PROPERTY COMPILE_DEFINITIONS BUILDING_PARSEC)
  set_property(TARGET atomics_inline
               APPEND PROPERTY COMPILE_OPTIONS ${PARSEC_ATOMIC_SUPPORT_OPTIONS})
endif(PARSEC_HAVE_ERAND48 AND PARSEC_HAVE_NRAND48 AND PARSEC_HAVE_LRAND48)
parsec_addtest_executable(C rwlock_inline SOURCES rwlock.c)
parsec_addtest_executable(C lifo_inline SOURCES lifo.c)
parsec_addtest_executable(C list_inline SOURCES list.c)
parsec_addtest_executable(C hash_inline SOURCES hash.c)
set_property(TARGET rwlock_inline lifo_inline list_inline hash_inline
  APPEND PROPERTY COMPILE_DEFINITIONS BUILDING_PARSEC)
set_property(TARGET rwlock_inline lifo_inline list_inline hash_inline
  APPEND PROPERTY COMPILE_OPTIONS ${PARSEC_ATOMIC_SUPPORT_OPTIONS})

