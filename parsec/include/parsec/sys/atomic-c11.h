/*
 * Copyright (c) 2016-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdatomic.h>
#include <time.h>


/**
 * Control the default memory order of all C11 atomic operations,
 * with the exception of parsec_atomic_lock operations, which use proper
 * acquire/release semantics, and the memory barriers.
 *
 * Change this define to memory_order_seq_cst to restore sequential
 * consistency of atomic operations.
 *
 * NOTE: semantically, we should use memory_order_consume to ensure that
 *       data dependencies on values produced from these operations are
 *       honored. Unfortunately, compilers still insert some barriers in
 *       this case. Almost all architectures will honor the data dependencies
 *       in hardware though (unless we're running on a DEC Alpha).
 *       See https://preshing.com/20140709/the-purpose-of-memory_order_consume-in-cpp11/
 *       Once the C standard has relaxed consume semantics, we should revisit this.
 */
#define PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER memory_order_relaxed

/* Memory Barriers */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence(void)
{
    atomic_thread_fence(memory_order_seq_cst);
}

#define PARSEC_ATOMIC_HAS_WMB
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    atomic_thread_fence(memory_order_release);
}

#define PARSEC_ATOMIC_HAS_RMB
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    /* Earlier GCC versions ignored the acquire thread fence.
     * It seems to be resolved in 8.3 so play it safe by
     * falling back to mfence for anything earlier than 8.3.
     */
#if !defined(__clang__) \
   && defined(__GNUC__) \
   && (__GNUC__ > 8 || (__GNUC__ == 8 && __GNUC_MINOR__ >= 3))
    atomic_thread_fence(memory_order_acquire);
#else
    parsec_mfence();
#endif
 }

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32(volatile int32_t* location,
                            int32_t old_value,
                            int32_t new_value)
{
    return atomic_compare_exchange_strong_explicit( (_Atomic int32_t*)location, &old_value, new_value,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64(volatile int64_t* location,
                            int64_t old_value,
                            int64_t new_value)
{
    return atomic_compare_exchange_strong_explicit( (_Atomic int64_t*)location, &old_value, new_value,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER );
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int128(volatile __int128_t* location,
                             __int128_t old_value,
                             __int128_t new_value)
{
    return atomic_compare_exchange_strong_explicit( (_Atomic __int128_t*)location, &old_value, new_value,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER,
                                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER );
}
#endif  /* defined(PARSEC_HAVE_INT128) */

/* Mask Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t or_value )
{
    return atomic_fetch_or_explicit((_Atomic int32_t*)location, or_value,
                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                      int64_t or_value)
{
    return atomic_fetch_or_explicit((_Atomic int64_t*)location, or_value,
                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                       int32_t and_value )
{
    return atomic_fetch_and_explicit((_Atomic int32_t*)location, and_value,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                       int64_t and_value )
{
    return atomic_fetch_and_explicit((_Atomic int64_t*)location, and_value,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_or_int128( volatile __int128_t* location,
                                          __int128_t or_value )
{
    return atomic_fetch_or_explicit((_Atomic __int128_t*)location, or_value,
                                    PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_and_int128( volatile __int128_t* location,
                                           __int128_t and_value )
{
    return atomic_fetch_and_explicit((_Atomic __int128_t*)location, and_value,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}
#endif

/* Integer Operations -- we use atomic_fetch_add for all, let atomic.h translate the rest */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32(volatile int32_t* l, int32_t v)
{
    return atomic_fetch_add_explicit((_Atomic int32_t*)l, v,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int64(volatile int64_t* l, int64_t v)
{
    return atomic_fetch_add_explicit((_Atomic int64_t*)l, v,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_add_int128(volatile __int128_t* l, __int128_t v)
{
    return atomic_fetch_add_explicit((_Atomic __int128_t*)l, v,
                                     PARSEC_ATOMIC_STDC_DEFAULT_MEMORDER);
}
#endif

/* Locks */

typedef volatile atomic_flag parsec_atomic_lock_t;

#  define PARSEC_ATOMIC_HAS_ATOMIC_INIT
ATOMIC_STATIC_INLINE
void parsec_atomic_lock_init( parsec_atomic_lock_t* atomic_lock )
{
    atomic_flag_clear_explicit(atomic_lock, memory_order_relaxed);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_LOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
    while( atomic_flag_test_and_set_explicit(atomic_lock, memory_order_acquire) )
        nanosleep( &ts, NULL ); /* less bandwidth consuming */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_UNLOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    atomic_flag_clear_explicit(atomic_lock, memory_order_release);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_TRYLOCK
ATOMIC_STATIC_INLINE
int parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return !atomic_flag_test_and_set_explicit(atomic_lock, memory_order_acquire);
}

#define PARSEC_ATOMIC_UNLOCKED ATOMIC_FLAG_INIT
