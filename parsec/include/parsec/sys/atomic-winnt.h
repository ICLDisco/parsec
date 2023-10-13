/*
 * Copyright (c) 2016-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <Windows.h>

/* Memory Barriers */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence(void)
{
    MemoryBarrier();
}

#define PARSEC_ATOMIC_HAS_WMB
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    _WriteBarrier();
}

#define PARSEC_ATOMIC_HAS_RMB
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    _ReadBarrier();
}

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32(volatile int32_t* location,
                            int32_t old_value,
                            int32_t new_value)
{
    return old_value == InterlockedCompareExchange(location, new_value, old_value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64(volatile int64_t* location,
                            int64_t old_value,
                            int64_t new_value)
{
    return old_value == InterlockedCompareExchange64(location, new_value, old_value);
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int128(volatile __int128_t* location,
                             __int128_t old_value,
                             __int128_t new_value)
{
    return old_value == InterlockedCompareExchange128(location, new_value, new_value, &old_value);
}
#endif  /* defined(PARSEC_HAVE_INT128) */

/* Mask Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t or_value )
{
    return InterlockedOr(location, or_value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                      int64_t or_value)
{
    return InterlockedOr64(location, or_value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                       int32_t and_value )
{
    return InterlockedAnd(location, and_value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                       int64_t and_value )
{
    return InterlockedAnd64(location, and_value);
}

/* Integer Operations -- we use atomic_fetch_add for all, let atomic.h translate the rest */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32(volatile int32_t* l, int32_t v)
{
    return InterlockedExchangeAdd(l, v);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int64(volatile int64_t* l, int64_t v)
{
    return InterlockedExchangeAdd64(l, v);
}

/* Locks */

typedef volatile atomic_flag parsec_atomic_lock_t;

#define PARSEC_ATOMIC_HAS_ATOMIC_LOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    while( atomic_flag_test_and_set(atomic_lock) )
        NtDelayExecution(FALSE, 100);  /* less bandwidth consuming */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_UNLOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    atomic_flag_clear(atomic_lock);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_TRYLOCK
ATOMIC_STATIC_INLINE
int parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return !atomic_flag_test_and_set(atomic_lock);
}

#define PARSEC_ATOMIC_UNLOCKED ATOMIC_FLAG_INIT
