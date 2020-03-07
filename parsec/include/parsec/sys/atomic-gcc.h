/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Memory Barriers */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence( void )
{
    __sync_synchronize();
}

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32( volatile int32_t* location,
                             int32_t old_value,
                             int32_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}

#if defined(PARSEC_ATOMIC_USE_GCC_64_BUILTINS)
#  define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64( volatile int64_t* location,
                             int64_t old_value,
                             int64_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}
#else
#  error "Use of 64b CAS using atomic-gcc without __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 set"
#endif

#if defined(PARSEC_HAVE_INT128)
#if defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int128( volatile __int128_t* location,
                              __int128_t old_value,
                              __int128_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}
#else
#  error "Requirement of CAS_128, but cannot use GCC_128_BUILTINS"
#endif /* defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS) */
#endif /* PARSEC_HAVE_INT128 */

/* Mask */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t value )
{
    return __sync_fetch_and_or(location, value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                       int32_t value )
{
    return __sync_fetch_and_and(location, value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                      int64_t value )
{
    return __sync_fetch_and_or(location, value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                       int64_t value )
{
    return __sync_fetch_and_and(location, value);
}

#if defined(PARSEC_HAVE_INT128) && defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
#if defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_or_int128( volatile __int128_t* location,
                                          __int128_t or_value )
{
    return __sync_fetch_and_or(location, or_value);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_and_int128( volatile __int128_t* location,
                                           __int128_t and_value )
{
    return __sync_fetch_and_and(location, and_value);
}
#else  /* defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS) */
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_or_int128( volatile __int128_t* location,
                                          __int128_t or_value )
{
    __int128_t old;
    do {
        old = *location;
    } while( !__sync_bool_compare_and_swap(location, old, old | or_value ) );
    return old;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_and_int128( volatile __int128_t* location,
                                           __int128_t and_value )
{
    __int128_t old;
    do {
        old = *location;
    } while( !__sync_bool_compare_and_swap(location, old, old & and_value) );
    return old;
}
#endif  /* defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS) */
#endif  /* defined(PARSEC_HAVE_INT128) && defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS) */

/* Integer -- we use __sync_fetch_and_add for all, let atomic.h translate the missing ones */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32(volatile int32_t* l, int32_t v)
{
    return __sync_fetch_and_add(l, v);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int64(volatile int64_t* l, int64_t v)
{
    return __sync_fetch_and_add(l, v);
}

#if defined(PARSEC_HAVE_INT128) && defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
#if defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_add_int128(volatile __int128_t* l, __int128_t v)
{
    return __sync_fetch_and_add(l, v);
}

#else  /* defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS) */
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_add_int128( volatile __int128_t* location,
                                           __int128_t v)
{
    __int128_t old;
    do {
        old = *location;
    } while( !__sync_bool_compare_and_swap(location, old, old + v) );
    return old;
}
#endif  /* defined(PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS) */
#endif  /* defined(PARSEC_HAVE_INT128) && defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS) */
