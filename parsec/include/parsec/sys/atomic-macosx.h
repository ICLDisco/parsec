/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_OSX
#error This file should only be included on MAC OS X > Snow Leopard
#endif

#if defined(PARSEC_HAVE_INT128)
/* INT128 support was detected, yet there is no 128 atomics on this architecture */
#error CMake Logic Error. INT128 support should be deactivated when using these atomics
#endif

#include <libkern/OSAtomic.h>

/* Memory Barriers */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence( void )
{
    OSMemoryBarrier();
}

/* Compare and swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32( volatile int32_t* location,
                             int32_t old_value,
                             int32_t new_value )
{
    return OSAtomicCompareAndSwap32( old_value, new_value, location );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64( volatile int64_t* location,
                             int64_t old_value,
                             int64_t new_value )
{
    return OSAtomicCompareAndSwap64( old_value, new_value, location );
}

#if defined(PARSEC_ATOMIC_USE_GCC_128_BUILTINS)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int128( volatile __int128_t* location,
                              __int128_t old_value,
                              __int128_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}
#endif

/* Mask */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t value )
{
    return OSAtomicOr32Orig( *(uint32_t*)&value, (uint32_t*)location );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                       int32_t value )
{
    return OSAtomicAnd32Orig( *(uint32_t*)&value, (uint32_t*)location );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                      int64_t value )
{
    int64_t ov, nv;
    do {
        ov = *location;
        nv = ov | value;
    } while( !parsec_atomic_cas_int64(location, ov, nv) );
    return ov;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                       int64_t value )
{
    int64_t ov, nv;
    do {
        ov = *location;
        nv = ov & value;
    } while( !parsec_atomic_cas_int64(location, ov, nv) );
    return ov;
}

/* Integer Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_inc_int32( volatile int32_t *location )
{
    return OSAtomicIncrement32( location ) - 1;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_dec_int32( volatile int32_t *location )
{
    return OSAtomicDecrement32( location ) + 1;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32( volatile int32_t *location, int32_t i )
{
    return OSAtomicAdd32( i, location ) - i;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_sub_int32( volatile int32_t *location, int32_t i )
{
    return OSAtomicAdd32( -i, location ) + i;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_inc_int64( volatile int64_t *location )
{
    return OSAtomicIncrement64( location ) - 1;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_dec_int64( volatile int64_t *location )
{
    return OSAtomicDecrement64( location ) + 1;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int64( volatile int64_t *location, int64_t i )
{
    return OSAtomicAdd64( i, location ) - i;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_sub_int64( volatile int64_t *location, int64_t i )
{
    return OSAtomicAdd64( -i, location ) + i;
}
