/*
 * Copyright (c) 2012-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#define parsec_mfence     __sync
#define PARSEC_ATOMIC_HAS_RMB
#define RMB              __lwsync
#define PARSEC_ATOMIC_HAS_WMB
#define WMB()            __eieio

static inline int parsec_atomic_bor_32b( volatile uint32_t* location,
                                        uint32_t value )
{
    uint32_t old_value = __fetch_and_or(location, value);
    return old_value | value;
}

static inline int parsec_atomic_band_32b( volatile uint32_t* location,
                                         uint32_t value )
{
    uint32_t old_value = __fetch_and_and(location, value);
    return old_value & value;
}

static inline int parsec_atomic_cas_32b( volatile uint32_t* location,
                                        uint32_t old_value,
                                        uint32_t new_value )
{
    int32_t old = (int32_t)old_value;
    return __compare_and_swap( (volatile int*)location, &old, new_value );
}

/**
 * Use the XLC intrinsics directly.
 */
#define PARSEC_HAVE_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_64b __ldarx
#define parsec_atomic_sc_64b __stdcx
#define parsec_atomic_ll_32b __lwarx
#define parsec_atomic_sc_32b __stwcx
#define parsec_atomic_ll_ptr parsec_atomic_ll_64b
#define parsec_atomic_sc_ptr parsec_atomic_sc_64b

#if defined(PARSEC_ATOMIC_USE_XLC_64_BUILTINS)
static inline int parsec_atomic_cas_64b( volatile uint64_t* location,
                                        uint64_t old_value,
                                        uint64_t new_value )
{
    int64_t old = (int64_t)old_value;
    return __compare_and_swaplp( (volatile long*)location, &old, new_value );
}
#else
#include "parsec/debug.h"
static inline int parsec_atomic_cas_64b( volatile uint64_t* location,
                                        uint64_t old_value,
                                        uint64_t new_value )
{
    parsec_fatal("Use of 64b CAS using atomic-xlc without compiler support\n ");
    return -1;
}
#endif

#define PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
static inline uint32_t parsec_atomic_add_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    __sync();
    do {
        old_val = __lwarx( (volatile int*)location );
        tmp_val = old_val + i;
    } while( !__stwcx( (volatile int*)location, tmp_val ) );

    return( tmp_val );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B
static inline uint32_t parsec_atomic_sub_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    __sync();
    do {
        old_val = __lwarx( (volatile int*)location );
        tmp_val = old_val - i;
    } while( !__stwcx( (volatile int*)location, tmp_val ) );

    return( tmp_val );
}

