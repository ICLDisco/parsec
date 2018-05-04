/*
 * Copyright (c) 2012-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence(void)
{
    __sync();
}

#define PARSEC_ATOMIC_HAS_WMB
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    __eieio();
}

#define PARSEC_ATOMIC_HAS_RMB
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    __lwsync();
}

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE int parsec_atomic_cas_int32( volatile int32_t *location,
                                                  int32_t old_value,
                                                  int32_t new_value )
{
    return __compare_and_swap( location, &old_value, new_value );
}

#if defined(PARSEC_ATOMIC_USE_XLC_64_BUILTINS)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE int parsec_atomic_cas_int64( volatile int64_t* location,
                                                  int64_t old_value,
                                                  int64_t new_value )
{
    return __compare_and_swaplp( location, &old_value, new_value );
}
#else
extern void parsec_fatal();
ATOMIC_STATIC_INLINE int parsec_atomic_cas_64b( volatile uint64_t* location,
                                        uint64_t old_value,
                                        uint64_t new_value )
{
    parsec_fatal("Use of 64b CAS using atomic-xlc without compiler support\n ");
    return -1;
}
#endif

#if defined(PARSEC_HAVE_INT128)
/* INT128 support was detected, yet there is no 128 atomics on this architecture */
#error CMake Logic Error. INT128 support should be deactivated when using XLC native atomics
#endif

/* Mask Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                                           int32_t value )
{
    unsigned int ret;
    ret = __fetch_and_or( (volatile unsigned int *)location, (unsigned int)value );
    return *(int32_t*)&ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                                           int64_t value )
{
    unsigned long ret;
    ret = __fetch_and_orlp( (volatile unsigned long *)location, (unsigned long)value );
    return *(int64_t*)&ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                                            int32_t value )
{
    unsigned int ret;
    ret = __fetch_and_and( (volatile unsigned int *)location, (unsigned int)value );
    return *(int32_t*)&ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                                            int64_t value )
{
    unsigned long ret;
    ret = __fetch_and_andlp( (volatile unsigned long *)location, (unsigned long)value );
    return *(int64_t*)&ret;
}

/* Linked Load / Store Conditional */

#define PARSEC_HAVE_ATOMIC_LLSC

#define PARSEC_HAVE_ATOMIC_LLSC_INT64
#define parsec_atomic_ll_int64 __ldarx
#define parsec_atomic_sc_int64 __stdcx

#define PARSEC_HAVE_ATOMIC_LLSC_INT32
#define parsec_atomic_ll_int32 __lwarx
#define parsec_atomic_sc_int32 __stwcx

#if PARSEC_SIZEOF_VOID_P == 4
#define PARSEC_HAVE_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_ptr parsec_atomic_ll_int32
#define parsec_atomic_sc_ptr parsec_atomic_sc_int32
#elif PARSEC_SIZEOF_VOID_P == 8
#define PARSEC_HAVE_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_ptr parsec_atomic_ll_int64
#define parsec_atomic_sc_ptr parsec_atomic_sc_int64
#else
/* No LLSC for PTR */
#warning CMake logic error: should not have selected these atomics on this architecture
#endif

/* Integer -- we use LL/SC for all, let atomic.h translate the missing ones */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE int32_t parsec_atomic_fetch_add_int32( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    __sync();
    do {
        old_val = __lwarx( location );
        tmp_val = old_val + i;
    } while( !__stwcx( location, tmp_val ) );

    return old_val;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE int64_t parsec_atomic_fetch_add_int64( volatile int64_t *location, int64_t i )
{
    register int64_t old_val, tmp_val;

    __sync();
    do {
        old_val = __ldarx( location );
        tmp_val = old_val + i;
    } while( !__stdcx( location, tmp_val ) );

    return old_val;
}
