/*
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdatomic.h>
#include <time.h>

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
    atomic_thread_fence(memory_order_acquire);
}

/**
 * This is extremely ugly but apparently it is the only way to correctly coherce
 * the compilers to convert to the correct type. Thanks to StackOverflow for the
 * tip (http://stackoverflow.com/questions/22851465/typeof-uses-in-c-besides-macros).
 */
#define parsec_atomic_bor(LOCATION, OR_VALUE)  \
    (((__typeof__(*(LOCATION)))atomic_fetch_or((_Atomic __typeof__(*(LOCATION))(*))(LOCATION), (OR_VALUE))) | (OR_VALUE))

ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_bor_32b( volatile uint32_t* location,
                                uint32_t or_value)
{
    return (or_value | atomic_fetch_or((atomic_uint*)location, or_value));
}

#define parsec_atomic_band(LOCATION, AND_VALUE)  \
    (((__typeof__(*(LOCATION)))atomic_fetch_and((_Atomic __typeof__(*(LOCATION))(*))(LOCATION), (AND_VALUE))) | (AND_VALUE))

ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_band_32b( volatile uint32_t* location,
                                uint32_t and_value)
{
    return (and_value & atomic_fetch_and((atomic_uint*)location, and_value));
}

ATOMIC_STATIC_INLINE
int32_t parsec_atomic_cas_32b(volatile uint32_t* location,
                              uint32_t old_value,
                              uint32_t new_value)
{
    return atomic_compare_exchange_strong( (atomic_uint*)location, &old_value, new_value );
}

ATOMIC_STATIC_INLINE
int32_t parsec_atomic_cas_64b(volatile uint64_t* location,
                              uint64_t old_value,
                              uint64_t new_value)
{
    if (sizeof(atomic_ulong) == sizeof(uint64_t))
        return atomic_compare_exchange_strong( (atomic_ulong*)location, (unsigned long*)&old_value, new_value );
    if (sizeof(atomic_ullong) == sizeof(uint64_t))
      return atomic_compare_exchange_strong( (atomic_ullong*)location, (unsigned long long*)&old_value, new_value );
    *((int*)0x0) = 0;  /* not good, there is no support */
    return 0;
}

#if defined(PARSEC_HAVE_UINT128)
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_cas_128b(volatile __uint128_t* location,
                               __uint128_t old_value,
                               __uint128_t new_value)
{
    return atomic_compare_exchange_strong( (_Atomic __uint128_t*)location, &old_value, new_value );
}
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B 1
#endif  /* defined(PARSEC_HAVE_UINT128b) */

#if PARSEC_SIZEOF_VOID_P == 4
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_32b((volatile uint32_t*)l, (uint32_t)o, (uint32_t)n);
}
#elif PARSEC_SIZEOF_VOID_P == 8
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_64b((volatile uint64_t*)l, (uint64_t)o, (uint64_t)n);
}
#else
#if defined(PARSEC_HAVE_UINT128)
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, __uint128_t o, __uint128_t n)
{
    return parsec_atomic_cas_128b((volatile __uint128_t*)l, o, n);
}
#else  /* defined(PARSEC_HAVE_UINT128b) */
#error Pointers are 128 bits long but no atomic operation on 128 bits are available
#endif  /* defined(PARSEC_HAVE_UINT128) */
#endif

#define PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_add_32b(volatile int32_t* l, int32_t v)
{
    return v + atomic_fetch_add((_Atomic uint32_t*)l, v);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_INC_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_inc_32b(volatile uint32_t* l)
{
    return 1 + atomic_fetch_add((_Atomic uint32_t*)l, 1);
}

#define PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_sub_32b(volatile int32_t* l, int32_t v)
{
    return atomic_fetch_sub((_Atomic uint32_t*)l, v) - v;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_DEC_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_dec_32b(volatile uint32_t* l)
{
    return atomic_fetch_add((_Atomic uint32_t*)l, -1) - 1;
}

typedef volatile atomic_flag parsec_atomic_lock_t;

ATOMIC_STATIC_INLINE
void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
    while( atomic_flag_test_and_set(atomic_lock) )
        nanosleep( &ts, NULL ); /* less bandwidth consuming */
}

ATOMIC_STATIC_INLINE
void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    atomic_flag_clear(atomic_lock);
}

ATOMIC_STATIC_INLINE
long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return !atomic_flag_test_and_set(atomic_lock);
}

#define PARSEC_ATOMIC_UNLOCKED 0

