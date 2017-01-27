/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_H_HAS_BEEN_INCLUDED
#define ATOMIC_H_HAS_BEEN_INCLUDED

#include "parsec_config.h"
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#if (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__)

#include <stdatomic.h>
#include <time.h>

#define parsec_mfence() atomic_thread_fence(memory_order_seq_cst)
/**
 * This is extremely ugly but apparently it is the only way to correctly coherce
 * the compilers to convert to the correct type. Thanks to StackOverflow for the
 * tip (http://stackoverflow.com/questions/22851465/typeof-uses-in-c-besides-macros).
 */
#define parsec_atomic_band(LOCATION, AND_VALUE)  \
    (((__typeof__(*(LOCATION)))atomic_fetch_and((_Atomic __typeof__(*(LOCATION))(*))(LOCATION), (AND_VALUE))) & (AND_VALUE))

#define parsec_atomic_bor(LOCATION, OR_VALUE)  \
    (((__typeof__(*(LOCATION)))atomic_fetch_or((_Atomic __typeof__(*(LOCATION))(*))(LOCATION), (OR_VALUE))) | (OR_VALUE))

#define parsec_atomic_set_mask(LOCATION, MASK) parsec_atomic_bor((LOCATION), (MASK))
#define parsec_atomic_clear_mask(LOCATION, MASK)  parsec_atomic_band((LOCATION), ~(MASK))

static inline int32_t parsec_atomic_cas_32b(volatile uint32_t* location,
                                            uint32_t old_value,
                                            uint32_t new_value)
{
    return atomic_compare_exchange_strong( (_Atomic uint32_t*)location, &old_value, new_value );
}

static inline int32_t parsec_atomic_cas_64b(volatile uint64_t* location,
                                            uint64_t old_value,
                                            uint64_t new_value)
{
    return atomic_compare_exchange_strong( (_Atomic uint64_t*)location, &old_value, new_value );
}

static inline int32_t parsec_atomic_cas_128b(volatile __uint128_t* location,
                                            __uint128_t old_value,
                                            __uint128_t new_value)
{
    return atomic_compare_exchange_strong( (_Atomic __uint128_t*)location, &old_value, new_value );
}

#if PARSEC_SIZEOF_VOID_P == 4
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_32b( (volatile uint32_t*)(L), \
                                                            (uint32_t)(O), (uint32_t)(N) )
#elif PARSEC_SIZEOF_VOID_P == 8
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_64b( (volatile uint64_t*)(L), \
                                                            (uint64_t)(O), (uint64_t)(N) )
#else
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_128b( (volatile __uint128_t*)(L), \
                                                             (__uint128_t)(O), (__uint128_t)(N) )
#endif

#define parsec_atomic_add_32b(LOCATION, VALUE)                           \
    _Generic((LOCATION),                                                \
             int32_t* : (atomic_fetch_add((_Atomic int32_t*)(LOCATION), (VALUE)) + (VALUE)), \
             uint32_t*: (atomic_fetch_add((_Atomic uint32_t*)(LOCATION), (VALUE)) + (VALUE)), \
             default: (atomic_fetch_add((_Atomic int32_t*)(LOCATION), (VALUE)) + (VALUE)))

#define parsec_atomic_sub_32b(LOCATION, VALUE)                           \
    _Generic((LOCATION),                                                \
             int32_t* : (atomic_fetch_sub((_Atomic int32_t*)(LOCATION), (VALUE)) - (VALUE)), \
             uint32_t*: (atomic_fetch_sub((_Atomic uint32_t*)(LOCATION), (VALUE)) - (VALUE)), \
             default: (atomic_fetch_sub((_Atomic int32_t*)(LOCATION), (VALUE)) - (VALUE)))

#define parsec_atomic_inc_32b(LOCATION)                                  \
    _Generic((LOCATION),                                                \
             int32_t* : (1 + atomic_fetch_add((_Atomic int32_t*)(LOCATION), 1)), \
             uint32_t*: (1 + atomic_fetch_add((_Atomic uint32_t*)(LOCATION), 1)), \
             default: (1 + atomic_fetch_add((_Atomic int32_t*)(LOCATION), 1)))

#define parsec_atomic_dec_32b(LOCATION)                                  \
    _Generic((LOCATION),                                                \
             int32_t* : (atomic_fetch_sub((_Atomic int32_t*)(LOCATION), 1) - 1), \
             uint32_t*: (atomic_fetch_sub((_Atomic uint32_t*)(LOCATION), 1) - 1), \
             default: (atomic_fetch_sub((_Atomic int32_t*)(LOCATION), 1) - 1))

typedef volatile atomic_flag parsec_atomic_lock_t;

static inline void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
    while( atomic_flag_test_and_set(atomic_lock) )
        nanosleep( &ts, NULL ); /* less bandwidth consuming */
}

static inline void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    atomic_flag_clear(atomic_lock);
}

static inline long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return !atomic_flag_test_and_set(atomic_lock);
}

#define PARSEC_ATOMIC_UNLOCKED 0
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B 1

#else  /* (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__) */

/**
 * If the compiler provides atomic primitives we prefer to use
 * them instead of our own atomic assembly.
 */
#if defined(__FUJITSU)
  #undef PARSEC_ATOMIC_USE_XLC_32_BUILTINS
#endif
#if defined(PARSEC_ATOMIC_USE_XLC_32_BUILTINS)
#  include "atomic-xlc.h"
#elif defined(PARSEC_OSX)
/* Temporary workaround until we integrate C11 atomics */
#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_12
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#  endif  /* defined(__clang__) */
#  include "atomic-macosx.h"
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  endif  /* defined(__clang__) */
#endif  /* MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_12 */
#elif defined(PARSEC_ARCH_PPC)
#  if defined(__bgp__)
#    include "atomic-ppc-bgp.h"
#  else
#    include "atomic-ppc.h"
#  endif
#elif defined(PARSEC_ATOMIC_USE_GCC_32_BUILTINS)
#  include "atomic-gcc.h"
#elif defined(PARSEC_ARCH_X86)
#  include "atomic-x86_32.h"
#elif defined(PARSEC_ARCH_X86_64)
#  include "atomic-x86_64.h"
#else
#  error "No safe atomics available"
#endif

static inline uint64_t parsec_atomic_bor_xxb( volatile void* location,
                                             uint64_t or_value,
                                             size_t type_size )
{
    assert( 4 == type_size );
    (void)type_size;
    return (uint64_t)parsec_atomic_bor_32b( (volatile uint32_t*)location,
                                           (uint32_t)or_value);
}

#define parsec_atomic_band(LOCATION, OR_VALUE)  \
    (__typeof__(*(LOCATION)))parsec_atomic_band_xxb(LOCATION, OR_VALUE, sizeof(*(LOCATION)) )

#define parsec_atomic_bor(LOCATION, OR_VALUE)  \
    (__typeof__(*(LOCATION)))parsec_atomic_bor_xxb(LOCATION, OR_VALUE, sizeof(*(LOCATION)) )

#if PARSEC_SIZEOF_VOID_P == 4
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_32b( (volatile uint32_t*)(L), \
                                                            (uint32_t)(O), (uint32_t)(N) )
#elif PARSEC_SIZEOF_VOID_P == 8
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_64b( (volatile uint64_t*)(L), \
                                                            (uint64_t)(O), (uint64_t)(N) )
#else
#define parsec_atomic_cas_ptr(L, O, N) parsec_atomic_cas_128b( (volatile __uint128_t*)(L), \
                                                             (__uint128_t)(O), (__uint128_t)(N) )
#endif

#define parsec_atomic_set_mask(LOCATION, MASK) parsec_atomic_bor((LOCATION), (MASK))
#define parsec_atomic_clear_mask(LOCATION, MASK)  parsec_atomic_band((LOCATION), ~(MASK))

#ifndef PARSEC_ATOMIC_HAS_ATOMIC_INC_32B
#define PARSEC_ATOMIC_HAS_ATOMIC_INC_32B /* We now have it ! */

#ifdef PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
#define parsec_atomic_inc_32b(l)  parsec_atomic_add_32b((int32_t*)l, 1)
#else
static inline uint32_t parsec_atomic_inc_32b( volatile uint32_t *location )
{
    uint32_t l;
    do {
        l = *location;
    } while( !parsec_atomic_cas_32b( location, l, l+1 ) );
    return l+1;
}
#endif  /* PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B */
#endif  /* PARSEC_ATOMIC_HAS_ATOMIC_INC_32B */

#ifndef PARSEC_ATOMIC_HAS_ATOMIC_DEC_32B
#define PARSEC_ATOMIC_HAS_ATOMIC_DEC_32B /* We now have it ! */

#ifdef PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B
#define parsec_atomic_dec_32b(l)  parsec_atomic_sub_32b((int32_t*)l, 1)
#else
static inline uint32_t parsec_atomic_dec_32b( volatile uint32_t *location )
{
    uint32_t l;
    do {
        l = *location;
    } while( !parsec_atomic_cas_32b( location, l, l-1 ) );
    return l-1;
}
#endif  /* PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B */
#endif  /* PARSEC_ATOMIC_HAS_ATOMIC_DEC_32B */

#ifndef PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
#define PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
static inline uint32_t parsec_atomic_add_32b( volatile uint32_t *location, int32_t d )
{
    uint32_t l, n;
    do {
        l = *location;
        n = (uint32_t)((int32_t)l + d);
    } while( !parsec_atomic_cas_32b( location, l, n ) );
    return n;
}
#endif /* PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B */

typedef volatile uint32_t parsec_atomic_lock_t;
/**
 * Enumeration of lock states
 */
enum {
    PARSEC_ATOMIC_UNLOCKED = 0,
    PARSEC_ATOMIC_LOCKED   = 1
};

static inline void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    while( !parsec_atomic_cas_32b( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    parsec_mfence();
    *atomic_lock = 0;
}

static inline long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return parsec_atomic_cas_32b( atomic_lock, 0, 1 );
}
#endif  /* (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__) */

#if !defined(PARSEC_ATOMIC_HAS_WMB)
#define parsec_atomic_wmb    parsec_mfence
#endif  /* !defined(PARSEC_ATOMIC_HAS_WMB) */
#if !defined(PARSEC_ATOMIC_HAS_RMB)
#define parsec_atomic_rmb    parsec_mfence
#endif  /* !defined(PARSEC_ATOMIC_HAS_RMB) */

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
