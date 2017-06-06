/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_H_HAS_BEEN_INCLUDED
#define ATOMIC_H_HAS_BEEN_INCLUDED

#include "parsec_config.h"
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

BEGIN_C_DECLS

#if !defined(BUILD_PARSEC)
#include "atomic-external.h"
#else  /* !defined(BUILD_PARSEC) */

#if !defined(ATOMIC_STATIC_INLINE)
#define ATOMIC_STATIC_INLINE static inline
#endif  /* !defined(ATOMIC_STATIC_INLINE) */

/*
 * This define will exists only in the PaRSEC build, and should remain undefined
 * in all other contexts.
 */
#define PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED 1

#if defined(PARSEC_ATOMIC_USE_C11_ATOMICS)
#include "atomic-c11.h"
#else  /* defined(PARSEC_ATOMIC_USE_C11_ATOMICS) */

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
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_128b((volatile __uint128_t*)l, (__uint128_t)o, (__uint128_t)n);
}
#endif

#ifndef PARSEC_ATOMIC_HAS_ATOMIC_INC_32B
#define PARSEC_ATOMIC_HAS_ATOMIC_INC_32B /* We now have it ! */

#ifdef PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_inc_32b( volatile uint32_t *location )
{
    return (uint32_t)parsec_atomic_add_32b((int32_t*)location, 1);
}
#else
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_inc_32b( volatile uint32_t *location )
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
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_dec_32b( volatile uint32_t *location )
{
    return parsec_atomic_sub_32b((int32_t*)location, 1);
}
#else
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_dec_32b( volatile uint32_t *location )
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
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_add_32b( volatile uint32_t *location, int32_t d )
{
    uint32_t l, n;
    do {
        l = *location;
        n = (uint32_t)((int32_t)l + d);
    } while( !parsec_atomic_cas_32b( location, l, n ) );
    return n;
}
#endif /* PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B */

#if !defined(PARSEC_ATOMIC_HAS_WMB)
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    parsec_atomic_mfence();
}
#endif  /* !defined(PARSEC_ATOMIC_HAS_WMB) */
#if !defined(PARSEC_ATOMIC_HAS_RMB)
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    parsec_atomic_mfence();
}

typedef volatile uint32_t parsec_atomic_lock_t;

/**
 * Enumeration of lock states
 */
enum {
    PARSEC_ATOMIC_UNLOCKED = 0,
    PARSEC_ATOMIC_LOCKED   = 1
};

ATOMIC_STATIC_INLINE
void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    while( !parsec_atomic_cas_32b( atomic_lock, 0, 1) )
        /* nothing */;
}

ATOMIC_STATIC_INLINE
void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    parsec_mfence();
    *atomic_lock = 0;
}

ATOMIC_STATIC_INLINE
long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return parsec_atomic_cas_32b( atomic_lock, 0, 1 );
}
#endif  /* (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__) */

#endif  /* !defined(BUILD_PARSEC) */

#endif  /* !defined(PARSEC_ATOMIC_HAS_RMB) */

END_C_DECLS

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
