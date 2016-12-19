/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_H_HAS_BEEN_INCLUDED
#define ATOMIC_H_HAS_BEEN_INCLUDED

#include "parsec_config.h"
#include <stdint.h>
#include <unistd.h>

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

#include <assert.h>

#if !defined(PARSEC_ATOMIC_HAS_WMB)
#define parsec_atomic_wmb    parsec_mfence
#endif  /* !defined(PARSEC_ATOMIC_HAS_WMB) */
#if !defined(PARSEC_ATOMIC_HAS_RMB)
#define parsec_atomic_rmb    parsec_mfence
#endif  /* !defined(PARSEC_ATOMIC_HAS_RMB) */

static inline int parsec_atomic_cas_xxb( volatile void* location,
                                        uint64_t old_value,
                                        uint64_t new_value,
                                        size_t type_size )
{
    switch(type_size){
    case 4:
        return parsec_atomic_cas_32b( (volatile uint32_t*)location,
                                     (uint32_t)old_value, (uint32_t)new_value );
    case 8:
        return parsec_atomic_cas_64b( (volatile uint64_t*)location,
                                     (uint64_t)old_value, (uint64_t)new_value );
    }
    return 0;
}

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

#define parsec_atomic_cas(LOCATION, OLD_VALUE, NEW_VALUE)               \
    parsec_atomic_cas_xxb((volatile void*)(LOCATION),                   \
                         (uint64_t)(OLD_VALUE), (uint64_t)(NEW_VALUE), \
                         sizeof(*(LOCATION)))


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
    while( !parsec_atomic_cas( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    parsec_mfence();
    *atomic_lock = 0;
}

static inline long parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return parsec_atomic_cas( atomic_lock, 0, 1 );
}

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
