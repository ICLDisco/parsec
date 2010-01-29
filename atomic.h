/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_H_HAS_BEEN_INCLUDED
#define ATOMIC_H_HAS_BEEN_INCLUDED

#if defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4) || (defined(__ICL) && (__ICC > 1100))
#include "atomic-gcc.h"
#elif defined(MAC_OS_X)
#include "atomic-macosx.h"
#elif defined(X86)
#include "atomic-x86_32.h"
#elif defined(X86_64)
#include "atomic-x86_64.h"
#else
#error "Using unsafe atomics"
#endif

#include <assert.h>

static inline int dplasma_atomic_cas_xxb( volatile void* location,
                                          uint64_t old_value,
                                          uint64_t new_value,
                                          size_t type_size )
{
    switch(type_size){
    case 4:
        return dplasma_atomic_cas_32b( (volatile uint32_t*)location,
                                       (uint32_t)old_value, (uint32_t)new_value );
    case 8:
        return dplasma_atomic_cas_64b( (volatile uint64_t*)location,
                                       (uint64_t)old_value, (uint64_t)new_value );
    }
    return 0;
}

static inline uint64_t dplasma_atomic_bor_xxb( volatile void* location,
                                               uint64_t or_value,
                                               size_t type_size )
{
    assert( 4 == type_size );
	(void)type_size;
    return (uint64_t)dplasma_atomic_bor_32b( (volatile uint32_t*)location,
                                             (uint32_t)or_value);
}

#define dplasma_atomic_band(LOCATION, OR_VALUE)  \
    (__typeof__(*(LOCATION)))dplasma_atomic_band_xxb(LOCATION, OR_VALUE, sizeof(*(LOCATION)) )

#define dplasma_atomic_bor(LOCATION, OR_VALUE)  \
    (__typeof__(*(LOCATION)))dplasma_atomic_bor_xxb(LOCATION, OR_VALUE, sizeof(*(LOCATION)) )

#define dplasma_atomic_cas(LOCATION, OLD_VALUE, NEW_VALUE)              \
    dplasma_atomic_cas_xxb((volatile uint32_t*)(LOCATION),              \
                           (uint64_t)(OLD_VALUE), (uint64_t)(NEW_VALUE), \
                           sizeof(*(LOCATION)))                         \
    
#define dplasma_atomic_set_mask(LOCATION, MASK) dplasma_atomic_bor((LOCATION), (MASK))
#define dplasma_atomic_clear_mask(LOCATION, MASK)  dplasma_atomic_band((LOCATION), ~(MASK))

#ifndef DPLASMA_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dplasma_atomic_inc_32b( volatile uint32_t *location )
{
    uint32_t l;
    do {
        l = *location;
    } while( !dplasma_atomic_cas_32b( location, l, l+1 ) );
    return l+1;
}
#endif  /* DPLASMA_ATOMIC_HAS_ATOMIC_INC_32B */

#ifndef DPLASMA_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dplasma_atomic_dec_32b( volatile uint32_t *location )
{
    uint32_t l;
    do {
        l = *location;
    } while( !dplasma_atomic_cas_32b( location, l, l-1 ) );
    return l-1;
}
#endif  /* DPLASMA_ATOMIC_HAS_ATOMIC_DEC_32B */

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
