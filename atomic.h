/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef ATOMIC_H_HAS_BEEN_INCLUDED
#define ATOMIC_H_HAS_BEEN_INCLUDED

#ifdef MAC_OS_X
#include "atomic-macosx.h"
#elif X86
#include "atomic-x86_32.h"
#elif X86_64
#include "atomic-x86_64.h"
#else
static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    *location |= value;
    return *location;
}

static inline int dplasma_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
    if( old_value == (*location) ) {
        *location = new_value;
        return 1;
    }
    return 0;
}

static inline int dplasma_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    if( old_value == (*location) ) {
        *location = new_value;
        return 1;
    }
    return 0;
}
#endif

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

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
