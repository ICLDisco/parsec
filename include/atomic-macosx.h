/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MAC_OS_X
#error This file should only be included on MAC OS X (Snow Leopard
#endif

#include <libkern/OSAtomic.h>

static inline void dague_mfence( void )
{
    OSMemoryBarrier();
}

static inline int dague_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    return OSAtomicOr32( value, location );
}

static inline int dague_atomic_band_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    return OSAtomicAnd32( value, location );
}

static inline int dague_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
    return OSAtomicCompareAndSwap32( old_value, new_value, (volatile int32_t*)location );
}

static inline int dague_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    return OSAtomicCompareAndSwap64( old_value, new_value, (volatile int64_t*)location );
}

#define DAGUE_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dague_atomic_inc_32b( volatile uint32_t *location )
{
    return OSAtomicIncrement32( (int32_t*)location );
}

#define DAGUE_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dague_atomic_dec_32b( volatile uint32_t *location )
{
    return OSAtomicDecrement32( (int32_t*)location );
}

