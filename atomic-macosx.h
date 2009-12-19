/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MAC_OS_X
#error This file should only be included on MAC OS X (Snow Leopard
#endif

#include <libkern/OSAtomic.h>

static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    return OSAtomicOr32( value, location );
}

static inline int dplasma_atomic_band_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    return OSAtomicAnd32( value, location );
}

static inline int dplasma_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
    return OSAtomicCompareAndSwap32( old_value, new_value, (volatile int32_t*)location );
}

static inline int dplasma_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    return OSAtomicCompareAndSwap64( old_value, new_value, (volatile int64_t*)location );
}
