/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

static inline void dague_mfence( void )
{
    __sync_synchronize();
}

static inline int dague_atomic_bor_32b( volatile uint32_t* location,
                                        uint32_t value )
{
    uint32_t old_value = __sync_fetch_and_or(location, value);
    return old_value | value;
}

static inline int dague_atomic_band_32b( volatile uint32_t* location,
                                         uint32_t value )
{
    uint32_t old_value = __sync_fetch_and_and(location, value);
    return old_value & value;
}

static inline int dague_atomic_cas_32b( volatile uint32_t* location,
                                        uint32_t old_value,
                                        uint32_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}

#if defined(DAGUE_ATOMIC_USE_GCC_64_BUILTINS)
static inline int dague_atomic_cas_64b( volatile uint64_t* location,
                                        uint64_t old_value,
                                        uint64_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}
#else
#include "debug.h"
static inline int dague_atomic_cas_64b( volatile uint64_t* location,
                                        uint64_t old_value,
                                        uint64_t new_value )
{
    ERROR(("Use of 64b CAS using atomic-gcc without __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 set\n \n"));
    (void)location; (void)old_value; (void)new_value;
    return -1;
}
#endif

#define DAGUE_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dague_atomic_inc_32b( volatile uint32_t *location )
{
    return __sync_add_and_fetch(location, (uint32_t)1);
}

#define DAGUE_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dague_atomic_dec_32b( volatile uint32_t *location )
{
    return __sync_sub_and_fetch(location, (uint32_t)1);
}

#define DAGUE_ATOMIC_HAS_ATOMIC_ADD_32B
static inline uint32_t dague_atomic_add_32b( volatile uint32_t *location, int32_t d )
{
    return (uint32_t)__sync_add_and_fetch((int32_t*)location, d);
}
