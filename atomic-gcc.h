/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    uint32_t old_value = __sync_fetch_and_or(location, value);
    return old_value | value;
}

static inline int dplasma_atomic_band_32b( volatile uint32_t* location,
                                           uint32_t value )
{
    uint32_t old_value = __sync_fetch_and_and(location, value);
    return old_value & value;
}

static inline int dplasma_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}

static inline int dplasma_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    return (__sync_bool_compare_and_swap(location, old_value, new_value) ? 1 : 0);
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dplasma_atomic_inc_32b( volatile uint32_t *location )
{
    return __sync_add_and_fetch(location, (uint32_t)1);
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dplasma_atomic_dec_32b( volatile uint32_t *location )
{
    return __sync_sub_and_fetch(location, (uint32_t)1);
}

