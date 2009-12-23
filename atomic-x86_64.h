/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    __asm__ __volatile__ (
                          "lock; orl %0,%1"
                          : : "r" (value), "m" (*(location))
                          : "memory");
    return *location;
}

static inline int dplasma_atomic_band_32b( volatile uint32_t* location,
                                           uint32_t value )
{
    __asm__ __volatile__ (
                          "lock; andl %0,%1"
                          : : "r" (value), "m" (*(location))
                          : "memory");
    return *location;
}

static inline int dplasma_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
    unsigned char ret;
    __asm__ __volatile__ (
                          "lock; cmpxchgl %3,%4   \n\t"
                          "      sete     %0      \n\t"
                          : "=qm" (ret), "=a" (old_value), "=m" (*location)
                          : "q"(new_value), "m"(*location), "1"(old_value)
                          : "memory", "cc");

    return (int)ret;
}

static inline int dplasma_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    unsigned char ret;
    __asm__ __volatile__ (
                          "lock; cmpxchgq %3,%4   \n\t"
                          "      sete     %0      \n\t"
                          : "=qm" (ret), "=a" (old_value), "=m" (*((volatile long*)location))
                          : "q"(new_value), "m"(*((volatile long*)location)), "1"(old_value)
                          : "memory", "cc");

   return (int)ret;
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dplasma_atomic_inc_32b( volatile uint32_t *location )
{
    __asm__ __volatile__ (
                          "lock; incl %0\n"
                          : "=m" (*(location))
                          : "m" (*(location)) );
    return (*location);
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dplasma_atomic_dec_32b( volatile uint32_t *location )
{
    __asm__ __volatile__ (
                          "lock; decl %0\n"
                          : "=m" (*(location))
                          : "m" (*(location)));
    return (*location);
}

