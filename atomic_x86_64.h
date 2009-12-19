/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
    __asm__ __volatile__ (
                          "lock; orl %1,%0"
                          : "+m" (*(location))
                          : "r" (value)
                          : "memory");
    return *location;
}

static inline int dplasma_atomic_band_32b( volatile uint32_t* location,
                                           uint32_t value )
{
    __asm__ __volatile__ (
                          "lock; andl %1,%0"
                          : "+m" (*(location))
                          : "r" (~(value))
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
                          "sete     %0      \n\t"
                          : "=qm" (ret), "+a" (oldval), "+m" (*addr)
                          : "q"(newval)
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
                          "sete     %0      \n\t"
                          : "=qm" (ret), "+a" (oldval), "+m" (*((volatile long*)addr))
                          : "q"(newval)
                            " "memory", "cc");

   return (int)ret;
}
