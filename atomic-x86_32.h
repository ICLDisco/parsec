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
   /*
    * Compare EDX:EAX with m64. If equal, set ZF and load ECX:EBX into
    * m64. Else, clear ZF and load m64 into EDX:EAX.
    */
    unsigned char ret;

    __asm__ __volatile__(
                    "push %%ebx            \n\t"
                    "movl %4, %%ebx        \n\t"
                    SMPLOCK "cmpxchg8b (%1)  \n\t"
                    "sete %0               \n\t"
                    "pop %%ebx             \n\t"
                    : "=qm"(ret)
                    : "D"(addr), "a"(ll_low(oldval)), "d"(ll_high(oldval)),
                      "r"(ll_low(newval)), "c"(ll_high(newval))
                    : "cc", "memory", "ebx");
    return (int) ret;
}
