/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Warning: 
 *  as of May 10, 2018, this file has not been tested, for lack of target architecture */

#if defined(PARSEC_HAVE_INT128)
/* INT128 support was detected, yet there is no 128 atomics on this architecture */
#error CMake Logic Error. INT128 support should be deactivated when using these atomics
#endif

/* Memory Barriers */

#define PARSEC_ATOMIC_HAS_MFENCE
ATOMIC_STATIC_INLINE
void parsec_mfence(void)
{
    __asm__ __volatile__ ("mfence\n\t":::"memory");
}

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32(volatile int32_t* location,
                            int32_t old_value,
                            int32_t new_value)
{
    unsigned char ret;
    __asm__ __volatile__ (
                          "lock; cmpxchgl %3,%2   \n\t"
                          "sete     %0      \n\t"
                          : "=qm" (ret), "+a" (old_value), "+m" (*location)
                          : "q"(new_value)
                          : "memory", "cc");

    return (int)ret;
}

#define ll_low(x)	*(((unsigned int *)&(x)) + 0)
#define ll_high(x)	*(((unsigned int *)&(x)) + 1)

ATOMIC_STATIC_INLINE
int64_t parsec_atomic_cas_int64(volatile int64_t* location,
                                int64_t old_value,
                                int64_t new_value)
{
   /*
    * Compare EDX:EAX with m64. If equal, set ZF and load ECX:EBX into
    * m64. Else, clear ZF and load m64 into EDX:EAX.
    */
    unsigned char ret;

    __asm__ __volatile__(
                    "push %%ebx            \n\t"
                    "movl %4, %%ebx        \n\t"
                    "lock cmpxchg8b (%1)  \n\t"
                    "sete %0               \n\t"
                    "pop %%ebx             \n\t"
                    : "=qm"(ret)
                    : "D"(location), "a"(ll_low(old_value)), "d"(ll_high(old_value)),
                      "r"(ll_low(new_value)), "c"(ll_high(new_value))
                    : "cc", "memory", "ebx");
    return (int) ret;
}

#undef ll_low
#undef ll_high

/* Mask */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32(volatile int32_t* location,
                                     int32_t value)
{
    int32_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_32b(location, old_value, (old_value|value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32(volatile int32_t* location,
                                      int32_t value)
{
    uint32_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_32b(location, old_value, (old_value&value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64(volatile int64_t* location,
                                     int64_t value)
{
    int64_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_64b(location, old_value, (old_value|value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64(volatile int64_t* location,
                                      int64_t value)
{
    uint64_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_64b(location, old_value, (old_value&value) ));
    return old_value;
}

/* Integer */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32(volatile int32_t* v, int32_t i)
{
    int32_t ret = i;
    /* See https://en.wikipedia.org/wiki/Fetch-and-add */
   __asm__ __volatile__(
                        "lock; xaddl %1,%0"
                        :"+m" (*v), "+r" (ret)
                        :
                        :"memory", "cc"
                        );
   return ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int364(volatile int64_t* v, int64_t i)
{
    int64_t ov, nv;

    do {
        ov = *v;
        nv = ov + i;
    } while( !parsec_atomic_cas_int64(v, ov, nv) );
    return ov;
}

