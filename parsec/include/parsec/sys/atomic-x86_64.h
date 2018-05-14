/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Warning: 
 *  as of May 10, 2018, this file has not been tested, for lack of target architecture */

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
                          "      sete     %0      \n\t"
                          : "=qm" (ret), "+a" (old_value), "+m" (*location)
                          : "q"(new_value)
                          : "memory", "cc");

    return (int)ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64(volatile int64_t* location,
                            int64_t old_value,
                            int64_t new_value)
{
    unsigned char ret;
    __asm__ __volatile__ (
                          "lock; cmpxchgq %3,%2   \n\t"
                          "      sete     %0      \n\t"
                          : "=qm" (ret), "+a" (old_value), "+m" (*location)
                          : "q"(new_value)
                          : "memory", "cc");

   return (int)ret;
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int128(volatile __int128_t* location,
                             __int128_t old_value,
                             __int128_t new_value)
{
    unsigned char ret;
    int64_t cmp_hi, cmp_lo, with_hi, with_lo;
    cmp_hi = (int64_t)(old_value >> 64);
    cmp_lo = *(uint64_t*)&old_value;
    with_hi = (int64_t)(new_value >> 64);
    with_lo = *(uint64_t*)&new_value;
    
    __asm__ __volatile__ (
                          "lock cmpxchg16b %1  \n\t"
                          "setz %0             \n\t"
                          : "=qm" ( ret ), "+m" ( *location ), "+d" ( cmp_hi ), "+a" ( cmp_lo )
                          : "c" ( with_hi ), "b" ( with_lo )
                          : "memory", "cc");
   
   return (int)ret;
}
#endif

/* Mask */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32(volatile int32_t* location,
                                     int32_t value)
{
    int32_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_int32(location, old_value, (old_value|value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32(volatile int32_t* location,
                                      int32_t value)
{
    int32_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_int32(location, old_value, (old_value&value) ));
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
    } while( !parsec_atomic_cas_int64(location, old_value, (old_value|value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64(volatile int64_t* location,
                                      int64_t value)
{
    int64_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_int64(location, old_value, (old_value&value) ));
    return old_value;
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_or_int128(volatile __int128_t* location,
                                         __int128_t value)
{
    __int128_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_int128(location, old_value, (old_value|value) ));
    return old_value;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_and_int128(volatile __int128_t* location,
                                          __int128_t value)
{
    __int128_t old_value;

    do {
        old_value = *location;
    } while( !parsec_atomic_cas_int128(location, old_value, (old_value&value) ));
    return old_value;
}
#endif

/* Integer */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32(volatile int32_t* v, int32_t i)
{
    int32_t ret = i;
    __asm__ __volatile__(
                        "lock; xaddl %1,%0"
                        :"+m" (*v), "+r" (ret)
                        :
                        :"memory", "cc");
   return ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int32(volatile int64_t* v, int64_t i)
{
    int64_t ret = i;
    __asm__ __volatile__(
                        "lock; xaddl %1,%0"
                        :"+m" (*v), "+r" (ret)
                        :
                        :"memory", "cc");
   return ret;
}

#if defined(PARSEC_HAVE_INT128)
#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_add_int128(volatile __int128_t* v, __int128_t i)
{
    __int128_t ov, nv;
    do {
        ov = *v;
        nv = ov + i;
    } while( !parsec_atomic_cas_int128(v, nv) );
    return ov;
}
#endif
