/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


/**
 * Based on shared Internet knowledge and the Power7 optimization book.
 * http://www.redbooks.ibm.com/redbooks/pdfs/sg248079.pdf
 */
#ifndef _ARCH_PPC
#warning This file is only for PowerPC
#endif  /* __ PPC */

#ifdef __xlC__
/* work-around bizzare xlc bug in which it sign-extends
   a pointer to a 32-bit signed integer */
#define PARSEC_ASM_ADDR(a) ((uintptr_t)a)
#else
#define PARSEC_ASM_ADDR(a) (a)
#endif

ATOMIC_STATIC_INLINE
void parsec_mfence( void )
{
    __asm__ __volatile__ ("sync\n\t":::"memory");
}

#define PARSEC_ATOMIC_HAS_RMB
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    __asm__ __volatile__ ("lwsync" : : : "memory");
}


#define PARSEC_ATOMIC_HAS_WMB
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    __asm__ __volatile__ ("lwsync" : : : "memory");
}

ATOMIC_STATIC_INLINE
int parsec_atomic_bor_32b( volatile uint32_t* location,
                           uint32_t mask )
{
#if !defined(__IBMC__)
   int32_t old, t;

   __asm__ __volatile__(
                        "1:   lwarx   %0,  0, %3   \n\t"
                        "     or      %1, %0, %2   \n\t"
                        "     stwcx.  %1,  0, %3   \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (old), "=&r" (t)
                        : "r" (mask), "r" PARSEC_ASM_ADDR(location)
                        : "cc", "memory");

   return t;
#else
   return mask | __fetch_and_or(location, mask);
#endif  /* !defined(__IBMC__) */
}

ATOMIC_STATIC_INLINE
int parsec_atomic_band_32b( volatile uint32_t* location,
                            uint32_t mask )
{
#if !defined(__IBMC__)
   int32_t old, t;

   __asm__ __volatile__(
                        "1:   lwarx   %0,  0, %3   \n\t"
                        "     andc    %1, %0, %2   \n\t"
                        "     stwcx.  %1,  0, %3   \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (old), "=&r" (t)
                        : "r" (mask), "r" PARSEC_ASM_ADDR(location)
                        : "cc", "memory");

   return t;
#else
   return mask & __fetch_and_and(location, mask);
#endif  /* !defined(__IBMC__) */
}

ATOMIC_STATIC_INLINE
int parsec_atomic_cas_32b( volatile uint32_t* location,
                           uint32_t old_value,
                           uint32_t new_value )
{
#if !defined(__IBMC__)
   int32_t ret;

   __asm__ __volatile__ (
                         "1: lwarx   %0, 0, %2  \n\t"
                         "   cmpw    0, %0, %3  \n\t"
                         "   bne-    2f         \n\t"
                         "   stwcx.  %4, 0, %2  \n\t"
                         "   bne-    1b         \n\t"
                         "2:"
                         : "=&r" (ret), "=m" (*location)
                         : "r" PARSEC_ASM_ADDR(location), "r" (old_value), "r" (new_value), "m" (*location)
                         : "cr0", "memory");

   return (ret == old_value);
#else
   return __compare_and_swap((volatile int*)location, (int*)&old_value, (int)new_value);
#endif  /* !defined(__IBMC__) */
}

ATOMIC_STATIC_INLINE
int parsec_atomic_cas_64b( volatile uint64_t* location,
                           uint64_t old_value,
                           uint64_t new_value )
{
#if !defined(__IBMC__)
   int64_t ret;

   __asm__ __volatile__ (
                         "1: ldarx   %0, 0, %2  \n\t"
                         "   cmpd    0, %0, %3  \n\t"
                         "   bne-    2f         \n\t"
                         "   stdcx.  %4, 0, %2  \n\t"
                         "   bne-    1b         \n\t"
                         "2:"
                         : "=&r" (ret), "=m" (*location)
                         : "r" (location), "r" PARSEC_ASM_ADDR(old_value), "r" (new_value), "m" (*location)
                         : "cr0", "memory");

   return (ret == old_value);
#else
   return __compare_and_swaplp((volatile long*)location, (long*)&old_value, (long)new_value);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_HAVE_ATOMIC_LLSC_PTR
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_ll_32b(volatile int32_t *location)
{
   int32_t ret;

   __asm__ __volatile__ ("lwarx   %0, 0, %1  \n\t"
                         : "=&r" (ret)
                         : "r" (location)
                         :);
   return ret;
}

ATOMIC_STATIC_INLINE
int64_t parsec_atomic_ll_64b(volatile int64_t *location)
{
   int64_t ret;

   __asm__ __volatile__ ("ldarx   %0, 0, %1  \n\t"
                         : "=&r" (ret)
                         : "r" (location)
                         :);
   return ret;
}
#define parsec_atomic_ll_ptr parsec_atomic_ll_64b

ATOMIC_STATIC_INLINE
int parsec_atomic_sc_32b(volatile int32_t *location, int32_t newval)
{
    int32_t ret, foo;

    __asm__ __volatile__ ("   stwcx.  %4, 0, %3  \n\t"
                          "   li      %0,0       \n\t"
                          "   bne-    1f         \n\t"
                          "   ori     %0,%0,1    \n\t"
                          "1:"
                          : "=r" (ret), "=m" (*location), "=r" (foo)
                          : "r" (location), "r" (newval)
                          : "cc", "memory");
    return ret;
}

ATOMIC_STATIC_INLINE
int parsec_atomic_sc_64b(volatile int64_t *location, int64_t newval)
{
    int32_t ret, foo;

    __asm__ __volatile__ ("   stdcx.  %4, 0, %3  \n\t"
                          "   li      %0,0       \n\t"
                          "   bne-    1f         \n\t"
                          "   ori     %0,%0,1    \n\t"
                          "1:"
                          : "=r" (ret), "=m" (*location), "=r" (foo)
                          : "r" (location), "r" (newval)
                          : "cc", "memory");
    return ret;
}
#define parsec_atomic_sc_ptr parsec_atomic_sc_64b

#define PARSEC_ATOMIC_HAS_ATOMIC_INC_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_inc_32b( volatile uint32_t *location )
{
#if !defined(__IBMC__)
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0, %1    \n\t"
                        "     addic   %0, %0, 1    \n\t"
                        "     stwcx.  %0, 0, %1    \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (t)
                        : "r" (location)
                        : "cc", "memory");

   return t;
#else
   return 1 + __fetch_and_add( (volatile int*)location, 1);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_DEC_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_dec_32b( volatile uint32_t *location )
{
#if !defined(__IBMC__)
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0,%1     \n\t"
                        "     addic   %0,%0,-1     \n\t"
                        "     stwcx.  %0,0,%1      \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (t)
                        : "r" (location)
                        : "cc", "memory");

   return t;
#else
   return __fetch_and_add( (volatile int*)location, -1) - 1;
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_add_32b( volatile int32_t *location, int32_t i )
{
#if !defined(__IBMC__)
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0, %3    \n\t"
                        "     add     %0, %2, %0   \n\t"
                        "     stwcx.  %0, 0, %3    \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (i), "r" PARSEC_ASM_ADDR(location), "m" (*location)
                        : "cc");

   return t;
#else
   return i + __fetch_and_add( (volatile int*)location, i);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B
ATOMIC_STATIC_INLINE
uint32_t parsec_atomic_sub_32b( volatile int32_t *location, int32_t i )
{
#if !defined(__IBMC__)
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0,0,%3      \n\t"
                        "     subf    %0,%2,%0     \n\t"
                        "     stwcx.  %0,0,%3      \n\t"
                        "     bne-    1b           \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (i), "r" PARSEC_ASM_ADDR(location)
                        : "cc");

   return t;
#else
   return __fetch_and_add( (volatile int*)location, i) - i;
#endif  /* !defined(__IBMC__) */
}
