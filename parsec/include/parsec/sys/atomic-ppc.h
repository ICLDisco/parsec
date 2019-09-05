/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
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

#if defined(PARSEC_HAVE_INT128)
/* INT128 support was detected, yet there is no 128 atomics on this architecture */
#error CMake Logic Error. INT128 support should be deactivated when using these atomics
#endif

/* Memory Barriers */

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

/* Linked Load / Store Conditional */

#ifdef __xlC__
/* work-around bizzare xlc bug in which it sign-extends
   a pointer to a 32-bit signed integer */
#define PARSEC_ASM_ADDR(a) ((uintptr_t)a)
#else
#define PARSEC_ASM_ADDR(a) (a)
#endif

#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC

#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_ll_int32(volatile int32_t *location)
{
   int32_t ret;

   __asm__ __volatile__ ("lwarx   %0, 0, %1  \n\t"
                         : "=&r" (ret)
                         : "r" PARSEC_ASM_ADDR(location)
                         :);
   return ret;
}

ATOMIC_STATIC_INLINE
int parsec_atomic_sc_int32(volatile int32_t *location, int32_t newval)
{
    int ret, foo;

    __asm__ __volatile__ ("   stwcx.  %4, 0, %3  \n\t"
                          "   li      %0,0       \n\t"
                          "   bne-    1f         \n\t"
                          "   ori     %0,%0,1    \n\t"
                          "1:"
                          : "=r" (ret), "=m" (*location), "=r" (foo)
                          : "r" PARSEC_ASM_ADDR(location), "r" (newval)
                          : "cc", "memory");
    return ret;
}

#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_ll_int64(volatile int64_t *location)
{
   int64_t ret;

   __asm__ __volatile__ ("ldarx   %0, 0, %1  \n\t"
                         : "=&r" (ret)
                         : "r" PARSEC_ASM_ADDR(location)
                         :);
   return ret;
}

ATOMIC_STATIC_INLINE
int parsec_atomic_sc_int64(volatile int64_t *location, int64_t newval)
{
    int ret, foo;

    __asm__ __volatile__ ("   stdcx.  %4, 0, %3  \n\t"
                          "   li      %0,0       \n\t"
                          "   bne-    1f         \n\t"
                          "   ori     %0,%0,1    \n\t"
                          "1:"
                          : "=r" (ret), "=m" (*location), "=r" (foo)
                          : "r" PARSEC_ASM_ADDR(location), "r" (newval)
                          : "cc", "memory");
    return ret;
}

#if PARSEC_SIZEOF_VOID_P == 4
#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_ptr parsec_atomic_ll_int32
#define parsec_atomic_sc_ptr parsec_atomic_sc_int32
#elif PARSEC_SIZEOF_VOID_P == 8
#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_ptr parsec_atomic_ll_int64
#define parsec_atomic_sc_ptr parsec_atomic_sc_int64
#else
/* No LLSC for PTR */
#warning CMake logic error: there is no Linked-Load / Store Conditional for the pointers of this architecture. Wrong atomics file selected.
#endif

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32( volatile int32_t* location,
                             int32_t old_value,
                             int32_t new_value )
{
#if !defined(__IBMC__)
   int32_t foo;
   foo = parsec_atomic_ll_int32(location);
   return ( foo == old_value && parsec_atomic_sc_int32(location, new_value) );
#else
   return __compare_and_swap(location, &old_value, new_value);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT64
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int64( volatile int64_t* location,
                             int64_t old_value,
                             int64_t new_value )
{
#if !defined(__IBMC__)
   int64_t foo;
   foo = parsec_atomic_ll_int64(location);
   return ( foo == old_value && parsec_atomic_sc_int64(location, new_value) );
#else
   return __compare_and_swap(location, &old_value, new_value);
#endif  /* !defined(__IBMC__) */
}

/* Mask Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t mask )
{
#if !defined(__IBMC__)
   int32_t old, t;
   do {
       old = parsec_atomic_ll_int32(location);
       t = old | mask;
   } while( !parsec_atomic_sc_int32(location, t) );
   return old;
#else
   return __fetch_and_or(location, mask);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_int32( volatile int32_t* location,
                                       int32_t mask )
{
#if !defined(__IBMC__)
   int32_t old, t;
   do {
       old = parsec_atomic_ll_int32(location);
       t = old & mask;
   } while( !parsec_atomic_sc_int32(location, t) );
   return old;
#else
   return __fetch_and_and(location, mask);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_OR_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_or_int64( volatile int64_t* location,
                                      int64_t mask )
{
#if !defined(__IBMC__)
   int64_t old, t;
   do {
       old = parsec_atomic_ll_int64(location);
       t = old | mask;
   } while( !parsec_atomic_sc_int64(location, t) );
   return old;
#else
   return __fetch_and_or(location, mask);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_AND_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_and_int64( volatile int64_t* location,
                                       int64_t mask )
{
#if !defined(__IBMC__)
   int64_t old, t;
   do {
       old = parsec_atomic_ll_int64(location);
       t = old & mask;
   } while( !parsec_atomic_sc_int64(location, t) );
   return old;
#else
   return __fetch_and_and(location, mask);
#endif  /* !defined(__IBMC__) */
}

/* Integer Operations */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32( volatile int32_t *location, int32_t val )
{
#if !defined(__IBMC__)
    int64_t old, t;
   do {
       old = parsec_atomic_ll_int32(location);
       t = old + val;
   } while( !parsec_atomic_sc_int32(location, t) );
   return old;
#else
   return __fetch_and_add(location, val);
#endif  /* !defined(__IBMC__) */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_add_int64( volatile int64_t *location, int64_t val )
{
#if !defined(__IBMC__)
    int64_t old, t;
   do {
       old = parsec_atomic_ll_int64(location);
       t = old + val;
   } while( !parsec_atomic_sc_int64(location, t) );
   return old;
#else
   return __fetch_and_add(location, val);
#endif  /* !defined(__IBMC__) */
}
