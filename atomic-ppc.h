/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __PPC
#warning This file is only for PowerPC
#endif  /* __ PPC */

static inline int dplasma_atomic_bor_32b( volatile uint32_t* location,
                                          uint32_t value )
{
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0, %3    \n\t"
                        "     add.    %0, %2, %0   \n\t"
                        "     stwcx.  %0, 0, %3    \n\t"
                        "     bne-    1b           \n\t"
                        "     mr      %3, %0       \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (value), "r" (location), "m" (*location)
                        : "cr0");

   return t;
}

static inline int dplasma_atomic_band_32b( volatile uint32_t* location,
                                          uint32_t value )
{
   int32_t t;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0, %3    \n\t"
                        "     or.     %0, %2, %0   \n\t"
                        "     stwcx.  %0, 0, %3    \n\t"
                        "     bne-    1b           \n\t"
                        "     mr      %3, %0       \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (value), "r" (location), "m" (*location)
                        : "cr0");

   return t;
}

static inline int dplasma_atomic_cas_32b( volatile uint32_t* location,
                                          uint32_t old_value,
                                          uint32_t new_value )
{
   int32_t ret;

   __asm__ __volatile__ (
                         "1: lwarx   %0, 0, %2  \n\t"
                         "   cmpw    0, %0, %3  \n\t"
                         "   bne-    2f         \n\t"
                         "   stwcx.  %4, 0, %2  \n\t"
                         "   bne-    1b         \n\t"
                         "2:"
                         : "=&r" (ret), "=m" (*location)
                         : "r" (location), "r" (old_value), "r" (new_value), "m" (*location)
                         : "cr0", "memory");

   return (ret == old_value);
}

static inline int dplasma_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
   int64_t ret;

   __asm__ __volatile__ (
                         "1: ldarx   %0, 0, %2  \n\t"
                         "   cmpd    0, %0, %3  \n\t"
                         "   bne-    2f         \n\t"
                         "   stdcx.  %4, 0, %2  \n\t"
                         "   bne-    1b         \n\t"
                         "2:"
                         : "=&r" (ret), "=m" (*location)
                         : "r" (location), "r" (old_value), "r" (new_value), "m" (*location)
                         : "cr0", "memory");

   return (ret == old_value);
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_INC_32B
static inline uint32_t dplasma_atomic_inc_32b( volatile uint32_t *location )
{
   int32_t t, inc = 1;

   __asm__ __volatile__(
                        "1:   lwarx   %0, 0, %3    \n\t"
                        "     add     %0, %2, %0   \n\t"
                        "     stwcx.  %0, 0, %3    \n\t"
                        "     bne-    1b           \n\t"
                        "     mr      %3, %0       \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (inc), "r" (location), "m" (*location)
                        : "cr0");

   return t;
}

#define DPLASMA_ATOMIC_HAS_ATOMIC_DEC_32B
static inline uint32_t dplasma_atomic_dec_32b( volatile uint32_t *location )
{
   int32_t t, dec = 1;

   __asm__ __volatile__(
                        "1:   lwarx   %0,0,%3      \n\t"
                        "     subf    %0,%2,%0     \n\t"
                        "     stwcx.  %0,0,%3      \n\t"
                        "     bne-    1b           \n\t"
                        "     mr      %3, %0       \n\t"
                        : "=&r" (t), "=m" (*location)
                        : "r" (dec), "r" (location), "m" (*location)
                        : "cr0");

   return t;
}

