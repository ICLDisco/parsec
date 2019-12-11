/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_ATOMIC_H_HAS_BEEN_INCLUDED
#define PARSEC_ATOMIC_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

BEGIN_C_DECLS

#if !defined(BUILDING_PARSEC)
#  include "atomic-external.h"
#else  /* !defined(BUILDING_PARSEC) */

#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#  if !defined(ATOMIC_STATIC_INLINE)
#    define ATOMIC_STATIC_INLINE static inline
#  endif  /* !defined(ATOMIC_STATIC_INLINE) */

/*
 * This define will exists only in the PaRSEC build, and should remain undefined
 * in all other contexts.
 */
#  define PARSEC_ATOMIC_ACCESS_TO_INTERNALS_ALLOWED 1

#  if defined(PARSEC_ATOMIC_USE_C11_ATOMICS)
#    include "atomic-c11.h"
#  else /* defined(PARSEC_ATOMIC_USE_C11_ATOMICS) */
/**
 * If the compiler provides atomic primitives we prefer to use
 * them instead of our own atomic assembly.
 */
#    if defined(__FUJITSU)
#      undef PARSEC_ATOMIC_USE_XLC_32_BUILTINS
#    endif
#    if defined(PARSEC_ATOMIC_USE_XLC_32_BUILTINS)
#      include "atomic-xlc.h"
#    elif defined(PARSEC_OSX)
#      if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_12
/* Intel compiler on OSX defined __clang__ but do not support the pragmas */
#        if defined(__clang__) && !defined(__ICC)
#          pragma clang diagnostic push
#          pragma clang diagnostic ignored "-Wdeprecated-declarations"
#        endif  /* defined(__clang__) && !defined(__ICC) */
#        include "atomic-macosx.h"
#        if defined(__clang__) && !defined(__ICC)
#          pragma clang diagnostic pop
#        endif  /* defined(__clang__) && !defined(__ICC) */
#      endif  /* MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_12 */
#    elif defined(PARSEC_ARCH_PPC)
#      if defined(__bgp__)
#        include "atomic-ppc-bgp.h"
#      else
#        include "atomic-ppc.h"
#      endif
#    elif defined(PARSEC_ATOMIC_USE_GCC_32_BUILTINS)
#      include "atomic-gcc.h"
#    elif defined(PARSEC_ARCH_X86)
#      include "atomic-x86_32.h"
#    elif defined(PARSEC_ARCH_X86_64)
#      include "atomic-x86_64.h"
#    else
#      error "No safe atomics available"
#    endif
#  endif /* defined(PARSEC_ATOMIC_USE_C11_ATOMICS) */

/*
 * Generic Alternative Methods, in case some routines are not directly defined
 */

/* parsec_atomic_cas_ptr */
#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_PTR)
#    if PARSEC_SIZEOF_VOID_P == 4
#    define PARSEC_ATOMIC_HAS_ATOMIC_CAS_PTR
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_int32((volatile int32_t*)l, (int32_t)o, (int32_t)n);
}
#    elif PARSEC_SIZEOF_VOID_P == 8
#    define PARSEC_ATOMIC_HAS_ATOMIC_CAS_PTR
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_int64((volatile int64_t*)l, (int64_t)o, (int64_t)n);
}
#    else
#      if defined(PARSEC_HAVE_INT128)
#        define PARSEC_ATOMIC_HAS_ATOMIC_CAS_PTR
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_ptr(volatile void* l, void* o, void* n)
{
    return parsec_atomic_cas_int128((volatile __int128_t*)l, (__int128_t)o, (__int128_t)n);
}
#      else  /* defined(PARSEC_HAVE_INT128) */
#        error Pointers are 128 bits long but no atomic operation on 128 bits are available
#      endif  /* defined(PARSEC_HAVE_INT128) */
#    endif
#  endif /* !defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_PTR) */

/* Memory Barriers */

#  if !defined(PARSEC_ATOMIC_HAS_WMB)
#  define PARSEC_ATOMIC_HAS_WMB
ATOMIC_STATIC_INLINE
void parsec_atomic_wmb(void)
{
    parsec_mfence();
}
#  endif  /* !defined(PARSEC_ATOMIC_HAS_WMB) */

#  if !defined(PARSEC_ATOMIC_HAS_RMB)
#  define PARSEC_ATOMIC_HAS_RMB
ATOMIC_STATIC_INLINE
void parsec_atomic_rmb(void)
{
    parsec_mfence();
}
#  endif  /* !defined(PARSEC_ATOMIC_HAS_RMB) */

/* Integer Operations */

/* Quite often, all atomic integer operations are based on the same
 * addition; here, we just manage the case where the architecture
 * does not have a better option than using add to define inc,
 * sub, and dec */

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT32)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_inc_int32(volatile int32_t* l)
{
    return parsec_atomic_fetch_add_int32(l, 1);
}
#    else
#      error No definition for parsec_atomic_fetch_inc_int32
#    endif
#  endif

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT32)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_sub_int32(volatile int32_t* l, int32_t v)
{
    return parsec_atomic_fetch_add_int32(l, -v);
}
#    else
#      error No definition for parsec_atomic_fetch_sub_int32
#    endif
#  endif

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT32)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_dec_int32(volatile int32_t* l)
{
    return parsec_atomic_fetch_add_int32(l, -1);
}
#    else
#      error No definition for parsec_atomic_fetch_dec_int32
#    endif
#  endif

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT64)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_inc_int64(volatile int64_t* l)
{
    return parsec_atomic_fetch_add_int64(l, 1);
}
#    endif
/* No error: 32 bits architectures do not need to define add/inc/sub/dec on 64 bits */
#  endif

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT64)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_sub_int64(volatile int64_t* l, int64_t v)
{
    return parsec_atomic_fetch_add_int64(l, -v);
}
#    endif
/* No error: 32 bits architectures do not need to define add/inc/sub/dec on 64 bits */
#  endif

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT64)
#    if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT64)
#      define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT64
ATOMIC_STATIC_INLINE
int64_t parsec_atomic_fetch_dec_int64(volatile int64_t* l)
{
    return parsec_atomic_fetch_add_int64(l, -1);
}
#    endif
/* No error: 32 bits architectures do not need to define add/inc/sub/dec on 64 bits */
#  endif

#  if defined(PARSEC_HAVE_INT128)
#    if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT128)
#      if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128)
#        define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_INC_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_inc_int128(volatile __int128_t* l)
{
    return parsec_atomic_fetch_add_int128(l, 1);
}
#      else
#        error No definition for parsec_atomic_fetch_inc_int128
#      endif
#    endif

#    if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT128)
#      if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128)
#        define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_SUB_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_sub_int128(volatile __int128_t* l, __int128_t v)
{
    return parsec_atomic_fetch_add_int128(l, -v);
}
#      else
#        error No definition for parsec_atomic_fetch_sub_int128
#      endif
#    endif

#    if !defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT128)
#      if defined(PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT128)
#        define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_DEC_INT128
ATOMIC_STATIC_INLINE
__int128_t parsec_atomic_fetch_dec_int128(volatile __int128_t* l)
{
    return parsec_atomic_fetch_add_int128(l, -1);
}
#      else
#        error No definition for parsec_atomic_fetch_dec_int128
#      endif
#    endif
#  endif  /* defined(PARSEC_HAVE_INT128) */

/* Locks */

#  if !defined(PARSEC_ATOMIC_HAS_ATOMIC_LOCK)
typedef volatile int32_t parsec_atomic_lock_t;
#  define PARSEC_ATOMIC_UNLOCKED 0
#  define PARSEC_ATOMIC_HAS_ATOMIC_LOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_lock( parsec_atomic_lock_t* atomic_lock )
{
    while( !parsec_atomic_cas_int32( atomic_lock, 0, 1) )
        /* nothing */;
}
#  define PARSEC_ATOMIC_HAS_ATOMIC_UNLOCK
ATOMIC_STATIC_INLINE
void parsec_atomic_unlock( parsec_atomic_lock_t* atomic_lock )
{
    parsec_mfence();
    *atomic_lock = 0;
}
#  define PARSEC_ATOMIC_HAS_ATOMIC_TRYLOCK
ATOMIC_STATIC_INLINE
int parsec_atomic_trylock( parsec_atomic_lock_t* atomic_lock )
{
    return parsec_atomic_cas_int32( atomic_lock, 0, 1 );
}
#  endif /* !defined(PARSEC_ATOMIC_HAS_ATOMIC_LOCK) */

#endif  /* !defined(BUILDING_PARSEC) */

END_C_DECLS

#endif  /* ATOMIC_H_HAS_BEEN_INCLUDED */
