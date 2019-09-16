/*
 * Copyright (c) 2011-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Warning: 
 *  as of May 10, 2018, this file has not been tested, for lack of target architecture */

#ifndef __PPC
#warning This file is only for PowerPC
#endif  /* __ PPC */

#ifndef __bgp__
#warning This file is only for the BG/P
#endif  /* __bgp__ */

#ifndef PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED
#define PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED

#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
/*#include <bpcore/ppc450_inlines.h>*/
#include <assert.h>

/* Memory barriers */

ATOMIC_STATIC_INLINE
void parsec_mfence( void )
{
    _bgp_msync();
}

/* Linked Load / Store Conditional */

#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC

#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_ll_int32(volatile int32_t *location)
{
    return _bgp_LoadReserved( location );
}

ATOMIC_STATIC_INLINE
int parsec_atomic_sc_int32(volatile int32_t *location, int32_t newval)
{
    return _bgp_StoreConditional( location, newval );
}

#if PARSEC_SIZEOF_VOID_P == 4
#define PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR
#define parsec_atomic_ll_ptr parsec_atomic_ll_int32
#define parsec_atomic_sc_ptr parsec_atomic_sc_int32
#else
/* No LLSC for PTR */
#error CMake logic error: there is no Linked-Load / Store Conditional for the pointers of this architecture. Wrong atomics file selected.
#endif

/* Compare and Swap */

#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT32
ATOMIC_STATIC_INLINE
int parsec_atomic_cas_int32( volatile int32_t* location,
                             int32_t old_value,
                             int32_t new_value )
{
   int32_t foo;
   foo = parsec_atomic_ll_int32(location);
   return ( foo == old_value && parsec_atomic_sc_int32(location, new_value) );
}

/* Mask */

#define PARSEC_ATOMIC_FETCH_OR_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_or_int32( volatile int32_t* location,
                                      int32_t mask )
{
    register int32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val | mask;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return old_val;
}

#define PARSEC_ATOMIC_FETCH_AND_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_and_32b( volatile int32_t* location,
                                     int32_t mask )
{
    register uint32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val & mask;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return old_val;
}

/* Integer */

#define PARSEC_ATOMIC_HAS_ATOMIC_FETCH_ADD_INT32
ATOMIC_STATIC_INLINE
int32_t parsec_atomic_fetch_add_int32( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val + i;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return old_val;
}

#endif  /* PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED */

