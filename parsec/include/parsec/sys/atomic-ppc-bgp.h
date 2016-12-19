/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __PPC
#warning This file is only for PowerPC
#endif  /* __ PPC */

#ifndef __bgp__
#warning This file is only for the BG/P
#endif  /* __bgp__ */

#ifndef PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED
#define PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED

#warning BGP atomic included

#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
/*#include <bpcore/ppc450_inlines.h>*/
#include <assert.h>

static inline void parsec_mfence( void )
{
    _bgp_msync();
}

static inline int parsec_atomic_bor_32b( volatile uint32_t* location,
                                        uint32_t mask )
{
    register uint32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val | mask;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

static inline int parsec_atomic_band_32b( volatile uint32_t* location,
                                          uint32_t mask )
{
    register uint32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val & mask;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

static inline int parsec_atomic_cas_32b( volatile uint32_t* location,
                                        uint32_t old_value,
                                        uint32_t new_value )
{
    uint32_t tmp_val;

    do {
        tmp_val = _bgp_LoadReserved( location );
        if( old_value != tmp_val ) {
            old_value = tmp_val;
            return( 0 );
        }
    } while( !_bgp_StoreConditional(location, new_value ) );

    return( 1 );
}

static inline int parsec_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    assert(0);  /* Not supported */
}

#define PARSEC_ATOMIC_HAS_ATOMIC_ADD_32B
static inline uint32_t parsec_atomic_add_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;


    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val + i;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

#define PARSEC_ATOMIC_HAS_ATOMIC_SUB_32B
static inline uint32_t parsec_atomic_sub_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val - i;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

#endif  /* PARSEC_ATOMIC_BGP_HAS_BEEN_INCLUDED */

