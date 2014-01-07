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

#ifndef DAGUE_ATOMIC_BGP_HAS_BEEN_INCLUDED
#define DAGUE_ATOMIC_BGP_HAS_BEEN_INCLUDED

#warning BGP atomic included

#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
/*#include <bpcore/ppc450_inlines.h>*/
#include <assert.h>

static inline void dague_mfence( void )
{
    _bgp_msync();
}

static inline int dague_atomic_bor_32b( volatile uint32_t* location,
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

static inline int dague_atomic_band_32b( volatile uint32_t* location,
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

static inline int dague_atomic_cas_32b( volatile uint32_t* location,
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

static inline int dague_atomic_cas_64b( volatile uint64_t* location,
                                          uint64_t old_value,
                                          uint64_t new_value )
{
    assert(0);  /* Not supported */
}

#define DAGUE_ATOMIC_HAS_ATOMIC_ADD_32B
static inline uint32_t dague_atomic_add_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;


    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val + i;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

#define DAGUE_ATOMIC_HAS_ATOMIC_SUB_32B
static inline uint32_t dague_atomic_sub_32b( volatile int32_t *location, int32_t i )
{
    register int32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = old_val - i;
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}

#define DAGUE_ATOMIC_HAS_ATOMIC_add_32B
static inline uint32_t dague_atomic_add_32b( volatile uint32_t *location, int32_t d )
{
    register uint32_t old_val, tmp_val;

    _bgp_msync();
    do {
        old_val = _bgp_LoadReserved( location );
        tmp_val = (uint32_t((int32_t)old_val + d);
    } while( !_bgp_StoreConditional( location, tmp_val ) );

    return( tmp_val );
}


#endif  /* DAGUE_ATOMIC_BGP_HAS_BEEN_INCLUDED */

