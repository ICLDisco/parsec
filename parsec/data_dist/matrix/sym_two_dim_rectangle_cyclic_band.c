/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic_band.h"
#include "parsec/data_dist/matrix/matrix_internal.h"

/* New rank_of for sym two dim block cyclic band */
static uint32_t sym_twoDBC_band_rank_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m-(int)n) < dc->band_size ) {
        /* New index */
        m = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.rank_of(&dc->band.super.super, m, n);
    }

    return dc->off_band.super.super.rank_of(&dc->off_band.super.super, m, n);
}

/* New vpid_of for sym two dim block cyclic band */
static int32_t sym_twoDBC_band_vpid_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m - (int)n) < dc->band_size ) {
        /* The new m in band */
        m = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.vpid_of(&dc->band.super.super, m, n);
    }

    return dc->off_band.super.super.vpid_of(&dc->off_band.super.super, m, n);
}

/* New data_of for sym two dim block cyclic band */
static parsec_data_t* sym_twoDBC_band_data_of(parsec_data_collection_t *desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc;
    dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m - (int)n) < dc->band_size ) {
        /* The new m in band */
        m = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.data_of(&dc->band.super.super, m, n);
    }

    return dc->off_band.super.super.data_of(&dc->off_band.super.super, m, n);
}

/* New rank_of_key for sym two dim block cyclic band */
static uint32_t sym_twoDBC_band_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return sym_twoDBC_band_rank_of(desc, m, n);
}

/* New vpid_of_key for two dim block cyclic band */
static int32_t sym_twoDBC_band_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return sym_twoDBC_band_vpid_of(desc, m, n);
}

/* New data_of_key for sym two dim block cyclic band */
static parsec_data_t* sym_twoDBC_band_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return sym_twoDBC_band_data_of(desc, m, n);
}

/*
 * sysm_two_dim_block_cyclic_band_t structure init
 * It inherits from off-band, so should be called after initialization of off_band
 */
void parsec_matrix_sym_block_cyclic_band_init( parsec_matrix_sym_block_cyclic_band_t *desc,
                                         int nodes, int myrank, int band_size ) {
    parsec_tiled_matrix_t *off_band = &desc->off_band.super;
    parsec_data_collection_t *dc = (parsec_data_collection_t*)desc;

    parsec_tiled_matrix_init( &desc->super, off_band->mtype, off_band->storage, off_band->dtype,
                                 nodes, myrank, off_band->mb, off_band->nb, off_band->lm, off_band->ln,
                                 off_band->i, off_band->j, off_band->m, off_band->n );

    desc->band_size  = band_size;
    dc->rank_of      = sym_twoDBC_band_rank_of;
    dc->vpid_of      = sym_twoDBC_band_vpid_of;
    dc->data_of      = sym_twoDBC_band_data_of;
    dc->rank_of_key  = sym_twoDBC_band_rank_of_key;
    dc->vpid_of_key  = sym_twoDBC_band_vpid_of_key;
    dc->data_of_key  = sym_twoDBC_band_data_of_key;
}
