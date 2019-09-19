/*
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "two_dim_rectangle_cyclic_band.h"

/* Get the rank */
inline int twoDBC_band_get_rank(two_dim_block_cyclic_t *dc,
                                unsigned int m, unsigned int n){
    unsigned int rr, cr, res;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = m % dc->grid.rows;
    cr = n % dc->grid.cols;
    res = rr * dc->grid.cols + cr;

    return res;
}

/* Offset of (i, j) and assert */
inline void twoDBC_band_offset(two_dim_block_cyclic_t *dc,
                               unsigned int *m, unsigned int *n){
    /* Offset by (i,j) to translate (m,n) in the global matrix */
    *m += dc->super.i / dc->super.mb;
    *n += dc->super.j / dc->super.nb;

    assert( *m < (unsigned int)dc->super.mt );
    assert( *n < (unsigned int)dc->super.nt );
}

/* New rank_of for two dim block cyclic band */
uint32_t twoDBC_band_rank_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m, n;
    va_list ap;
    two_dim_block_cyclic_band_t * dc = (two_dim_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Check tile location */
    if( (unsigned int)abs(m-n) < dc->band_size ){
        /* The new m in band */
        m = m - n + dc->band_size - 1;

        if( (dc->band.grid.strows != 1) || (dc->band.grid.stcols != 1) ){
            m = st_compute_m(&dc->band, m);
            n = st_compute_n(&dc->band, n);
        }

        /* Offset of (i, j) and assert */
        twoDBC_band_offset(&dc->band, &m, &n);

        return twoDBC_band_get_rank(&dc->band, m, n);
    }
    else{
        if( (dc->super.grid.strows != 1) || (dc->super.grid.stcols != 1) ){
            m = st_compute_m(&dc->super, m);
            n = st_compute_n(&dc->super, n);
        }

        /* Offset of (i, j) and assert */
        twoDBC_band_offset(&dc->super, &m, &n);

        return twoDBC_band_get_rank(&dc->super, m, n);
    }
}

/* New data_of for two dim block cyclic band */
parsec_data_t* twoDBC_band_data_of(parsec_data_collection_t *desc, ...)
{
    unsigned int m, n, position, key;
    va_list ap;
    two_dim_block_cyclic_band_t * dc;
    dc = (two_dim_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(desc->myrank == twoDBC_band_rank_of(desc, m, n));
#endif

    /* Compute the key */
    key = (n * dc->super.super.lmt) + m;

    /* Check tile location */
    if( (unsigned int)abs(m-n) < dc->band_size ){
        /* The new m in band */
        m = m - n + dc->band_size - 1;

        if( (dc->band.grid.strows != 1) || (dc->band.grid.stcols != 1) ){
            m = st_compute_m(&dc->band, m);
            n = st_compute_n(&dc->band, n);
        }

        /* Offset of (i, j) and assert */
        twoDBC_band_offset(&dc->band, &m, &n);

        /* Get position in data_map */
        position = twoDBC_coordinates_to_position(&dc->band, m, n);

        if( NULL == dc->band.mat )
            return parsec_matrix_create_data( &dc->band.super, NULL, position, key );
        else
            return parsec_matrix_create_data( &dc->band.super,
                                              (char*)dc->band.mat + position * dc->band.super.bsiz
                                              * parsec_datadist_getsizeoftype(dc->band.super.mtype),
                                              position, key );
    }
    else{
        if( (dc->super.grid.strows != 1) || (dc->super.grid.stcols != 1) ){
            m = st_compute_m(&dc->super, m);
            n = st_compute_n(&dc->super, n);
        }

        /* Offset of (i, j) and assert */
        twoDBC_band_offset(&dc->super, &m, &n);

        /* Get position in data_map */
        position = twoDBC_coordinates_to_position(&dc->super, m, n);

        if( NULL == dc->super.mat )
            return parsec_matrix_create_data( &dc->super.super, NULL, position, key );
        else
            return parsec_matrix_create_data( &dc->super.super,
                                              (char*)dc->super.mat + position * dc->super.super.bsiz
                                              * parsec_datadist_getsizeoftype(dc->super.super.mtype),
                                              position, key );
    }
}

/* New rank_of_key for two dim block cyclic band */
uint32_t twoDBC_band_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    twoDBC_key_to_coordinates(desc, key, &m, &n);
    return twoDBC_band_rank_of(desc, m, n);
}

/* New data_of_key for two dim block cyclic band */
parsec_data_t* twoDBC_band_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{   
    int m, n;
    twoDBC_key_to_coordinates(desc, key, &m, &n);
    return twoDBC_band_data_of(desc, m, n);
}
