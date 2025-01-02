/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"

#include <stdlib.h>
#include <stdio.h>
#ifdef PARSEC_HAVE_STRING_H
#include <string.h>
#endif
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif
#include <assert.h>
#ifdef PARSEC_HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef PARSEC_HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include <assert.h>
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/matrix_internal.h"
#include "parsec/mca/device/device.h"
#include "parsec/vpmap.h"
#include "parsec/data.h"

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

static int sym_twoDBC_memory_register(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_sym_block_cyclic_t * sym_twodbc = (parsec_matrix_sym_block_cyclic_t *)desc;
    if( NULL == sym_twodbc->mat ) {
        return PARSEC_SUCCESS;
    }
    return device->memory_register(device, desc,
                                   sym_twodbc->mat,
                                   ((size_t)sym_twodbc->super.nb_local_tiles * (size_t)sym_twodbc->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(sym_twodbc->super.mtype)));
}

static int sym_twoDBC_memory_unregister(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_sym_block_cyclic_t * sym_twodbc = (parsec_matrix_sym_block_cyclic_t *)desc;
    if( NULL == sym_twodbc->mat ) {
        return PARSEC_SUCCESS;
    }
    return device->memory_unregister(device, desc, sym_twodbc->mat);
}

static uint32_t sym_twoDBC_rank_of(parsec_data_collection_t * desc, ...)
{
    int cr, m, n;
    int rr;
    int res;
    va_list ap;
    parsec_matrix_sym_block_cyclic_t * dc;
    dc = (parsec_matrix_sym_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

    assert( (dc->uplo == PARSEC_MATRIX_LOWER && m>=n) ||
            (dc->uplo == PARSEC_MATRIX_UPPER && n>=m) );
    if ( ((dc->uplo == PARSEC_MATRIX_LOWER) && (m < n)) ||
         ((dc->uplo == PARSEC_MATRIX_UPPER) && (m > n)) )
    {
        return UINT_MAX;
    }

    /* for tile (m,n), first find coordinate of process in
     process grid which possess the tile in block cyclic dist */
    rr = m % dc->grid.rows;
    cr = n % dc->grid.cols;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * dc->grid.cols + cr;

    return res;
}

static void sym_twoDBC_key_to_coordinates(parsec_data_collection_t *desc, parsec_data_key_t key, int *m, int *n)
{
    int _m, _n;
    parsec_tiled_matrix_t * dc;

    dc = (parsec_tiled_matrix_t *)desc;

    _m = key % dc->lmt;
    _n = key / dc->lmt;
    *m = _m - dc->i / dc->mb;
    *n = _n - dc->j / dc->nb;
}

static uint32_t sym_twoDBC_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_rank_of(desc, m, n);
}

static parsec_data_t* sym_twoDBC_data_of(parsec_data_collection_t *desc, ...)
{
    int m, n, position;
    parsec_matrix_sym_block_cyclic_t * dc;
    size_t pos = 0;
    va_list ap;

    dc = (parsec_matrix_sym_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif
    assert( dc->super.storage == PARSEC_MATRIX_TILE );
    assert( (dc->uplo == PARSEC_MATRIX_LOWER && m>=n) ||
            (dc->uplo == PARSEC_MATRIX_UPPER && n>=m) );

    position = parsec_matrix_sym_block_cyclic_coord2pos(dc, m, n);

    /* If mat allocatd, set pos to the right position for each tile */
    if( NULL != dc->mat )
        pos = position;

    return parsec_tiled_matrix_create_data( &dc->super,
                                     (char*)dc->mat + pos * dc->super.bsiz * parsec_datadist_getsizeoftype(dc->super.mtype),
                                     position, (n * dc->super.lmt) + m );
}

static parsec_data_t* sym_twoDBC_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_data_of(desc, m, n);
}

static int32_t sym_twoDBC_vpid_of(parsec_data_collection_t *desc, ...)
{
    int m, n, p, q, pq;
    int local_m, local_n;
    parsec_matrix_sym_block_cyclic_t * dc;
    va_list ap;
    int32_t vpid;
    dc = (parsec_matrix_sym_block_cyclic_t *)desc;

    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    q = dc->grid.vp_q;
    p = dc->grid.vp_p;
    assert(p*q == pq);


    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif
    assert( (dc->uplo == PARSEC_MATRIX_LOWER && m>=n) ||
            (dc->uplo == PARSEC_MATRIX_UPPER && n>=m) );

    /* Compute the local tile row */
    local_m = m / dc->grid.rows;
    assert( (m % dc->grid.rows) == dc->grid.rrank );

    /* Compute the local column */
    local_n = n / dc->grid.cols;
    assert( (n % dc->grid.cols) == dc->grid.crank );

    vpid = (local_n % q) * p + (local_m % p);
    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static int32_t sym_twoDBC_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_vpid_of(desc, m, n);
}

void parsec_matrix_sym_block_cyclic_init(parsec_matrix_sym_block_cyclic_t * dc,
                                   parsec_matrix_type_t mtype,
                                   int myrank,
                                   int mb,   int nb,   /* Tile size */
                                   int lm,   int ln,   /* Global matrix size (what is stored)*/
                                   int i,    int j,    /* Staring point in the global matrix */
                                   int m,    int n,    /* Submatrix size (the one concerned by the computation */
                                   int P,    int Q,    /* process process grid */
                                   parsec_matrix_uplo_t uplo )
{
    int nb_elem, total;
    int nodes = P*Q;
    /* Initialize the tiled_matrix descriptor */
    parsec_data_collection_t *o = &(dc->super.super);

    parsec_tiled_matrix_init( &(dc->super), mtype, PARSEC_MATRIX_TILE,
                            parsec_matrix_sym_block_cyclic_type,
                            nodes, myrank,
                            mb, nb, lm, ln, i, j, m, n );
    dc->mat = NULL;  /* No data associated with the matrix yet */

    o->rank_of     = sym_twoDBC_rank_of;
    o->rank_of_key = sym_twoDBC_rank_of_key;
    o->vpid_of     = sym_twoDBC_vpid_of;
    o->vpid_of_key = sym_twoDBC_vpid_of_key;
    o->data_of     = sym_twoDBC_data_of;
    o->data_of_key = sym_twoDBC_data_of_key;

    o->register_memory   = sym_twoDBC_memory_register;
    o->unregister_memory = sym_twoDBC_memory_unregister;

    if(nodes < P) {
        parsec_warning("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);
        P = nodes;
    }
    if(nodes != P*Q)
        parsec_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);
    parsec_grid_2Dcyclic_init(&dc->grid, myrank, P, Q, 1, 1, 0, 0);

    /* Extra parameters */
    dc->uplo = uplo;

    /* find the number of tiles this process will handle */
    total = 0; /* number of tiles handled by the process */
    if ( uplo == PARSEC_MATRIX_LOWER ) {
        int column = dc->grid.crank; /* tile column considered */
        int nb_elem_col = (dc->super.lmt) / (dc->grid.rows); //nb of tile associated to that proc in a full column
        if( (dc->super.lmt) % (dc->grid.rows) > dc->grid.rrank )
            nb_elem_col++;

        /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
        while(column < dc->super.lnt)
        {
            nb_elem = column / (dc->grid.rows);
            if ( (column % (dc->grid.rows)) > dc->grid.rrank)
                nb_elem++;

            total += (nb_elem_col - nb_elem);
            column += dc->grid.cols;
        }
    } else { /* Upper */
        int row = dc->grid.rrank; /* tile row considered */
        int nb_elem_row = (dc->super.lnt) / (dc->grid.cols); //nb of tile associated to that proc in a full row
        if( (dc->super.lnt) % (dc->grid.cols) > dc->grid.crank )
            nb_elem_row++;

        /* for each row of tiles in memory before searched element, compute the number of tile for displacement */
        while(row < dc->super.lmt)
        {
            nb_elem = row / (dc->grid.cols);
            if ( (row % (dc->grid.cols)) > dc->grid.crank)
                nb_elem++;

            total += (nb_elem_row - nb_elem);
            row += dc->grid.rows;
        }
    }

    dc->super.nb_local_tiles = total;
    dc->super.data_map = (parsec_data_t**)calloc(dc->super.nb_local_tiles, sizeof(parsec_data_t*));
}

size_t parsec_matrix_sym_block_cyclic_coord2pos(
    parsec_matrix_sym_block_cyclic_t *dc,
    int m,
    int n)
{
    size_t pos;
    int nb_elem, nb_elem_col, column;

    pos = 0; /* current position (as number of tile) in the buffer */
    column = dc->grid.crank; /* tile column considered */

    /**********************************/
    if(dc->uplo == PARSEC_MATRIX_LOWER ) {
        nb_elem_col = (dc->super.lmt) / (dc->grid.rows); //nb of tile associated to that proc in a full column
        if( (dc->super.lmt) % (dc->grid.rows) > dc->grid.rrank )
            nb_elem_col++;

        while(column != n) {
            /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
            nb_elem = column / (dc->grid.rows);
            if ( (column % (dc->grid.rows)) > dc->grid.rrank)
                nb_elem++;

            pos += (nb_elem_col - nb_elem);
            column += dc->grid.cols;
        }

        pos += ((m - n) / (dc->grid.rows));
    } else {
        while(column != n) {
            /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
            nb_elem = (column + 1) / (dc->grid.rows);
            if ( ( (column + 1) % (dc->grid.rows)) > dc->grid.rrank)
                nb_elem++;

            pos += nb_elem;
            column += dc->grid.cols;
        }

        pos += (m / (dc->grid.rows));
    }
    return pos;
}

