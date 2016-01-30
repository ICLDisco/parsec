/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "dague/debug.h"

#include <stdlib.h>
#include <stdio.h>
#ifdef DAGUE_HAVE_STRING_H
#include <string.h>
#endif
#ifdef DAGUE_HAVE_LIMITS_H
#include <limits.h>
#endif
#include <assert.h>
#ifdef DAGUE_HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef DAGUE_HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include <assert.h>
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "dague/devices/device.h"
#include "dague/vpmap.h"
#include "dague/data.h"

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

static int sym_twoDBC_memory_register(dague_ddesc_t* desc, struct dague_device_s* device)
{
    sym_two_dim_block_cyclic_t * sym_twodbc = (sym_two_dim_block_cyclic_t *)desc;
    return device->device_memory_register(device, desc,
                                          sym_twodbc->mat,
                                          ((size_t)sym_twodbc->super.nb_local_tiles * (size_t)sym_twodbc->super.bsiz *
                                           (size_t)dague_datadist_getsizeoftype(sym_twodbc->super.mtype)));
}

static int sym_twoDBC_memory_unregister(dague_ddesc_t* desc, struct dague_device_s* device)
{
    sym_two_dim_block_cyclic_t * sym_twodbc = (sym_two_dim_block_cyclic_t *)desc;
    return device->device_memory_unregister(device, desc, sym_twodbc->mat);
}

static uint32_t sym_twoDBC_rank_of(dague_ddesc_t * desc, ...)
{
    int cr, m, n;
    int rr;
    int res;
    va_list ap;
    sym_two_dim_block_cyclic_t * Ddesc;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    assert( m < Ddesc->super.mt );
    assert( n < Ddesc->super.nt );

    assert( (Ddesc->uplo == matrix_Lower && m>=n) ||
            (Ddesc->uplo == matrix_Upper && n>=m) );
    if ( ((Ddesc->uplo == matrix_Lower) && (m < n)) ||
         ((Ddesc->uplo == matrix_Upper) && (m > n)) )
    {
        return UINT_MAX;
    }

    /* for tile (m,n), first find coordinate of process in
     process grid which possess the tile in block cyclic dist */
    rr = m % Ddesc->grid.rows;
    cr = n % Ddesc->grid.cols;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * Ddesc->grid.cols + cr;

    return res;
}

static void sym_twoDBC_key_to_coordinates(dague_ddesc_t *desc, dague_data_key_t key, int *m, int *n)
{
    int _m, _n;
    tiled_matrix_desc_t * Ddesc;

    Ddesc = (tiled_matrix_desc_t *)desc;

    _m = key % Ddesc->lmt;
    _n = key / Ddesc->lmt;
    *m = _m - Ddesc->i / Ddesc->mb;
    *n = _n - Ddesc->j / Ddesc->nb;
}

static uint32_t sym_twoDBC_rank_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_rank_of(desc, m, n);
}

static dague_data_t* sym_twoDBC_data_of(dague_ddesc_t *desc, ...)
{
    int m, n;
    sym_two_dim_block_cyclic_t * Ddesc;
    size_t pos;
    va_list ap;

    Ddesc = (sym_two_dim_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    assert( m < Ddesc->super.mt );
    assert( n < Ddesc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == sym_twoDBC_rank_of(desc, m, n));
#endif
    assert( Ddesc->super.storage == matrix_Tile );
    assert( (Ddesc->uplo == matrix_Lower && m>=n) ||
            (Ddesc->uplo == matrix_Upper && n>=m) );

    pos = sym_twoDBC_coordinates_to_position(Ddesc, m, n);

    return dague_matrix_create_data( &Ddesc->super,
                                     (char*)Ddesc->mat + pos * Ddesc->super.bsiz * dague_datadist_getsizeoftype(Ddesc->super.mtype),
                                     pos, (n * Ddesc->super.lmt) + m );
}

static dague_data_t* sym_twoDBC_data_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_data_of(desc, m, n);
}

static int32_t sym_twoDBC_vpid_of(dague_ddesc_t *desc, ...)
{
    int m, n, p, q, pq;
    int local_m, local_n;
    sym_two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    int32_t vpid;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;

    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    q = Ddesc->grid.vp_q;
    p = Ddesc->grid.vp_p;
    assert(p*q == pq);


    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    assert( m < Ddesc->super.mt );
    assert( n < Ddesc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == sym_twoDBC_rank_of(desc, m, n));
#endif
    assert( (Ddesc->uplo == matrix_Lower && m>=n) ||
            (Ddesc->uplo == matrix_Upper && n>=m) );

    /* Compute the local tile row */
    local_m = m / Ddesc->grid.rows;
    assert( (m % Ddesc->grid.rows) == Ddesc->grid.rrank );

    /* Compute the local column */
    local_n = n / Ddesc->grid.cols;
    assert( (n % Ddesc->grid.cols) == Ddesc->grid.crank );

    vpid = (local_n % q) * p + (local_m % p);
    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static int32_t sym_twoDBC_vpid_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    sym_twoDBC_key_to_coordinates(desc, key, &m, &n);
    return sym_twoDBC_vpid_of(desc, m, n);
}

void sym_two_dim_block_cyclic_init(sym_two_dim_block_cyclic_t * Ddesc,
                                   enum matrix_type mtype,
                                   int nodes, int myrank,
                                   int mb,   int nb,   /* Tile size */
                                   int lm,   int ln,   /* Global matrix size (what is stored)*/
                                   int i,    int j,    /* Staring point in the global matrix */
                                   int m,    int n,    /* Submatrix size (the one concerned by the computation */
                                   int P, int uplo )
{
    int nb_elem, total;
    int Q;
    /* Initialize the tiled_matrix descriptor */
    dague_ddesc_t *o = &(Ddesc->super.super);

    tiled_matrix_desc_init( &(Ddesc->super), mtype, matrix_Tile,
                            sym_two_dim_block_cyclic_type,
                            nodes, myrank,
                            mb, nb, lm, ln, i, j, m, n );
    Ddesc->mat = NULL;  /* No data associated with the matrix yet */

    o->rank_of     = sym_twoDBC_rank_of;
    o->rank_of_key = sym_twoDBC_rank_of_key;
    o->vpid_of     = sym_twoDBC_vpid_of;
    o->vpid_of_key = sym_twoDBC_vpid_of_key;
    o->data_of     = sym_twoDBC_data_of;
    o->data_of_key = sym_twoDBC_data_of_key;

    o->register_memory   = sym_twoDBC_memory_register;
    o->unregister_memory = sym_twoDBC_memory_unregister;

    if(nodes < P)
        dague_abort("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);
    Q = nodes / P;
    if(nodes != P*Q)
        dague_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, 1, 1);

    /* Extra parameters */
    Ddesc->uplo = uplo;

    /* find the number of tiles this process will handle */
    total = 0; /* number of tiles handled by the process */
    if ( uplo == matrix_Lower ) {
        int column = Ddesc->grid.crank; /* tile column considered */
        int nb_elem_col = (Ddesc->super.lmt) / (Ddesc->grid.rows); //nb of tile associated to that proc in a full column
        if( (Ddesc->super.lmt) % (Ddesc->grid.rows) > Ddesc->grid.rrank )
            nb_elem_col++;

        /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
        while(column < Ddesc->super.lnt)
        {
            nb_elem = column / (Ddesc->grid.rows);
            if ( (column % (Ddesc->grid.rows)) > Ddesc->grid.rrank)
                nb_elem++;

            total += (nb_elem_col - nb_elem);
            column += Ddesc->grid.cols;
        }
    } else { /* Upper */
        int row = Ddesc->grid.rrank; /* tile row considered */
        int nb_elem_row = (Ddesc->super.lnt) / (Ddesc->grid.cols); //nb of tile associated to that proc in a full row
        if( (Ddesc->super.lnt) % (Ddesc->grid.cols) > Ddesc->grid.crank )
            nb_elem_row++;

        /* for each row of tiles in memory before searched element, compute the number of tile for displacement */
        while(row < Ddesc->super.lmt)
        {
            nb_elem = row / (Ddesc->grid.cols);
            if ( (row % (Ddesc->grid.cols)) > Ddesc->grid.crank)
                nb_elem++;

            total += (nb_elem_row - nb_elem);
            row += Ddesc->grid.rows;
        }
    }

    Ddesc->super.nb_local_tiles = total;
    Ddesc->super.data_map = (dague_data_t**)calloc(Ddesc->super.nb_local_tiles, sizeof(dague_data_t*));
}
