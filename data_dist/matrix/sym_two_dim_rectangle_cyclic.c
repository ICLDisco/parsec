/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"

#include <stdlib.h>
#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#include <assert.h>
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include <assert.h>
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "dague/devices/device.h"
#include "data.h"

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

static int sym_twoDBC_memory_register(dague_ddesc_t* desc, struct dague_device_s* device)
{
    sym_two_dim_block_cyclic_t * sym_twodbc = (sym_two_dim_block_cyclic_t *)desc;
    return device->device_memory_register(device,
                                          sym_twodbc->mat,
                                          (sym_twodbc->super.nb_local_tiles * sym_twodbc->super.bsiz *
                                           dague_datadist_getsizeoftype(sym_twodbc->super.mtype)));
}

static int sym_twoDBC_memory_unregister(dague_ddesc_t* desc, struct dague_device_s* device)
{
    sym_two_dim_block_cyclic_t * sym_twodbc = (sym_two_dim_block_cyclic_t *)desc;
    return device->device_memory_unregister(device, sym_twodbc->mat);
}

static uint32_t sym_twoDBC_rank_of(dague_ddesc_t * desc, ...)
{
    unsigned int cr, m, n;
    unsigned int rr;
    unsigned int res;
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

    assert( (Ddesc->uplo == MatrixLower && m>=n) ||
            (Ddesc->uplo == MatrixUpper && n>=m) );
    if ( ((Ddesc->uplo == MatrixLower) && (m < n)) ||
         ((Ddesc->uplo == MatrixUpper) && (m > n)) )
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

#if defined(DISTRIBUTED)
    assert(desc->myrank == sym_twoDBC_rank_of(desc, m, n));
#endif
    assert( Ddesc->super.storage == matrix_Tile );
    assert( (Ddesc->uplo == MatrixLower && m>=n) ||
            (Ddesc->uplo == MatrixUpper && n>=m) );

    pos = sym_twoDBC_coordinates_to_position(Ddesc, m, n);

    return dague_matrix_create_data( &Ddesc->super,
                                     (char*)Ddesc->mat + pos * Ddesc->super.bsiz * dague_datadist_getsizeoftype(Ddesc->super.mtype),
                                     pos, (n * Ddesc->super.lmt) + m );
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

#if defined(DISTRIBUTED)
    assert(desc->myrank == sym_twoDBC_rank_of(desc, m, n));
#endif
    assert( (Ddesc->uplo == MatrixLower && m>=n) ||
            (Ddesc->uplo == MatrixUpper && n>=m) );

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


#ifdef DAGUE_PROF_TRACE
/* return a unique key (unique only for the specified dague_ddesc_t) associated to a data */
static uint32_t sym_twoDBC_data_key(dague_ddesc_t *desc, ...)
{
    unsigned int m, n;
    sym_two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    return ((n * Ddesc->super.lmt) + m);
}

/* return a string meaningful for profiling about data */
static int  sym_twoDBC_key_to_string(dague_ddesc_t *desc, uint32_t datakey, char * buffer, uint32_t buffer_size)
{
    sym_two_dim_block_cyclic_t * Ddesc;
    unsigned int row, column;
    int res;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;
    column = datakey / Ddesc->super.lmt;
    row = datakey % Ddesc->super.lmt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", row, column);
    if (res < 0) {
        printf("error in key_to_string for tile (%u, %u) key: %u\n", row, column, datakey);
    }
    return res;
}
#endif /* DAGUE_PROF_TRACE */


void sym_two_dim_block_cyclic_init(sym_two_dim_block_cyclic_t * Ddesc,
                                   enum matrix_type mtype,
                                   int nodes, int cores, int myrank,
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
    o->rank_of       = sym_twoDBC_rank_of;
    o->vpid_of       = sym_twoDBC_vpid_of;
    o->data_of       = sym_twoDBC_data_of;
#if defined(DAGUE_PROF_TRACE)
    o->data_key      = sym_twoDBC_data_key;
    o->key_to_string = sym_twoDBC_key_to_string;
    o->key_dim       = NULL;
    o->key_base      = NULL;
#endif
    o->register_memory   = sym_twoDBC_memory_register;
    o->unregister_memory = sym_twoDBC_memory_unregister;
    tiled_matrix_desc_init( &(Ddesc->super), mtype, matrix_Tile,
                            sym_two_dim_block_cyclic_type,
                            nodes, cores, myrank,
                            mb, nb, lm, ln, i, j, m, n );

    if(nodes < P)
        ERROR(("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d\n", nodes, P));
    Q = nodes / P;
    if(nodes != P*Q)
        WARNING(("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d\n", nodes, P, Q));
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, 1, 1);

    /* Extra parameters */
    Ddesc->uplo = uplo;

    /* find the number of tiles this process will handle */
    total = 0; /* number of tiles handled by the process */
    if ( uplo == MatrixLower ) {
        int column = Ddesc->grid.crank; /* tile column considered */
        int nb_elem_col = (Ddesc->super.lmt) / (Ddesc->grid.rows); //nb of tile associated to that proc in a full column
        if( (Ddesc->super.lmt) % (Ddesc->grid.rows) > Ddesc->grid.rrank )
            nb_elem_col++;

        while(column < Ddesc->super.lnt) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
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

        while(row < Ddesc->super.lmt) /* for each row of tiles in memory before searched element, compute the number of tile for displacement */
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
