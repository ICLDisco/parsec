/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/vector_two_dim_cyclic.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>

static uint32_t vector_twoDBC_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t vector_twoDBC_vpid_of(dague_ddesc_t* ddesc, ...);
static void* vector_twoDBC_data_of(dague_ddesc_t* ddesc, ...);

static uint32_t vector_twoDBC_stview_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t vector_twoDBC_stview_vpid_of(dague_ddesc_t* ddesc, ...);
static void* vector_twoDBC_stview_data_of(dague_ddesc_t* ddesc, ...);

#if defined(DAGUE_HARD_SUPERTILE)
static uint32_t vector_twoDBC_st_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t vector_twoDBC_st_vpid_of(dague_ddesc_t* ddesc, ...);
static void* vector_twoDBC_st_data_of(dague_ddesc_t* ddesc, ...);
#endif

#if defined(DAGUE_PROF_TRACE)
static uint32_t vector_twoDBC_data_key(struct dague_ddesc *desc, ...);
static int  vector_twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size);
#endif


void vector_two_dim_cyclic_init(vector_two_dim_cyclic_t * Ddesc,
                                enum matrix_type mtype,
                                enum matrix_storage storage,
                                int nodes, int cores, int myrank,
                                int mb,   /* Tile size */
                                int lm,   /* Global matrix size (what is stored)*/
                                int i,    /* Staring point in the global matrix */
                                int m,    /* Submatrix size (the one concerned by the computation */
                                int nrst, /* Super-tiling size */
                                int P )
{
    int nb_elem_c, temp;
    int Q;
    dague_ddesc_t *o = &(Ddesc->super.super);
#if defined(DAGUE_PROF_TRACE)
    o->data_key      = vector_twoDBC_data_key;
    o->key_to_string = vector_twoDBC_key_to_string;
    o->key_dim       = NULL;
    o->key           = NULL;
#endif

    /* Initialize the tiled_matrix descriptor */
    tiled_matrix_desc_init( &(Ddesc->super), mtype, storage, two_dim_block_cyclic_type,
                            nodes, cores, myrank,
                            mb, 1, lm, 1, i, 0, m, 1 );

    if(nodes < P)
        ERROR(("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d\n", nodes, P));
    Q = nodes / P;
    if(nodes != P*Q)
        WARNING(("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d\n", nodes, P, Q));

#if defined(DAGUE_HARD_SUPERTILE)
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, nrst, 1);
#else
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, 1, 1);
#endif /* DAGUE_HARD_SUPERTILE */

    /* Compute the number of rows handled by the local process */
    Ddesc->nb_elem_r = 0;
    temp = Ddesc->grid.rrank * Ddesc->grid.strows; /* row coordinate of the first tile to handle */
    while( temp < Ddesc->super.lmt ) {
        if( (temp + (Ddesc->grid.strows)) < Ddesc->super.lmt ) {
            Ddesc->nb_elem_r += (Ddesc->grid.strows);
            temp += ((Ddesc->grid.rows) * (Ddesc->grid.strows));
            continue;
        }
        Ddesc->nb_elem_r += ((Ddesc->super.lmt) - temp);
        break;
    }

    /* Total number of tiles stored locally */
    nb_elem_c = ( myrank % Q == 0 ) ? 1 : 0;
    Ddesc->super.nb_local_tiles = nb_elem_c * Ddesc->nb_elem_r;

    /* Update llm and lln */
    Ddesc->super.llm = Ddesc->nb_elem_r * mb;
    Ddesc->super.lln = 1;

    /* set the methods */
    if( (nrst == 1) ) {
        o->rank_of      = vector_twoDBC_rank_of;
        o->vpid_of      = vector_twoDBC_vpid_of;
        o->data_of      = vector_twoDBC_data_of;
    } else {
#if defined(DAGUE_HARD_SUPERTILE)
        o->rank_of      = vector_twoDBC_st_rank_of;
        o->vpid_of      = vector_twoDBC_st_vpid_of;
        o->data_of      = vector_twoDBC_st_data_of;
#else
        vector_two_dim_cyclic_supertiled_view(Ddesc, Ddesc, nrst);
#endif /* DAGUE_HARD_SUPERTILE */
    }

    DEBUG3(("vector_two_dim_cyclic_init: \n"
           "      Ddesc = %p, mtype = %d, nodes = %u, cores = %u, myrank = %d, \n"
           "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
           "      nrst = %d, ncst = %d, P = %d, Q = %d\n",
           Ddesc, Ddesc->super.mtype, Ddesc->super.super.nodes, Ddesc->super.super.cores,
           Ddesc->super.super.myrank,
           Ddesc->super.mb, Ddesc->super.nb,
           Ddesc->super.lm, Ddesc->super.ln,
           Ddesc->super.i,  Ddesc->super.j,
           Ddesc->super.m,  Ddesc->super.n,
           Ddesc->grid.strows, Ddesc->grid.stcols,
           P, Q));
}




/*
 *
 * Set of functions with no super-tiles
 *
 */
static uint32_t vector_twoDBC_rank_of(dague_ddesc_t * desc, ...)
{
    unsigned int m;
    unsigned int rr;
    unsigned int res;
    va_list ap;
    vector_two_dim_cyclic_t * Ddesc;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = m % Ddesc->grid.rows;
    res = rr * Ddesc->grid.cols;

    return res;
}

static int32_t vector_twoDBC_vpid_of(dague_ddesc_t *desc, ...)
{
    int m, p, pq;
    int local_m;
    vector_two_dim_cyclic_t * Ddesc;
    va_list ap;
    int32_t vpid;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* If 1 VP, always return 0 */
    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    p = Ddesc->grid.vp_p;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    local_m = m / Ddesc->grid.rows;
    assert( (m % Ddesc->grid.rows) == Ddesc->grid.rrank );

    vpid = (local_m % p);
    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static void *vector_twoDBC_data_of(dague_ddesc_t *desc, ...)
{
    int m;
    size_t pos;
    int local_m;
    va_list ap;
    vector_two_dim_cyclic_t * Ddesc;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    local_m = m / Ddesc->grid.rows;
    assert( (m % Ddesc->grid.rows) == Ddesc->grid.rrank );

    pos = local_m * Ddesc->super.mb;

    pos *= dague_datadist_getsizeoftype(Ddesc->super.mtype);
    return &(((char *) Ddesc->mat)[pos]);
}


/****
 * Set of functions with Supertiled view of the distribution
 ****/

void vector_two_dim_cyclic_supertiled_view( vector_two_dim_cyclic_t* target,
                                            vector_two_dim_cyclic_t* origin,
                                            int rst )
{
    assert( (origin->grid.strows == 1) && (origin->grid.stcols == 1) );
    target = origin;
    target->grid.strows = rst;
    target->grid.stcols = 1;
    target->super.super.rank_of = vector_twoDBC_stview_rank_of;
    target->super.super.data_of = vector_twoDBC_stview_data_of;
    target->super.super.vpid_of = vector_twoDBC_stview_vpid_of;
}

static inline unsigned int st_compute_m(vector_two_dim_cyclic_t* desc, unsigned int m)
{
    unsigned int p, ps, mt;
    p = desc->grid.rows;
    ps = desc->grid.strows;
    mt = desc->super.mt;
    do {
        m = m-m%(p*ps) + (m%ps)*p + (m/ps)%p;
    } while(m >= mt);
    return m;
}

static uint32_t vector_twoDBC_stview_rank_of(dague_ddesc_t* ddesc, ...)
{
    unsigned int m, sm;
    vector_two_dim_cyclic_t* desc = (vector_two_dim_cyclic_t*)ddesc;
    va_list ap;
    va_start(ap, ddesc);
    m = va_arg(ap, unsigned int);
    va_end(ap);
    sm = st_compute_m(desc, m);

    DEBUG3(("SuperTiledView: rankof(%d)=%d converted to rankof(%d)=%d\n", m, vector_twoDBC_rank_of(ddesc,m), sm, vector_twoDBC_rank_of(ddesc,sm)));

    return vector_twoDBC_rank_of(ddesc, sm);
}

static int32_t vector_twoDBC_stview_vpid_of(dague_ddesc_t* ddesc, ...)
{
    unsigned int m;
    vector_two_dim_cyclic_t* desc = (vector_two_dim_cyclic_t*)ddesc;
    va_list ap;
    va_start(ap, ddesc);
    m = va_arg(ap, unsigned int);
    va_end(ap);
    m = st_compute_m(desc, m);
    return vector_twoDBC_vpid_of(ddesc, m);
}

static void* vector_twoDBC_stview_data_of(dague_ddesc_t* ddesc, ...)
{
    unsigned int m;
    vector_two_dim_cyclic_t* desc = (vector_two_dim_cyclic_t*)ddesc;
    va_list ap;
    va_start(ap, ddesc);
    m = va_arg(ap, unsigned int);
    va_end(ap);
    m = st_compute_m(desc, m);
    return vector_twoDBC_data_of(ddesc, m);
}





#if defined(DAGUE_HARD_SUPERTILE)
/*
 *
 * Set of functions with super-tiles
 *
 */
static uint32_t vector_twoDBC_st_rank_of(dague_ddesc_t * desc, ...)
{
    unsigned int m;
    unsigned int str, rr;
    unsigned int res;
    va_list ap;
    vector_two_dim_cyclic_t * Ddesc;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    /* (m,n) is in super-tile (str, stc)*/
    str = m / Ddesc->grid.strows;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = str % Ddesc->grid.rows;
    res = rr * Ddesc->grid.cols;

    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
    /*            m, n, res, rr, cr, Ddesc->grid.rows, Ddesc->grid.cols); */
    return res;
}

static int32_t vector_twoDBC_st_vpid_of(dague_ddesc_t *desc, ...)
{
    int m, p, q, pq;
    int local_m;
    vector_two_dim_cyclic_t * Ddesc;
    va_list ap;
    int32_t vpid;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* If no vp, always return 0 */
    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    q = Ddesc->grid.vp_q;
    p = Ddesc->grid.vp_p;
    assert(p*q == pq);

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_st_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    local_m = ( m / (Ddesc->grid.strows * Ddesc->grid.rows) ) * Ddesc->grid.strows;
    m = m % (Ddesc->grid.strows * Ddesc->grid.rows);
    assert( m / Ddesc->grid.strows == Ddesc->grid.rrank);
    local_m += m % Ddesc->grid.strows;

    vpid = (local_m % p);
    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static void *vector_twoDBC_st_data_of(dague_ddesc_t *desc, ...)
{
    size_t pos;
    int m, local_m;
    va_list ap;
    vector_two_dim_cyclic_t * Ddesc;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_st_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    local_m = ( m / (Ddesc->grid.strows * Ddesc->grid.rows) ) * Ddesc->grid.strows;
    m = m % (Ddesc->grid.strows * Ddesc->grid.rows);
    assert( m / Ddesc->grid.strows == Ddesc->grid.rrank);
    local_m += m % Ddesc->grid.strows;

    if( Ddesc->super.storage == matrix_Tile ) {
        pos = local_m;
        pos *= (size_t)Ddesc->super.bsiz;
    } else {
        pos = local_m * Ddesc->super.mb;
    }

    pos *= dague_datadist_getsizeoftype(Ddesc->super.mtype);
    return &(((char *) Ddesc->mat)[pos]);
}

#endif /* DAGUE_HARD_SUPERTILE */

/*
 * Common functions
 */
#ifdef DAGUE_PROF_TRACE
/* return a unique key (unique only for the specified dague_ddesc) associated to a data */
static uint32_t vector_twoDBC_data_key(struct dague_ddesc *desc, ...)
{
    unsigned int m;
    vector_two_dim_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;

    return m;
}

/* return a string meaningful for profiling about data */
static int  vector_twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size)
{
    vector_two_dim_cyclic_t * Ddesc;
    int res;

    Ddesc = (vector_two_dim_cyclic_t *)desc;
    res = snprintf(buffer, buffer_size, "(%u)", datakey);
    if (res < 0)
        {
            printf("error in key_to_string for tile (%u) key: %u\n", datakey, datakey);
        }
    return res;
}
#endif /* DAGUE_PROF_TRACE */
