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
#include "dague/debug.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/local_rectangle_cyclic.h"
#include "dague/devices/device.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>

static uint32_t localBC_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t localBC_vpid_of(dague_ddesc_t* ddesc, ...);
static dague_data_t* localBC_data_of(dague_ddesc_t* ddesc, ...);
static uint32_t localBC_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t localBC_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* localBC_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

/*
static uint32_t localBC_stview_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t localBC_stview_vpid_of(dague_ddesc_t* ddesc, ...);
static dague_data_t* localBC_stview_data_of(dague_ddesc_t* ddesc, ...);
static uint32_t localBC_stview_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t localBC_stview_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* localBC_stview_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

#if defined(DAGUE_HARD_SUPERTILE)
static uint32_t localBC_st_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t localBC_st_vpid_of(dague_ddesc_t* ddesc, ...);
static dague_data_t* localBC_st_data_of(dague_ddesc_t* ddesc, ...);
static uint32_t localBC_st_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t localBC_st_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* localBC_st_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
#endif
*/

static int localBC_memory_register(dague_ddesc_t* desc, struct dague_device_s* device)
{
    local_block_cyclic_t * localbc = (local_block_cyclic_t *)desc;
    return device->device_memory_register(device, desc,
                                          localbc->mat,
                                          ((size_t)localbc->super.nb_local_tiles * (size_t)localbc->super.bsiz *
                                           (size_t)dague_datadist_getsizeoftype(localbc->super.mtype)));
}

static int localBC_memory_unregister(dague_ddesc_t* desc, struct dague_device_s* device)
{
    local_block_cyclic_t * localbc = (local_block_cyclic_t *)desc;
    return device->device_memory_unregister(device, desc, localbc->mat);
}

void local_block_cyclic_init(local_block_cyclic_t * Ddesc,
                               enum matrix_type mtype,
                               enum matrix_storage storage,
                               int nodes, int myrank,
                               int mb,   int nb,   /* Tile size */
                               int lm,   int ln,   /* Global matrix size (what is stored)*/
                               int i,    int j,    /* Staring point in the global matrix */
                               int m,    int n,    /* Submatrix size (the one concerned by the computation */
                               int nrst, int ncst, /* Super-tiling size */
                               int P )
{
    dague_ddesc_t *o = &(Ddesc->super.super);
    int Q;

    /* Initialize the tiled_matrix descriptor */
    tiled_matrix_desc_init( &(Ddesc->super), mtype, storage, two_dim_block_cyclic_type,
                            nodes, myrank,
                            mb, nb, lm, ln, i, j, m, n );
    Ddesc->mat = NULL;  /* No data associated with the matrix yet */

    if(nodes < P)
        ERROR(("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d\n", nodes, P));
    Q = nodes / P;
    if(nodes != P*Q)
        WARNING(("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d\n", nodes, P, Q));
#if defined(DAGUE_HARD_SUPERTILE)
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, nrst, ncst);
#else
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, 1, 1);
#endif /* DAGUE_HARD_SUPERTILE */

    Ddesc->grid.rrank = 0;
    Ddesc->grid.crank = 0;

    /* Compute the number of rows handled by the local process */
    Ddesc->nb_elem_r = Ddesc->super.lmt;

    /* Compute the number of columns handled by the local process */
    Ddesc->nb_elem_c = Ddesc->super.lnt;

    /* Total number of tiles stored locally */
    Ddesc->super.nb_local_tiles = Ddesc->nb_elem_r * Ddesc->nb_elem_c;
    Ddesc->super.data_map = (dague_data_t**)calloc(Ddesc->super.nb_local_tiles, sizeof(dague_data_t*));

    /* Update llm and lln */
    Ddesc->super.llm = Ddesc->nb_elem_r * mb;
    Ddesc->super.lln = Ddesc->nb_elem_c * nb;

    /* set the methods */
    if( (nrst == 1) && (ncst == 1) ) {
        o->rank_of      = localBC_rank_of;
        o->vpid_of      = localBC_vpid_of;
        o->data_of      = localBC_data_of;
        o->rank_of_key  = localBC_rank_of_key;
        o->vpid_of_key  = localBC_vpid_of_key;
        o->data_of_key  = localBC_data_of_key;
    } 

    o->register_memory   = localBC_memory_register;
    o->unregister_memory = localBC_memory_unregister;

    DEBUG3(("two_dim_block_cyclic_init: \n"
           "      Ddesc = %p, mtype = %d, nodes = %u, myrank = %d, \n"
           "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
           "      nrst = %d, ncst = %d, P = %d, Q = %d\n",
           Ddesc, Ddesc->super.mtype, Ddesc->super.super.nodes,
           Ddesc->super.super.myrank,
           Ddesc->super.mb, Ddesc->super.nb,
           Ddesc->super.lm, Ddesc->super.ln,
           Ddesc->super.i,  Ddesc->super.j,
           Ddesc->super.m,  Ddesc->super.n,
           Ddesc->grid.strows, Ddesc->grid.stcols,
           P, Q));
}

static void localBC_key_to_coordinates(dague_ddesc_t *desc, dague_data_key_t key, int *m, int *n)
{
    int _m, _n;
    tiled_matrix_desc_t * Ddesc;

    Ddesc = (tiled_matrix_desc_t *)desc;

    _m = key % Ddesc->lmt;
    _n = key / Ddesc->lmt;
    *m = _m - Ddesc->i / Ddesc->mb;
    *n = _n - Ddesc->j / Ddesc->nb;
}

/*
 *
 * Set of functions with no super-tiles
 *
 */
static uint32_t localBC_rank_of(dague_ddesc_t * desc, ...)
{
    return desc->myrank;
}

static uint32_t localBC_rank_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    localBC_key_to_coordinates(desc, key, &m, &n);
    return localBC_rank_of(desc, m, n);
}

static int32_t localBC_vpid_of(dague_ddesc_t *desc, ...)
{
    (void)desc;
    return 0;
}

static int32_t localBC_vpid_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    localBC_key_to_coordinates(desc, key, &m, &n);
    return localBC_vpid_of(desc, m, n);
}

/*
 * Do not change this function without updating the inverse function:
 * twoDBC_position_to_coordinates()
 * Other files (zhebut) depend on the inverse function.
 */
inline int localBC_coordinates_to_position(local_block_cyclic_t *Ddesc, int m, int n){
    int position, local_m, local_n;

    /* Compute the local tile row */
    local_m = m / Ddesc->grid.rows;
    assert( (m % Ddesc->grid.rows) == Ddesc->grid.rrank );

    /* Compute the local column */
    local_n = n / Ddesc->grid.cols;
    assert( (n % Ddesc->grid.cols) == Ddesc->grid.crank );

    position = Ddesc->nb_elem_r * local_n + local_m;

    return position;
}

/*
 * This is the inverse function of: twoDBC_coordinates_to_position()
 * Please keep them in sync, other files (zhebut) depend on this function.
 */
inline void localBC_position_to_coordinates(local_block_cyclic_t *Ddesc, int position, int *m, int *n){
    int local_m, local_n, sanity_check;

    local_m = position%(Ddesc->nb_elem_r);
    local_n = position/(Ddesc->nb_elem_r);

    *m = local_m*(Ddesc->grid.rows) + Ddesc->grid.rrank;
    *n = local_n*(Ddesc->grid.cols) + Ddesc->grid.crank;

    sanity_check = localBC_coordinates_to_position(Ddesc, *m, *n);
    assert( sanity_check == position ); (void)sanity_check;

    return;
}

static dague_data_t* localBC_data_of(dague_ddesc_t *desc, ...)
{
    int m, n, position;
    size_t pos;
    va_list ap;
    local_block_cyclic_t * Ddesc;
    Ddesc = (local_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == localBC_rank_of(desc, m, n));
#endif

    position = localBC_coordinates_to_position(Ddesc, m, n);

    if( Ddesc->super.storage == matrix_Tile ) {
        pos = position;
        pos *= (size_t)Ddesc->super.bsiz;
    } else {
        int local_m = m / Ddesc->grid.rows;
        int local_n = n / Ddesc->grid.cols;
        pos = (local_n * Ddesc->super.nb) * Ddesc->super.llm
            +  local_m * Ddesc->super.mb;
    }

    return dague_matrix_create_data( &Ddesc->super,
                                     (char*)Ddesc->mat + pos * dague_datadist_getsizeoftype(Ddesc->super.mtype),
                                     position, (n * Ddesc->super.lmt) + m );
}

static dague_data_t* localBC_data_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    int m, n;
    localBC_key_to_coordinates(desc, key, &m, &n);
    return localBC_data_of(desc, m, n);
}
