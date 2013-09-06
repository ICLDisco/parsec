
/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
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
#include "data_dist/matrix/two_dim_tabular.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>

#include "dague.h"
#include "data.h"

static uint32_t         twoDTD_rank_of(dague_ddesc_t* ddesc, ...);
static uint32_t         twoDTD_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t          twoDTD_vpid_of(dague_ddesc_t* ddesc, ...);
static int32_t          twoDTD_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t*    twoDTD_data_of(dague_ddesc_t* ddesc, ...);
static dague_data_t*    twoDTD_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_key_t twoDTD_data_key(dague_ddesc_t *desc, ...);

#if defined(DAGUE_PROF_TRACE)
static int twoDTD_key_to_string(dague_ddesc_t * desc, dague_data_key_t datakey, char * buffer, uint32_t buffer_size);
#endif

/*
 * Tiles are stored in column major order
 */
static uint32_t twoDTD_rank_of(dague_ddesc_t * desc, ...)
{
    int m, n, res;
    va_list ap;
    two_dim_tabular_t   * Ddesc;

    Ddesc = (two_dim_tabular_t*)desc;

    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i;
    n += Ddesc->super.j;

    res = (Ddesc->super.lmt * n) + m;
    assert( res >= 0 && res < Ddesc->tiles_table->nbelem );
    return Ddesc->tiles_table->elems[res].rank;
}

static uint32_t twoDTD_rank_of_key(dague_ddesc_t *ddesc, dague_data_key_t key)
{
    assert( key >= 0 && key < ((two_dim_tabular_t*)ddesc)->tiles_table->nbelem );

    return ((two_dim_tabular_t*)ddesc)->tiles_table->elems[key].rank;
}

static int32_t twoDTD_vpid_of(dague_ddesc_t * desc, ...)
{
    int m, n, res;
    va_list ap;
    two_dim_tabular_t   * Ddesc;

    Ddesc = (two_dim_tabular_t*)desc;

    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i;
    n += Ddesc->super.j;

    res = (Ddesc->super.lmt * n) + m;
    assert( res >= 0 && res < Ddesc->tiles_table->nbelem );
    return Ddesc->tiles_table->elems[res].vpid;
}

static int32_t twoDTD_vpid_of_key(dague_ddesc_t *ddesc, dague_data_key_t key)
{
    assert( key >= 0 && key < ((two_dim_tabular_t*)ddesc)->tiles_table->nbelem );

    return ((two_dim_tabular_t*)ddesc)->tiles_table->elems[key].vpid;
}


static dague_data_t* twoDTD_data_of(dague_ddesc_t* ddesc, ...)
{
    int m, n, res;
    va_list ap;
    two_dim_tabular_t * Ddesc;
    Ddesc = (two_dim_tabular_t *)ddesc;

    va_start(ap, ddesc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += Ddesc->super.i;
    n += Ddesc->super.j;

    res = (Ddesc->super.lmt * n) + m;
    assert( res >= 0 && res < Ddesc->tiles_table->nbelem );

    return Ddesc->tiles_table->elems[res].tile;
}

static dague_data_t* twoDTD_data_of_key(dague_ddesc_t *ddesc, dague_data_key_t key)
{
    assert( key >= 0 && key < ((two_dim_tabular_t*)ddesc)->tiles_table->nbelem );

    return ((two_dim_tabular_t*)ddesc)->tiles_table->elems[key].tile;
}

static dague_data_key_t twoDTD_data_key(dague_ddesc_t *ddesc, ...)
{
    int m, n;
    two_dim_tabular_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_tabular_t *)ddesc;
    va_start(ap, ddesc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    return ((n * Ddesc->super.lmt) + m);
}

#ifdef DAGUE_PROF_TRACE
static int twoDTD_key_to_string(dague_ddesc_t *ddesc, dague_data_key_t datakey, char * buffer, uint32_t buffer_size)
{
    two_dim_tabular_t * Ddesc;
    unsigned int row, column;
    int res;
    Ddesc = (two_dim_tabular_t *)ddesc;
    column = datakey / Ddesc->super.lmt;
    row = datakey % Ddesc->super.lmt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", row, column);
    if (res < 0)
        {
            printf("error in key_to_string for tile (%u, %u) key: %u\n", row, column, datakey);
        }
    return res;
}
#endif /* DAGUE_PROF_TRACE */

void two_dim_tabular_init(two_dim_tabular_t * Ddesc,
                          enum matrix_type mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          two_dim_td_table_t *table )
{
    // Filling matrix description with user parameter
    tiled_matrix_desc_init(&Ddesc->super,
                           mtype, matrix_Tile, two_dim_tabular_type,
                           nodes, myrank,
                           mb, nb, lm, ln, i, j, m, n);
    Ddesc->tiles_table = NULL;
    Ddesc->super.nb_local_tiles = 0;

    if( NULL != table ) {
        two_dim_tabular_set_table( Ddesc, table );
    }
    Ddesc->super.super.rank_of  = twoDTD_rank_of;
    Ddesc->super.super.rank_of_key = twoDTD_rank_of_key;
    Ddesc->super.super.vpid_of  = twoDTD_vpid_of;
    Ddesc->super.super.vpid_of_key = twoDTD_vpid_of_key;
    Ddesc->super.super.data_of  = twoDTD_data_of;
    Ddesc->super.super.data_of_key = twoDTD_data_of_key;
    Ddesc->super.super.data_key = twoDTD_data_key;

#ifdef DAGUE_PROF_TRACE
    Ddesc->super.super.key_to_string = twoDTD_key_to_string;
    Ddesc->super.super.key = NULL;
    asprintf(&Ddesc->super.super.key_dim, "(%d, %d)", Ddesc->super.mt, Ddesc->super.nt);
#endif /* DAGUE_PROF_TRACE */
}

void two_dim_tabular_set_table(two_dim_tabular_t *Ddesc, two_dim_td_table_t *table)
{
    int i;
    assert( Ddesc->tiles_table == NULL );
    assert( table != NULL );

    Ddesc->tiles_table = table;
    Ddesc->super.nb_local_tiles = 0;
    for(i = 0; i < table->nbelem; i++) {
        if( table->elems[i].rank == Ddesc->super.super.myrank )
            Ddesc->super.nb_local_tiles++;
    }
}

void two_dim_tabular_set_random_table(two_dim_tabular_t *Ddesc,
                                      unsigned int seed)
{
    int nbvp;
    unsigned int rankseed, vpseed;
    uint32_t nbtiles;
    two_dim_td_table_t *table;
    int m, n, p;
    dague_data_copy_t* dcopy;

    nbtiles = Ddesc->super.lmt * Ddesc->super.lnt;

    table = (two_dim_td_table_t*)malloc( sizeof(two_dim_td_table_t) + (nbtiles-1)*sizeof(two_dim_td_table_elem_t) );
    table->nbelem = nbtiles;

    nbvp = vpmap_get_nb_vp();

    rankseed = rand_r(&seed);
    vpseed   = rand_r(&seed);

    for(n = 0; n < Ddesc->super.lnt; n++) {
        for(m = 0; m < Ddesc->super.lmt; m++) {
            p = ((n * Ddesc->super.lmt) + m);
            table->elems[p].rank = (int)floor(((double)Ddesc->super.super.nodes * (double)rand_r(&rankseed)) / (double)RAND_MAX);
            if( table->elems[p].rank == Ddesc->super.super.myrank ) {
                table->elems[p].vpid = (int)floor(((double)nbvp * (double)rand_r(&vpseed)) / (double)RAND_MAX);
                table->elems[p].tile = dague_data_new();
                dcopy = dague_data_copy_new(table->elems[p].tile, 0);
                dcopy->device_private = dague_data_allocate( (size_t)Ddesc->super.bsiz *
                                                             (size_t)dague_datadist_getsizeoftype(Ddesc->super.mtype) );
            } else {
                table->elems[p].vpid = -1;
                table->elems[p].tile = NULL;
            }
        }
    }

    two_dim_tabular_set_table(Ddesc, table);
}

void two_dim_td_table_clone_table_structure(two_dim_tabular_t *Src, two_dim_tabular_t *Dst)
{
    int nbvp;
    uint32_t nbtiles;
    two_dim_td_table_t *table;
    int m, n, p;
    dague_data_copy_t* dcopy;

    /* Safety check: check that we can indeed clone the structure */
    assert( Src->super.lmt == Dst->super.lmt );
    assert( Src->super.lnt == Dst->super.lnt );
    assert( Src->super.i   == Dst->super.i   );
    assert( Src->super.j   == Dst->super.j   );
    assert( Src->super.mt  == Dst->super.mt  );
    assert( Src->super.nt  == Dst->super.nt  );

    assert( Src->super.super.nodes == Dst->super.super.nodes );

    nbtiles = Dst->super.lmt * Dst->super.lnt;

    table = (two_dim_td_table_t*)malloc( sizeof(two_dim_td_table_t) + (nbtiles-1)*sizeof(two_dim_td_table_elem_t) );
    table->nbelem = nbtiles;

    nbvp = vpmap_get_nb_vp();

    for(n = 0; n < Dst->super.lnt; n++) {
        for(m = 0; m < Dst->super.lmt; m++) {
            p = ((n * Dst->super.lmt) + m);
            table->elems[p].rank = Src->tiles_table->elems[p].rank;
            assert( table->elems[p].rank >= 0 && table->elems[p].rank < Dst->super.super.nodes );

            table->elems[p].vpid = Src->tiles_table->elems[p].vpid;
            assert( table->elems[p].vpid < nbvp );

            if( table->elems[p].rank == Src->super.super.myrank ) {
                table->elems[p].tile = dague_data_new();
                dcopy = dague_data_copy_new(table->elems[p].tile, 0);
                dcopy->device_private = dague_data_allocate( (size_t)Dst->super.bsiz *
                                                             (size_t)dague_datadist_getsizeoftype(Dst->super.mtype) );
            } else {
                table->elems[p].tile = NULL;
            }
        }
    }

    two_dim_tabular_set_table(Dst, table);
}

void two_dim_tabular_free_table(two_dim_td_table_t *table)
{
    int i;
    dague_data_copy_t* dcopy;

    for(i = 0; i < table->nbelem; i++) {
        if( NULL != table->elems[i].tile ) {
            dcopy = dague_data_get_copy(table->elems[i].tile, 0);
            dague_data_free(dcopy->device_private);
            dague_data_copy_release(dcopy);
            dague_data_delete(table->elems[i].tile);
            table->elems[i].tile = NULL;
        }
    }
    free(table);
}
