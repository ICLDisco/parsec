
/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/two_dim_tabular.h"
#include "dague/vpmap.h"
#include "dague.h"
#include "data.h"

#include <math.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

static uint32_t      twoDTD_rank_of(    dague_ddesc_t* ddesc, ... );
static uint32_t      twoDTD_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static int32_t       twoDTD_vpid_of(    dague_ddesc_t* ddesc, ... );
static int32_t       twoDTD_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* twoDTD_data_of(    dague_ddesc_t* ddesc, ... );
static dague_data_t* twoDTD_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

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
    assert( key < (dague_data_key_t)(((two_dim_tabular_t*)ddesc)->tiles_table->nbelem) );

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
    assert( key < (dague_data_key_t)(((two_dim_tabular_t*)ddesc)->tiles_table->nbelem) );

    return ((two_dim_tabular_t*)ddesc)->tiles_table->elems[key].vpid;
}


static dague_data_t* twoDTD_data_of(dague_ddesc_t* ddesc, ...)
{
    int m, n, res;
    va_list ap;
    two_dim_tabular_t * Ddesc;
    two_dim_td_table_elem_t *elem;
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
    elem = &(Ddesc->tiles_table->elems[res]);
    assert(elem->pos >= 0);

    return dague_matrix_create_data( &Ddesc->super, elem->data, elem->pos, res );
}

static dague_data_t* twoDTD_data_of_key(dague_ddesc_t *ddesc, dague_data_key_t key)
{
    two_dim_tabular_t       *tddesc = (two_dim_tabular_t*)ddesc;
    two_dim_td_table_elem_t *elem;
    assert( key < (dague_data_key_t)( tddesc->tiles_table->nbelem ) );

    elem = &(tddesc->tiles_table->elems[key]);
    return dague_matrix_create_data( &tddesc->super, elem->data, elem->pos, key );
}

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

    Ddesc->super.super.rank_of     = twoDTD_rank_of;
    Ddesc->super.super.rank_of_key = twoDTD_rank_of_key;
    Ddesc->super.super.vpid_of     = twoDTD_vpid_of;
    Ddesc->super.super.vpid_of_key = twoDTD_vpid_of_key;
    Ddesc->super.super.data_of     = twoDTD_data_of;
    Ddesc->super.super.data_of_key = twoDTD_data_of_key;

    if( NULL != table ) {
        two_dim_tabular_set_table( Ddesc, table );
    }
}

void two_dim_tabular_destroy(two_dim_tabular_t *tddesc)
{
    two_dim_td_table_elem_t *elem;
    two_dim_td_table_t *table = tddesc->tiles_table;
    int i;

    for(i = 0, elem = &(table->elems[0]);
        i < table->nbelem;
        i++, elem++)
    {
        if( elem->data != NULL ) {
            dague_data_free(elem->data);
            elem->data = NULL;
        }
    }
    free(tddesc->tiles_table);

    tiled_matrix_desc_destroy( &(tddesc->super) );
}

void two_dim_tabular_set_table(two_dim_tabular_t *Ddesc, two_dim_td_table_t *table)
{
    int i;
    assert( Ddesc->tiles_table == NULL );
    assert( table != NULL );
    assert( table->nbelem == Ddesc->super.lmt * Ddesc->super.lnt );

    Ddesc->tiles_table = table;
    Ddesc->super.nb_local_tiles = 0;
    for(i = 0; i < table->nbelem; i++) {
        if( table->elems[i].rank == Ddesc->super.super.myrank )
        {
            table->elems[i].pos  = Ddesc->super.nb_local_tiles;
            table->elems[i].data = dague_data_allocate( (size_t)Ddesc->super.bsiz *
                                                        (size_t)dague_datadist_getsizeoftype(Ddesc->super.mtype) );
            Ddesc->super.nb_local_tiles++;
        }
        else {
            table->elems[i].pos  = -1;
            table->elems[i].vpid = -1;
            table->elems[i].data = NULL;
        }
    }

    Ddesc->super.data_map = (dague_data_t**)calloc(Ddesc->super.nb_local_tiles, sizeof(dague_data_t*));
}

void two_dim_tabular_set_random_table(two_dim_tabular_t *Ddesc,
                                      unsigned int seed)
{
    int nbvp;
    unsigned int rankseed, vpseed;
    uint32_t nbtiles;
    two_dim_td_table_t *table;
    int m, n, p;

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
            }
        }
    }

    two_dim_tabular_set_table(Ddesc, table);
}

void two_dim_td_table_clone_table_structure(two_dim_tabular_t *Src, two_dim_tabular_t *Dst)
{
    size_t tablesize;
    two_dim_td_table_t *table;

    /* Safety check: check that we can indeed clone the structure */
    assert( Src->super.lmt == Dst->super.lmt );
    assert( Src->super.lnt == Dst->super.lnt );
    assert( Src->super.i   == Dst->super.i   );
    assert( Src->super.j   == Dst->super.j   );
    assert( Src->super.mt  == Dst->super.mt  );
    assert( Src->super.nt  == Dst->super.nt  );

    assert( Src->super.super.nodes == Dst->super.super.nodes );

    tablesize = (Dst->super.lmt * Dst->super.lnt - 1) * sizeof(two_dim_td_table_elem_t)
        + sizeof(two_dim_td_table_t);

    table = (two_dim_td_table_t*)malloc( tablesize );
    memcpy( table, Src->tiles_table, tablesize );

    two_dim_tabular_set_table(Dst, table);
}
