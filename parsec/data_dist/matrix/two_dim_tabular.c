
/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_tabular.h"
#include "parsec/vpmap.h"
#include "parsec/runtime.h"
#include "parsec/data.h"
#include <string.h>
#if defined(__WINDOWS__)
#define _CRT_RAND_S
#include <stdlib.h>
#endif

static uint32_t      twoDTD_rank_of(    parsec_data_collection_t* dc, ... );
static uint32_t      twoDTD_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t       twoDTD_vpid_of(    parsec_data_collection_t* dc, ... );
static int32_t       twoDTD_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* twoDTD_data_of(    parsec_data_collection_t* dc, ... );
static parsec_data_t* twoDTD_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);

/*
 * Tiles are stored in column major order
 */
static uint32_t twoDTD_rank_of(parsec_data_collection_t * desc, ...)
{
    int m, n, res;
    va_list ap;
    parsec_matrix_tabular_t   * dc;

    dc = (parsec_matrix_tabular_t*)desc;

    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    res = (dc->super.lmt * n) + m;
    assert( res >= 0 && res < dc->tiles_table->nbelem );
    return dc->tiles_table->elems[res].rank;
}

static uint32_t twoDTD_rank_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    assert( key < (parsec_data_key_t)(((parsec_matrix_tabular_t*)dc)->tiles_table->nbelem) );

    return ((parsec_matrix_tabular_t*)dc)->tiles_table->elems[key].rank;
}

static int32_t twoDTD_vpid_of(parsec_data_collection_t * desc, ...)
{
    int m, n, res;
    va_list ap;
    parsec_matrix_tabular_t   * dc;

    dc = (parsec_matrix_tabular_t*)desc;

    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    res = (dc->super.lmt * n) + m;
    assert( res >= 0 && res < dc->tiles_table->nbelem );
    return dc->tiles_table->elems[res].vpid;
}

static int32_t twoDTD_vpid_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    assert( key < (parsec_data_key_t)(((parsec_matrix_tabular_t*)dc)->tiles_table->nbelem) );

    return ((parsec_matrix_tabular_t*)dc)->tiles_table->elems[key].vpid;
}


static parsec_data_t* twoDTD_data_of(parsec_data_collection_t* dc, ...)
{
    int m, n, res;
    va_list ap;
    parsec_matrix_tabular_t * tdc;
    parsec_two_dim_td_table_elem_t *elem;
    tdc = (parsec_matrix_tabular_t *)dc;

    va_start(ap, dc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += tdc->super.i / tdc->super.mb;
    n += tdc->super.j / tdc->super.nb;

    res = (tdc->super.lmt * n) + m;
    assert( res >= 0 && res < tdc->tiles_table->nbelem );
    elem = &(tdc->tiles_table->elems[res]);
    assert(elem->pos >= 0);

    return parsec_tiled_matrix_create_data( &tdc->super, elem->data, elem->pos, res );
}

static parsec_data_t* twoDTD_data_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    parsec_matrix_tabular_t       *tdc = (parsec_matrix_tabular_t*)dc;
    parsec_two_dim_td_table_elem_t *elem;
    assert( key < (parsec_data_key_t)( tdc->tiles_table->nbelem ) );

    elem = &(tdc->tiles_table->elems[key]);
    return parsec_tiled_matrix_create_data( &tdc->super, elem->data, elem->pos, key );
}

void parsec_matrix_tabular_init(parsec_matrix_tabular_t * dc,
                          parsec_matrix_type_t mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          parsec_two_dim_td_table_t *table )
{
    // Filling matrix description with user parameter
    parsec_tiled_matrix_init(&dc->super,
                           mtype, PARSEC_MATRIX_TILE, parsec_matrix_tabular_type,
                           nodes, myrank,
                           mb, nb, lm, ln, i, j, m, n);
    dc->tiles_table = NULL;
    dc->super.nb_local_tiles = 0;

    dc->super.super.rank_of     = twoDTD_rank_of;
    dc->super.super.rank_of_key = twoDTD_rank_of_key;
    dc->super.super.vpid_of     = twoDTD_vpid_of;
    dc->super.super.vpid_of_key = twoDTD_vpid_of_key;
    dc->super.super.data_of     = twoDTD_data_of;
    dc->super.super.data_of_key = twoDTD_data_of_key;

    if( NULL != table ) {
        parsec_matrix_tabular_set_table( dc, table );
    }
}

void parsec_matrix_tabular_destroy(parsec_matrix_tabular_t *tdc)
{
    parsec_two_dim_td_table_elem_t *elem;
    parsec_two_dim_td_table_t *table = tdc->tiles_table;
    int i;

    for(i = 0, elem = &(table->elems[0]);
        i < table->nbelem;
        i++, elem++)
    {
        if( elem->data != NULL ) {
            if(tdc->user_table == 0)
                parsec_data_free(elem->data);
            elem->data = NULL;
        }
    }
    free(tdc->tiles_table);

    parsec_tiled_matrix_destroy( &(tdc->super) );
}

void parsec_matrix_tabular_set_user_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
{
    int i;
    assert( dc->tiles_table == NULL );
    assert( table != NULL );
    assert( table->nbelem == dc->super.lmt * dc->super.lnt );
    dc->tiles_table = table;
    dc->super.nb_local_tiles = 0;
    for(i = 0; i < table->nbelem; i++) {
        if( table->elems[i].rank == dc->super.super.myrank ) {
            assert(table->elems[i].data != NULL);
            assert(table->elems[i].pos == dc->super.nb_local_tiles);
            dc->super.nb_local_tiles++;
        }
    }
    dc->user_table = 1;
    dc->super.data_map = (parsec_data_t**)calloc(dc->super.nb_local_tiles, sizeof(parsec_data_t*));
}

void parsec_matrix_tabular_set_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
{
    int i;
    assert( dc->tiles_table == NULL );
    assert( table != NULL );
    assert( table->nbelem == dc->super.lmt * dc->super.lnt );

    dc->tiles_table = table;
    dc->super.nb_local_tiles = 0;
    for(i = 0; i < table->nbelem; i++) {
        if( table->elems[i].rank == dc->super.super.myrank )
        {
            table->elems[i].pos  = dc->super.nb_local_tiles;
            table->elems[i].data = parsec_data_allocate( (size_t)dc->super.bsiz *
                                                        (size_t)parsec_datadist_getsizeoftype(dc->super.mtype) );
            dc->super.nb_local_tiles++;
        }
        else {
            table->elems[i].pos  = -1;
            table->elems[i].vpid = -1;
            table->elems[i].data = NULL;
        }
    }
    dc->user_table = 0;
    dc->super.data_map = (parsec_data_t**)calloc(dc->super.nb_local_tiles, sizeof(parsec_data_t*));
}


void parsec_matrix_tabular_set_random_table(parsec_matrix_tabular_t *dc,
                                      unsigned int seed)
{
    int nbvp;
    unsigned int rankseed, vpseed;
    uint32_t nbtiles;
    parsec_two_dim_td_table_t *table;
    int m, n, p;

    nbtiles = dc->super.lmt * dc->super.lnt;

    table = (parsec_two_dim_td_table_t*)malloc( sizeof(parsec_two_dim_td_table_t)
                                                + (nbtiles-1)*sizeof(parsec_two_dim_td_table_elem_t) );
    table->nbelem = nbtiles;

    nbvp = vpmap_get_nb_vp();
#if defined(__WINDOWS__)
    rand_s(&rankseed);
    rand_s(&vpseed);

    for(n = 0; n < dc->super.lnt; n++) {
        for(m = 0; m < dc->super.lmt; m++) {
            p = ((n * dc->super.lmt) + m);
            rand_s(&rankseed);
            table->elems[p].rank = (int)(((double)dc->super.super.nodes * (double)rankseed) / ((double) UINT_MAX + 1));

            if( table->elems[p].rank == dc->super.super.myrank ) {
                rand_s(&vpseed);
                table->elems[p].vpid = (int)(((double)nbvp * (double)vpseed) / ((double) UINT_MAX + 1));
            }
        }
    }
    (void)seed;  /* stop the compiler from complaining about unused data */
#elif defined(PARSEC_HAVE_RAND_R)
    rankseed = rand_r(&seed);
    vpseed   = rand_r(&seed);

    for(n = 0; n < dc->super.lnt; n++) {
        for(m = 0; m < dc->super.lmt; m++) {
            p = ((n * dc->super.lmt) + m);
            table->elems[p].rank = (int)(((double)dc->super.super.nodes * (double)rand_r(&rankseed)) / (double)RAND_MAX);

            if( table->elems[p].rank == dc->super.super.myrank ) {
                table->elems[p].vpid = (int)(((double)nbvp * (double)rand_r(&vpseed)) / (double)RAND_MAX);
            }
        }
    }
#else
#error Missing support for the platform random number generator similar to POSIX rand_r
#endif
    parsec_matrix_tabular_set_table(dc, table);
}

void parsec_matrix_tabular_clone_table_structure(parsec_matrix_tabular_t *Src, parsec_matrix_tabular_t *Dst)
{
    size_t tablesize;
    parsec_two_dim_td_table_t *table;

    /* Safety check: check that we can indeed clone the structure */
    assert( Src->super.lmt == Dst->super.lmt );
    assert( Src->super.lnt == Dst->super.lnt );
    assert( Src->super.i   == Dst->super.i   );
    assert( Src->super.j   == Dst->super.j   );
    assert( Src->super.mt  == Dst->super.mt  );
    assert( Src->super.nt  == Dst->super.nt  );

    assert( Src->super.super.nodes == Dst->super.super.nodes );

    tablesize = (Dst->super.lmt * Dst->super.lnt - 1) * sizeof(parsec_two_dim_td_table_elem_t)
        + sizeof(parsec_two_dim_td_table_t);

    table = (parsec_two_dim_td_table_t*)malloc( tablesize );
    memcpy( table, Src->tiles_table, tablesize );

    parsec_matrix_tabular_set_table(Dst, table);
}
