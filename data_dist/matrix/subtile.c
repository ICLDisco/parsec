/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/subtile.h"

static uint32_t      subtile_rank_of(parsec_ddesc_t* ddesc, ...);
static int32_t       subtile_vpid_of(parsec_ddesc_t* ddesc, ...);
static parsec_data_t* subtile_data_of(parsec_ddesc_t* ddesc, ...);
static uint32_t      subtile_rank_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key);
static int32_t       subtile_vpid_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key);
static parsec_data_t* subtile_data_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key);

subtile_desc_t *subtile_desc_create( const tiled_matrix_desc_t *tdesc,
                                     int mt, int nt,   /* Tile in tdesc */
                                     int mb, int nb,   /* sub-tiles size  */
                                     int i,  int j,    /* Starting point in the tile */
                                     int m,  int n)    /* Submatrix size (the one concerned by the computation) */
{
    subtile_desc_t *sdesc = malloc( sizeof(subtile_desc_t) );
    parsec_ddesc_t *o = &(sdesc->super.super);
    (void)mt; (void)nt;

    /* Initialize the tiled_matrix descriptor */
    tiled_matrix_desc_init( &(sdesc->super), tdesc->mtype, matrix_Lapack, 0,
                            tdesc->super.nodes, tdesc->super.myrank,
                            mb, nb, BLKLDD( tdesc, mt ), tdesc->nb,
                            i, j, m, n );

    sdesc->super.nb_local_tiles = sdesc->super.lmt * sdesc->super.lnt;
    sdesc->super.data_map = (parsec_data_t**)calloc(sdesc->super.nb_local_tiles, sizeof(parsec_data_t*));

    sdesc->mat = NULL;  /* No data associated with the matrix yet */
    //sdesc->mat  = tdesc->super.data_of( (parsec_ddesc_t*)tdesc, mt, nt );
    sdesc->vpid = 0;

    /* set the methods */
    o->rank_of      = subtile_rank_of;
    o->vpid_of      = subtile_vpid_of;
    o->data_of      = subtile_data_of;
    o->rank_of_key  = subtile_rank_of_key;
    o->vpid_of_key  = subtile_vpid_of_key;
    o->data_of_key  = subtile_data_of_key;

    /* Memory is allready registered at direct upper level */
    o->register_memory   = NULL;
    o->unregister_memory = NULL;

    return sdesc;
}

static inline void subtile_key_to_coordinates(parsec_ddesc_t *desc, parsec_data_key_t key, int *m, int *n)
{
    int _m, _n;
    tiled_matrix_desc_t *tdesc;

    tdesc = (tiled_matrix_desc_t *)desc;

    _m = key % tdesc->lmt;
    _n = key / tdesc->lmt;
    *m = _m - tdesc->i / tdesc->mb;
    *n = _n - tdesc->j / tdesc->nb;
}

/*
 *
 * Set of functions with no super-tiles
 *
 */
static uint32_t subtile_rank_of(parsec_ddesc_t * desc, ...)
{
    return desc->myrank;
}

static uint32_t subtile_rank_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    (void)key;
    return desc->myrank;
}

static int32_t subtile_vpid_of(parsec_ddesc_t *desc, ...)
{
    return ((subtile_desc_t*)desc)->vpid;
}

static int32_t subtile_vpid_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    (void)key;
    return ((subtile_desc_t*)desc)->vpid;
}

static parsec_data_t* subtile_data_of(parsec_ddesc_t *desc, ...)
{
    int m, n, position;
    size_t pos;
    va_list ap;
    subtile_desc_t * sdesc;
    sdesc = (subtile_desc_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += sdesc->super.i / sdesc->super.mb;
    n += sdesc->super.j / sdesc->super.nb;

    position = sdesc->super.lmt * n + m;

    pos = (n * sdesc->super.nb) * sdesc->super.lm
        +  m * sdesc->super.mb;
    pos *= parsec_datadist_getsizeoftype(sdesc->super.mtype);

    return parsec_matrix_create_data( &sdesc->super,
                                     (char*)sdesc->mat + pos,
                                     position, position );
}

static parsec_data_t* subtile_data_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    int m, n;
    subtile_key_to_coordinates(desc, key, &m, &n);
    return subtile_data_of(desc, m, n);
}
