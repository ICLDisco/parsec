/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/matrix_internal.h"
#include "parsec/mca/device/device.h"
#include "parsec/vpmap.h"

static uint32_t twoDBC_rank_of(parsec_data_collection_t* dc, ...);
static int32_t twoDBC_vpid_of(parsec_data_collection_t* dc, ...);
static parsec_data_t* twoDBC_data_of(parsec_data_collection_t* dc, ...);
static uint32_t twoDBC_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t twoDBC_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* twoDBC_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);

static uint32_t twoDBC_kview_rank_of(parsec_data_collection_t* dc, ...);
static int32_t twoDBC_kview_vpid_of(parsec_data_collection_t* dc, ...);
static parsec_data_t* twoDBC_kview_data_of(parsec_data_collection_t* dc, ...);
static uint32_t twoDBC_kview_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t twoDBC_kview_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* twoDBC_kview_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);

#if !PARSEC_KCYCLIC_WITH_VIEW
static uint32_t twoDBC_kcyclic_rank_of(parsec_data_collection_t* dc, ...);
static int32_t twoDBC_kcyclic_vpid_of(parsec_data_collection_t* dc, ...);
static parsec_data_t* twoDBC_kcyclic_data_of(parsec_data_collection_t* dc, ...);
static uint32_t twoDBC_kcyclic_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t twoDBC_kcyclic_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* twoDBC_kcyclic_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key);
#endif

static int twoDBC_memory_register(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_block_cyclic_t * twodbc = (parsec_matrix_block_cyclic_t *)desc;
    if( (NULL == twodbc->mat ) || (twodbc->super.nb_local_tiles == 0)) {
        return PARSEC_SUCCESS;
    }
    return device->memory_register(device, desc,
                                   twodbc->mat,
                                   ((size_t)twodbc->super.llm * (size_t)twodbc->super.lln *
                                   (size_t)parsec_datadist_getsizeoftype(twodbc->super.mtype)));
}

static int twoDBC_memory_unregister(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_block_cyclic_t * twodbc = (parsec_matrix_block_cyclic_t *)desc;
    if( (NULL == twodbc->mat ) || (twodbc->super.nb_local_tiles == 0)) {
        return PARSEC_SUCCESS;
    }
    return device->memory_unregister(device, desc, twodbc->mat);
}


void parsec_matrix_block_cyclic_lapack_init(parsec_matrix_block_cyclic_t * dc,
                               parsec_matrix_type_t mtype,
                               parsec_matrix_storage_t storage,
                               int myrank,
                               int mb,   int nb,   /* Tile size */
                               int lm,   int ln,   /* Global matrix size (what is stored)*/
                               int i,    int j,    /* Staring point in the global matrix */
                               int m,    int n,    /* Submatrix size (the one concerned by the computation */
                               int P,    int Q,    /* process process grid */
                               int kp,   int kq,   /* k-cyclicity */
                               int ip,   int jq,   /* starting point on the process grid */
                               int mloc, int nloc){/* number of local rows and cols of the matrix */

    assert(storage == PARSEC_MATRIX_LAPACK);

    parsec_matrix_block_cyclic_init(dc,
                              mtype,
                              storage,
                              myrank,
                              mb, nb,  /* Tile size */
                              lm, ln,  /* Global matrix size (what is stored)*/
                              i,  j,   /* Staring point in the global matrix */
                              m,  n,   /* Submatrix size (the one concerned by the computation */
                              P,  Q,   /* process process grid */
                              kp, kq,  /* k-cyclicity */
                              ip, jq); /* starting point on the process grid */

    parsec_tiled_matrix_t *tdesc = &(dc->super);
    tdesc->lln = nloc;
    tdesc->llm = mloc;

    /* Generate default dtt for LAPACK storage */
    parsec_data_collection_t *o = (parsec_data_collection_t*)tdesc;
    parsec_datatype_t elem_dt = PARSEC_DATATYPE_NULL;
    ptrdiff_t extent;
    parsec_translate_matrix_type( tdesc->mtype, &elem_dt );
    parsec_type_free(&o->default_dtt);
    /* Default type is MBxNB if we have enough rows&cols */
    if( PARSEC_SUCCESS != parsec_matrix_define_datatype(&o->default_dtt, elem_dt,
                                              PARSEC_MATRIX_FULL, 1 /*diag*/,
                                              (tdesc->mb > tdesc->m ? tdesc->m : tdesc->mb),
                                              (tdesc->nb > tdesc->n ? tdesc->n : tdesc->nb),
                                              tdesc->llm/*ld*/,
                                              -1/*resized*/, &extent)){
        parsec_fatal("Unable to create a datatype for the data collection.");
    }
}

void parsec_matrix_block_cyclic_init(parsec_matrix_block_cyclic_t * dc,
                               parsec_matrix_type_t mtype,
                               parsec_matrix_storage_t storage,
                               int myrank,
                               int mb,    int nb,   /* Tile size */
                               int lm,    int ln,   /* Global matrix size (what is stored)*/
                               int i,     int j,    /* Staring point in the global matrix */
                               int m,     int n,    /* Submatrix size (the one concerned by the computation */
                               int P,     int Q,    /* process process grid */
                               int kp,    int kq,   /* k-cyclicity */
                               int ip,    int jq)   /* starting point on the process grid */
{
    int temp;
    int nodes = P*Q;
    parsec_data_collection_t *o     = &(dc->super.super);
    parsec_tiled_matrix_t *tdesc = &(dc->super);

    /* Initialize the tiled_matrix descriptor */
    parsec_tiled_matrix_init( tdesc, mtype, storage, parsec_matrix_block_cyclic_type,
                                 nodes, myrank,
                                 mb, nb, lm, ln, i, j, m, n );
    dc->mat = NULL;  /* No data associated with the matrix yet */

#if !PARSEC_KCYCLIC_WITH_VIEW
    parsec_grid_2Dcyclic_init(&dc->grid, myrank, P, Q, kp, kq, ip, jq);
#else
    parsec_grid_2Dcyclic_init(&dc->grid, myrank, P, Q, 1, 1, ip, jq);
#endif /* PARSEC_KCYCLIC_WITH_VIEW */

    if(storage == PARSEC_MATRIX_LAPACK) {
        tdesc->slm = tdesc->sln = 0;
    }

    /* Compute the number of rows handled by the local process */
    dc->nb_elem_r = 0;
    temp = dc->grid.rrank * dc->grid.krows; /* row coordinate of the first tile to handle */
    while( temp < tdesc->lmt ) {
        if(storage == PARSEC_MATRIX_LAPACK) {
            tdesc->slm += temp == 0          ? ((i % mb) == 0 ? mb : mb - i) /* first row */
                        : temp == tdesc->lmt ? ( m % mb)                     /* last row */
                        : mb; /* middle row */
        }
        if( (temp + (dc->grid.krows)) < tdesc->lmt ) {
            dc->nb_elem_r += (dc->grid.krows);
            temp += ((dc->grid.rows) * (dc->grid.krows));
            continue;
        }
        dc->nb_elem_r += ((tdesc->lmt) - temp);
        break;
    }

    /* Compute the number of columns handled by the local process */
    dc->nb_elem_c = 0;
    temp = dc->grid.crank * dc->grid.kcols;
    while( temp < tdesc->lnt ) {
        if(storage == PARSEC_MATRIX_LAPACK) {
            tdesc->sln += temp == 0          ? ((j % nb) == 0 ? nb : nb - j) /* first col */
                        : temp == tdesc->lnt ? ( n % nb)                     /* last col*/
                        : nb; /* middle col*/
        }
        if( (temp + (dc->grid.kcols)) < tdesc->lnt ) {
            dc->nb_elem_c += (dc->grid.kcols);
            temp += (dc->grid.cols) * (dc->grid.kcols);
            continue;
        }
        dc->nb_elem_c += ((tdesc->lnt) - temp);
        break;
    }

    /* If rows or cols are 0, then no elemns, set
     * both to 0.
     * */
    if(dc->nb_elem_r == 0) dc->nb_elem_c = 0;
    if(dc->nb_elem_c == 0) dc->nb_elem_r = 0;
    /* Total number of tiles stored locally */
    tdesc->nb_local_tiles = dc->nb_elem_r * dc->nb_elem_c;
    tdesc->data_map = (parsec_data_t**)calloc(tdesc->nb_local_tiles, sizeof(parsec_data_t*));

    /* Update llm and lln */
    if(storage != PARSEC_MATRIX_LAPACK) {
        tdesc->slm = tdesc->llm = dc->nb_elem_r * mb;
        tdesc->sln = tdesc->lln = dc->nb_elem_c * nb;
    }

    /* set the methods */
    if( (kp == 1) && (kq == 1) ) {
        o->rank_of      = twoDBC_rank_of;
        o->vpid_of      = twoDBC_vpid_of;
        o->data_of      = twoDBC_data_of;
        o->rank_of_key  = twoDBC_rank_of_key;
        o->vpid_of_key  = twoDBC_vpid_of_key;
        o->data_of_key  = twoDBC_data_of_key;
    } else {
#if !PARSEC_KCYCLIC_WITH_VIEW
        o->rank_of      = twoDBC_kcyclic_rank_of;
        o->vpid_of      = twoDBC_kcyclic_vpid_of;
        o->data_of      = twoDBC_kcyclic_data_of;
        o->rank_of_key  = twoDBC_kcyclic_rank_of_key;
        o->vpid_of_key  = twoDBC_kcyclic_vpid_of_key;
        o->data_of_key  = twoDBC_kcyclic_data_of_key;
#else
        parsec_matrix_block_cyclic_kview(dc, dc, kp, kq);
#endif /* PARSEC_KCYCLIC_WITH_VIEW */
    }
    o->register_memory   = twoDBC_memory_register;
    o->unregister_memory = twoDBC_memory_unregister;

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "parsec_matrix_block_cyclic_init: \n"
           "      dc = %p, mtype = %d, nodes = %u, myrank = %d, \n"
           "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
           "      mt = %d, nt = %d, lmt = %d, lnt = %d, llm = %d, lln = %d, slm = %d, sln = %d, nb_local_tile = %d \n"
           "      kp = %d, kq = %d, ip = %d, jq = %d, P = %d, Q = %d, rowrank %d, colrank %d",
           dc, tdesc->mtype, tdesc->super.nodes,
           tdesc->super.myrank,
           tdesc->mb, tdesc->nb,
           tdesc->lm, tdesc->ln,
           tdesc->i,  tdesc->j,
           tdesc->m,  tdesc->n,
           tdesc->mt, tdesc->nt,
           tdesc->lmt, tdesc->lnt,
           tdesc->llm, tdesc->lln,
           tdesc->slm, tdesc->sln,
           tdesc->nb_local_tiles,
           dc->grid.krows, dc->grid.kcols,
           dc->grid.ip, dc->grid.jq,
           dc->grid.rows, dc->grid.cols,
           dc->grid.rrank, dc->grid.crank);
}

void parsec_matrix_block_cyclic_key2coords(parsec_data_collection_t *desc,
                                                   parsec_data_key_t key,
                                                   int *m, int *n)
{
    int _m, _n;
    parsec_tiled_matrix_t * dc;

    dc = (parsec_tiled_matrix_t *)desc;

    _m = key % dc->lmt;
    _n = key / dc->lmt;
    *m = _m - dc->i / dc->mb;
    *n = _n - dc->j / dc->nb;
}

/*
 *
 * Set of functions with no k-cyclicity support
 *
 */
static uint32_t twoDBC_rank_of(parsec_data_collection_t * desc, ...)
{
    int cr, m, n;
    int rr;
    int res;
    va_list ap;
    parsec_matrix_block_cyclic_t * dc = (parsec_matrix_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Assert using local info */
    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = (m % dc->grid.rows + dc->grid.ip) % dc->grid.rows;
    cr = (n % dc->grid.cols + dc->grid.jq) % dc->grid.cols;
    res = rr * dc->grid.cols + cr;

    return res;
}

static uint32_t twoDBC_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_rank_of(desc, m, n);
}

static int32_t twoDBC_vpid_of(parsec_data_collection_t *desc, ...)
{
    int m, n, p, q, pq;
    int local_m, local_n;
    parsec_matrix_block_cyclic_t * dc;
    va_list ap;
    int32_t vpid;
    dc = (parsec_matrix_block_cyclic_t *)desc;

    /* If 1 VP, always return 0 */
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

    /* Assert using local info */
    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

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

static int32_t twoDBC_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_vpid_of(desc, m, n);
}

static inline int twoDBC_coordinates_to_position(parsec_matrix_block_cyclic_t *dc, int m, int n){
    int position, local_m, local_n;

    /* Compute the local tile row */
    local_m = m / dc->grid.rows;
    assert( (m % dc->grid.rows) == dc->grid.rrank );

    /* Compute the local column */
    local_n = n / dc->grid.cols;
    assert( (n % dc->grid.cols) == dc->grid.crank );

    assert(dc->nb_elem_r <= dc->super.lmt);
    position = dc->nb_elem_r * local_n + local_m;

    return position;
}

static parsec_data_t* twoDBC_data_of(parsec_data_collection_t *desc, ...)
{
    int m, n, position;
    size_t pos = 0;
    va_list ap;
    parsec_matrix_block_cyclic_t * dc;
    dc = (parsec_matrix_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Assert using local info */
    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    position = twoDBC_coordinates_to_position(dc, m, n);

    /* If mat allocatd, set pos to the right position for each tile */
    if( NULL != dc->mat ) {
        if( dc->super.storage == PARSEC_MATRIX_TILE ) {
            pos = position;
            pos *= (size_t)dc->super.bsiz;
        } else {
            int local_m = m / dc->grid.rows;
            int local_n = n / dc->grid.cols;
            pos = (((size_t)local_n) * ((size_t)dc->super.nb)) * ((size_t)dc->super.llm)
                +  ((size_t)local_m) * ((size_t)dc->super.mb);
        }
    }

    return parsec_tiled_matrix_create_data( &dc->super,
                                     (char*)dc->mat + pos * parsec_datadist_getsizeoftype(dc->super.mtype),
                                     position, (n * dc->super.lmt) + m );
}

static parsec_data_t* twoDBC_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_data_of(desc, m, n);
}

/****
 * Set of functions with a pseudo k-cyclic view of the distribution
 ****/

void parsec_matrix_block_cyclic_kview( parsec_matrix_block_cyclic_t* target,
                                           parsec_matrix_block_cyclic_t* origin,
                                           int kp, int kq )
{
    assert( (origin->grid.krows == 1) && (origin->grid.kcols == 1) );
    *target = *origin;
    target->grid.krows = kp;
    target->grid.kcols = kq;
    target->super.super.rank_of     = twoDBC_kview_rank_of;
    target->super.super.data_of     = twoDBC_kview_data_of;
    target->super.super.vpid_of     = twoDBC_kview_vpid_of;
    target->super.super.rank_of_key = twoDBC_kview_rank_of_key;
    target->super.super.data_of_key = twoDBC_kview_data_of_key;
    target->super.super.vpid_of_key = twoDBC_kview_vpid_of_key;
}

static inline unsigned int kview_compute_m(parsec_matrix_block_cyclic_t* desc, unsigned int m)
{
    unsigned int p, ps, mt;
    p = desc->grid.rows;
    ps = desc->grid.krows;
    mt = desc->super.mt;
    do {
        m = m-m%(p*ps) + (m%ps)*p + (m/ps)%p;
    } while(m >= mt);
    return m;
}

static inline unsigned int kview_compute_n(parsec_matrix_block_cyclic_t* desc, unsigned int n)
{
    unsigned int q, qs, nt;
    q = desc->grid.cols;
    qs = desc->grid.kcols;
    nt = desc->super.nt;
    do {
        n = n-n%(q*qs) + (n%qs)*q + (n/qs)%q;
    } while(n >= nt);
    return n;
}

static uint32_t twoDBC_kview_rank_of(parsec_data_collection_t* dc, ...)
{
    unsigned int m, n, sm, sn;
    parsec_matrix_block_cyclic_t* desc = (parsec_matrix_block_cyclic_t*)dc;
    va_list ap;
    va_start(ap, dc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);
    sm = kview_compute_m(desc, m);
    sn = kview_compute_n(desc, n);
    return twoDBC_rank_of(dc, sm, sn);
}

static uint32_t twoDBC_kview_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kview_rank_of(desc, m, n);
}

static int32_t twoDBC_kview_vpid_of(parsec_data_collection_t* dc, ...)
{
    unsigned int m, n;
    parsec_matrix_block_cyclic_t* desc = (parsec_matrix_block_cyclic_t*)dc;
    va_list ap;
    va_start(ap, dc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);
    m = kview_compute_m(desc, m);
    n = kview_compute_n(desc, n);
    return twoDBC_vpid_of(dc, m, n);
}

static int32_t twoDBC_kview_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kview_vpid_of(desc, m, n);
}

static parsec_data_t* twoDBC_kview_data_of(parsec_data_collection_t* dc, ...)
{
    unsigned int m, n;
    parsec_matrix_block_cyclic_t* desc = (parsec_matrix_block_cyclic_t*)dc;
    va_list ap;
    va_start(ap, dc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);
    m = kview_compute_m(desc, m);
    n = kview_compute_n(desc, n);
    return twoDBC_data_of(dc, m, n);
}

static parsec_data_t* twoDBC_kview_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kview_data_of(desc, m, n);
}

#if !PARSEC_KCYCLIC_WITH_VIEW
/*
 *
 * Set of functions with k-cyclicity support
 *
 */
static uint32_t twoDBC_kcyclic_rank_of(parsec_data_collection_t * desc, ...)
{
    unsigned int stc, cr, m, n;
    unsigned int str, rr;
    unsigned int res;
    va_list ap;
    parsec_matrix_block_cyclic_t * dc;
    dc = (parsec_matrix_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    /* (m,n) is in k-cyclic tile (str, stc)*/
    str = m / dc->grid.krows;
    stc = n / dc->grid.kcols;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = (str % dc->grid.rows + dc->grid.ip) % dc->grid.rows;
    cr = (stc % dc->grid.cols + dc->grid.jq) % dc->grid.cols;
    res = rr * dc->grid.cols + cr;

    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
    /*            m, n, res, rr, cr, dc->grid.rows, dc->grid.cols); */
    return res;
}

static uint32_t twoDBC_kcyclic_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kcyclic_rank_of(desc, m, n);
}

static int32_t twoDBC_kcyclic_vpid_of(parsec_data_collection_t *desc, ...)
{
    int m, n, p, q, pq;
    int local_m, local_n;
    parsec_matrix_block_cyclic_t * dc;
    va_list ap;
    int32_t vpid;
    dc = (parsec_matrix_block_cyclic_t *)desc;

    /* If no vp, always return 0 */
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

    /* Assert using local info */
#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    /* Compute the local tile row */
    local_m = ( m / (dc->grid.krows * dc->grid.rows) ) * dc->grid.krows;
    m = m % (dc->grid.krows * dc->grid.rows);
    assert( m / dc->grid.krows == dc->grid.rrank);
    local_m += m % dc->grid.krows;

    /* Compute the local column */
    local_n = ( n / (dc->grid.kcols * dc->grid.cols) ) * dc->grid.kcols;
    n = n % (dc->grid.kcols * dc->grid.cols);
    assert( n / dc->grid.kcols == dc->grid.crank);
    local_n += n % dc->grid.kcols;

    vpid = (local_n % q) * p + (local_m % p);
    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static int32_t twoDBC_kcyclic_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kcyclic_vpid_of(desc, m, n);
}

static parsec_data_t* twoDBC_kcyclic_data_of(parsec_data_collection_t *desc, ...)
{
    size_t pos = 0;
    int m, n, local_m, local_n, position;
    va_list ap;
    parsec_matrix_block_cyclic_t * dc;
    dc = (parsec_matrix_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Assert using local info */
#if defined(DISTRIBUTED)
    assert(desc->myrank == desc->rank_of(desc, m, n));
#endif

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    /* Compute the local tile row */
    local_m = ( m / (dc->grid.krows * dc->grid.rows) ) * dc->grid.krows;
    m = m % (dc->grid.krows * dc->grid.rows);
    assert( m / dc->grid.krows == dc->grid.rrank);
    local_m += m % dc->grid.krows;

    /* Compute the local column */
    local_n = ( n / (dc->grid.kcols * dc->grid.cols) ) * dc->grid.kcols;
    n = n % (dc->grid.kcols * dc->grid.cols);
    assert( n / dc->grid.kcols == dc->grid.crank);
    local_n += n % dc->grid.kcols;

    position = dc->nb_elem_r * local_n + local_m;;

    /* If mat allocatd, set pos to the right position for each tile */
    if( NULL != dc->mat ) {
        if( dc->super.storage == PARSEC_MATRIX_TILE ) {
            pos = position;
            pos *= (size_t)dc->super.bsiz;
        } else {
            pos = (((size_t)local_n) * ((size_t)dc->super.nb)) * ((size_t)dc->super.llm)
                +  ((size_t)local_m) * ((size_t)dc->super.mb);
        }
    }

    return parsec_tiled_matrix_create_data( &dc->super,
                                     (char*)dc->mat + pos * parsec_datadist_getsizeoftype(dc->super.mtype),
                                     position, (n * dc->super.lmt) + m );
}

static parsec_data_t* twoDBC_kcyclic_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return twoDBC_kcyclic_data_of(desc, m, n);
}

#endif /* PARSEC_KCYCLIC_WITH_VIEW */
