/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/debug.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/vector_two_dim_cyclic.h"
#include "parsec/vpmap.h"

static uint32_t vector_twoDBC_rank_of(parsec_data_collection_t* dc, ...);
static int32_t  vector_twoDBC_vpid_of(parsec_data_collection_t* dc, ...);
static parsec_data_t* vector_twoDBC_data_of(parsec_data_collection_t* dc, ...);

#if defined(PARSEC_PROF_TRACE) || defined(PARSEC_HAVE_CUDA)
static uint32_t vector_twoDBC_data_key(struct parsec_data_collection_s *desc, ...);
#endif /* defined(PARSEC_PROF_TRACE) || defined(PARSEC_HAVE_CUDA) */

#if defined(PARSEC_PROF_TRACE)
static int      vector_twoDBC_key_to_string(struct parsec_data_collection_s * desc, uint32_t datakey, char * buffer, uint32_t buffer_size);
#endif

static inline int gcd(int a, int b){
    int x, y, t;
    x = a;
    y = b;
    while( y != 0 ) {
        t = y;
        y = x%y;
        x = t;
    }
    return x;
}

static inline int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

void vector_two_dim_cyclic_init( vector_two_dim_cyclic_t * dc,
                                 enum matrix_type mtype,
                                 enum vector_distrib distrib,
                                 int nodes, int myrank,
                                 int mb,   /* Segment size                                           */
                                 int lm,   /* Global vector size (what is stored)                    */
                                 int i,    /* Staring point in the global vector                     */
                                 int m,    /* Sub-vector size (the one concerned by the computation) */
                                 int P )
{
    int Q;
    parsec_data_collection_t *o = &(dc->super.super);

    /* Initialize the tiled_matrix descriptor */
    parsec_tiled_matrix_dc_init( &(dc->super), mtype, matrix_Tile, two_dim_block_cyclic_type,
                            nodes, myrank,
                            mb, 1, lm, 1, i, 0, m, 1 );
    dc->mat = NULL;  /* No data associated with the vector yet */

    if(nodes < P) {
        parsec_warning("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);
        P = nodes;
    }
    Q = nodes / P;
    if(nodes != P*Q)
        parsec_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

    grid_2Dcyclic_init(&dc->grid, myrank, P, Q, 1, 1);

    dc->super.nb_local_tiles = 0;
    dc->distrib = distrib;

    switch ( distrib ) {
    case PlasmaVectorDiag:
    {
        int pmq   = dc->grid.crank - dc->grid.rrank;
        int gcdpq = gcd( P, Q );
        int lcmpq = lcm( P, Q );

        dc->lcm = lcmpq;

        /*
         * Compute the number of segment stored locally
         * Segments are owned only if pmq is part of the corpse gcdZ.
         */
        if ( pmq % gcdpq == 0 ) {
            int drank = pmq;

            dc->super.nb_local_tiles = dc->super.lmt / lcmpq;

            /* Compute rank on the diagonal */
            while ( drank % Q != 0 ) {
                drank += Q;
            }
            drank = drank + dc->grid.rrank;

            if ( drank < (dc->super.lmt % lcmpq) )
                (dc->super.nb_local_tiles)++;
        }
    }
    break;

    case PlasmaVectorRow:
    {
        dc->lcm = Q;

        if ( dc->grid.rrank == 0 ) {
            dc->super.nb_local_tiles = dc->super.lmt / Q;

            if ( dc->grid.rrank < (dc->super.lmt % Q) )
                (dc->super.nb_local_tiles)++;
        }
    }
    break;

    case PlasmaVectorCol:
    default:
        dc->lcm = P;

        if ( dc->grid.crank == 0 ) {
            dc->super.nb_local_tiles = dc->super.lmt / P;

            if ( dc->grid.crank < (dc->super.lmt % P) )
                (dc->super.nb_local_tiles)++;
        }
    }

    /* Update llm and lln */
    dc->super.llm = dc->super.nb_local_tiles * mb;
    dc->super.lln = 1;

    /* set the methods */
    o->rank_of = vector_twoDBC_rank_of;
    o->vpid_of = vector_twoDBC_vpid_of;
    o->data_of = vector_twoDBC_data_of;

#if defined(PARSEC_PROF_TRACE) || defined(PARSEC_HAVE_CUDA)
    o->data_key      = vector_twoDBC_data_key;
#endif
#if defined(PARSEC_PROF_TRACE)
    o->key_to_string = vector_twoDBC_key_to_string;
    o->key_dim       = NULL;
    o->key           = NULL;
    asprintf(&(o->key_dim), "(%d)", dc->super.lmt);
#endif
    dc->super.data_map = (parsec_data_t**)calloc(dc->super.nb_local_tiles, sizeof(parsec_data_t*));

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "vector_two_dim_cyclic_init: \n"
            "      dc = %p, mtype = %d, nodes = %u, myrank = %d, \n"
            "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
            "      nrst = %d, ncst = %d, P = %d, Q = %d",
            dc, dc->super.mtype, dc->super.super.nodes, dc->super.super.myrank,
            dc->super.mb, dc->super.nb,
            dc->super.lm, dc->super.ln,
            dc->super.i,  dc->super.j,
            dc->super.m,  dc->super.n,
            dc->grid.strows, dc->grid.stcols,
            P, Q);
}


/*
 *
 * Set of functions with no super-tiles
 *
 */
static uint32_t vector_twoDBC_rank_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m;
    unsigned int rr = 0;
    unsigned int cr = 0;
    unsigned int res;
    va_list ap;
    vector_two_dim_cyclic_t * dc;
    dc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;

    /* P(rr, cr) has the tile, compute the rank*/
    if ( dc->distrib != PlasmaVectorCol )
        rr = m % dc->grid.rows;

    if ( dc->distrib != PlasmaVectorRow )
        cr = m % dc->grid.cols;

    res = rr * dc->grid.cols + cr;

    return res;
}

static int32_t vector_twoDBC_vpid_of(parsec_data_collection_t *desc, ...)
{
    int m, p, q, pq;
    int local_m = 0;
    int local_n = 0;
    vector_two_dim_cyclic_t * dc;
    va_list ap;
    int32_t vpid;
    dc = (vector_two_dim_cyclic_t *)desc;

    /* If 1 VP, always return 0 */
    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    p = dc->grid.vp_p;
    q = dc->grid.vp_q;
    assert(p*q == pq);

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    if ( dc->distrib != PlasmaVectorCol )
        local_m = (m / dc->grid.rows) % p;

    if ( dc->distrib != PlasmaVectorRow )
        local_n = (m / dc->grid.cols) % q;

    vpid = local_m * q + local_n;

    assert( vpid < vpmap_get_nb_vp() );
    return vpid;
}

static parsec_data_t* vector_twoDBC_data_of(parsec_data_collection_t *desc, ...)
{
    int m;
    size_t pos;
    int local_m;
    va_list ap;
    vector_two_dim_cyclic_t * dc;
    dc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == vector_twoDBC_rank_of(desc, m));
#endif

    /* Compute the local tile row */
    assert( dc->super.bsiz == dc->super.mb );

    local_m = m / dc->lcm;
    pos = local_m * dc->super.mb;

    pos *= parsec_datadist_getsizeoftype(dc->super.mtype);
    return parsec_matrix_create_data(&dc->super,
                                    (char*)dc->mat + pos,
                                    local_m, m);
}

/*
 * Common functions
 */
#if defined(PARSEC_PROF_TRACE) || defined(PARSEC_HAVE_CUDA)
/* return a unique key (unique only for the specified parsec_dc) associated to a data */
static uint32_t vector_twoDBC_data_key(struct parsec_data_collection_s *desc, ...)
{
    unsigned int m;
    vector_two_dim_cyclic_t * dc;
    va_list ap;
    dc = (vector_two_dim_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;

    return m;
}
#endif /* defined(PARSEC_PROF_TRACE) || defined(PARSEC_HAVE_CUDA) */

#if defined(PARSEC_PROF_TRACE)
/* return a string meaningful for profiling about data */
static int  vector_twoDBC_key_to_string(struct parsec_data_collection_s * desc, uint32_t datakey, char * buffer, uint32_t buffer_size)
{
    int res;
    (void)desc;

    res = snprintf(buffer, buffer_size, "(%u)", datakey);
    if (res < 0)
    {
        printf("error in key_to_string for tile (%u) key: %u\n", datakey, datakey);
    }
    return res;
}
#endif /* PARSEC_PROF_TRACE */
