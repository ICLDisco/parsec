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

static uint32_t vector_twoDBC_rank_of(dague_ddesc_t* ddesc, ...);
static int32_t  vector_twoDBC_vpid_of(dague_ddesc_t* ddesc, ...);
static void*    vector_twoDBC_data_of(dague_ddesc_t* ddesc, ...);

#if defined(DAGUE_PROF_TRACE)
static uint32_t vector_twoDBC_data_key(struct dague_ddesc *desc, ...);
static int      vector_twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size);
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

void vector_two_dim_cyclic_init( vector_two_dim_cyclic_t * Ddesc,
                                 enum matrix_type mtype,
                                 enum vector_distrib distrib,
                                 int nodes, int cores, int myrank,
                                 int mb,   /* Segment size                                           */
                                 int lm,   /* Global vector size (what is stored)                    */
                                 int i,    /* Staring point in the global vector                     */
                                 int m,    /* Sub-vector size (the one concerned by the computation) */
                                 int P )
{
    int Q;
    dague_ddesc_t *o = &(Ddesc->super.super);

    /* Initialize the tiled_matrix descriptor */
    tiled_matrix_desc_init( &(Ddesc->super), mtype, matrix_Tile, two_dim_block_cyclic_type,
                            nodes, cores, myrank,
                            mb, 1, lm, 1, i, 0, m, 1 );

    if(nodes < P)
        ERROR(("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d\n", nodes, P));
    Q = nodes / P;
    if(nodes != P*Q)
        WARNING(("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d\n", nodes, P, Q));

    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, 1, 1);

    Ddesc->super.nb_local_tiles = 0;
    Ddesc->distrib = distrib;

    switch ( distrib ) {
    case PlasmaVectorDiag:
    {
        int pmq   = Ddesc->grid.crank - Ddesc->grid.rrank;
        int gcdpq = gcd( P, Q );
        int lcmpq = lcm( P, Q );

        Ddesc->lcm = lcmpq;

        /*
         * Compute the number of segment stored locally
         * Segments are owned only if pmq is part of the corpse gcdZ.
         */
        if ( pmq % gcdpq == 0 ) {
            int drank = pmq;

            Ddesc->super.nb_local_tiles = Ddesc->super.lmt / lcmpq;

            /* Compute rank on the diagonal */
            while ( drank % Q != 0 ) {
                drank += Q;
            }
            drank = drank + Ddesc->grid.rrank;

            if ( drank < (Ddesc->super.lmt % lcmpq) )
                (Ddesc->super.nb_local_tiles)++;
        }
    }
    break;

    case PlasmaVectorRow:
    {
        Ddesc->lcm = Q;

        if ( Ddesc->grid.rrank == 0 ) {
            Ddesc->super.nb_local_tiles = Ddesc->super.lmt / Q;

            if ( Ddesc->grid.rrank < (Ddesc->super.lmt % Q) )
                (Ddesc->super.nb_local_tiles)++;
        }
    }
    break;

    case PlasmaVectorCol:
    default:
        Ddesc->lcm = P;

        if ( Ddesc->grid.crank == 0 ) {
            Ddesc->super.nb_local_tiles = Ddesc->super.lmt / P;

            if ( Ddesc->grid.crank < (Ddesc->super.lmt % P) )
                (Ddesc->super.nb_local_tiles)++;
        }
    }

    /* Update llm and lln */
    Ddesc->super.llm = Ddesc->super.nb_local_tiles * mb;
    Ddesc->super.lln = 1;

    /* set the methods */
    o->rank_of = vector_twoDBC_rank_of;
    o->vpid_of = vector_twoDBC_vpid_of;
    o->data_of = vector_twoDBC_data_of;

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
    unsigned int rr = 0;
    unsigned int cr = 0;
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
    if ( Ddesc->distrib != PlasmaVectorCol )
        rr = m % Ddesc->grid.rows;

    if ( Ddesc->distrib != PlasmaVectorRow )
        cr = m % Ddesc->grid.cols;

    res = rr * Ddesc->grid.cols + cr;

    return res;
}

static int32_t vector_twoDBC_vpid_of(dague_ddesc_t *desc, ...)
{
    int m, p, q, pq;
    int local_m = 0;
    int local_n = 0;
    vector_two_dim_cyclic_t * Ddesc;
    va_list ap;
    int32_t vpid;
    Ddesc = (vector_two_dim_cyclic_t *)desc;

    /* If 1 VP, always return 0 */
    pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    p = Ddesc->grid.vp_p;
    q = Ddesc->grid.vp_q;
    assert(p*q == pq);

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
    if ( Ddesc->distrib != PlasmaVectorCol )
        local_m = (m / Ddesc->grid.rows) % p;

    if ( Ddesc->distrib != PlasmaVectorRow )
        local_n = (m / Ddesc->grid.cols) % q;

    vpid = local_m * q + local_n;

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
    assert( Ddesc->super.bsiz == Ddesc->super.mb );

    local_m = m / Ddesc->lcm;
    pos = local_m * Ddesc->super.mb;

    pos *= dague_datadist_getsizeoftype(Ddesc->super.mtype);
    return &(((char *) Ddesc->mat)[pos]);
}

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
