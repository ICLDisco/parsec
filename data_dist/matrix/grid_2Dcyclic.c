/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dague_config.h"
#include "dague.h"
#include "grid_2Dcyclic.h"
#include "dague/vpmap.h"
#include "dague/debug.h"

#include <math.h>
#include <assert.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

int default_vp_data_dist();

void grid_2Dcyclic_init(grid_2Dcyclic_t *grid, int myrank, int P, int Q, int nrst, int ncst)
{
    /* Filling matrix description woth user parameter */
    grid->rank = myrank ;
    grid->rows = P;
    grid->cols = Q;
    grid->strows = nrst;
    grid->stcols = ncst;

    /* computing colRANK and rowRANK */
    grid->rrank = myrank / Q;
    grid->crank = myrank % Q;

    grid->rloc = 0;
    grid->cloc = 0;

    /* VPMAP data distribution */
    /* TODO:: Users should be able to define it through parameters */
    grid->vp_q = default_vp_data_dist();
    grid->vp_p = vpmap_get_nb_vp()/default_vp_data_dist();
}


int default_vp_data_dist()
{
    int p, q, pq;
    /* default: q >= p, worst case p=1 */
    pq = vpmap_get_nb_vp();
    q = (int)ceilf(sqrtf( (float)pq));

    assert(q > 0);
    p = pq / q;

    /* if the VP number is not a square, find p and q to use all of them */
    while ( p*q != pq) {
        q++;
        p = pq / q;
    }
    DAGUE_DEBUG_VERBOSE(20, dague_debug_output,  "Default data distribution between VP defined by \"pxq\" %ix%i", p, q );
    return q;
}


