/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"
#include "parsec/vpmap.h"
#include "parsec/utils/debug.h"

#include <math.h>
#include <assert.h>

static
int default_vp_data_dist();

void parsec_grid_2Dcyclic_init(parsec_grid_2Dcyclic_t *grid, int myrank, int P, int Q, int kp, int kq, int ip, int jq)
{
    /* Filling matrix description with user parameter */
    grid->rank = myrank ;
    grid->rows = P;
    grid->cols = Q;
    grid->krows = kp;
    grid->kcols = kq;

    grid->ip = ip;
    grid->jq = jq;

    /* computing colRANK and rowRANK */
    grid->rrank = ((myrank / Q) + (grid->rows - grid->ip)) % grid->rows;
    grid->crank = ((myrank % Q) + (grid->cols - grid->jq)) % grid->cols;

    grid->rloc = 0;
    grid->cloc = 0;

    /* VPMAP data distribution */
    /* TODO:: Users should be able to define it through parameters */
    grid->vp_q = default_vp_data_dist();
    grid->vp_p = vpmap_get_nb_vp()/default_vp_data_dist();
}

static
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
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,  "Default data distribution between VP defined by \"pxq\" %ix%i", p, q );
    return q;
}


