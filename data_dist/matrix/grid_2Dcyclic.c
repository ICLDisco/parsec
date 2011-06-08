/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "dague.h"
#include "grid_2Dcyclic.h"

void grid_2Dcyclic_init(grid_2Dcyclic_t *grid, int myrank, int P, int Q, int nrst, int ncst)
{
    // Filling matrix description woth user parameter
    grid->rank = myrank ;
    grid->rows = P;
    grid->cols = Q;
    grid->strows = nrst;
    grid->stcols = ncst;

    /* computing colRANK and rowRANK */
    grid->rrank = myrank/Q;
    grid->crank = myrank%Q;
}

