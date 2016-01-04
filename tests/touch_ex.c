/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "dague/data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"

#define BLOCK 10
#define N     100

extern int touch_finalize(void);
extern dague_handle_t* touch_initialize(int block, int n);

int main( int argc, char** argv )
{
    dague_context_t* dague;
    dague_handle_t* handle;

#ifdef HAVE_MPI
    MPI_Init(NULL, NULL);
#endif
    dague = dague_init(1, &argc, &argv);
    assert( NULL != dague );

    handle = touch_initialize(BLOCK, N);

    dague_enqueue( dague, handle );

    dague_context_wait(dague);

    dague_fini( &dague);

    touch_finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
