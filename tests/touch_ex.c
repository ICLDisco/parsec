/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"

#define BLOCK 10
#define N     100

extern int touch_finalize(void);
extern parsec_handle_t* touch_initialize(int block, int n);

int main( int argc, char** argv )
{
    parsec_context_t* parsec;
    parsec_handle_t* handle;

#ifdef PARSEC_HAVE_MPI
    MPI_Init(NULL, NULL);
#endif
    parsec = parsec_init(1, &argc, &argv);
    if( NULL != parsec ) {
        handle = touch_initialize(BLOCK, N);

        parsec_enqueue( parsec, handle );

        parsec_context_wait(parsec);

        parsec_fini( &parsec);

        touch_finalize();
    }
#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
