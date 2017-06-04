/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec_config.h"
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
    int rc;
    parsec_context_t* parsec;
    parsec_handle_t* handle;

#ifdef PARSEC_HAVE_MPI
    MPI_Init(NULL, NULL);
#endif
    parsec = parsec_init(1, &argc, &argv);
    if( NULL != parsec ) {
        handle = touch_initialize(BLOCK, N);

        rc = parsec_enqueue( parsec, handle );
        PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        parsec_fini( &parsec);

        touch_finalize();
    }
#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
