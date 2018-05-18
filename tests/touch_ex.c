/*
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define BLOCK 10
#define N     100

extern int touch_finalize(void);
extern parsec_taskpool_t* touch_initialize(int block, int n);

int main( int argc, char** argv )
{
    parsec_context_t* parsec;
    parsec_taskpool_t* tp;
    int i = 1, rc, verbose;

#ifdef PARSEC_HAVE_MPI
    MPI_Init(NULL, NULL);
#endif

    while( NULL != argv[i] ) {
        if( 0 == strncmp(argv[i], "-v=", 3) ) {
            verbose = strtol(argv[i]+3, NULL, 10);
            goto move_and_continue;
        }
        i++;  /* skip this one */
        continue;
    move_and_continue:
        memmove(&argv[i], &argv[i+1], (argc - 1) * sizeof(char*));
        argc -= 1;
    }

    parsec = parsec_init(1, &argc, &argv);
    if( NULL == parsec ) {
        exit(-2);
    }
    tp = touch_initialize(BLOCK, N);

    rc = parsec_context_add_taskpool( parsec, tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_fini( &parsec);

    touch_finalize();
    if( verbose >= 5 ) {
    }

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
