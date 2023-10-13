/*
 * Copyright (c) 2013-2023 The University of Tennessee and The University
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
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &rc);
#endif

    int pargc = 0; char **pargv = NULL;
    for( i = 1; i < argc; i++) {
        if( 0 == strncmp(argv[i], "--", 3) ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
        if( 0 == strncmp(argv[i], "-v=", 3) ) {
            verbose = strtol(argv[i]+3, NULL, 10);
            continue;
        }
    }

    parsec = parsec_init(1, &pargc, &pargv);
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

    parsec_taskpool_free(tp);

    parsec_fini( &parsec);

    touch_finalize();
    if( verbose >= 5 ) {
    }

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
