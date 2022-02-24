/**
 * Copyright (c) 2019-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "nvlink.h"
#include "nvlink_wrapper.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int size = 1;
    int rank = 0;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* DISTRIBUTED */

    parsec = parsec_init(&argc, &argv);

    tp = testing_nvlink_New(parsec, 10, 512);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif /* DISTRIBUTED */
    return 0;
}
