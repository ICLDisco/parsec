/*
 * Copyright (c) 2019-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/utils/mca_param.h"

#include "stress.h"
#include "stress_wrapper.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int size = 1;
    int rank = 0, nb_gpus = 1;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    {
        MPI_Comm local_comm;
        int local_rank, local_size;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);
        int gpu_mask = 0;
        for (int i = 0; i < nb_gpus; i++)
        {
            gpu_mask |= ((1 << local_rank) << i);
        }
        char *value;
        asprintf(&value, "%d", gpu_mask);
        parsec_setenv_mca_param("device_cuda_mask", value, &environ);
        free(value);
        value = NULL;
    }
#endif /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)*/
#endif /* DISTRIBUTED */

    parsec = parsec_init(-1, &argc, &argv);

    tp = testing_stress_New(parsec, 80, 1024);
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
