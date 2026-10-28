#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "reallocate_tile_example.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* The generated internal taskpool class derives from this public taskpool type. */
PARSEC_OBJ_CLASS_INSTANCE(parsec_reallocate_tile_example_taskpool_t, parsec_taskpool_t,
                          NULL, NULL);

int main(int argc, char **argv)
{
    parsec_context_t *parsec = NULL;
    parsec_reallocate_tile_example_taskpool_t *tp = NULL;
    parsec_matrix_block_cyclic_t dcA;
    int rank = 0;
    int ret = 0;
    const int tile_size = 32; /* 32x32 = 1024 doubles in one tile */

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    parsec = parsec_init(1, &argc, &argv);
    if( NULL == parsec ) {
        return EXIT_FAILURE;
    }

    /* This test validates a CUDA BODY path and should be skipped without GPU. */
    if( 0 == parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA) ) {
        parsec_warning("reallocate_tile_example requires at least one CUDA device");
        printf("TEST SKIPPED\n");
        parsec_fini(&parsec);
#if defined(DISTRIBUTED)
        MPI_Finalize();
#endif
        return EXIT_SUCCESS;
    }

    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                    rank,
                                    tile_size, tile_size,
                                    tile_size, tile_size,
                                    0, 0,
                                    tile_size, tile_size,
                                    1, 1, 1, 1, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    assert(NULL != dcA.mat);
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcA, "dcA_reallocate");

    tp = parsec_reallocate_tile_example_new((parsec_tiled_matrix_t *)&dcA, &ret);
    assert(NULL != tp);

    parsec_context_add_taskpool(parsec, &tp->super);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);
    parsec_taskpool_free(&tp->super);

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcA);
    parsec_fini(&parsec);

#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif

    if( 0 == ret ) {
        printf("reallocate_tile_example: TEST PASSED\n");
        return EXIT_SUCCESS;
    }

    printf("reallocate_tile_example: TEST FAILED (%d)\n", ret);
    return EXIT_FAILURE;
}
