#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"

#include "common_data.h"
#include "common_timing.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

enum regions {
               TILE_FULL,
             };

int
reduce0( parsec_execution_stream_t    *es,
         parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
reduce1( parsec_execution_stream_t    *es,
         parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data,
                          UNPACK_DATA,  &second_data);

    *second_data += *data;

    return PARSEC_HOOK_RETURN_DONE;
}

int
bcast0( parsec_execution_stream_t    *es,
        parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
bcast1( parsec_execution_stream_t    *es,
        parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data,
                          UNPACK_DATA,  &second_data);

    printf( "My rank: %d, data: %d\n", this_task->taskpool->context->my_rank, *data );

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rc;
    int rank, world, cores;
    int nb, nt;
    parsec_tiled_matrix_dc_t *dcA;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    nb = 1; /* tile_size */
    nt = world; /* total no. of tiles */
    cores = 8;

    parsec = parsec_init( cores, &argc, &argv );

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;
    int root = 0, i;

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_enqueue( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    root = 0;

//Reduce:
// *********************
    key = A->data_key(A, rank, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = rank;

    for( i = 0; i < world; i ++ ) {
        if( root != i ) {
            parsec_dtd_taskpool_insert_task( dtd_tp, reduce0,    0,  "reduce0",
                                PASSED_BY_REF,    TILE_OF_KEY(A, i), INOUT | TILE_FULL | AFFINITY,
                                0 );

            parsec_dtd_taskpool_insert_task( dtd_tp, reduce1,    0,  "reduce1",
                                PASSED_BY_REF,    TILE_OF_KEY(A, i),    INOUT | TILE_FULL,
                                PASSED_BY_REF,    TILE_OF_KEY(A, root), INOUT | TILE_FULL | AFFINITY,
                                0 );
        }
    }

    parsec_dtd_taskpool_wait( parsec, dtd_tp );
// *********************

//Broadcast:
// *********************
    if( rank == root) {
        printf("Root: %d\n\n", root );
    }

    parsec_dtd_taskpool_insert_task( dtd_tp, bcast0,    0,  "bcast0",
                        PASSED_BY_REF,    TILE_OF_KEY(A, root), INOUT | TILE_FULL | AFFINITY,
                        0 );

    if( rank == root ) {
        for( i = 0; i < world; i++ ) {
            if( i != root ) {
                parsec_dtd_taskpool_insert_task( dtd_tp, bcast1,    0,  "bcast1",
                                    PASSED_BY_REF,    TILE_OF_KEY(A, root),  INPUT | TILE_FULL,
                                    PASSED_BY_REF,    TILE_OF_KEY(A, i),     INOUT | TILE_FULL | AFFINITY,
                                    0 );
            }
        }

    } else {
        parsec_dtd_taskpool_insert_task( dtd_tp, bcast1,    0,  "bcast1",
                            PASSED_BY_REF,    TILE_OF_KEY(A, root),    INPUT | TILE_FULL,
                            PASSED_BY_REF,    TILE_OF_KEY(A, rank), INOUT | TILE_FULL | AFFINITY,
                            0 );
    }
//******************

    parsec_dtd_data_flush_all( dtd_tp, A );

    parsec_dtd_taskpool_wait( parsec, dtd_tp );
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_taskpool_free( dtd_tp );

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
