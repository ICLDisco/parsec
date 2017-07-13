#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"
#ifdef PARSEC_VTRACE
#include "parsec/vt_user.h"
#endif

#include "common_data.h"
#include "common_timing.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "dplasma/testing/common_timing.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

int
task_to_check_dont_track(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    int *data;

    parsec_dtd_unpack_args( this_task,
                            UNPACK_DATA,  &data
                          );

    printf("%d\n", *data);
    *data += 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
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

    if( world != 1 ) {
        parsec_fatal( "Nope! world is not right, we need exactly one MPI process. "
                      "Try with \"mpirun -np 1 .....\"\n" );
    }

    cores = 8;
    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init( cores, &argc, &argv );

    /****** Checking Dont track flag ******/
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    int i, total_tasks = 20;
    nb = 1; /* size of each tile */
    nt = 1; /* total tiles */

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;
    key = A->data_key(A, 0, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = 0;


    if( 0 == rank ) {
        parsec_output( 0, "\nChecking DONT_TRACK flag. "
                      "We insert %d tasks and increase a counter to see if %d task executed sequentially or not\n\n", total_tasks, total_tasks );
    }

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, dtd_tp );

    parsec_context_start(parsec);

    for( i = 0; i < total_tasks; i++ ) {
        /* This task does not have any data associated with it, so it will be inserted in all mpi processes */
        parsec_dtd_taskpool_insert_task( dtd_tp, task_to_check_dont_track,    0,  "sample_task",
                            PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | DONT_TRACK | AFFINITY,
                            0 );
    }

    parsec_dtd_data_flush_all( dtd_tp, A );

    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    parsec_context_wait(parsec);

    if( 0 == rank ) {
        parsec_output( 0, "Test passed if we do not see 0-%d printed sequentially in order\n\n", total_tasks-1 );
    }

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
