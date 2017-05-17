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
task_to_check_dont_track(parsec_execution_unit_t *context, parsec_task_t *this_task)
{
    (void)context; (void)this_task;
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
    tiled_matrix_desc_t *ddescA;

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
    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new(  );

    int i, total_tasks = 20;
    nb = 1; /* size of each tile */
    nt = 1; /* total tiles */

    ddescA = create_and_distribute_data(rank, world, nb, nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    parsec_dtd_ddesc_init(A);

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
    parsec_enqueue( parsec, parsec_dtd_handle );

    parsec_context_start(parsec);

    for( i = 0; i < total_tasks; i++ ) {
        /* This task does not have any data associated with it, so it will be inserted in all mpi processes */
        parsec_insert_task( parsec_dtd_handle, task_to_check_dont_track,    0,  "sample_task",
                            PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | DONT_TRACK | AFFINITY,
                            0 );
    }

    parsec_dtd_data_flush_all( parsec_dtd_handle, A );

    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

    parsec_context_wait(parsec);

    if( 0 == rank ) {
        parsec_output( 0, "Test passed if we do not see 0-%d printed sequentially in order\n\n", total_tasks-1 );
    }

    parsec_dtd_ddesc_fini( A );
    free_data(ddescA);

    parsec_handle_free( parsec_dtd_handle );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
