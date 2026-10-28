/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

static int nb_errors = 0;

int
task_to_check_dont_track(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *data;
    int value;

    parsec_dtd_unpack_args( this_task, &data, &value );

    /* The scheduler does not order the tasks carrying this flag, so they may
     * reach the counter concurrently and the body has to protect it. */
    printf("%d -> %d\n", parsec_atomic_fetch_inc_int32(data), value);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_complete_dont_track(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *data;
    int nb_tasks;

    parsec_dtd_unpack_args( this_task, &data, &nb_tasks );

    if( nb_tasks != *data ) {
        parsec_warning( "This task was supposed to become active only once all %d "
                        "ATOMIC_WRITE tasks have been executed, but the counter is %d",
                        nb_tasks, *data );
        nb_errors++;
    }
    *data = 0;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, nt, rc;
    parsec_tiled_matrix_t *dcA;

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

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init( cores, &argc, &argv );

    /****** Checking Don't track flag ******/
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
    int32_t *real_data;
    int key;
    key = A->data_key(A, 0, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = 0;

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking PARSEC_ATOMIC_WRITE flag. "
                       "We insert %d unordered tasks that each increase a counter, followed by a "
                       "tracked task that must not run before all of them have\n\n", total_tasks );
    }

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( i = 0; i < total_tasks; i++ ) {
        parsec_dtd_insert_task(dtd_tp, task_to_check_dont_track, 0, PARSEC_DEV_CPU, "parallel ATOMIC_WRITE tasks",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE | PARSEC_AFFINITY,
                               sizeof(int), &i, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }

    /* A regular tracked task closing the unordered group: it reads the counter
     * and must therefore wait for every task above. */
    parsec_dtd_insert_task(dtd_tp, task_to_complete_dont_track, 0, PARSEC_DEV_CPU, "successor of all ATOMIC_WRITE tasks",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | PARSEC_AFFINITY,
                           sizeof(int), &total_tasks, PARSEC_VALUE,
                           PARSEC_DTD_ARG_END);

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    if( nb_errors > 0 )
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
