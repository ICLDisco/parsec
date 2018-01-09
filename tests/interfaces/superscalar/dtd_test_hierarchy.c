#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"

#include "common_timing.h"
#include "common_data.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

/* This testing shows graph pruning as well as hierarchical execution.
 * The only restriction is the parsec_taskpool_wait() before parsec_context_wait()
 */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

int count = 0;

enum regions {
               TILE_FULL,
             };

int
test_task( parsec_execution_stream_t *es,
           parsec_task_t *this_task )
{
    (void)es;

    int amount_of_work;
    parsec_dtd_unpack_args( this_task, &amount_of_work);
    int i, j, bla;
    for( i = 0; i < amount_of_work; i++ ) {
        //for( j = 0; j < *amount_of_work; j++ ) {
        for( j = 0; j < 2; j++ ) {
            bla = j*2;
            bla = j + 20;
            bla = j*2+i+j+i*i;
        }
    }
    count++;
    (void)bla;
    return PARSEC_HOOK_RETURN_DONE;
}

int
test_task_generator( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    (void)es;

    parsec_tiled_matrix_dc_t *dcB, *tmp;
    int amount = 0, nb, nt;
    int rank = es->virtual_process->parsec_context->my_rank;
    int world = es->virtual_process->parsec_context->nb_nodes, i;

    parsec_dtd_unpack_args( this_task, &nb, &nt, &tmp);

    dcB = create_and_distribute_empty_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcB, "B");
    parsec_data_collection_t *B = (parsec_data_collection_t *)dcB;
    parsec_dtd_data_collection_init(B);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( es->virtual_process->parsec_context, dtd_tp );

    for( i = 0; i < 100; i++ ) {
        parsec_dtd_taskpool_insert_task(dtd_tp, test_task,    0,  "Test_Task",
                                        sizeof(int),       &amount,    VALUE,
                                        PASSED_BY_REF,     TILE_OF_KEY(B, rank),      INOUT | AFFINITY,
                                        PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush(dtd_tp, TILE_OF_KEY(B, rank));

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( es->virtual_process->parsec_context, dtd_tp );

    parsec_dtd_data_collection_fini(B);
    free_data(dcB);

    parsec_taskpool_free( dtd_tp );

    count++;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 8, rc;

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

    int m;
    int nb, nt;
    parsec_tiled_matrix_dc_t *dcA;
    parsec_taskpool_t *dtd_tp;

    parsec = parsec_init( cores, &argc, &argv );

    dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_enqueue( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
    rc = parsec_context_start( parsec );
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    nb = 1; /* size of each tile */
    nt = world; /* total tiles */

    dcA = create_and_distribute_empty_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                           nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                           MPI_INT);
#endif

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    SYNC_TIME_START();

    for( m = 0; m < nt; m++ ) {
        parsec_dtd_taskpool_insert_task(dtd_tp, test_task_generator,    0,  "Test_Task_generator",
                                        sizeof(int),       &nb,                 VALUE,
                                        sizeof(int),       &nt,                 VALUE,
                                        PASSED_BY_REF,     TILE_OF_KEY(A, m),   INOUT | AFFINITY,
                                        PARSEC_DTD_ARG_END);

        parsec_dtd_data_flush(dtd_tp, TILE_OF_KEY(A, m));
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    parsec_output( 0, "Successfully executed %d tasks in rank %d\n", count, parsec->my_rank );

    SYNC_TIME_PRINT(rank, ("\n") );

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
