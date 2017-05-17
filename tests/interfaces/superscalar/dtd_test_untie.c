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

#include "common_timing.h"
#include "common_data.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

int count = 0;

enum regions {
               TILE_FULL,
             };

int
test_task( parsec_execution_unit_t    *context,
           parsec_task_t *this_task )
{
    (void)context;

    int *amount_of_work;
    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE,  &amount_of_work
                          );
    int i, j, bla;
    for( i = 0; i < *amount_of_work; i++ ) {
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
test_task_generator( parsec_execution_unit_t    *context,
                     parsec_task_t *this_task )
{
    (void)context;

    tiled_matrix_desc_t *ddescA;
    parsec_handle_t *parsec_dtd_handle = this_task->parsec_handle;
    int *total, *step, *iteration, *n;
    int *amount_of_work;
    int i;

    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE,  &n,
                           UNPACK_VALUE,  &amount_of_work,
                           UNPACK_VALUE,  &total,
                           UNPACK_VALUE,  &step,
                           UNPACK_VALUE,  &iteration,
                           UNPACK_SCRATCH, &ddescA
                          );

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    for( i = 0; *iteration < *total; *iteration += 1, i++ ) {
        if( i > *step ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        } else {
            parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                                sizeof(int),      amount_of_work,    VALUE,
                                PASSED_BY_REF,    TILE_OF_KEY(A, *n), INOUT  ,
                                0 );
        }
    }
    parsec_dtd_data_flush(parsec_dtd_handle, TILE_OF_KEY(A, *n));

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 20;

    if(argc > 1) {
        cores = atoi(argv[1]);
        if( 0 >= cores )
            cores = 1;  /* fix it to a sane number */
    }

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

    int m, n;
    int no_of_chain;
    int nb, nt;
    tiled_matrix_desc_t *ddescA;
    int amount_of_work[3] = {1000, 10000, 100000};
    parsec_handle_t *parsec_dtd_handle;

    no_of_chain = cores;
    int tasks_in_each_chain[3] = {1000, 10000, 100000};

    parsec = parsec_init( cores, &argc, &argv );

    parsec_dtd_handle = parsec_dtd_handle_new(  );

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );
    parsec_context_start( parsec );

    nb = 1; /* size of each tile */
    nt = no_of_chain; /* total tiles */

    ddescA = create_and_distribute_data(rank, world, nb, nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    parsec_dtd_ddesc_init(A);
    int i;
    int work_index = 0;

    for( i = 0; i < 3; i++ ) {

        SYNC_TIME_START();
        for( n = 0; n < no_of_chain; n++ ) {
            for( m = 0; m < tasks_in_each_chain[i]; m++ ) {
                parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                                    sizeof(int),      &amount_of_work[work_index],    VALUE,
                                    PASSED_BY_REF,    TILE_OF_KEY(A, n), INOUT  ,
                                    0 );
            }
            parsec_dtd_data_flush(parsec_dtd_handle, TILE_OF_KEY(A, n));
        }
        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        SYNC_TIME_PRINT(rank, ("No of chains : %d, No of tasks in each chain: %d,  Amount of work: %d\n", no_of_chain, tasks_in_each_chain[i], amount_of_work[work_index]));
    }

    count = 0;

    for( i = 0; i < 3; i++ ) {

        SYNC_TIME_START();
        int step = dtd_window_size, iteration = 0;

        for( n = 0; n < no_of_chain; n++ ) {
            parsec_insert_task( parsec_dtd_handle, test_task_generator,    0,  "Test_Task_Generator",
                                sizeof(int),      &n,                     VALUE,
                                sizeof(int),      &amount_of_work[work_index],     VALUE,
                                sizeof(int),      &tasks_in_each_chain[i],   VALUE,
                                sizeof(int),      &step,                  VALUE,
                                sizeof(int),      &iteration,             VALUE,
                                sizeof(tiled_matrix_desc_t*),    ddescA,  SCRATCH,
                                0 );

        }

        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        SYNC_TIME_PRINT(rank, ("No of chains : %d, No of tasks in each chain: %d,  Amount of work: %d\n", no_of_chain, tasks_in_each_chain[i], amount_of_work[work_index]));
    }

    parsec_context_wait(parsec);

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_ddesc_fini( A );
    free_data(ddescA);

    parsec_handle_free( parsec_dtd_handle );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
