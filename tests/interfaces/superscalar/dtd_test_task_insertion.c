#include "parsec_config.h"

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

int
test_task( parsec_execution_unit_t    *context,
           parsec_execution_context_t *this_task )
{
    (void)context;

    int *amount_of_work;
    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE,  &amount_of_work
                          );

    (void)parsec_atomic_inc_32b(&count);

    int i, j;
    for( i = 0; i < *amount_of_work; i++ ) {
        for( j = 0; j < *amount_of_work/2; j++ ) {
            i = j*2;
            j = j + 20;
            i = j*2;
        }
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int
test_task_generator( parsec_execution_unit_t    *context,
                     parsec_execution_context_t *this_task )
{
    (void)context;

    parsec_handle_t *parsec_dtd_handle = this_task->parsec_handle;
    int *total, *step, *iteration, *amount_of_work;
    int i;

    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE,  &amount_of_work,
                           UNPACK_VALUE,  &total,
                           UNPACK_VALUE,  &step,
                           UNPACK_VALUE,  &iteration
                          );

    for( i = 0; *iteration < *total; *iteration += 1, i++ ) {
        if( i > *step ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        } else {
            parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                               sizeof(int),      amount_of_work,    VALUE,
                               0 );

        }
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 20;

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
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
    int no_of_tasks = 500000;
    int amount_of_work[3] = {100, 1000, 10000};
    parsec_handle_t *parsec_dtd_handle;

    parsec = parsec_init( cores, &argc, &argv );

    parsec_dtd_handle = parsec_dtd_handle_new(  );

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );
    parsec_context_start( parsec );

    if( rank == 0 ) {
        parsec_output( 0, "In all the tests we insert tasks "
                         "that does varying amount of work. The operation is constant, we vary "
                         "the number of times we do the operation\n\n" );
    }

    int tmp_window_size, tmp_threshold_size;
    tmp_window_size    = dtd_window_size;
    tmp_threshold_size = dtd_threshold_size;

    dtd_window_size    = no_of_tasks;
    dtd_threshold_size = no_of_tasks;


/****** Inserting tasks using main thread while others execute ******/
    if( rank == 0 ) {
        parsec_output( 0, "\nWe now insert %d tasks using the main thread while the others %d cores "
                         "executes them simultaneously, main thread joins after all tasks are inserted "
                         "\n\n", no_of_tasks, cores-1 );
    }

    for( n = 0; n < 3; n++ ) {
        count = 0;

        TIME_START();

        for( m = 0; m < no_of_tasks; m++ ) {
            parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                               sizeof(int),      &amount_of_work[n], VALUE,
                               0 );
        }

        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        TIME_PRINT(rank, ("Tasks executed : %d : Amount of work: %d\n", count, amount_of_work[n]));
    }
/****** END ******/

    count = 0;
    dtd_window_size    = tmp_window_size;
    dtd_threshold_size = tmp_threshold_size;

/****** Inserting tasks using main thread while others execute ******/
    if( rank == 0 ) {
        parsec_output( 0, "\nWe now insert %d tasks using the main thread while the other %d cores "
                         "executes them simultaneously, the main thread also joins the others to "
                         "execute following a sliding window\n\n", no_of_tasks, cores-1 );
    }

    for( n = 0; n < 3; n++ ) {
        count = 0;

        TIME_START();

        for( m = 0; m < no_of_tasks; m++ ) {
            parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                               sizeof(int),      &amount_of_work[n], VALUE,
                               0 );
        }

        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        TIME_PRINT(rank, ("Tasks executed : %d : Amount of work: %d\n", count, amount_of_work[n]));
    }
/****** END ******/


/****** All threads insert and all threads execute ******/
    if( rank == 0 ) {
        parsec_output( 0, "\nWe now insert %d tasks using all threads and is also executed "
                         "by the all of them\n\n", no_of_tasks, cores-1 );
    }

    for( n = 0; n < 3; n++ ) {
        count = 0;

        TIME_START();

        int step = dtd_window_size, iteration = 0;
        parsec_insert_task( parsec_dtd_handle, test_task_generator,    0,  "Test_Task",
                           sizeof(int),      &amount_of_work[n],     VALUE,
                           sizeof(int),      &no_of_tasks,           VALUE,
                           sizeof(int),      &step,                  VALUE,
                           sizeof(int),      &iteration,             VALUE,
                           0 );

        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        TIME_PRINT(rank, ("Tasks executed : %d : Amount of work: %d\n", count, amount_of_work[n]));

    }
/****** END ******/


    parsec_context_wait(parsec);

    parsec_handle_free( parsec_dtd_handle );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
