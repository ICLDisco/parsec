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
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed;
double sync_time_elapsed;

int
real_task( parsec_execution_unit_t    *context,
           parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;

    parsec_output( 0, "I am %d and I am executing a real task\n", context->th_id );
    usleep(10);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_insert_task( parsec_execution_unit_t    *context,
                     parsec_execution_context_t *this_task )
{
    (void)context;

    parsec_handle_t* parsec_dtd_handle = this_task->parsec_handle;
    int *total, *increment, *count, i;

    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE, &total,
                           UNPACK_VALUE, &count,
                           UNPACK_VALUE, &increment
                         );

    parsec_output( 0, "Task inserting task by thread: %d count: %d Total: %d increment: %d total_inserted: %d\n", context->th_id, *count, *total, *increment, *count-1 );

    for( i = 0; *count < *total; i++, *count += 1 ) {
        if( i > *increment ) {
            /* Return some kind of rescheduling signal */
            return PARSEC_HOOK_RETURN_AGAIN;
        }
        /* Inserting real task */
        parsec_insert_task( parsec_dtd_handle, real_task,    0,  "Real_Task",
                           0 );
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 8;
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

    int m;
    int no_of_tasks = 1;

    parsec = parsec_init( cores, &argc, &argv );

    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new(  );

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue(parsec, (parsec_handle_t *)parsec_dtd_handle);

    parsec_context_start(parsec);

    SYNC_TIME_START();

    int total_tasks = 20;
    int increment   = 5;
    int count       = 0;

    for( m = 0; m < no_of_tasks; m++ ) {
        parsec_insert_task( parsec_dtd_handle, task_to_insert_task,    0,  "Task_inserting_Task",
                           sizeof(int),      &total_tasks,        VALUE,
                           sizeof(int),      &count,              VALUE,
                           sizeof(int),      &increment,          VALUE,
                           0 );
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

    SYNC_TIME_PRINT(rank, ("\n"));

    parsec_context_wait(parsec);

    parsec_handle_free( parsec_dtd_handle );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
