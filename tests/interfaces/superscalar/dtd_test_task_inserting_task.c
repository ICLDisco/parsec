/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "common_timing.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed;
double sync_time_elapsed;

int
real_task( parsec_execution_stream_t *es,
           parsec_task_t *this_task )
{
    (void)es; (void)this_task;

    parsec_output( 0, "I am %d and I am executing a real task\n", es->th_id );
    usleep(10);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_insert_task( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    (void)es;

    parsec_taskpool_t* dtd_tp = this_task->taskpool;
    int total, increment, *count, i;

    parsec_dtd_unpack_args(this_task, &total, &count, &increment);

    parsec_output( 0, "Task inserting task by thread: %d count: %d Total: %d increment: %d total_inserted: %d\n",
                   es->th_id, *count, total, increment, *count-1 );

    for( i = 0; *count < total; i++, *count += 1 ) {
        if( i > increment ) {
            /* Return some kind of rescheduling signal */
            return PARSEC_HOOK_RETURN_AGAIN;
        }
        /* Inserting real task */
        parsec_dtd_taskpool_insert_task( dtd_tp, real_task,    0,  "Real_Task",
                           PARSEC_DTD_ARG_END );
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1, rc;

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

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool(parsec, (parsec_taskpool_t *)dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    SYNC_TIME_START();

    int total_tasks = 20;
    int increment   = 5;
    int count       = 0;

    for( m = 0; m < no_of_tasks; m++ ) {
        parsec_dtd_taskpool_insert_task( dtd_tp, task_to_insert_task,    0,  "Task_inserting_Task",
                           sizeof(int),      &total_tasks,        VALUE,
                           sizeof(int),      &count,              REF,
                           sizeof(int),      &increment,          VALUE,
                           PARSEC_DTD_ARG_END );
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_dtd_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    SYNC_TIME_PRINT(rank, ("\n"));

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
