/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
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
task_to_check_generation(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1, rc;
    int parsec_argc;
    char** parsec_argv;

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

    parsec_argv = &argv[1];
    parsec_argc = argc - 1;
    if(argv[1] != NULL) {
        cores = atoi(argv[1]);
        parsec_argv++;
        parsec_argc--;
    }

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init( cores, &parsec_argc, &parsec_argv );
    if( NULL == parsec ) {
        exit(-1);
    }

    /****** Checking task generation ******/
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    int i, j, total_tasks = 100000;

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking task generation using dtd interface. "
                       "We insert 10000 tasks and atomically increase a global counter to see if %d task executed\n\n", total_tasks );
    }

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( i = 0; i < 6; i++ ) {
        SYNC_TIME_START();
        for( j = 0; j < total_tasks; j++ ) {
            /* This task does not have any data associated with it, so it will be inserted in all mpi processes */
            parsec_dtd_insert_task(dtd_tp, task_to_check_generation, 0, PARSEC_DEV_CPU, "sample_task",
                                   PARSEC_DTD_ARG_END);
        }

        rc = parsec_taskpool_wait( dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
        SYNC_TIME_PRINT(rank, ("\n"));
    }

    parsec_taskpool_free( dtd_tp );

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
