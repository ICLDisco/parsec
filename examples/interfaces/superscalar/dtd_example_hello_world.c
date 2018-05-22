/**
 * Copyright (c) 2015-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file dtd_example_hello_world.c
 *
 * @version 2.0.0
 *
 */

/* parsec headers */
#include "parsec.h"

/* we need the DTD internals to get access to the context members. */
#include "parsec/interfaces/superscalar/insert_function_internal.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */


/* Task that prints "Hello World" */
int
task_hello_world( parsec_execution_stream_t *es,
                  parsec_task_t *this_task )
{
    (void)es; (void)this_task;

    printf("Hello World my rank is: %d\n", this_task->taskpool->context->my_rank);

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 1;

    /* Initializing MPI */
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

    /* Initializing parsec context */
    parsec = parsec_init( cores, &argc, &argv );

    /* Initializing parsec handle(collection of tasks) */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Registering the dtd_handle with PARSEC context */
    parsec_context_add_taskpool( parsec, dtd_tp );
    /* Starting the parsec_context */
    parsec_context_start( parsec );

    /* Inserting task to print Hello World
     * and the rank of the process
     */
    parsec_dtd_taskpool_insert_task(dtd_tp, task_hello_world,
                                    0,   "Hello_World_task",
                                    PARSEC_DTD_ARG_END);

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    /* Waiting on the context */
    parsec_context_wait(parsec);

    /* Cleaning the parsec handle */
    parsec_taskpool_free( dtd_tp );

    /* Cleaning up parsec context */
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
