/**
 * Copyright (c) 2015-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file dtd_example_hello_world_untied.c
 *
 * @version 2.0.0
 *
 */

/* parsec headers */
#include "parsec.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

/*
 * By this example we are trying to show the capability of the dtd interface to
 * recursively insert task from another task. We esseentially untie which thread
 * inserts task using this recursive task insertion.
 */

/* Task that prints "Hello World" */
int
task_hello_world( parsec_execution_stream_t *es,
                  parsec_task_t *this_task )
{
    int i;
    (void)es; (void)this_task;

    parsec_dtd_unpack_args( this_task, &i);
    printf("Hello World %d/%d\n",
           this_task->taskpool->context->my_rank, i);

    return PARSEC_HOOK_RETURN_DONE;
}

/* Task that inserts task to print "Hello World" */
int
task_to_insert_task_hello_world( parsec_execution_stream_t *es,
                                 parsec_task_t *this_task )
{
    int n;
    (void)es; (void)this_task;
    parsec_taskpool_t *dtd_tp = this_task->taskpool;

    parsec_dtd_unpack_args( this_task, &n);
    printf("I am inserting task to print \"Hello World\", and my rank is: %d\n", this_task->taskpool->context->my_rank);

    for( int i = 0; i < n; i++ ) {
        parsec_dtd_taskpool_insert_task(dtd_tp, task_hello_world,
                                        0,  "Hello_World_task",
                                        sizeof(int), &i, VALUE,
                                        PARSEC_DTD_ARG_END);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 1, number_of_tasks = 10;

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

    /* Inserting task to insert task that
     * will print Hello World and the
     * rank of the process
     */
    parsec_dtd_taskpool_insert_task(dtd_tp, task_to_insert_task_hello_world,
                                    0,  "Task_inserting_task",
                                    sizeof(int), &number_of_tasks, VALUE,
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
