/**
 * Copyright (c) 2015-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file dtd_example_hello_world_untied.c
 *
 * @version 4.0
 * @email parsec-users@icl.utk.edu
 *
 */

/* parsec headers */
#include "parsec.h"

/* we need the DTD internals to get access to the context members. */
#include "parsec/interfaces/dtd/insert_function_internal.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

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
    int i, n, how_many = 5;
    (void)es; (void)this_task;
    parsec_taskpool_t *dtd_tp = this_task->taskpool;

    parsec_dtd_unpack_args( this_task, &n);
    printf("I am inserting %d tasks to print \"Hello World\", and my rank is: %d\n",
           how_many, this_task->taskpool->context->my_rank);

    for( i = this_task->locals[0].value;
         (i < n) && i < (this_task->locals[0].value + how_many); i++ ) {
        parsec_dtd_insert_task(dtd_tp, task_hello_world,
                               0, PARSEC_DEV_CPU, "Hello_World_task",
                               sizeof(int), &i, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }
    this_task->locals[0].value = i;
    printf("Up to %d tasks out of %d tasks generated\n",
           this_task->locals[0].value, n);
    return (i < n) ? PARSEC_HOOK_RETURN_AGAIN : PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rc, rank, world, cores = 1, number_of_tasks = 10;

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
    if( NULL == parsec ) {
        printf("Cannot initialize PaRSEC\n");
        exit(-1);
    }

    /* Initializing parsec handle(collection of tasks) */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    /* Starting the parsec_context */
    rc = parsec_context_start( parsec );
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    /* Inserting task to insert task that
     * will print Hello World and the
     * rank of the process
     */
    parsec_dtd_insert_task(dtd_tp, task_to_insert_task_hello_world,
                           0, PARSEC_DEV_CPU, "Task_inserting_task",
                           sizeof(int), &number_of_tasks, PARSEC_VALUE,
                           PARSEC_DTD_ARG_END);

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    /* Waiting on the context */
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    /* Cleaning the parsec handle */
    parsec_taskpool_free( dtd_tp );

    /* Cleaning up parsec context */
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
