/**
 * Copyright (c) 2015-2017 The University of Tennessee and The University
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

#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec headers */
#include "parsec.h"
#include "parsec/profiling.h"
#ifdef PARSEC_VTRACE
#include "parsec/vt_user.h"
#endif

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
task_hello_world( parsec_execution_unit_t    *context,
                  parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;

    printf("Hello World my rank is: %d\n", this_task->parsec_handle->context->my_rank);

    return PARSEC_HOOK_RETURN_DONE;
}

/* Task that inserts task to print "Hello World" */
int
task_to_insert_task_hello_world( parsec_execution_unit_t    *context,
                                 parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;
    parsec_handle_t *parsec_dtd_handle = this_task->parsec_handle;

    printf("I am inserting task to print \"Hello World\", and my rank is: %d\n", this_task->parsec_handle->context->my_rank);

    parsec_insert_task( parsec_dtd_handle, task_hello_world,    0,  "Hello_World_task", 0 );

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
    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new(  );

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );
    /* Starting the parsec_context */
    parsec_context_start( parsec );

    /* Inserting task to insert task that
     * will print Hello World and the
     * rank of the process
     */
    parsec_insert_task( parsec_dtd_handle, task_to_insert_task_hello_world,    0,  "Task_inserting_task", 0 );

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

    /* Waiting on the context */
    parsec_context_wait(parsec);

    /* Cleaning the parsec handle */
    parsec_handle_free( parsec_dtd_handle );

    /* Cleaning up parsec context */
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
