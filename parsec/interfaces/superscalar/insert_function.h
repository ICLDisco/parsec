/**
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 **/
/**
 *
 * @file insert_function.h
 *
 * @version 2.0.0
 *
 **/

#ifndef INSERT_FUNCTION_H_HAS_BEEN_INCLUDED
#define INSERT_FUNCTION_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"

BEGIN_C_DECLS

/**
 * To see examples please look at testing_zpotrf_dtd.c, testing_zgeqrf_dtd.c,
 * testing_zgetrf_incpiv_dtd.c files in the directory "root_of_PaRSEC/dplasma/testing/".
 * Very simple example of inserting just one task can be found in
 * "root_of_PaRSEC/example/interfaces/superscalar/"
 **/

/*
 * The following is a definition of the flags, for usage please check usage of parsec_dtd_taskpool_insert_task() below.
 *
 *   **  Details of Flags **
 *
 *  INPUT:          Data is used in read-only mode, no modification is done.
 *  OUTPUT:         Data is used in write-only, written only, not read.
 *  INOUT:          Data is read and written both.
 *  ATOMIC_WRITE:   Data is used like OUTPUT but the ordering of the tasks having this flag is not maintained
 *                  by the scheduler.
 *                  It is the responsibility of the user to make sure data is written atomically.
 *                  Treated like INPUT by the scheduler.
 *                  This flag is not currently VALID, please refrain from using it.
 *  SCRATCH:        Will be used by the task as scratch pad, does not effect the DAG, tells the runtime
 *                  to allocate memory specified by the user.
 *                  This flag can also be used to pass pointer of any variable. Please look at the usage below.
 *  VALUE:          Tells the runtime to copy the value as a parameter of the task.
 *
 */
#define GET_OP_TYPE 0xf00000
typedef enum { INPUT=0x100000,
               OUTPUT=0x200000,
               INOUT=0x300000,
               ATOMIC_WRITE=0x400000, /* DO NOT USE ,Iterate_successors do not support this at this point */
               SCRATCH=0x500000,
               VALUE=0x600000,
               REF=0x700000
             } parsec_dtd_op_type;

/*
 * The following is a definition of the flags, for usage please check usage of parsec_dtd_taskpool_insert_task() below.
 *
 *   **  Details of Flags **
 *
 *  AFFINITY:       Indicates where to place a task. This flag should be provided with a data and the
 *                  runtime will place the task in the rank where the data, this flag was provided with,
 *                  resides.
 *
 *  DONT_TRACK:     This flag indicates to the runtime to not track any dependency associated with the
 *                  data this flag was provided with.
 *
 */

#define GET_OTHER_FLAG_INFO 0xf0000
typedef enum { AFFINITY=1<<16, /* Data affinity */
               DONT_TRACK=1<<17, /* Drop dependency tracking */
             } parsed_dtd_other_flag_type;

/*
 * Describes different regions to express more specific dependency.
 * All regions are mutually exclusive.
 */
#define GET_REGION_INFO 0xffff
#define PARSEC_DTD_ARG_END -111

/*
 * Array of arenas to hold the data region shape and other information.
 * Currently only 16 types of different regions are supported at a time.
 */
extern parsec_arena_t **parsec_dtd_arenas;

/*
 * Users can use this two variables to control the sliding window of task insertion.
 * This is set using a default number or the number set by the mca_param.
 * The command line to set the value of window size and threshold size are:
 * "-- --mca parsec_dtd_window_size 4000 --mca parsec_dtd_threshold_size 2000"
 * This will set the window size to be 4000 tasks. This means the main thread
 * will insert 4000 tasks and then retire from it and join the workers.
 * The parsec_dtd_threshold_size indicates the number of tasks, reaching which
 * the main thread will resume inserting tasks again.
 * The threshold should always be smaller than the window size.
 */
extern int parsec_dtd_window_size;
extern int parsec_dtd_threshold_size;

#define PASSED_BY_REF                1
#define MAX_FLOW                    25 /* Max number of flows allowed per task */
#define PARSEC_DTD_NB_TASK_CLASSES  25 /* Max number of task classes allowed */

typedef struct parsec_dtd_tile_s         parsec_dtd_tile_t;
typedef struct parsec_dtd_task_s         parsec_dtd_task_t;
typedef struct parsec_dtd_taskpool_s     parsec_dtd_taskpool_t;

/*
 * Function pointer typeof  kernel pointer pased as parameter to insert_function().
 * This is the prototype of the function in which the actual operations of each task
 * is implemented by the User. The actual computation will be performed in functions
 * having this prototype.
 * 1. parsec_execution_stream_t *
 * 2. parsec_task_t * -> this gives access to the actual task the User inserted
 *                                    using this interface.
 * This function should return one of the following:
 *  PARSEC_HOOK_RETURN_DONE    : This execution succeeded
 *  PARSEC_HOOK_RETURN_AGAIN   : Reschedule later
 *  PARSEC_HOOK_RETURN_NEXT    : Try next variant [if any]
 *  PARSEC_HOOK_RETURN_DISABLE : Disable the device, something went wrong
 *  PARSEC_HOOK_RETURN_ASYNC   : The task is outside our reach, the completion will
 *                               be triggered asynchronously.
 *  PARSEC_HOOK_RETURN_ERROR   : Some other major error happened
 *
 */
typedef int (parsec_dtd_funcptr_t)(parsec_execution_stream_t *, parsec_task_t *);

/*
 * This function is used to retrieve the parameters passed during insertion of a task.
 * This function takes variadic parameters.
 * 1. parsec_task_t * -> The parameter list is attached with this structure.
 *                                     The User needs to pass a FLAG to specify what sort of value needs to be
 *                                     unpacked. Three types of FLAGS are supported:
 *                                     - UNPACK_VALUE
 *                                     - UNPACK_SCRATCH
 *                                     - UNPACK_DATA
 *                                     Following each FLAG the pointer to the memory location where the paramter
 *                                     will be copied needs to be given.
 *
 * There is no way to unpack individual parameters. e.g. If user wants to unpack the 3rd parameter only, they have to
 * unpack at least the first three to maintain the order in which whey were inserted. However user can unpack
 * a partial amount of parameters. To do that correctly user needs to pass 0 as the last parameter.
 *
 *  ******* THE ORDER IN WHICH THE PARAMETERS WERE PASSED DURING INSERTION NEEDS TO BE *******
 *                              STRICTLY FOLLOWED WHILE UNPACKING
 */
void
parsec_dtd_unpack_args( parsec_task_t *this_task, ... );

/*
 * The following macro is very specific to two dimensional matrix.
 * The parameters to pass to get pointer to data
 * 1. parsec_data_collection_t *
 * 2. m (coordinates of the data in the matrix)
 * 3. n (coordinates of the data in the matrix)
 */
#define TILE_OF(DC, I, J) \
    parsec_dtd_tile_of(&(__dc##DC->super.super), (&(__dc##DC->super.super))->data_key(&(__dc##DC->super.super), I, J))

/*
 * This macro is for any type of data. The user needs to provide the
 * data-descriptor and the key. The dc and the key will allow us
 * to uniquely identify the data a task is supposed to use.
 */
#define TILE_OF_KEY(DC, KEY) \
    parsec_dtd_tile_of(DC, KEY)

parsec_dtd_tile_t *
parsec_dtd_tile_of( parsec_data_collection_t *dc, parsec_data_key_t key );

/*
 * Using this function users can insert task in PaRSEC
 * 1. The parsec taskpool (parsec_dtd_taskpool_t *)
 * 2. The function pointer which will be executed as the "real computation task" being inserted.
 *    This function should include the actual computation the user wants to perform on the data.
 * 3. The priority of the task, if not sure user should provide 0.
 * 4. The name of the task.
 * 5. Variadic type paramter. User can pass any number of paramters. The runtime will pack the
 *    parameters and attach them to the task they belong to. User can later use unpakcing
 *    fuction provided to get access to the parametrs.
 *    Each paramter to pass to a task should be expressed in the form of a triplet. e.g
 *
 *    1.      sizeof(int),             &uplo,                           VALUE,
 *
 *         (size in bytes),    (pointer to the variable),        (VALUE will result in the value
 *                                                                of the variable "uplo" to be copied)
 *
 *    2.  sizeof(double) * 100,         NULL,                          SCRATCH,
 *
 *         (size in bytes),    (memory of specified size         (runtime will allocate memory
 *                              will be allocated, no pointer     requested by the first of the trio)
 *                              needs to be passed),
 *
 *                                              /
 *
 *         sizeof(double *),           &pointer_to_double,             SCRATCH,
 *
 *          (size in bytes),    (the pointer of the pointer       (runtime will copy the address of
 *                               vairable we want to pass to the   the pointer which the task can later
 *                               task. This pointer will be        retrieve)
 *                               copied),
 *
 *
 *    3.    PASSED_BY_REF,         TILE_OF(dc, i, j),               INOUT/INPUT/OUTPUT,
 *                                         /                                    /
 *                                 TILE_OF_KEY(dc, key),            INOUT | REGION_INFO,
 *                                                                              /
 *                                                                     INOUT | AFFINITY/DONT_TRACK,
 *                                                                              /
 *                                                                     INOUT | REGION_INFO | AFFINITY/DONT_TRACK,
 *
 *          (To specify we        (We call tile_of with            (First shows the essential flag
 *           are passing only      data-descriptor and either       INPUT/INOUT/OUTPUT to indicate the type
 *           reference of data),   a key or indices in a 2D         of operation the task will be performing
 *                                 matrix),                         on the data. The other flags are combined
 *                                                                  with this flag. REGION_INFO states the index
 *                                                                  of parsec_dtd_arenas array this data belongs
 *                                                                  to. AFFINITY flag is a must in distributed
 *                                                                  environemnt. This is the only way for the runtime
 *                                                                  to place a task in the process grid. It must be
 *                                                                  provided with only one data indicating to place
 *                                                                  the task in the rank the data resides. If
 *                                                                  this flag is provided with multiple data, the
 *                                                                  task is placed in the rank where the last data
 *                                                                  in order is situated)
 *
 *      *******  THIS PARAMETER MUST BE PROVIDED *******
 *      4. "0" indicates the end of paramter list. This should always be the last parameter.
 *
 */
void
parsec_dtd_taskpool_insert_task(parsec_taskpool_t  *tp,
                                parsec_dtd_funcptr_t *fpointer, int priority,
                                const char *name_of_kernel, ...);

/*
 * This function behaves exactly like parsec_dtd_taskpool_insert_task()
 * except it does not insert the task in PaRSEC and just returns it.
 * Users will need to use parsec_insert_dtd_task() to insert the task
 */
parsec_task_t *
parsec_dtd_taskpool_create_task(parsec_taskpool_t  *tp,
                                parsec_dtd_funcptr_t *fpointer, int priority,
                                const char *name_of_kernel, ...);

/* This function allows users to insert a properly formed DTD task in
 * PaRSEC.
 */
void
parsec_insert_dtd_task(parsec_task_t *this_task);

/*
 * This macros should be called anytime users
 * are using data in their parsec-dtd runs.
 * This functions intializes/cleans necessary
 * structures in a data-descriptor(dc). The
 * init should be called after a valid dc
 * has been acquired, and the fini before
 * the dc is cleaned.
 */
#define DTD_DC_INIT(DC) \
    parsec_dtd_data_collection_init(&(__dc##DC->super.super))
void
parsec_dtd_data_collection_init( parsec_data_collection_t *dc );

#define DTD_DC_FINI(DC) \
    parsec_dtd_data_collection_fini(&(__dc##DC->super.super))
void
parsec_dtd_data_collection_fini( parsec_data_collection_t *dc );

/*
 * This function create and returns a PaRSEC DTD taskpool. The
 * taskpool is not associated with any context, contains no
 * tasks, and has no callback associated with.
 */
parsec_taskpool_t*
parsec_dtd_taskpool_new();

/*
 * This function will block until all the tasks inserted
 * so far is completed.
 * User can call this function multiple times
 * between a parsec_dtd_taskpool_new() and parsec_taskpool_free()
 * Takes a parsec context and a parsec taskpool as input.
 */
int
parsec_dtd_taskpool_wait( parsec_context_t *parsec,
                          parsec_taskpool_t  *tp );

/*
 * This function flushes a specific data,
 * it indicates to the engine that this data
 * will no longer be used by any further tasks.
 * This indication optimizes the reuse of memory
 * related to that piece of data. This also
 * results in transmission of the last version of
 * data from the rank that last edited it
 * to the rank that owns it. So we end up with the
 * same data distribution as we started with.
 * The tile of a data can be acqiured using the
 * TILE_OF or TILE_OF_KEY macro.
 * To ensure consistent and correct behavior, user
 * must wait on the taskpool before inserting
 * new task using this data after the flush.
 */
void
parsec_dtd_data_flush( parsec_taskpool_t   *tp,
                       parsec_dtd_tile_t *tile );

/*
 * This function flushes all the data of a dc(data collection).
 * This function must be called for all dc(s) before
 * parsec_context_wait() is called.
 */
void
parsec_dtd_data_flush_all( parsec_taskpool_t *tp,
                           parsec_data_collection_t  *dc );

/*
 * This function returns the taskpool a task bekongs to.
 */
parsec_taskpool_t *
parsec_dtd_get_taskpool(parsec_task_t *this_task);

END_C_DECLS

#endif  /* INSERT_FUNCTION_H_HAS_BEEN_INCLUDED */
