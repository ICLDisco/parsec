/**
 * Copyright (c) 2015-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 **/
/**
 *
 * @file insert_function.h
 *
 **/

#ifndef PARSEC_INSERT_FUNCTION_H_HAS_BEEN_INCLUDED
#define PARSEC_INSERT_FUNCTION_H_HAS_BEEN_INCLUDED

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"

BEGIN_C_DECLS

/**
 * @addtogroup DTD_INTERFACE
 *  @{
 */

/**
 * To see examples please look at testing_zpotrf_dtd.c, testing_zgeqrf_dtd.c,
 * testing_zgetrf_incpiv_dtd.c files in the directory "root_of_DPLASMA/tests/testing/".
 * Very simple example of inserting just one task can be found in
 * "root_of_PaRSEC/example/interfaces/dtd/"
 **/

/**
 * The following is a definition of the flags, for usage please check usage of parsec_dtd_insert_task() below.
 *
 *   **  Details of Flags **
 *
 *  PARSEC_INPUT:        Data is used in read-only mode, no modification is done.
 *  PARSEC_OUTPUT:       Data is used in write-only, written only, not read.
 *  PARSEC_INOUT:        Data is read and written both.
 *  PARSEC_ATOMIC_WRITE: Data is used like OUTPUT but the ordering of the tasks having this flag is not maintained
 *                       by the scheduler.
 *                       It is the responsibility of the user to make sure data is written atomically.
 *                       Treated like PARSEC_INPUT by the scheduler.
 *                       This flag is not currently VALID, please refrain from using it.
 *  PARSEC_SCRATCH:      Will be used by the task as scratch pad, does not effect the DAG, tells the runtime
 *                       to allocate memory specified by the user.
 *                       This flag can also be used to pass pointer of any variable. Please look at the usage below.
 *  PARSEC_VALUE:        Tells the runtime to copy the value as a parameter of the task.
 *  PARSEC_REF:          Tells the runtime to reference the user pointer (i.e., not a PaRSEC data)
 *
 *  PARSEC_AFFINITY:     Indicates where to place a task. This flag should be provided with a data and the
 *                       runtime will place the task in the rank where the data, this flag was provided with,
 *                       resides.
 *  PARSEC_DONT_TRACK:   This flag indicates to the runtime to not track any dependency associated with the
 *                       data this flag was provided to.
 *
 *  Lower 16 bits:       Index (arbitrary value) for different REGIONS to express more specific dependency.
 *                       Regions indices are user provided and must be mutually exclusive for the tile.
 */
typedef enum { PARSEC_INPUT =      0x100000,
               PARSEC_OUTPUT =     0x200000,
               PARSEC_INOUT =      0x300000,
               PARSEC_ATOMIC_WRITE=0x400000, /* DO NOT USE ,Iterate_successors do not support this at this point */
               PARSEC_SCRATCH =    0x500000,
               PARSEC_VALUE =      0x600000,
               PARSEC_REF =        0x700000,
               PARSEC_GET_OP_TYPE =0xf00000, /* MASK: not an actual value, used to filter the relevant enum values */
               PARSEC_AFFINITY    =1<<16, /* Data affinity */
               PARSEC_DONT_TRACK  =1<<17, /* Drop dependency tracking */
               PARSEC_PUSHOUT     =1<<18, /* Push GPU data back to CPU */
               PARSEC_PULLIN      =1<<19, /* Pull data on the CPU */
               PARSEC_GET_OTHER_FLAG_INFO=0xf0000, /* MASK: not an actual value, used to filter the relevant enum values */
               PARSEC_GET_REGION_INFO=0xffff /* MASK: not an actual value, used to filter the relevant enum values */
             } parsec_dtd_op_t;
typedef enum { PASSED_BY_REF=-2,
               PARSEC_DTD_ARG_END=-1,
               PARSEC_DTD_EMPTY_FLAG=0
             } parsec_dtd_size_t;

typedef struct {
    parsec_dtd_op_t op;
    long            size;
} parsec_dtd_param_t;

typedef struct parsec_dtd_task_class_s  parsec_dtd_task_class_t;

/**
 * @brief Create a new Arena Datatype for DTD
 * @details Create a new unique Arena Datatype identifier,
 *    @id, and a new Arena Datatype; Associates the new Arena
 *    Datatype with the context @p ctx and the unique id @p id.
 *    This function is thread-safe per context, but not
 *    thread-safe if called in parallel on the same context.
 *
 * @param ctx the context in which the arena datatype exists.
 * @param id Ignored as input. As output: the unique ID assigned
 *    to this new Arena Datatype.
 * @return the new Arena Datatype, or NULL if there was an error
 *    (e.g. no more memory or too many arena datatypes created
 *    within this context.) id is untouched if NULL is returned.
 */
parsec_arena_datatype_t *parsec_dtd_create_arena_datatype(parsec_context_t *ctx, int *id);

/**
 * @brief returns the Arena Datatype associated with this identifier
 *    in this context
 * @details This function is thread-safe as long as the Arena Datatype
 *    associated with this @p id in this @p ctx is not removed in parallel.
 * @param ctx the context in which the Arena Datatype exists
 * @param id the unique identifier of the Arena Datatype
 * @return the existing Arena Datatype, or NULL if there was an
 *    error.
 */
parsec_arena_datatype_t *parsec_dtd_get_arena_datatype(parsec_context_t *ctx, int id);

/**
 * @brief release the Arena Datatype and its association to the
 *    unique identifier
 * @details This function is only valid if @p id is the unique
 *    identifier associated to an Arena Datatype in @p ctx. This
 *    function releases the Arena Datatype and its association with the
 *    identifier.
 * @param ctx the context in which the Arena Datatype exists
 * @param id the unique identifier of the Arena Datatype to release
 * @return PARSEC_SUCCESS in case of success, or an error code
 *    otherwise.
 */
int parsec_dtd_destroy_arena_datatype(parsec_context_t *ctx, int id);

/**
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


typedef struct parsec_dtd_tile_s         parsec_dtd_tile_t;
typedef struct parsec_dtd_task_s         parsec_dtd_task_t;
typedef struct parsec_dtd_taskpool_s     parsec_dtd_taskpool_t;

/**
 * Hard limit on the number of parameters a task inserted with DTD
 * can take
 */
#define PARSEC_DTD_MAX_PARAMS 64

/**
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
typedef parsec_hook_return_t (parsec_dtd_funcptr_t)(parsec_execution_stream_t *, parsec_task_t *);

/**
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

/**
 * The following macro is very specific to two dimensional matrix.
 * The parameters to pass to get pointer to data
 * 1. parsec_data_collection_t *
 * 2. m (coordinates of the data in the matrix)
 * 3. n (coordinates of the data in the matrix)
 */
#define PARSEC_DTD_TILE_OF(DC, I, J) \
    parsec_dtd_tile_of(&(__dc##DC->super.super), (&(__dc##DC->super.super))->data_key(&(__dc##DC->super.super), I, J))

/**
 * This macro is for any type of data. The user needs to provide the
 * data-descriptor and the key. The dc and the key will allow us
 * to uniquely identify the data a task is supposed to use.
 */
#define PARSEC_DTD_TILE_OF_KEY(DC, KEY) \
    parsec_dtd_tile_of(DC, KEY)

/**
 * Returns the tile belonging to data collection dc from its key
 */
parsec_dtd_tile_t *
parsec_dtd_tile_of( parsec_data_collection_t *dc, parsec_data_key_t key );

/**
 * Using this function users can insert task in PaRSEC
 * 1. The parsec taskpool (parsec_dtd_taskpool_t *)
 * 2. The function pointer which will be executed as the "real computation task" being inserted.
 *    This function should include the actual computation the user wants to perform on the data.
 * 3. The priority of the task, if not sure user should provide 0.
 * 4. The name of the task.
 * 5. Variadic type parameter. User can pass any number of paramters. The runtime will pack the
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
 *    3.    PASSED_BY_REF,         PARSEC_DTD_TILE_OF(dc, i, j),       PARSEC_INOUT/PARSEC_INPUT/PARSEC_OUTPUT,  /
 *                                 PARSEC_DTD_TILE_OF_KEY(dc, key),    PARSEC_INOUT | REGION_INFO, /
 *                                                                     PARSEC_INOUT | PARSEC_AFFINITY/PARSEC_DONT_TRACK, /
 *                                                                     PARSEC_INOUT | PARSEC_REGION_INFO | PARSEC_AFFINITY/PARSEC_DONT_TRACK,
 *
 *          (To specify we        (We call tile_of with            (First shows the essential flag
 *           are passing only      data-descriptor and either       PARSEC_INPUT/PARSEC_INOUT/PARSEC_OUTPUT to indicate the type
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
parsec_dtd_insert_task(parsec_taskpool_t  *tp,
                       parsec_dtd_funcptr_t *fpointer, int priority,
                       int device_type,
                       const char *name_of_kernel, ...);

/**
 * Return pointer on the device pointer associated with the i-th flow
 * of `this_task`.
 **/
void*
parsec_dtd_get_dev_ptr(parsec_task_t *this_task, int i);
   
/**
 * This function behaves exactly like parsec_dtd_insert_task()
 * except it does not insert the task in PaRSEC and just returns it.
 * Users will need to use parsec_insert_dtd_task() to insert the task
 */
parsec_task_t *
parsec_dtd_create_task(parsec_taskpool_t  *tp,
                       parsec_dtd_funcptr_t *fpointer, int priority,
                       int device_type,
                       const char *name_of_kernel, ...);

/**
 * This function allows users to insert a properly formed DTD task in
 * PaRSEC.
 */
void
parsec_insert_dtd_task(parsec_task_t *this_task);

/**
 * This function should be called anytime users
 * are using data in their parsec-dtd runs.
 * This functions intializes/cleans necessary
 * structures in a data-descriptor(dc). The
 * init should be called after a valid dc
 * has been acquired, and the fini before
 * the dc is cleaned.
 */
void
parsec_dtd_data_collection_init( parsec_data_collection_t *dc );

void
parsec_dtd_data_collection_fini( parsec_data_collection_t *dc );

/**
 * This function create and returns a PaRSEC DTD taskpool. The
 * taskpool is not associated with any context, contains no
 * tasks, and has no callback associated with.
 */
parsec_taskpool_t*
parsec_dtd_taskpool_new(void);

/**
 * This function will block until all the tasks inserted
 * so far is completed.
 * User can call this function multiple times
 * between a parsec_dtd_taskpool_new() and parsec_taskpool_free()
 * Takes a parsec context and a parsec taskpool as input.
 */
int
parsec_dtd_taskpool_wait( parsec_taskpool_t  *tp );

/**
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
 * PARSEC_DTD_TILE_OF or PARSEC_DTD_TILE_OF_KEY macro.
 * To ensure consistent and correct behavior, user
 * must wait on the taskpool before inserting
 * new task using this data after the flush.
 */
int
parsec_dtd_data_flush( parsec_taskpool_t   *tp,
                       parsec_dtd_tile_t *tile );

/**
 * This function flushes all the data of a dc(data collection).
 * This function must be called for all dc(s) before
 * parsec_context_wait() is called.
 */
int
parsec_dtd_data_flush_all( parsec_taskpool_t *tp,
                           parsec_data_collection_t  *dc );

/**
 * This function returns the taskpool a task bekongs to.
 */
parsec_taskpool_t *
parsec_dtd_get_taskpool(parsec_task_t *this_task);

/**
 * This function allows explicit dequeue of a taskpool.
 * Taskpools are automatically dequeued in parsec_context_wait()
 */
int
parsec_dtd_dequeue_taskpool(parsec_taskpool_t *tp);

parsec_task_class_t*
parsec_dtd_create_task_class( parsec_taskpool_t *__tp,
                              const char* name,
                              ... );

parsec_dtd_task_class_t *
parsec_dtd_create_task_classv(parsec_dtd_taskpool_t *dtd_tp,
                              const char *name,
                              int nb_params,
                              const parsec_dtd_param_t *params);

void
parsec_dtd_task_class_release(parsec_taskpool_t *tp, parsec_task_class_t *tc );

int parsec_dtd_task_class_add_chore(parsec_taskpool_t *tp,
                                    parsec_task_class_t *tc,
                                    int device_type,
                                    void *function);

void
parsec_dtd_insert_task_with_task_class(parsec_taskpool_t *tp,
                                       parsec_task_class_t *tc, int priority,
                                       int device_type,
                                       ...);

void
parsec_dtd_register_task_class(parsec_taskpool_t *tp,
                               uint64_t key,
                               parsec_task_class_t *value);

/**
 * @}
 */

END_C_DECLS

#endif  /* PARSEC_INSERT_FUNCTION_H_HAS_BEEN_INCLUDED */
