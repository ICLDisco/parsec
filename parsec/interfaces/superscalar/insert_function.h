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

BEGIN_C_DECLS

#include <stdarg.h>
#include "parsec.h"
#include "parsec/data_distribution.h"

/**
 * To see examples please look at testing_zpotrf_dtd.c, testing_zgeqrf_dtd.c,
 * testing_zgetrf_incpiv_dtd.c files in the directory "root_of_PaRSEC/dplasma/testing/"
 **/

/*
    **  Details of Flags **

 *  INPUT:          Data is used in read-only mode, no modification is done.
 *  OUTPUT:         Data is used in write-only, written only, not read.
 *  INOUT:          Data is read and written both.
 *  ATOMIC_WRITE:   Data is used like OUTPUT but the ordering of the tasks having this flag is not maintained
                    by the scheduler.
                    It is the responsibility of the user to make sure data is written atomically.
                    Treated like INPUT by the scheduler.
 *  SCRATCH:        Will be used by the task as scratch pad, does not effect the DAG, tells the runtime
                    to allocate memory specified by the user.
 *  VALUE:          Tells the runtime to copy the value as a parameter of the task.

 */

#define GET_OP_TYPE 0xf00000
typedef enum { INPUT=0x100000,
               OUTPUT=0x200000,
               INOUT=0x300000,
               ATOMIC_WRITE=0x400000, /* DO NOT USE ,Iterate_successors do not support this at this point */
               SCRATCH=0x500000,
               VALUE=0x600000
             } parsec_dtd_op_type;

#define GET_OTHER_FLAG_INFO 0xf0000
typedef enum { AFFINITY=1<<16, /* Data affinity */
               DONT_TRACK=1<<17, /* Drop dependency tracking */
             } parsed_dtd_other_flag_type;

/* Describes different regions to express more specific dependency.
 * All regions are mutually exclusive.
 */
#define GET_REGION_INFO 0xffff

extern int dtd_window_size;
extern int dtd_threshold_size;
/* Array of arenas to hold the data region shape and other information.
 * Currently only 16 types of different regions are supported at a time.
 */
extern parsec_arena_t **parsec_dtd_arenas;

#define PASSED_BY_REF            1
#define UNPACK_VALUE             1
#define UNPACK_DATA              2
#define UNPACK_SCRATCH           3
#define MAX_FLOW                 25
#define PARSEC_DTD_NB_FUNCTIONS  25 /* Max number of task classes allowed */

/* The parameters to pass to get pointer to data
 * 1. parsec_dtd_handle_t*
 * 2. parsec_ddesc_t *
 * 3. m (coordinates of the data in the matrix)
 * 4. n (coordinates of the data in the matrix)
 */
#define TILE_OF(DDESC, I, J) \
    parsec_dtd_tile_of(&(__ddesc##DDESC->super.super), (&(__ddesc##DDESC->super.super))->data_key(&(__ddesc##DDESC->super.super), I, J))

#define TILE_OF_KEY(DDESC, KEY) \
    parsec_dtd_tile_of(DDESC, KEY)

typedef struct parsec_dtd_tile_s       parsec_dtd_tile_t;
typedef struct parsec_dtd_task_s       parsec_dtd_task_t;
typedef struct parsec_dtd_handle_s     parsec_dtd_handle_t;

/* Function pointer typeof  kernel pointer pased as parameter to insert_function() */
/* This is the prototype of the function in which the actual operations of each task
 * is implemented by the User.
 * 1. parsec_execution_unit_t *
 * 2. parsec_execution_context_t * -> this gives access to the actual task the User inserted
                                     using this interface.
 * This function should always return PARSEC_HOOK_RETURN_DONE or 0.
 */
typedef int (parsec_dtd_funcptr_t)(parsec_execution_unit_t *, parsec_execution_context_t *);

/* This function is used to retrieve the parameters passed during insertion of this task.
 * This function takes variadic parameters.
 * 1. parsec_execution_context_t * -> The parameter list is attached with this structure.
                                     The User needs to pass a FLAG to specify what sort of value needs to be
                                     unpacked. Three types of FLAGS are supported:
                                     - UNPACK_VALUE
                                     - UNPACK_SCRATCH
                                     - UNPACK_DATA
                                     Following each FLAG the pointer to the memory location where the paramter
                                     will be copied needs to be given.
                                   **The order in which the parameters were passed during insertion needs to be
                                     strictly followed while unpacking.
 */
void parsec_dtd_unpack_args(parsec_execution_context_t *this_task, ...);

parsec_dtd_tile_t *
parsec_dtd_tile_of( parsec_ddesc_t *ddesc, parsec_data_key_t key );

/* Using this function Users can insert task in PaRSEC
 * 1. The parsec handle (parsec_dtd_handle_t *)
 * 2. The function pointer which will be executed as the "task" being inserted. This function should include
      the actual calculation the User wants to perform on the data. (The body of the task)
 * 3. The priority of the task, if not sure user should provide 0.
 * 4. String, stating the name of the task.
 * 5. Variadic type paramter. User can pass any number of paramters.
      Currently 3 type of paramters can be passed as a paramter of a task.
        - VALUE -> Will be copied according to the size(in bytes) specified.
        - SCRATCH -> Memory(in bytes) will be allocated and passed.
        - DATA -> Actual data or the matrix. Should be allocated before as only the reference is passed.
      Each paramter to pass to a task should be expressed in the form of a triplet. e.g

      1.              sizeof(int),             &uplo,              VALUE,
                    (size in bytes),  (pointer to the variable), (FLAG to specify how to handle this paramter)

      2.        sizeof(parsec_complex64_t)*ib*100,     NULL,                          SCRATCH,
                    (size in byte),               (as memory of specified size      (FLAG to specify how to
                                                   will be allocated, no pointer      handle this paramter)
                                                   needs to be passed),

      3.          PASSED_BY_REF,    TILE_OF(PARSEC_dtd_handle, A, k, k),     INOUT | REGION_FULL,
              (To specify we        ( we provide the handle,              ( The type of operation
               are passing only       descriptor and the                    the task being inserted will
               reference),            co-ordinates ),                       perform on this data and the
                                                                            region information to track more
                                                                            specific data dependency.)
      4. "0" indicates the end of paramter list. Must be provided.
 */
void
parsec_insert_task( parsec_handle_t  *parsec_handle,
                   parsec_dtd_funcptr_t *fpointer, int priority,
                   char *name_of_kernel, ... );

#define DTD_DDESC_INIT(DDESC) \
    parsec_dtd_ddesc_init(&(__ddesc##DDESC->super.super))

void
parsec_dtd_ddesc_init( parsec_ddesc_t *ddesc );

#define DTD_DDESC_FINI(DDESC) \
    parsec_dtd_ddesc_fini(&(__ddesc##DDESC->super.super))

void
parsec_dtd_ddesc_fini( parsec_ddesc_t *ddesc );

/* This function will create a handle and return it. Provide the corresponding
 * parsec context, so that the new handle is associated with.
 */
parsec_handle_t*
parsec_dtd_handle_new();

/* Makes the PaRSEC context wait on the handle passed. The context will wait untill all the
 * tasks attached to this handle are over.
 * User can call this function multiple times in between a parsec_dtd_handle_new() and parsec_dtd_handle_destruct()
 */
void
parsec_dtd_handle_wait( parsec_context_t *parsec,
                        parsec_handle_t  *parsec_handle );

/* This function flushes a specific data, it indicates to the engine that this data
 * will no longer be used by any tasks, this indication optimizes the reuse of memory attached to data.
 */
void
parsec_dtd_data_flush( parsec_handle_t *parsec_handle,
                       parsec_dtd_tile_t *tile );

/* This function flushes all data the runtime has discovered belonging to the ddesc(data descriptor) provided.
 * By flushing, it means that we call parsec_dtd_data_flush() for each data belonging to this ddesc.
 */
void
parsec_dtd_data_flush_all( parsec_handle_t *parsec_handle,
                           parsec_ddesc_t *ddesc );

END_C_DECLS

#endif  /* INSERT_FUNCTION_H_HAS_BEEN_INCLUDED */
