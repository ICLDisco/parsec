/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 **/
/**
 *
 * @file insert_function.h
 *
 * @version 2.0.0
 * @author Reazul Hoque
 *
 **/

#ifndef INSERT_FUNCTION_H_HAS_BEEN_INCLUDED
#define INSERT_FUNCTION_H_HAS_BEEN_INCLUDED

BEGIN_C_DECLS

#include <stdarg.h>
#include "dague.h"
#include "dague/data_distribution.h"

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

#define GET_OP_TYPE 0xf00
typedef enum {  INPUT=0x100,
                OUTPUT=0x200,
                INOUT=0x300,
                ATOMIC_WRITE=0x400, /* DO NOT USE ,Iterate_successors do not support this at this point */
                SCRATCH=0x500,
                VALUE=0x600
             } dtd_op_type;

/* Describes different regions to express more specific dependency.
 * All regions are mutually exclusive.
 */
#define GET_REGION_INFO 0xff
typedef enum {  REGION_FULL=1<<0,/* 0x1 is reserved for default(FULL tile) */
                REGION_L=1<<1, /* Lower triangle */
                REGION_D=1<<2, /* Diagonal */
                REGION_U=1<<3, /* Upper Triangle */
             } dtd_regions;

#define DAGUE_dtd_NB_FUNCTIONS  25 /* Max number of task classes allowed */
#define PASSED_BY_REF           1
#define UNPACK_VALUE            1
#define UNPACK_DATA             2
#define UNPACK_SCRATCH          3
#define MAX_FLOW                25

/* The parameters to pass to get pointer to data
 * 1. dague_dtd_handle_t*
 * 2. dague_ddesc_t *
 * 3. m (coordinates of the data in the matrix)
 * 4. n (coordinates of the data in the matrix)
 */
#define TILE_OF(DAGUE, DDESC, I, J) \
    dague_dtd_tile_of(DAGUE, &(__ddesc##DDESC->super.super), I, J)

typedef struct dague_dtd_task_param_s dague_dtd_task_param_t;
typedef struct dague_dtd_task_s       dague_dtd_task_t;
typedef struct dague_dtd_tile_s       dague_dtd_tile_t;
typedef struct dague_dtd_handle_s     dague_dtd_handle_t;
typedef struct dague_dtd_function_s   dague_dtd_function_t;

/* Function pointer typeof  kernel pointer pased as parameter to insert_function() */
/* This is the prototype of the function in which the actual operations of each task
 * is implemented by the User.
 * 1. dague_execution_unit_t *
 * 2. dague_execution_context_t * -> this gives access to the actual task the User inserted
                                     using this interface.
 * This function should always return DAGUE_HOOK_RETURN_DONE or 0.
 */
typedef int (dague_dtd_funcptr_t)(dague_execution_unit_t *, dague_execution_context_t *);

/* This function is used to retrieve the parameters passed during insertion of this task.
 * This function takes variadic parameters.
 * 1. dague_execution_context_t * -> The parameter list is attached with this structure.
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
void dague_dtd_unpack_args(dague_execution_context_t *this_task, ...);

dague_dtd_tile_t* dague_dtd_tile_of(dague_dtd_handle_t *dague_dtd_handle,
                                    dague_ddesc_t *ddesc, int i, int j);

/* Using this function Users can insert task in PaRSEC
 * 1. The dague handle (dague_dtd_handle_t *)
 * 2. The function pointer which will be executed as the "task" being inserted. This function should include
      the actual calculation the User wants to perform on the data. (The body of the task)
 * 3. String, stating the name of the task.
 * 4. Variadic type paramter. User can pass any number of paramters.
      Currently 3 type of paramters can be passed as a paramter of a task.
        - VALUE -> Will be copied according to the size(in bytes) specified.
        - SCRATCH -> Memory(in bytes) will be allocated and passed.
        - DATA -> Actual data or the matrix. Should be allocated before as only the reference is passed.
      Each paramter to pass to a task should be expressed in the form of a triplet. e.g

      1.              sizeof(int),             &uplo,              VALUE,
                    (size in bytes),  (pointer to the variable), (FLAG to specify how to handle this paramter)

      2.        sizeof(dague_complex64_t)*ib*100,     NULL,                          SCRATCH,
                    (size in byte),               (as memory of specified size      (FLAG to specify how to
                                                   will be allocated, no pointer      handle this paramter)
                                                   needs to be passed),

      3.          PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, k, k),     INOUT | REGION_FULL,
              (To specify we        ( we provide the handle,              ( The type of operation
               are passing only       descriptor and the                    the task being inserted will
               reference),            co-ordinates ),                       perform on this data and the
                                                                            region information to track more
                                                                            specific data dependency.)
      4. "0" indicates the end of paramter list. Must be provided.
 */
void insert_task_generic_fptr(dague_dtd_handle_t *,
                              dague_dtd_funcptr_t *, char *, ...);

/* This function will create a handle and return it. Provide the corresponding
 * dague context, so that the new handle is associated with.
 */
dague_dtd_handle_t* dague_dtd_handle_new(dague_context_t *);

/* Destroys the DAGUE  handle
 * Should be called after all tasks are done.
 */
void dague_dtd_handle_destruct(dague_dtd_handle_t *);

/* Makes the Dague context wait on the handle passed. The context will wait untill all the
 * tasks attached to this handle is over.
 * User can call this function multiple times in between a dague_dtd_handle_new() and dague_dtd_handle_destruct()
 */
void dague_dtd_handle_wait( dague_context_t     *dague,
                            dague_dtd_handle_t  *dague_handle );

/* User should call this function right before they intend to destroy a handle.
 * Should be called once for each handle.
 * User can not call this multiple times in between a dague_dtd_handle_new() and dague_dtd_handle_destruct().
 * This function can be called exactly once per handle.
 */
void dague_dtd_context_wait_on_handle( dague_context_t     *dague,
                                       dague_dtd_handle_t  *dague_handle );

/* Initiate and Finish dtd environment
 * dague_dtd_init () should be called right after dague_context_init()
 * dague_dtd_fini () right before dague_context_fini()
 */
void dague_dtd_init();
void dague_dtd_fini();

END_C_DECLS

#endif  /* INSERT_FUNCTION_H_HAS_BEEN_INCLUDED */
