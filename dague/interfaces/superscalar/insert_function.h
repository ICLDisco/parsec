/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef INSERT_FUNCTION_H_HAS_BEEN_INCLUDED
#define INSERT_FUNCTION_H_HAS_BEEN_INCLUDED

BEGIN_C_DECLS

#include <stdarg.h>
#include "dague.h"
#include "dague/data_distribution.h"

/* TODO: REMOVE */
extern double time_double;

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
                ATOMIC_WRITE=0x400,
                SCRATCH=0x500,
                VALUE=0x600
             } dtd_op_type;

#define GET_REGION_INFO 0xff
typedef enum {  REGION_FULL=1<<0,/* 0x1 is reserved for default(FULL tile) */
                REGION_L=1<<1,
                REGION_D=1<<2,
                REGION_U=1<<3,
                LOCALITY=1<<4
             } dtd_regions;

#define DAGUE_dtd_NB_FUNCTIONS  25
#define DTD_TASK_COUNT          10000
#define PASSED_BY_REF           1
#define UNPACK_VALUE            1
#define UNPACK_DATA             2
#define UNPACK_SCRATCH          3
#define MAX_DESC                25

#define OVERLAP                 1

#define TILE_OF(DAGUE, DDESC, I, J) \
    tile_manage(DAGUE, &(__ddesc##DDESC->super.super), I, J)

typedef struct dague_dtd_task_param_s dague_dtd_task_param_t;
typedef struct dague_dtd_task_s       dague_dtd_task_t;
typedef struct dague_dtd_tile_s       dague_dtd_tile_t;
typedef struct dague_dtd_handle_s     dague_dtd_handle_t;
typedef struct dague_dtd_function_s   dague_dtd_function_t;

/* Function pointer typeof  kernel pointer pased as parameter to insert_function() */
typedef int (dague_dtd_funcptr_t)(dague_execution_unit_t *, dague_execution_context_t *);

dague_dtd_tile_t* tile_manage(dague_dtd_handle_t *dague_dtd_handle,
                              dague_ddesc_t *ddesc, int i, int j);

dague_dtd_handle_t* dague_dtd_new(dague_context_t *, int, int, int* );

void insert_task_generic_fptr(dague_dtd_handle_t *,
                              dague_dtd_funcptr_t *, char *, ...);

void dague_dtd_unpack_args(dague_execution_context_t *this_task, ...);

/* This should not use intrenql structure ot should not be public */
typedef struct  __dague_dtd_internal_handle_s __dague_dtd_internal_handle_t;
void dtd_destructor(__dague_dtd_internal_handle_t * handle);

void increment_task_counter(dague_dtd_handle_t *);

END_C_DECLS

#endif  /* INSERT_FUNCTION_H_HAS_BEEN_INCLUDED */
