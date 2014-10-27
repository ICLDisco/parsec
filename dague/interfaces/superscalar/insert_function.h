#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include "data_distribution.h"
#include "dague.h"

#define INPUT 0x1
#define OUTPUT 0x2
#define INOUT 0x3
#define VALUE 0x20
#define AFFINITY 0x4
#define DEFAULT 0x1
#define LOWER_TILE 0x2
#define LITTLE_T 0x3
#define DAGUE_dtd_NB_FUNCTIONS 5
#define DTD_TASK_COUNT 10000
#define PASSED_BY_REF 1
#define UNPACK_VALUE 1
#define UNPACK_DATA 2

#define TILE_OF(DAGUE, DDESC, I, J) \
    tile_manage(DAGUE, &(ddesc##DDESC.super.super), I, J)

typedef struct generic_hash_table hash_table;
typedef struct task_param_s task_param_t;
typedef struct dtd_task_s dtd_task_t;
typedef struct dtd_tile_s dtd_tile_t;
typedef struct dague_dtd_handle_s dague_dtd_handle_t;

typedef int (task_func)(dague_execution_context_t*); /* Function pointer typeof  kernel pointer pased as parameter to insert_function() */

dtd_tile_t* tile_manage(dague_dtd_handle_t *dague_dtd_handle, 
                        dague_ddesc_t *ddesc, int i, int j);

dague_dtd_handle_t* dague_dtd_new(int, int, int* );
    
void insert_task_generic_fptr(dague_dtd_handle_t *, 
                              task_func *, char *, ...);

void dague_dtd_unpack_args(dague_execution_context_t *this_task, ...);
