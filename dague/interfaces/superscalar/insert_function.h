#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include "data_distribution.h"
#include "dague.h"

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
