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

typedef int (task_func)(dague_execution_context_t*); /* Function pointer typeof  kernel pointer pased as parameter to insert_function() */

/* Structure used to pack arguments of insert_task() */
struct task_param_s {
    void *pointer_to_tile;
    int operation_type;
    int tile_type_index;
    task_param_t *next;
};

/* Task structure derived from dague_execution_context_t */
struct descendant_info { /* All the fields store info about the descendant except op_type_parent(operation type ex. INPUT, INPUt or OUTPUT) */
    uint8_t op_type_parent; /* Info about the current_task and not about descendant */
    uint8_t op_type; 
    uint8_t flow_index; 
    dtd_task_t *task;
};
struct dtd_task_s {
    dague_execution_context_t super;
    task_func* fpointer;
    uint32_t ref_count;
    uint32_t task_id;
    int total_flow;
    struct descendant_info desc[10];
    int flow_count;
    int ready_mask;
    char *name;
    uint8_t belongs_to_function;
    task_param_t *param_list;
};

#if 0
struct dtd_tile_s {
    uint32_t rank;
    int32_t vp_id;
    dague_data_key_t key;
    dague_data_copy_t *data_copy;
    dague_data_t *data;
    dague_ddesc_t *ddesc;
    struct user last_user;
};
#endif

/**
 * internal_dague_handle
 */
typedef struct dague_dtd_handle_s {
    dague_handle_t super; 
    /* The array of datatypes LOWER_TILE, LITTLE_T, DEFAULT and the others */
    dague_arena_t **arenas;
    int arenas_size;
    int *INFO; //zpotrf specific; should be removed    
    int tile_hash_table_size;
    int task_hash_table_size;
    uint8_t function_hash_table_size;
    hash_table *task_h_table; // hash_table for tasks
    hash_table *function_h_table; // hash_table for master function structure
    hash_table *tile_h_table; // ready task list head
    dtd_task_t *ready_task; //ring of initial ready tasks 
    int total_task_class;
} dague_dtd_handle_t;


dtd_tile_t* tile_manage(dague_dtd_handle_t *dague_dtd_handle, 
                        dague_ddesc_t *ddesc, int i, int j);

dague_dtd_handle_t* dague_dtd_new(int, int, int* );
    
void insert_task_generic_fptr(dague_dtd_handle_t *, 
                              task_func *, char *, ...);
