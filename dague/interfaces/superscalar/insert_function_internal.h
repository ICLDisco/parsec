#include "dague_internal.h"
#include "data.h"
#include "datarepo.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include "data_distribution.h"
#include "dague/interfaces/superscalar/insert_function.h"

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


#if 0
#define TILE_OF(DAGUE, DDESC, I, J) \
    tile_manage(DAGUE, &(ddesc##DDESC.super.super), I, J)


typedef struct generic_hash_table hash_table;

#endif
typedef struct bucket_element_f_s bucket_element_f_t;
typedef struct bucket_element_tile_s bucket_element_tile_t;
typedef struct bucket_element_task_s bucket_element_task_t;

#if 0
typedef struct dtd_tile_s dtd_tile_t;

typedef struct task_param_s task_param_t;
typedef struct dtd_task_s dtd_task_t;

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
#endif

DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dtd_task_t); /* For creating objects of class dtd_task_t */

/** Tile structure **/
struct user { 
    uint8_t flow_index;
    uint8_t op_type;
    dtd_task_t *task;
};

struct dtd_tile_s {
    uint32_t rank;
    int32_t vp_id;
    dague_data_key_t key;
    dague_data_copy_t *data_copy;
    dague_data_t *data;
    dague_ddesc_t *ddesc;
    struct user last_user;
};

/** Function Hash table elements **/
struct bucket_element_f_s {
    task_func* key;
    dague_function_t *dtd_function;
    bucket_element_f_t *next;
};

/** Tile Hash table elements **/
struct bucket_element_tile_s {
    dague_data_key_t key;
    dtd_tile_t *tile;
    bucket_element_tile_t *next;
};

/** Task hashtable elements **/
struct bucket_element_task_s {
    int key;
    dtd_task_t *task;
    bucket_element_task_t *next;
};

/* One type of hash table for task, tiles and functions */
struct generic_hash_table {
    int size;
    void **buckets;
};


/**
 * internal_dague_handle
 */
#if 0
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
#endif
typedef struct __dague_dtd_internal_handle_s {
    dague_dtd_handle_t super;
    data_repo_t *dtd_data_repository;
} __dague_dtd_internal_handle_t;


/* Function prototypes */
dague_ontask_iterate_t  dtd_release_dep_fct(struct dague_execution_unit_s *eu,
                                            const dague_execution_context_t *newcontext,
                                            const dague_execution_context_t *oldcontext,
                                            const dep_t* dep,
                                            dague_dep_data_description_t *data,
                                            int rank_src, int rank_dst, int vpid_dst,
                                            void *param);

hash_table* create_task_table(int);

int test_hook_of_dtd_task(dague_execution_unit_t *context, 
                          dague_execution_context_t *this_task);

int dtd_is_ready(const dtd_task_t *,const int);
   
#if 0 
dtd_tile_t* tile_manage(dague_dtd_handle_t *dague_dtd_handle, 
                        dague_ddesc_t *ddesc, int i, int j);

dague_dtd_handle_t* dague_dtd_new(int, int, int* );
    
void insert_task_generic_fptr(dague_dtd_handle_t *, 
                              task_func *, char *, ...);
#endif
void _internal_insert_task(dague_dtd_handle_t *, 
                           task_func*, task_param_t *, 
                           char *);

void iterate_successors_of_dtd_task(dague_execution_unit_t *eu,
                                    const dague_execution_context_t *this_task, 
                                    uint32_t action_mask, 
                                    dague_ontask_function_t *ontask, 
                                    void *ontask_arg);

int release_deps_of_dtd(struct dague_execution_unit_s *,
                        dague_execution_context_t *,
                        uint32_t, dague_remote_deps_t *);

dague_hook_return_t complete_hook_of_dtd(struct dague_execution_unit_s *, 
                                         dague_execution_context_t *);

int dtd_startup_tasks(dague_context_t *context, 
                      __dague_dtd_internal_handle_t *__dague_handle, 
                      dague_execution_context_t **pready_list);

void dtd_startup(dague_context_t *context, 
                 dague_handle_t *dague_handle, 
                 dague_execution_context_t **pready_list);

dtd_tile_t* find_tile(hash_table *tile_h_table, 
                      uint32_t key, int h_size);

void tile_insert_h_t(hash_table *tile_h_table, 
                     uint32_t key, dtd_tile_t *tile, 
                     int h_size);

int data_lookup_of_dtd_task(dague_execution_unit_t *, 
                            dague_execution_context_t *);
