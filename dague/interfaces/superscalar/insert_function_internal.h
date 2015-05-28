#include "dague/dague_internal.h"
#include "dague/data.h"
#include "dague/data_internal.h"
#include "dague/datarepo.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include "dague/data_distribution.h"
#include "dague/interfaces/superscalar/insert_function.h"

int dump_traversal_info; /* For printing traversal info */
int dump_function_info; /* For printing function_structure info */

dague_dtd_handle_t *__dtd_handle; /* for testing purpose of automatic insertion from Awesome PTG approach */

typedef struct bucket_element_f_s bucket_element_f_t;
typedef struct bucket_element_tile_s bucket_element_tile_t;
typedef struct bucket_element_task_s bucket_element_task_t;

typedef struct dtd_successor_list_s dtd_successor_list_t;

/* Structure used to pack arguments of insert_task() */
struct task_param_s {
    void *pointer_to_tile;
    int operation_type;
    int tile_type_index;
    task_param_t *next;
};

/* Task structure derived from dague_execution_context_t */
struct descendant_info { /* All the fields store info about the descendant except op_type_parent(operation type ex. INPUT, INOUT or OUTPUT) */
    int op_type_parent; /* Info about the current_task and not about descendant */
    int op_type;
    uint8_t flow_index;
    dtd_task_t *task;
    dtd_tile_t *tile;
};

/* Structure to hold list of Read-ONLY successors of a task */
struct dtd_successor_list_s { /* Structure to be used for correct ordering strategy 1 in multi-threaded env */
    dtd_task_t *task;
    dep_t *deps;
    int flow_index;
    dtd_successor_list_t *next;
};

struct dtd_task_s {
    dague_execution_context_t super;
    task_func* fpointer;
    uint32_t ref_count;
    uint32_t task_id;
    int total_flow;
    struct descendant_info desc[MAX_DESC];
    int flow_count;
    int flow_satisfied;
    int ready_mask;
    char *name;
    uint8_t belongs_to_function;
    uint8_t dont_skip_releasing_data[MAX_DESC]; /* Saves flow index for which we have to release data of a TASK 
                                  with INPUT and ATOMIC_WRITE operation */
    dague_execution_context_t *orig_task; //for testing purpose will be removed
    task_param_t *param_list;
};

DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dtd_task_t); /* For creating objects of class dtd_task_t */

/** Tile structure **/
struct user {
    uint8_t flow_index;
    int op_type;
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
    dague_ddesc_t* belongs_to;
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

/* for testing abstraction for PaRsec */
struct hook_info{
    dague_hook_t *hook;
};

/**
 * internal_dague_handle
 */
struct dague_dtd_handle_s {
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
    int tasks_created;
    int tasks_scheduled;
    struct hook_info actual_hook[DAGUE_dtd_NB_FUNCTIONS];/* from here to end is for the testing interface */
    int total_tasks_to_be_exec;
    //hash_table *tile_ctl_table; // ready task list head
};

struct __dague_dtd_internal_handle_s {
    dague_dtd_handle_t super;
    data_repo_t *dtd_data_repository;
};

/**
 * Extension of dague_function_t class 
 *
 */
struct dague_dtd_function_s {
    dague_function_t super;
    dague_mempool_t *context_mempool;
    int count_of_params;
    long unsigned int size_of_param;
};

/* Function prototypes */
dague_ontask_iterate_t  dtd_release_dep_fct(struct dague_execution_unit_s *eu,
                                            const dague_execution_context_t *newcontext,
                                            const dague_execution_context_t *oldcontext,
                                            const dep_t* dep,
                                            dague_dep_data_description_t *data,
                                            int rank_src, int rank_dst, int vpid_dst,
                                            void *param);

hash_table* create_task_table(int);

void dtd_startup(dague_context_t *context,
                 dague_handle_t *dague_handle,
                 dague_execution_context_t **pready_list);

dtd_tile_t* find_tile(hash_table *tile_h_table,
                      uint32_t key, int h_size,
                      dague_ddesc_t *belongs_to);

void tile_insert_h_t(hash_table *tile_h_table,
                     uint32_t key, dtd_tile_t *tile,
                     int h_size, dague_ddesc_t *belongs_to);

int data_lookup_of_dtd_task(dague_execution_unit_t *,
                            dague_execution_context_t *);


void copy_chores(dague_handle_t *handle, dague_dtd_handle_t *dtd_handle);
