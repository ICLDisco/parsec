/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/**
 *
 * @file insert_function_internal.h
 *
 * @version 2.0.0
 * @author Reazul Hoque
 *
 **/

#ifndef INSERT_FUNCTION_INTERNAL_H_HAS_BEEN_INCLUDED
#define INSERT_FUNCTION_INTERNAL_H_HAS_BEEN_INCLUDED

BEGIN_C_DECLS

#include "dague/dague_internal.h"
#include "dague/data.h"
#include "dague/data_internal.h"
#include "dague/datarepo.h"
#include "dague/data_distribution.h"
#include "dague/interfaces/superscalar/insert_function.h"

extern int dump_traversal_info; /* For printing traversal info */
extern int dump_function_info; /* For printing function_structure info */
extern int testing_ptg_to_dtd; /* to detect ptg testing dtd */
extern int dtd_window_size;
extern int my_rank;

#define GET_TASK_PTR(TASK) (dague_dtd_task_t *)(((uintptr_t)TASK) & 0xFFFFFFFFFFFFFFF0)
#define GET_FLOW_IND(TASK) (int)(((uintptr_t)TASK) & 0x0F)

#define OVERLAP    1 /* enables window */
#define LOCAL_DATA 200 /* function_id is uint8_t */
#define DEBUG_HEAVY 1

/* Structure used to pack arguments of insert_task() */
struct dague_dtd_task_param_s {
    void            *pointer_to_tile;
    int             operation_type;
    int             tile_type_index;
    dague_dtd_task_param_t    *next;
};

/* Contains info about each flow of a task
 * We assume each task will have at most MAX_FLOW
 * number of flows
 */
typedef struct dague_dtd_flow_info_s {
    int               op_type;  /* Operation type on the data */
    dague_dtd_tile_t *tile;
}dague_dtd_flow_info_t;

/* All the fields store info about the descendant
 */
typedef struct descendant_info_s {
    int               op_type;
    uint8_t           flow_index;
    dague_dtd_task_t *task;
}descendant_info_t;

struct dague_dtd_task_s {
    dague_execution_context_t   super;
    dague_dtd_funcptr_t        *fpointer;
    uint32_t                    ref_count;
    int                         flow_count;
    /* Saves flow index for which we have to release data of a TASK
       with INPUT and ATOMIC_WRITE operation
     */
    uint8_t                     dont_skip_releasing_data[MAX_FLOW];
    /* for testing PTG inserting task in DTD */
    dague_execution_context_t  *orig_task;
    descendant_info_t           desc[MAX_FLOW];
    dague_dtd_flow_info_t       flow[MAX_FLOW];
    dague_dtd_task_param_t     *param_list;
};
/* For creating objects of class dague_dtd_task_t */
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_dtd_task_t);

/** Tile structure **/
struct user {
    uint8_t     flow_index;
    int         op_type;
    dague_dtd_task_t  *task;
};

struct dague_dtd_tile_s {
    dague_hashtable_item_t   super;
    uint32_t            rank;
    int32_t             vp_id;
    dague_data_key_t    key;
    dague_data_copy_t   *data_copy;
    dague_data_t        *data;
    dague_ddesc_t       *ddesc;
    struct user         last_user;
};
/* For creating objects of class dague_dtd_tile_t */
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_dtd_tile_t);

/* for testing abstraction for PaRsec */
struct hook_info{
    dague_hook_t *hook;
};

/**
 * internal_dague_handle
 */
struct dague_dtd_handle_s {
    dague_handle_t  super;
    dague_thread_mempool_t *mempool_owner;
    /* The array of datatypes, the region_info */
    dague_arena_t   **arenas;
    int             arenas_size;
    int             task_id;
    int             task_window_size;
    uint32_t        task_threshold_size;
    uint8_t         function_counter;
    uint8_t         flow_set_flag[DAGUE_dtd_NB_FUNCTIONS];
    dague_mempool_t *tile_mempool;
    dague_mempool_t *hash_table_bucket_mempool;
    hash_table      *task_h_table;
    hash_table      *function_h_table;
    hash_table      *tile_h_table;
    /* ring of initial ready tasks */
    dague_execution_context_t      **startup_list;
    /* from here to end is for the testing interface */
    struct          hook_info actual_hook[DAGUE_dtd_NB_FUNCTIONS];
    int             total_tasks_to_be_exec;
};

/*
 * Extension of dague_function_t class
 */
struct dague_dtd_function_s {
    dague_function_t     super;
    dague_dtd_funcptr_t *fpointer;
    dague_mempool_t     *context_mempool;
    int                  count_of_params;
    long unsigned int    size_of_param;
};

/* Function prototypes */
dague_ontask_iterate_t  dtd_release_dep_fct(struct dague_execution_unit_s *eu,
                                            const dague_execution_context_t *newcontext,
                                            const dague_execution_context_t *oldcontext,
                                            const dep_t* dep,
                                            dague_dep_data_description_t *data,
                                            int rank_src, int rank_dst, int vpid_dst,
                                            void *param);

void dtd_startup(dague_context_t *context,
                 dague_handle_t *dague_handle,
                 dague_execution_context_t **pready_list);

int data_lookup_of_dtd_task(dague_execution_unit_t *,
                            dague_execution_context_t *);

void ordering_correctly_1(dague_execution_unit_t * eu,
                     const dague_execution_context_t * this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t * ontask,
                     void *ontask_arg);

void ordering_correctly_2(dague_execution_unit_t * eu,
                     const dague_execution_context_t * this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t * ontask,
                     void *ontask_arg);

dague_dtd_task_t *
create_fake_writer_task( dague_dtd_handle_t  *__dague_handle, dague_dtd_tile_t *tile );

void
set_task(dague_dtd_task_t *temp_task, void *tmp, dague_dtd_tile_t *tile, int *satisfied_flow,
         int tile_op_type, dague_dtd_task_param_t *current_param,
         uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS], void **current_val,
         dague_dtd_handle_t *__dague_handle, int *flow_index, int *next_arg);

void
schedule_tasks(dague_dtd_handle_t *__dague_handle);

/* Function to remove tile from hash_table
 */
void
dague_dtd_tile_remove
( dague_dtd_handle_t *dague_handle, uint32_t key,
  dague_ddesc_t      *ddesc );

/* Function to find tile in hash_table
 */
dague_dtd_tile_t *
dague_dtd_tile_find
( dague_dtd_handle_t *dague_handle, uint32_t key,
  dague_ddesc_t      *ddesc );

void
dague_dtd_tile_release
(dague_dtd_handle_t *dague_handle, dague_dtd_tile_t *tile);

uint32_t
hash_key (uintptr_t key, int size);

void
dague_dtd_tile_insert( dague_dtd_handle_t *dague_handle, uint32_t key,
                       dague_dtd_tile_t   *tile,
                       dague_ddesc_t      *ddesc );

dague_dtd_function_t *
dague_dtd_function_find( dague_dtd_handle_t  *dague_handle,
                         dague_dtd_funcptr_t *key );

dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, dague_dtd_funcptr_t* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param, int flow_count);

void
add_profiling_info(dague_dtd_handle_t *__dague_handle,
                   dague_function_t *function, char* name,
                   int flow_count);

void
dague_dtd_task_release( dague_dtd_handle_t  *dague_handle,
                        uint32_t             key );

void
dague_dtd_task_insert( dague_dtd_handle_t   *dague_handle,
                       dague_dtd_task_t     *value );

void
dague_execute_and_come_back(dague_context_t *context,
                            dague_handle_t *dague_handle);

END_C_DECLS

#endif  /* INSERT_FUNCTION_INTERNAL_H_HAS_BEEN_INCLUDED */
