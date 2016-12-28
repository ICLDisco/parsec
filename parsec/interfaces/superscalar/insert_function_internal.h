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

#include "parsec/parsec_internal.h"
#include "parsec/data.h"
#include "parsec/data_internal.h"
#include "parsec/datarepo.h"
#include "parsec/data_distribution.h"
#include "parsec/interfaces/superscalar/insert_function.h"

extern int dump_traversal_info; /* For printing traversal info */
extern int dump_function_info; /* For printing function_structure info */
extern int dtd_window_size;
extern int my_rank;

#define GET_HEAD_OF_PARAM_LIST(TASK) (parsec_dtd_task_param_t *) (((char *)TASK) + sizeof(parsec_dtd_task_t))
#define GET_VALUE_BLOCK(HEAD, PARAM_COUNT) ((char *)HEAD) + PARAM_COUNT * sizeof(parsec_dtd_task_param_t)

#define OVERLAP    1 /* enables window */
#define LOCAL_DATA 200 /* function_id is uint8_t */
//#define DEBUG_HEAVY 1
#define WILL_USE_IN_DISTRIBUTED
#define PARSEC_DEFAULT_ARENA     0

#define NOT_OVERLAPPED 1
#define OVERLAPPED     0

#define TASK_IS_ALIVE       (uint8_t)1
#define TASK_IS_NOT_ALIVE   (uint8_t)0

/* Structure used to pack arguments of insert_task() */
struct parsec_dtd_task_param_s {
    void                       *pointer_to_tile;
    parsec_dtd_task_param_t    *next;
};

/* Contains info about each flow of a task
 * We assume each task will have at most MAX_FLOW
 * number of flows
 */
typedef struct parsec_dtd_flow_info_s {
    uint8_t            op_type;  /* Operation type on the data */
    parsec_dtd_tile_t *tile;
}parsec_dtd_flow_info_t;

/* All the fields store info about the descendant
 */
typedef struct descendant_info_s {
    uint8_t            op_type;
    uint8_t            flow_index;
    parsec_dtd_task_t *task;
}descendant_info_t;

typedef struct parsec_dtd_parent_info_s {
    uint8_t              op_type;
    uint8_t              flow_index;
    parsec_dtd_task_t   *task;
} parsec_dtd_parent_info_t;

struct parsec_dtd_task_s {
    parsec_execution_context_t   super;
    uint32_t                    ref_count;
    int                         flow_count;
    /* Saves flow index for which we have to release data of a TASK
       with INPUT and ATOMIC_WRITE operation
     */
    uint8_t                     dont_skip_releasing_data[MAX_FLOW];
    /* for testing PTG inserting task in DTD */
    parsec_execution_context_t  *orig_task;
    descendant_info_t           desc[MAX_FLOW];
    parsec_dtd_parent_info_t     parent[MAX_FLOW];
    parsec_dtd_flow_info_t       flow[MAX_FLOW];
    parsec_dtd_task_param_t     *param_list;
};
/* For creating objects of class parsec_dtd_task_t */
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_dtd_task_t);

/** Tile structure **/
typedef struct parsec_dtd_tile_user_s {
    uint8_t              flow_index;
    uint8_t              op_type;
    uint8_t              alive;
    parsec_dtd_task_t   *task;
    parsec_atomic_lock_t atomic_lock;
}parsec_dtd_tile_user_t;

struct parsec_dtd_tile_s {
    parsec_hashtable_item_t super;
    uint32_t               rank;
    int32_t                vp_id;
    parsec_data_key_t       key;
    parsec_data_copy_t     *data_copy;
    parsec_data_t          *data;
    parsec_ddesc_t         *ddesc;
    parsec_dtd_tile_user_t  last_user;
};
/* For creating objects of class parsec_dtd_tile_t */
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_dtd_tile_t);

/* for testing abstraction for PaRsec */
struct hook_info{
    parsec_hook_t *hook;
};

/**
 * internal_parsec_handle
 */
struct parsec_dtd_handle_s {
    parsec_handle_t  super;
    parsec_thread_mempool_t *mempool_owner;
    /* The array of datatypes, the region_info */
    parsec_arena_t   **arenas;
    uint8_t         mode;
    int             arenas_size;
    int             task_id;
    int             task_window_size;
    int32_t         task_threshold_size;
    uint8_t         function_counter;
    uint8_t         flow_set_flag[PARSEC_dtd_NB_FUNCTIONS];
    parsec_mempool_t *tile_mempool;
    parsec_mempool_t *hash_table_bucket_mempool;
    hash_table      *task_h_table;
    hash_table      *function_h_table;
    hash_table      *tile_h_table;
    /* ring of initial ready tasks */
    parsec_execution_context_t      **startup_list;
    /* from here to end is for the testing interface */
    struct          hook_info actual_hook[PARSEC_dtd_NB_FUNCTIONS];
    int             total_tasks_to_be_exec;
};

/*
 * Extension of parsec_function_t class
 */
struct parsec_dtd_function_s {
    parsec_function_t     super;
    parsec_dtd_funcptr_t *fpointer;
    parsec_mempool_t     *context_mempool;
    int                  dep_datatype_index;
    int                  dep_out_index;
    int                  dep_in_index;
    int                  count_of_params;
    long unsigned int    size_of_param;
};

/* Function prototypes */
parsec_dtd_task_t *
create_and_initialize_dtd_task( parsec_dtd_handle_t *parsec_dtd_handle,
                                parsec_function_t   *function);

void
set_params_of_task( parsec_dtd_task_t *this_task, parsec_dtd_tile_t *tile,
                    int tile_op_type, int *flow_index, void **current_val,
                    parsec_dtd_task_param_t *current_param, int *next_arg );

void
parsec_insert_dtd_task( parsec_dtd_task_t *this_task );

void dtd_startup(parsec_context_t *context,
                 parsec_handle_t *parsec_handle,
                 parsec_execution_context_t **pready_list);

int data_lookup_of_dtd_task(parsec_execution_unit_t *,
                            parsec_execution_context_t *);

void ordering_correctly_1(parsec_execution_unit_t * eu,
                     const parsec_execution_context_t * this_task,
                     uint32_t action_mask,
                     parsec_ontask_function_t * ontask,
                     void *ontask_arg);

void
schedule_tasks(parsec_dtd_handle_t *__parsec_handle);

/* Function to remove tile from hash_table
 */
void
parsec_dtd_tile_remove
( parsec_dtd_handle_t *parsec_handle, uint32_t key,
  parsec_ddesc_t      *ddesc );

/* Function to find tile in hash_table
 */
parsec_dtd_tile_t *
parsec_dtd_tile_find
( parsec_dtd_handle_t *parsec_handle, uint32_t key,
  parsec_ddesc_t      *ddesc );

void
parsec_dtd_tile_release
(parsec_dtd_handle_t *parsec_handle, parsec_dtd_tile_t *tile);

uint32_t
hash_key (uintptr_t key, int size);

void
parsec_dtd_tile_insert( parsec_dtd_handle_t *parsec_handle, uint32_t key,
                       parsec_dtd_tile_t   *tile,
                       parsec_ddesc_t      *ddesc );

parsec_dtd_function_t *
parsec_dtd_function_find( parsec_dtd_handle_t  *parsec_handle,
                         parsec_dtd_funcptr_t *key );

parsec_function_t*
create_function(parsec_dtd_handle_t *__parsec_handle, parsec_dtd_funcptr_t* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param, int flow_count);

void
add_profiling_info(parsec_dtd_handle_t *__parsec_handle,
                   parsec_function_t *function, char* name,
                   int flow_count);

void
parsec_dtd_task_release( parsec_dtd_handle_t  *parsec_handle,
                        uint32_t             key );

void
parsec_dtd_task_insert( parsec_dtd_handle_t   *parsec_handle,
                       parsec_dtd_task_t     *value );

void
parsec_execute_and_come_back(parsec_context_t *context,
                            parsec_handle_t *parsec_handle);

/***************************************************************************//**
 *
 * Function to lock last_user of a tile
 *
 * @param[in,out]   last_user
 *                      User we are trying to lock
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
static inline void
parsec_dtd_last_user_lock( parsec_dtd_tile_user_t *last_user )
{
    parsec_atomic_lock(&last_user->atomic_lock);
}

/***************************************************************************//**
 *
 * Function to unlock last_user of a tile
 *
 * @param[in,out]   last_user
 *                      User we are trying to unlock
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
static inline void
parsec_dtd_last_user_unlock( parsec_dtd_tile_user_t *last_user )
{
    parsec_atomic_unlock(&last_user->atomic_lock);
}

END_C_DECLS

#endif  /* INSERT_FUNCTION_INTERNAL_H_HAS_BEEN_INCLUDED */
