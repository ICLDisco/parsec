/**
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file insert_function.c
 *
 * @version 2.0.0
 *
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup DTD_INTERFACE Dynamic Task Discovery interface for PaRSEC
 * @ingroup parsec_public
 *
 * These functions are available from the PaRSEC library for the
 * scheduling of kernel routines.
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup DTD_INTERFACE_INTERNAL Dynamic Task Discovery functions for PaRSEC
 * @ingroup parsec_internal
 *
 * These functions are not available from the PaRSEC library for the
 * scheduling of kernel routines.
 */

#include <stdlib.h>
#include <sys/time.h>

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"
#include "parsec/remote_dep.h"
#include "parsec/devices/device.h"
#include "parsec/constants.h"
#include "parsec/vpmap.h"
#include "parsec/utils/mca_param.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/interfaces/interface.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/parsec_prof_grapher.h"
#include "parsec/mca/pins/pins.h"
#include "data_dist/matrix/matrix.h"

int dtd_init                    = 0;      /**< Indicates init of dtd environment */
int dtd_window_size             = 4000;   /**< Default window size */
int dtd_threshold_size          = 2000;   /**< Default threshold size of tasks for master thread to wait on */
static int task_hash_table_size = 104729; /**< Default task hash table size */
static int tile_hash_table_size = 104729; /**< Default tile hash table size */
static int parsec_dtd_no_of_arenas = 16;

int dump_traversal_info; /**< For printing traversal info */
int dump_function_info; /**< For printing function_structure info */
int insert_task_trace_keyin;
int insert_task_trace_keyout;
int hashtable_trace_keyin;
int hashtable_trace_keyout;

parsec_arena_t **parsec_dtd_arenas;

extern parsec_sched_module_t *current_scheduler;

/* Global mempool for all the parsec handles that will be created for a run */
parsec_mempool_t *handle_mempool = NULL;

/* Global mempool for all tiles */
parsec_mempool_t *parsec_dtd_tile_mempool = NULL;

/**
 * All the static functions should be declared before being defined.
 */
static int
hook_of_dtd_task(parsec_execution_unit_t *context,
                      parsec_execution_context_t *this_task);

static void
iterate_successors_of_dtd_task(parsec_execution_unit_t *eu,
                               const parsec_execution_context_t *this_task,
                               uint32_t action_mask,
                               parsec_ontask_function_t *ontask,
                               void *ontask_arg);

static int
release_deps_of_dtd(parsec_execution_unit_t *,
                    parsec_execution_context_t *,
                    uint32_t, parsec_remote_deps_t *);

static parsec_hook_return_t
complete_hook_of_dtd(parsec_execution_unit_t *,
                     parsec_execution_context_t *);

static uint32_t hash_key (uintptr_t key, void *data);

inline int parsec_dtd_task_is_local(parsec_dtd_task_t *task) { return task->rank == task->super.parsec_handle->context->my_rank;}
inline int parsec_dtd_task_is_remote(parsec_dtd_task_t *task) { return !parsec_dtd_task_is_local(task);}

/* Copied from parsec/scheduling.c, will need to be exposed */
#define TIME_STEP 5410
#define MIN(x, y) ( (x)<(y)?(x):(y) )
static inline unsigned long exponential_backoff(uint64_t k)
{
    unsigned int n = MIN( 64, k );
    unsigned int r = (unsigned int) ((double)n * ((double)rand()/(double)RAND_MAX));
    return r * TIME_STEP;
}

void
parsec_detach_all_dtd_handles_from_context( parsec_context_t *context )
{
    /* Here we wait on all dtd handles registered with us */
    if( NULL != context->object_array && 0 < context->object_array_occupied ) {
        int iterator;
        for( iterator = 0; iterator < context->object_array_occupied; iterator++ ) {
            parsec_dtd_handle_t *handle = (parsec_dtd_handle_t *)context->object_array[iterator];
            if( NULL != handle ) {
                if( handle->enqueue_flag ) {
                    parsec_handle_update_runtime_nbtask( (parsec_handle_t *)handle, -1 );
                }
                context->object_array[iterator] = NULL;
            }
        }
        context->object_array_occupied = 0;
    }
}

void
parsec_dtd_attach_handle_to_context( parsec_handle_t  *parsec_handle,
                                     parsec_context_t *parsec_context )
{
    if( (NULL == parsec_context->object_array) ||
        (parsec_context->object_array_occupied >= parsec_context->object_array_size) ) {
        parsec_context->object_array_size <<= 1;
        parsec_context->object_array = (parsec_handle_t**)realloc(parsec_context->object_array, parsec_context->object_array_size * sizeof(parsec_handle_t*) );
        /* NULLify all the new elements */
        for( int32_t i = (parsec_context->object_array_size>>1); i < parsec_context->object_array_size;
             parsec_context->object_array[i++] = NULL );
    }
    parsec_context->object_array[parsec_context->object_array_occupied++] = parsec_handle;
}

/* enqueue wrapper for dtd */
int
parsec_dtd_enqueue_handle( parsec_handle_t *parsec_handle, void *parsec_context )
{
    (void)parsec_context;

    parsec_dtd_handle_t *handle = (parsec_dtd_handle_t *)parsec_handle;
    handle->super.nb_pending_actions = 1;  /* For the future tasks that will be inserted */
    handle->super.nb_tasks -= PARSEC_RUNTIME_RESERVED_NB_TASKS;
    handle->super.nb_tasks += 1;  /* For the bounded window, starting with +1 task */
    handle->enqueue_flag    = 1;

    parsec_dtd_handle_retain((parsec_handle_t *)parsec_handle);

    /* Attaching the reference of this handle to the parsec context */
    parsec_dtd_attach_handle_to_context( parsec_handle, parsec_handle->context );
    return 0;
}

/* To create object of class parsec_dtd_task_t that inherits parsec_execution_context_t
 * class
 */
OBJ_CLASS_INSTANCE(parsec_dtd_task_t, parsec_execution_context_t,
                   NULL, NULL);

/* To create object of class .list_itemdtd_tile_t that inherits parsec_list_item_t
 * class
 */
OBJ_CLASS_INSTANCE(parsec_dtd_tile_t, parsec_list_item_t,
                   NULL, NULL);

/* To create object of class parsec_handle_t that inherits parsec_list_t
 * class
 */
OBJ_CLASS_INSTANCE(parsec_handle_t, parsec_list_item_t,
                   NULL, NULL);

/***************************************************************************//**
 *
 * Constructor of PaRSEC's DTD handle.
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to handle which will be constructed
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
void parsec_dtd_handle_constructor(parsec_dtd_handle_t *parsec_handle)
{
    int i;

    parsec_handle->startup_list = (parsec_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(parsec_execution_context_t*));

    parsec_handle->function_counter = 0;

    parsec_handle->two_hash_table = (parsec_dtd_two_hash_table_t *) malloc(sizeof(parsec_dtd_two_hash_table_t));
    parsec_atomic_unlock(&parsec_handle->two_hash_table->atomic_lock);

    parsec_handle->two_hash_table->task_and_rem_dep_h_table = OBJ_NEW(hash_table_t);
    hash_table_init(parsec_handle->two_hash_table->task_and_rem_dep_h_table,
                    offsetof(dtd_hash_table_pointer_item_t, ht_item),
                    task_hash_table_size,
                    &hash_key,
                    parsec_handle->two_hash_table->task_and_rem_dep_h_table);
    parsec_handle->function_h_table = OBJ_NEW(hash_table_t);
    hash_table_init(parsec_handle->function_h_table,
                    offsetof(dtd_hash_table_pointer_item_t, ht_item),
                    PARSEC_DTD_NB_FUNCTIONS,
                    &hash_key,
                    parsec_handle->function_h_table);

    parsec_handle->super.startup_hook    = parsec_dtd_startup;
    parsec_handle->super.destructor      = (parsec_destruct_fn_t)parsec_dtd_handle_destruct;
    parsec_handle->super.functions_array = (const parsec_function_t **) malloc( PARSEC_DTD_NB_FUNCTIONS * sizeof(parsec_function_t *));

    for( i = 0; i < PARSEC_DTD_NB_FUNCTIONS; i++ ) {
        parsec_handle->super.functions_array[i] = NULL;
    }

    parsec_handle->super.dependencies_array  = calloc(PARSEC_DTD_NB_FUNCTIONS, sizeof(parsec_dependencies_t *));

#if defined(PARSEC_PROF_TRACE)
    parsec_handle->super.profiling_array     = calloc (2 * PARSEC_DTD_NB_FUNCTIONS , sizeof(int));
#endif /* defined(PARSEC_PROF_TRACE) */

    /* Initializing hash_table_bucket mempool */
    parsec_handle->hash_table_bucket_mempool = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct( parsec_handle->hash_table_bucket_mempool,
                              NULL, sizeof(dtd_hash_table_pointer_item_t),
                              offsetof(dtd_hash_table_pointer_item_t, mempool_owner),
                              1/* no. of threads*/ );
}

/***************************************************************************//**
 *
 * Destructor of PaRSEC's DTD handle.
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to handle which will be destroyed
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
void
parsec_dtd_handle_destructor
(parsec_dtd_handle_t *parsec_handle)
{
    int i;
#if defined(PARSEC_PROF_TRACE)
    free((void *)parsec_handle->super.profiling_array);
#endif /* defined(PARSEC_PROF_TRACE) */

    free(parsec_handle->super.functions_array);
    parsec_handle->super.nb_functions = 0;

    /* Destroy the data repositories for this object */
    for (i = 0; i <PARSEC_DTD_NB_FUNCTIONS; i++) {
        parsec_destruct_dependencies(parsec_handle->super.dependencies_array[i]);
        parsec_handle->super.dependencies_array[i] = NULL;
    }

    free(parsec_handle->super.dependencies_array);
    parsec_handle->super.dependencies_array = NULL;

    /* Unregister the handle from the devices */
    for (i = 0; i < (int)parsec_nb_devices; i++) {
        if (!(parsec_handle->super.devices_mask & (1 << i)))
            continue;
        parsec_handle->super.devices_mask ^= (1 << i);
        parsec_device_t *device = parsec_devices_get(i);
        if ((NULL == device) || (NULL == device->device_handle_unregister))
            continue;
        if (PARSEC_SUCCESS != device->device_handle_unregister(device, &parsec_handle->super))
            continue;
    }

    /* dtd_handle specific */
    parsec_mempool_destruct(parsec_handle->hash_table_bucket_mempool);
    free(parsec_handle->hash_table_bucket_mempool);
    free(parsec_handle->startup_list);

    hash_table_fini(parsec_handle->two_hash_table->task_and_rem_dep_h_table);
    free(parsec_handle->two_hash_table);

    hash_table_fini(parsec_handle->function_h_table);
}

/* To create object of class parsec_dtd_handle_t that inherits parsec_handle_t
 * class
 */
OBJ_CLASS_INSTANCE(parsec_dtd_handle_t, parsec_handle_t,
                   parsec_dtd_handle_constructor, parsec_dtd_handle_destructor);


/* **************************************************************************** */
/**
 * Init function of Dynamic Task Discovery Interface.
 *
 * Here a global(per node/process) handle mempool for PaRSEC's DTD handle
 * is constructed. The mca_params passed to the runtime are also scanned
 * and set here.
 * List of mca options available for DTD interface are:
 *  - dtd_traversal_info (default=0 off):   This prints the DAG travesal
 *                                          info for each node in the DAG.
 *  - dtd_function_info (default=0 off):    This prints the DOT compliant
 *                                          output to check the relationship
 *                                          between the master structure,
 *                                          which represent each task class.
 *  - dtd_tile_hash_size (default=104729):  This sets the tile hash table
 *                                          size.
 *  - dtd_task_hash_size (default=11):      This sets the size of task hash
 *                                          table.
 *  - dtd_window_size (default:2048):       To set the window size for the
 *                                          execution.
 *  - dtd_threshold_size (default:2048):    This sets the threshold task
 *                                          size up to which the master
 *                                          thread will wait before going
 *                                          back and inserting task into the
 *                                          engine.
 * @ingroup DTD_INTERFACE
 */
void
parsec_dtd_init()
{
    parsec_dtd_handle_t  *parsec_handle;

    /* Registering mca param for printing out traversal info */
    (void)parsec_mca_param_reg_int_name("dtd", "traversal_info",
                                       "Show graph traversal info",
                                       false, false, 0, &dump_traversal_info);

    /* Registering mca param for printing out function_structure info */
    (void)parsec_mca_param_reg_int_name("dtd", "function_info",
                                       "Show master structure info",
                                       false, false, 0, &dump_function_info);

    /* Registering mca param for tile hash table size */
    (void)parsec_mca_param_reg_int_name("dtd", "tile_hash_size",
                                       "Registers the supplied size overriding the default size of tile hash table",
                                       false, false, tile_hash_table_size, &tile_hash_table_size);

    /* Registering mca param for task hash table size */
    (void)parsec_mca_param_reg_int_name("dtd", "task_hash_size",
                                       "Registers the supplied size overriding the default size of task hash table",
                                       false, false, task_hash_table_size, &task_hash_table_size);

    /* Registering mca param for window size */
    (void)parsec_mca_param_reg_int_name("dtd", "window_size",
                                       "Registers the supplied size overriding the default size of window size",
                                       false, false, dtd_window_size, &dtd_window_size);

    /* Registering mca param for threshold size */
    (void)parsec_mca_param_reg_int_name("dtd", "threshold_size",
                                       "Registers the supplied size overriding the default size of threshold size",
                                       false, false, dtd_threshold_size, &dtd_threshold_size);

    handle_mempool          = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct( handle_mempool,
                              OBJ_CLASS(parsec_dtd_handle_t), sizeof(parsec_dtd_handle_t),
                              offsetof(parsec_dtd_handle_t, mempool_owner),
                              1/* no. of threads*/ );

    parsec_handle = (parsec_dtd_handle_t *)parsec_thread_mempool_allocate(handle_mempool->thread_mempools);
    parsec_thread_mempool_free( handle_mempool->thread_mempools, parsec_handle );

    /* Initializing the tile mempool and attaching it to the parsec_handle */
    parsec_dtd_tile_mempool = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct( parsec_dtd_tile_mempool,
                              OBJ_CLASS(parsec_dtd_tile_t), sizeof(parsec_dtd_tile_t),
                              offsetof(parsec_dtd_tile_t, mempool_owner),
                              1/* no. of threads*/ );

    int i;
    parsec_dtd_arenas = (parsec_arena_t **) malloc(parsec_dtd_no_of_arenas * sizeof(parsec_arena_t *));
    for (i = 0; i < parsec_dtd_no_of_arenas; i++) {
        parsec_dtd_arenas[i] = (parsec_arena_t *) calloc(1, sizeof(parsec_arena_t));
    }
}

/* **************************************************************************** */
/**
 * Fini function of Dynamic Task Discovery Interface.
 *
 * The global mempool of parsec_dtd_handle is destroyed here.
 *
 * @ingroup DTD_INTERFACE
 */
void
parsec_dtd_fini()
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert(handle_mempool != NULL);
#endif
    int i;
    for( i = 0; i < parsec_dtd_no_of_arenas; i++ ) {
        if (parsec_dtd_arenas[i] != NULL) {
            free(parsec_dtd_arenas[i]);
            parsec_dtd_arenas[i] = NULL;
        }
    }
    free(parsec_dtd_arenas);

    parsec_mempool_destruct( parsec_dtd_tile_mempool );
    free( parsec_dtd_tile_mempool );

    parsec_mempool_destruct( handle_mempool );
    free( handle_mempool );
}

/**
 * Body of function to copy data to the original matrix, coming from remote tasks.
 */
int
parsec_dtd_copy_data_to_matrix(parsec_execution_unit_t *eu,
                               parsec_execution_context_t *this_task)
{
    (void)eu;
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_tile_t *tile = (FLOW_OF(current_task, 0))->tile;

    assert(tile != NULL);

#if defined(DISTRIBUTED)
    if( current_task->super.data[0].data_in != tile->data_copy ) {
        int16_t arena_index = (FLOW_OF(current_task, 0))->arena_index;
        parsec_dep_data_description_t data;
        data.data   = current_task->super.data[0].data_in;
        data.arena  = parsec_dtd_arenas[arena_index];
        data.layout = data.arena->opaque_dtt;
        data.count  = 1;
        data.displ  = 0;
        parsec_remote_dep_memcpy(this_task->parsec_handle,
                                 tile->data_copy, current_task->super.data[0].data_in, &data);
    }
#endif

    parsec_dtd_release_local_task( current_task );

    return 0;
}

void
parsec_dtd_data_flush(parsec_handle_t *parsec_handle, parsec_dtd_tile_t *tile)
{
    assert(tile->flushed == NOT_FLUSHED);
    parsec_dtd_tile_retain(tile);
    parsec_insert_task( parsec_handle, parsec_dtd_copy_data_to_matrix, 0, "Copy_data_in_dist",
                        PASSED_BY_REF,    tile, INOUT | AFFINITY,
                        0 );
    tile->flushed = FLUSHED;
    parsec_dtd_tile_remove( tile->ddesc, tile->key );
    parsec_dtd_tile_release( tile );
}

void
parsec_dtd_data_flush_all(parsec_handle_t *parsec_handle, parsec_ddesc_t *ddesc)
{
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)parsec_handle;
    hash_table_t *hash_table   = (hash_table_t *)ddesc->tile_h_table;

    PINS(parsec_dtd_handle->super.context->virtual_processes[0]->execution_units[0], DATA_FLUSH_BEGIN, NULL);

    hash_table_for_all( hash_table, (hash_elem_fct_t)parsec_dtd_data_flush, parsec_handle);

    PINS(parsec_dtd_handle->super.context->virtual_processes[0]->execution_units[0], DATA_FLUSH_END, NULL);
}

/* **************************************************************************** */
/**
 * Master thread calls this to join worker threads in executing tasks.
 *
 * Master thread, at the end of each window, calls this function to
 * join the worker thread(s) in executing tasks and takes a break
 * from inserting tasks. It(master thread) remains in this function
 * till the total number of pending tasks in the engine reaches a
 * threshold (see dtd_threshold_size). It goes back to inserting task
 * once the number of pending tasks in the engine reaches the
 * threshold size.
 *
 * @param[in]   context
 *                  The PaRSEC context (pointer to the runtime instance)
 * @param[in]   parsec_handle
 *                  PaRSEC dtd handle
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
parsec_execute_and_come_back( parsec_context_t *context,
                              parsec_handle_t *parsec_handle,
                              int task_threshold_count )
{
    uint64_t misses_in_a_row;
    parsec_execution_unit_t* eu_context = context->virtual_processes[0]->execution_units[0];
    parsec_execution_context_t* exec_context;
    int rc, nbiterations = 0, distance;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    /* Checking if the context has been started or not */
    /* The master thread might not have to trigger the barrier if the other
     * threads have been activated by a previous start.
     */
    if( !(PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags) ) {
        (void)parsec_remote_dep_on(context);
        /* Mark the context so that we will skip the initial barrier during the _wait */
        context->flags |= PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE;
        /* Wake up the other threads */
        parsec_barrier_wait( &(context->barrier) );
    }

    /* we wait for only all the tasks inserted in the handle and not for all the communication
     * invoked by those tasks.
     */
    while(parsec_handle->nb_tasks > task_threshold_count) {
        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }

        exec_context = current_scheduler->module.select(eu_context, &distance);

        if( exec_context != NULL ) {
            PINS(eu_context, SELECT_END, exec_context);
            misses_in_a_row = 0;

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
            {
                uint32_t my_idx = parsec_atomic_inc_32b(&sched_priority_trace_counter);
                if(my_idx < PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
                    sched_priority_trace[my_idx].step = eu_context->sched_nb_tasks_done++;
                    sched_priority_trace[my_idx].thread_id = eu_context->th_id;
                    sched_priority_trace[my_idx].vp_id     = eu_context->virtual_process->vp_id;
                    sched_priority_trace[my_idx].priority  = exec_context->priority;
                }
            }
#endif

            rc = PARSEC_HOOK_RETURN_DONE;
            if(exec_context->status <= PARSEC_TASK_STATUS_PREPARE_INPUT) {
                PINS(eu_context, PREPARE_INPUT_BEGIN, exec_context);
                rc = exec_context->function->prepare_input(eu_context, exec_context);
                PINS(eu_context, PREPARE_INPUT_END, exec_context);
            }
            switch(rc) {
            case PARSEC_HOOK_RETURN_DONE: {
                if(exec_context->status <= PARSEC_TASK_STATUS_HOOK) {
                    rc = __parsec_execute( eu_context, exec_context );
                }
                /* We're good to go ... */
                switch(rc) {
                case PARSEC_HOOK_RETURN_DONE:    /* This execution succeeded */
                    exec_context->status = PARSEC_TASK_STATUS_COMPLETE;
                    __parsec_complete_execution( eu_context, exec_context );
                    break;
                case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
                    exec_context->status = PARSEC_TASK_STATUS_HOOK;
                    if(0 == exec_context->priority) {
                        SET_LOWEST_PRIORITY(exec_context, parsec_execution_context_priority_comparator);
                    } else
                        exec_context->priority /= 10;  /* demote the task */
                    PARSEC_LIST_ITEM_SINGLETON(exec_context);
                    __parsec_schedule(eu_context, exec_context, distance + 1);
                    exec_context = NULL;
                    break;
                case PARSEC_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                                 * even try to change it's state, the completion
                                                 * will be triggered asynchronously. */
                    break;
                case PARSEC_HOOK_RETURN_NEXT:    /* Try next variant [if any] */
                case PARSEC_HOOK_RETURN_DISABLE: /* Disable the device, something went wrong */
                case PARSEC_HOOK_RETURN_ERROR:   /* Some other major error happened */
                    assert( 0 ); /* Internal error: invalid return value */
                }
                nbiterations++;
                break;
            }
            case PARSEC_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                             * even try to change it's state, the completion
                                             * will be triggered asynchronously. */
                break;
            case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
                if(0 == exec_context->priority) {
                    SET_LOWEST_PRIORITY(exec_context, parsec_execution_context_priority_comparator);
                } else
                    exec_context->priority /= 10;  /* demote the task */
                PARSEC_LIST_ITEM_SINGLETON(exec_context);
                __parsec_schedule(eu_context, exec_context, distance + 1);
                exec_context = NULL;
                break;
            default:
                assert( 0 ); /* Internal error: invalid return value for data_lookup function */
            }

            // subsequent select begins
            PINS(eu_context, SELECT_BEGIN, NULL);
        } else {
            misses_in_a_row++;
        }
    }
}

/* **************************************************************************** */
/**
 * Function to call when PaRSEC context should wait on a specific handle.
 *
 * This function is called to execute a task collection attached to the
 * handle by the user. This function will schedule all the initially ready
 * tasks in the engine and return when all the pending tasks are executed.
 * Users should call this function everytime they insert a bunch of tasks.
 * Users can call this function once per handle.
 *
 * @param[in]       parsec
 *                      The PaRSEC context
 * @param[in,out]   parsec_handle
 *                      PaRSEC dtd handle
 *
 * @ingroup         DTD_INTERFACE
 */
int
parsec_dtd_handle_wait( parsec_context_t *parsec,
                        parsec_handle_t  *parsec_handle )
{
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)parsec_handle;
    parsec_dtd_schedule_tasks( parsec_dtd_handle );
    parsec_dtd_handle->wait_func( parsec, parsec_handle );
    return 0;
}

static void
parsec_dtd_handle_wait_func( parsec_context_t *parsec,
                             parsec_handle_t  *parsec_handle )
{
    (void)parsec;
    parsec_execute_and_come_back( parsec_handle->context, parsec_handle, 1 );
}

/* **************************************************************************** */
/**
 * This function unpacks the parameters of a task
 *
 * Unpacks all the parameters of a task, the variables(in which the
 * actual values will be copied) are passed from the body(function that does
 * what this_task is supposed to compute) of this task and the parameters of each
 * task is copied back on the passed variables
 *
 * @param[in]   this_task
 *                  The task we are trying to unpack the parameters for
 * @param[out]  variadic paramter
 *                  The variables where the paramters will be unpacked
 *
 * @ingroup DTD_INTERFACE
 */
void
parsec_dtd_unpack_args(parsec_execution_context_t *this_task, ...)
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_task_param_t *current_param = GET_HEAD_OF_PARAM_LIST(current_task);
    int next_arg;
    int i = 0;
    void **tmp;
    va_list arguments;
    va_start(arguments, this_task);
    next_arg = va_arg(arguments, int);

    while( current_param != NULL && next_arg != 0) {
        tmp = va_arg(arguments, void**);
        if(UNPACK_VALUE == next_arg) {
            *tmp = current_param->pointer_to_tile;
        }else if (UNPACK_DATA == next_arg) {
            /* Let's return directly the usable pointer to the user */
            *tmp = PARSEC_DATA_COPY_GET_PTR(this_task->data[i].data_in);
            i++;
        }else if (UNPACK_SCRATCH == next_arg) {
            *tmp = current_param->pointer_to_tile;
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}

#if defined(PARSEC_PROF_TRACE)
/* **************************************************************************** */
/**
 * This function returns a unique color
 *
 * This function takes a index and a colorspace and queries unique_color()
 * for a hex color code and prepends "fills:" to that color string.
 *
 * @param[in]   index
 * @param[in]   colorspace
 * @return
 *                  String containing "fill:" followed by color code returned
 *                  by unique_color()
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static inline char*
fill_color(int index, int colorspace)
{
    char *str, *color;
    str = (char *)calloc(12,sizeof(char));
    color = unique_color(index, colorspace);
    snprintf(str,12,"fill:%s",color+1); /* need to remove the prepended '#' */
    free(color);
    return str;
}

/* **************************************************************************** */
/**
 * This function adds info about a task class into a global dictionary
 * used for profiling.
 *
 * @param[in]   __parsec_handle
 *                  The pointer to the DTD handle
 * @param[in]   function_id
 *                  Id of master structure representing a
 *                  task class
 * @param[in]   name
 *                  Name of the task class
 * @param[in]   flow_count
 *                  Total number of flows of the task class
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_add_profiling_info( parsec_handle_t *parsec_handle,
                    int function_id, char* name )
{
    char *str = fill_color(function_id, PARSEC_DTD_NB_FUNCTIONS);
    parsec_profiling_add_dictionary_keyword(name, str,
                                           sizeof(parsec_profile_ddesc_info_t),
                                           PARSEC_PROFILE_DDESC_INFO_CONVERTOR,
                                           (int *) &parsec_handle->profiling_array[0 +
                                                                                  2 *
                                                                                  function_id
                                                                                  /* start key */
                                                                                  ],
                                           (int *) &parsec_handle->profiling_array[1 +
                                                                                  2 *
                                                                                  function_id
                                                                                  /*  end key */
                                                                                  ]);
    free(str);
}

void
parsec_dtd_add_profiling_info_generic( parsec_handle_t *parsec_handle,
                            char* name,
                            int *keyin, int *keyout)
{
    (void)parsec_handle;
    char *str = fill_color(*keyin, PARSEC_DTD_NB_FUNCTIONS);
    parsec_profiling_add_dictionary_keyword(name, str,
                                           sizeof(parsec_profile_ddesc_info_t),
                                           PARSEC_PROFILE_DDESC_INFO_CONVERTOR,
                                           keyin,
                                           keyout);
    free(str);
}

#endif /* defined(PARSEC_PROF_TRACE) */

/* **************************************************************************** */

/**
 * This function produces a hash from a key and a hash_table_t
 *
 * This function returns a hash for a key. The hash is produced
 * by the following operation key % size
 *
 * @param[in]   key
 *                  The key to be hashed
 * @param[in]   ht
 *                  the hash table
 * @return
 *              The hash for the key provided
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static uint32_t hash_key (uintptr_t key, void *data)
{
    int size = ((hash_table_t*)data)->size;
    uint32_t hash_val = ((uint64_t)key) % size;
    return hash_val;
}

void
parsec_dtd_insert_task( parsec_dtd_handle_t *parsec_handle,
                        uint64_t             key,
                        void                *value )
{
    dtd_hash_table_pointer_item_t *item = (dtd_hash_table_pointer_item_t *)parsec_thread_mempool_allocate(parsec_handle->hash_table_bucket_mempool->thread_mempools);

    hash_table_t *hash_table = parsec_handle->two_hash_table->task_and_rem_dep_h_table;

    item->ht_item.key   = key;
    item->mempool_owner = parsec_handle->hash_table_bucket_mempool->thread_mempools;
    item->value         = (void *)value;

    hash_table_nolock_insert( hash_table, &item->ht_item );
}

void
parsec_dtd_insert_remote_dep( parsec_dtd_handle_t *parsec_handle,
                              uint64_t             key,
                              void                *value )
{
    parsec_dtd_insert_task( parsec_handle, key, value );
}

void
parsec_dtd_remove_task( parsec_dtd_handle_t  *parsec_handle,
                        uint64_t              key )
{
    hash_table_t *hash_table = parsec_handle->two_hash_table->task_and_rem_dep_h_table;

    hash_table_nolock_remove( hash_table, key );
}

void
parsec_dtd_remove_remote_deps( parsec_dtd_handle_t  *parsec_handle,
                               uint64_t              key )
{
    parsec_dtd_remove_task( parsec_handle, key );
}

void *
parsec_dtd_find_task( parsec_dtd_handle_t *parsec_handle,
                      uint64_t             key )
{
    hash_table_t *hash_table = parsec_handle->two_hash_table->task_and_rem_dep_h_table;

    dtd_hash_table_pointer_item_t *item = (dtd_hash_table_pointer_item_t *)hash_table_nolock_find( hash_table, key );
    if( NULL == item ) return NULL;
    else return item->value;
}

void *
parsec_dtd_find_remote_dep( parsec_dtd_handle_t *parsec_handle,
                            uint64_t             key )
{
    return parsec_dtd_find_task( parsec_handle, key );
}

void *
parsec_dtd_find_and_remove_task( parsec_dtd_handle_t *parsec_handle,
                                 uint64_t             key )
{
    hash_table_t *hash_table = parsec_handle->two_hash_table->task_and_rem_dep_h_table;
    void *value;

    dtd_hash_table_pointer_item_t *item = (dtd_hash_table_pointer_item_t *)hash_table_nolock_find( hash_table, key );
    if( NULL == item ) return NULL;
    else {
        parsec_dtd_remove_task( parsec_handle, key );
        value = item->value;
        parsec_thread_mempool_free( parsec_handle->hash_table_bucket_mempool->thread_mempools, item );
        return value;
    }
}

void *
parsec_dtd_find_and_remove_remote_dep( parsec_dtd_handle_t *parsec_handle,
                                       uint64_t             key )
{
    return parsec_dtd_find_and_remove_task( parsec_handle, key );
}

/* **************************************************************************** */
/**
 * This function inserts master structure in hash table
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The function-pointer to the body of task-class
 *                      is treated as the key
 * @param[in]       value
 *                      The pointer to the master structure
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_function_insert( parsec_dtd_handle_t   *parsec_handle,
                            uint64_t  key,
                            parsec_dtd_function_t *value )
{
    dtd_hash_table_pointer_item_t *item  =  (dtd_hash_table_pointer_item_t *)parsec_thread_mempool_allocate(parsec_handle->hash_table_bucket_mempool->thread_mempools);

    hash_table_t *hash_table          =  parsec_handle->function_h_table;

    item->ht_item.key   = (uint64_t)key;
    item->mempool_owner = parsec_handle->hash_table_bucket_mempool->thread_mempools;
    item->value         = (void *)value;

    hash_table_insert ( hash_table, &item->ht_item );
}

/* **************************************************************************** */
/**
 * This function removes master structure from hash table
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The function-pointer to the body of task-class
 *                      is treated as the key
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_function_remove( parsec_dtd_handle_t  *parsec_handle,
                            uint64_t key )
{
    hash_table_t *hash_table    =  parsec_handle->function_h_table;

    hash_table_remove( hash_table, (uint64_t)key );
}

/* **************************************************************************** */
/**
 * This function searches for master-structure in hash table
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The function-pointer to the body of task-class
 *                      is treated as the key
 * @return
 *                  The pointer to the master-structure is returned if found,
 *                  NULL otherwise
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void *
parsec_dtd_function_find_internal( parsec_dtd_handle_t  *parsec_handle,
                                   uint64_t key )
{
    hash_table_t *hash_table      =  parsec_handle->function_h_table;

    return hash_table_nolock_find ( hash_table, key );
}

/* **************************************************************************** */
/**
 * This function internal API to search for master-structure in hash table
 *
 * @see             parsec_dtd_function_find_internal()
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The function-pointer to the body of task-class
 *                      is treated as the key
 * @return
 *                  The pointer to the master-structure is returned if found,
 *                  NULL otherwise
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
parsec_dtd_function_t *
parsec_dtd_function_find( parsec_dtd_handle_t  *parsec_handle,
                          uint64_t key )
{
    dtd_hash_table_pointer_item_t *item = parsec_dtd_function_find_internal( parsec_handle, key );
    if( item != NULL ) {
        return (parsec_dtd_function_t *)item->value;
    }
    return NULL;
}

/* **************************************************************************** */
/**
 * This function inserts DTD tile into tile hash table
 *
 * The actual key for each tile is formed by OR'ing the last 32 bits of
 * address of the data descriptor (ddesc) with the 32 bit key. So, the actual
 * key is of 64 bits,
 * first 32 bits: last 32 bits of ddesc pointer + last 32 bits: 32 bit key.
 * This is done as PaRSEC key for tiles are unique per ddesc.
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The key of the tile
 * @param[in]       tile
 *                      Pointer to the tile structure
 * @param[in]       ddesc
 *                      Pointer to the ddesc the tile belongs to
 *
 * @ingroup         DTD_ITERFACE_INTERNAL
 */
void
parsec_dtd_tile_insert( uint64_t key,
                        parsec_dtd_tile_t *tile,
                        parsec_ddesc_t    *ddesc )
{
    hash_table_t *hash_table = (hash_table_t *)ddesc->tile_h_table;

    tile->ht_item.key = (uint64_t)key;

    hash_table_insert( hash_table, &tile->ht_item );
}

/* **************************************************************************** */
/**
 * This function removes DTD tile from tile hash table
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The key of the tile
 * @param[in]       ddesc
 *                      Pointer to the ddesc the tile belongs to
 *
 * @ingroup         DTD_ITERFACE_INTERNAL
 */
void
parsec_dtd_tile_remove( parsec_ddesc_t *ddesc, uint64_t key )
{
    hash_table_t *hash_table = (hash_table_t *)ddesc->tile_h_table;

    hash_table_remove( hash_table, (uint64_t)key );
}

/* **************************************************************************** */
/**
 * This function searches a DTD tile in the tile hash table
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The key of the tile
 * @param[in]       ddesc
 *                      Pointer to the ddesc the tile belongs to
 *
 * @ingroup         DTD_ITERFACE_INTERNAL
 */
parsec_dtd_tile_t *
parsec_dtd_tile_find( parsec_ddesc_t *ddesc, uint64_t key )
{
    hash_table_t *hash_table   = (hash_table_t *)ddesc->tile_h_table;
    assert(hash_table != NULL);
    parsec_dtd_tile_t *tile  = (parsec_dtd_tile_t *)hash_table_nolock_find( hash_table, (uint64_t)key );

    return tile;
}

/* **************************************************************************** */
/**
 * This function releases tile from tile hash table and pushes them back in
 * tile mempool
 *
 * Here we at first we try to remove the tile from the tile hash table,
 * if we are successful we push the tile back in the tile mempool.
 * This function is called for each flow of a task, after the task is
 * completed.
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       tile
 *                      Tile to be released
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_tile_release( parsec_dtd_tile_t *tile )
{
    assert(tile->super.super.obj_reference_count>1);
    if( 1 == parsec_atomic_add_32b( &tile->super.super.obj_reference_count, -1 ) ) {
        assert(tile->flushed == FLUSHED);
        parsec_thread_mempool_free( parsec_dtd_tile_mempool->thread_mempools, tile );
        tile = NULL;
    }
}

void
parsec_dtd_tile_retain( parsec_dtd_tile_t *tile )
{
    OBJ_RETAIN(tile);
}

/* **************************************************************************** */
/**
 * This function releases the master-structure and pushes them back in mempool
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The function pointer to the body of the task class
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_function_release( parsec_dtd_handle_t *parsec_handle,
                             uint64_t key )
{
    dtd_hash_table_pointer_item_t *item = parsec_dtd_function_find_internal( parsec_handle, key );
#if defined(PARSEC_DEBUG_PARANOID)
    assert (item != NULL);
#endif
    parsec_dtd_function_remove( parsec_handle, key );
    parsec_thread_mempool_free( parsec_handle->hash_table_bucket_mempool->thread_mempools, item );
}

void
parsec_dtd_ddesc_init( parsec_ddesc_t *ddesc )
{
    ddesc->tile_h_table = OBJ_NEW(hash_table_t);
    hash_table_init( ddesc->tile_h_table,
                     offsetof(parsec_dtd_tile_t, ht_item),
                     tile_hash_table_size,
                     &hash_key,
                     ddesc->tile_h_table);
}

void
parsec_dtd_ddesc_fini( parsec_ddesc_t *ddesc )
{
    hash_table_fini( ddesc->tile_h_table );
}

/* **************************************************************************** */
/**
 * Function to recover tiles inserted by insert_task()
 *
 * This function search for a tile if already inserted in the system,
 * and if not returns the freshly created tile.
 *
 * @param[in,out]   parsec handle
 *                      Pointer to the DTD handle
 * @param[in]       ddesc
 *                      Data descriptor
 * @param[in]       i,j
 *                      The co-ordinates of the tile in the matrix
 * @return
 *                  The tile representing the data in specified co-ordinate
 *
 * @ingroup         DTD_INTERFACE
 */
parsec_dtd_tile_t*
parsec_dtd_tile_of( parsec_ddesc_t *ddesc, parsec_data_key_t key )
{
    parsec_dtd_tile_t *tile = parsec_dtd_tile_find ( ddesc, (uint64_t)key );
    if( NULL == tile ) {
        /* Creating Tile object */
        tile = (parsec_dtd_tile_t *) parsec_thread_mempool_allocate( parsec_dtd_tile_mempool->thread_mempools );
        tile->ddesc                 = ddesc;
        tile->arena_index           = -1;
        tile->key                   = (uint64_t) 0x00000000 | key;
        tile->rank                  = ddesc->rank_of_key(ddesc, tile->key);
        tile->flushed               = NOT_FLUSHED;
        if( tile->rank == (int)ddesc->myrank ) {
            tile->data_copy         = (ddesc->data_of_key(ddesc, tile->key))->device_copies[0];
#if defined(PARSEC_HAVE_CUDA)
            tile->data_copy->readers = 0;
#endif
        } else {
            tile->data_copy         = NULL;
        }

        SET_LAST_ACCESSOR(tile);

        parsec_dtd_tile_retain(tile);
        parsec_dtd_tile_insert( tile->key,
                                tile, ddesc );
    }
    assert(tile->flushed == NOT_FLUSHED);
#if defined(PARSEC_DEBUG_PARANOID)
    assert(tile->super.super.obj_reference_count > 0);
#endif
    return tile;
}

/* **************************************************************************** */
/**
 * This function acts as the hook to connect the PaRSEC task with
 * the actual task.
 *
 * The function users passed while inserting task in PaRSEC is called
 * in this procedure.
 * Called internally by the scheduler
 *
 * @param[in]   context
 *                  The execution unit
 * @param[in]   this_task
 *                  The DTD task to be executed
 * @return
 *              Returns PARSEC_HOOK_RETURN_DONE if the task was
 *              successfully executed, anything else otherwise
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static int
hook_of_dtd_task( parsec_execution_unit_t    *context,
                  parsec_execution_context_t *this_task )
{
    parsec_dtd_task_t *dtd_task = (parsec_dtd_task_t*)this_task;
    assert(parsec_dtd_task_is_local(dtd_task));
    int rc = 0;

    PARSEC_TASK_PROF_TRACE(context->eu_profile,
                          this_task->parsec_handle->profiling_array[2 * this_task->function->function_id],
                          this_task);

#if !defined(PARSEC_PROF_DRY_BODY)
#if !defined(PARSEC_DRY_RUN)
    rc = ((parsec_dtd_function_t *)(dtd_task->super.function))->fpointer(context, this_task);
#endif /* !defined(PARSEC_DRY_RUN) */
#endif

    return rc;
}

/* chores and parsec_function_t structure initialization */
static const __parsec_chore_t dtd_chore[] = {
    {.type      = PARSEC_DEV_CPU,
     .evaluate  = NULL,
     .hook      = hook_of_dtd_task },
    {.type      = PARSEC_DEV_NONE,
     .evaluate  = NULL,
     .hook      = NULL},             /* End marker */
};

/* for GRAPHER purpose */
static symbol_t symb_dtd_taskid = {
    .name           = "task_id",
    .context_index  = 0,
    .min            = NULL,
    .max            = NULL,
    .cst_inc        = 1,
    .expr_inc       = NULL,
    .flags          = 0x0
};

/**
 * To make it consistent with PaRSEC we need to intialize and have this function
 * we do not use this at this point
 */
static inline uint64_t DTD_identity_hash(const parsec_dtd_handle_t * __parsec_handle,
                                         const assignment_t * assignments)
{
    (void)__parsec_handle;
    return (uint64_t)assignments[0].value;
}

void
parsec_dtd_dequeue_handle( parsec_handle_t *parsec_handle )
{
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)parsec_handle;
    int remaining = parsec_atomic_dec_32b( (uint32_t*)&parsec_handle->nb_tasks );
    if( 0 == remaining ) {
        (void)parsec_atomic_cas_32b((volatile uint32_t *)&parsec_handle->nb_tasks, 0, PARSEC_RUNTIME_RESERVED_NB_TASKS);
        parsec_dtd_handle->enqueue_flag = 0;
    }
    assert(0 == remaining);

    parsec_dtd_handle_release( parsec_handle );
    return;  /* we're done in all cases */
}

int
parsec_dtd_update_runtime_task( parsec_handle_t *parsec_handle, int32_t count )
{
    int remaining;
    remaining = parsec_atomic_add_32b( (int32_t*)&(parsec_handle->nb_pending_actions), count );
    assert( 0<= remaining );

    if( 0 == remaining && 1 == parsec_handle->nb_tasks ) {
        parsec_dtd_dequeue_handle( parsec_handle );
    }

    return remaining;
}

/* **************************************************************************** */
/**
 * Intializes all the needed members and returns the DTD handle
 *
 * For correct profiling the task_class_counter should be correct
 *
 * @param[in]   context
 *                  The PARSEC context
 * @param[in]   arena_count
 *                  The count of the task class DTD handle will deal with
 * @return
 *              The PARSEC DTD handle
 *
 * @ingroup     DTD_INTERFACE
 */
parsec_handle_t *
parsec_dtd_handle_new()
{
    if( !dtd_init ) {
        parsec_dtd_init();
        dtd_init = 1;
    }

    if (dump_traversal_info) {
        parsec_output(parsec_debug_output, "\n\n------ New Handle -----\n\n\n");
    }

    parsec_dtd_handle_t *__parsec_handle;
    int i;

#if defined(PARSEC_DEBUG_PARANOID)
    assert( handle_mempool != NULL );
#endif
    __parsec_handle = (parsec_dtd_handle_t *)parsec_thread_mempool_allocate(handle_mempool->thread_mempools);

    parsec_dtd_handle_retain((parsec_handle_t *)__parsec_handle);

    __parsec_handle->super.context          = NULL;
    __parsec_handle->super.on_enqueue       = parsec_dtd_enqueue_handle;
    __parsec_handle->super.on_enqueue_data  = NULL;
    __parsec_handle->super.on_complete      = NULL;
    __parsec_handle->super.on_complete_data = NULL;
    __parsec_handle->super.devices_mask     = PARSEC_DEVICES_ALL;
    __parsec_handle->super.nb_tasks         = PARSEC_RUNTIME_RESERVED_NB_TASKS;
    __parsec_handle->super.handle_type         = 1;  /* Indicating this is a handle for dtd tasks */
    __parsec_handle->super.nb_pending_actions  = 0;  /* For the future tasks that will be inserted */
    __parsec_handle->super.nb_functions        = 0;
    __parsec_handle->super.update_nb_runtime_task = parsec_dtd_update_runtime_task;

    for(i = 0; i < vpmap_get_nb_vp(); i++) {
        __parsec_handle->startup_list[i] = NULL;
    }

    /* Keeping track of total tasks to be executed per handle for the window */
    for (i=0; i<PARSEC_DTD_NB_FUNCTIONS; i++) {
        __parsec_handle->flow_set_flag[i]  = 0;
        /* Added new */
        __parsec_handle->super.functions_array[i] = NULL;
    }

    __parsec_handle->wait_func           = parsec_dtd_handle_wait_func;
    __parsec_handle->task_id             = 0;
    __parsec_handle->task_window_size    = 1;
    __parsec_handle->task_threshold_size = dtd_threshold_size;
    __parsec_handle->local_task_inserted = 0;
    __parsec_handle->function_counter    = 0;
    __parsec_handle->enqueue_flag        = 0;
    parsec_atomic_unlock(&__parsec_handle->two_hash_table->atomic_lock);

    (void)parsec_handle_reserve_id((parsec_handle_t *) __parsec_handle);
    (void)parsec_handle_enable((parsec_handle_t *)__parsec_handle, NULL, NULL, NULL,
                               __parsec_handle->super.nb_pending_actions);

#if defined(PARSEC_PROF_TRACE) /* TODO: should not be per handle */
    parsec_dtd_add_profiling_info_generic((parsec_handle_t *)__parsec_handle, "Insert_task",
                               &insert_task_trace_keyin, &insert_task_trace_keyout );
    parsec_dtd_add_profiling_info_generic((parsec_handle_t *)__parsec_handle, "Hash_table_duration",
                               &hashtable_trace_keyin, &hashtable_trace_keyout);
#endif

    return (parsec_handle_t *)__parsec_handle;
}

/* **************************************************************************** */
/**
 * Clean up function to clean memory allocated dynamically for the run
 *
 * @param[in,out]   parsec_handle
 *                      Pointer to the DTD handle
 *
 * @ingroup         DTD_INTERFACE
 */
void
parsec_dtd_handle_destruct(parsec_handle_t *parsec_handle)
{
    parsec_dtd_handle_release( parsec_handle );
}

void
parsec_dtd_handle_retain( parsec_handle_t *parsec_handle )
{
    OBJ_RETAIN(parsec_handle);
}

void
parsec_dtd_handle_release( parsec_handle_t *parsec_handle )
{
    if( 1 == parsec_atomic_add_32b( &(parsec_handle->super.super.obj_reference_count) , -1 ) ) {
        parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)parsec_handle;
        int i;

        for(i = 0; i < PARSEC_DTD_NB_FUNCTIONS; i++) {
            const parsec_function_t   *func = parsec_dtd_handle->super.functions_array[i];
            parsec_dtd_function_t *dtd_func = (parsec_dtd_function_t *)func;


            /* Have we reached the end of known functions for this handle ? */
            if( NULL == func ) {
                assert(parsec_dtd_handle->function_counter == i);
                break;
            }

            uint64_t fkey = (uint64_t)(uintptr_t)dtd_func->fpointer + func->nb_flows;
            parsec_dtd_function_release( parsec_dtd_handle, fkey );

            parsec_dtd_template_release( func );
        }

        parsec_handle_unregister( parsec_handle );
        parsec_thread_mempool_free( handle_mempool->thread_mempools, parsec_dtd_handle );
    }
}

/* **************************************************************************** */
/**
 * This is the hook that connects the function to start initial ready
 * tasks with the context. Called internally by PaRSEC.
 *
 * @param[in]   context
 *                  PARSEC context
 * @param[in]   parsec_handle
 *                  Pointer to DTD handle
 * @param[in]   pready_list
 *                  Lists of ready tasks for each core
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_startup( parsec_context_t            *context,
                    parsec_handle_t             *parsec_handle,
                    parsec_execution_context_t **pready_list )
{
    uint32_t supported_dev = 0;
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *) parsec_handle;

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    parsec_dtd_handle->super.context = context;
#endif /* defined(PINS_ENABLE) */

    uint32_t wanted_devices = parsec_handle->devices_mask;
    parsec_handle->devices_mask = 0;

    for (uint32_t _i = 0; _i < parsec_nb_devices; _i++) {
        if (!(wanted_devices & (1 << _i)))
            continue;
        parsec_device_t *device = parsec_devices_get(_i);

        if (NULL == device)
            continue;
        if (NULL != device->device_handle_register)
            if (PARSEC_SUCCESS != device->device_handle_register(device, (parsec_handle_t *) parsec_handle))
                continue;

        supported_dev |= device->type;
        parsec_handle->devices_mask |= (1 << _i);
    }
    (void)pready_list;

    parsec_dtd_schedule_tasks( parsec_dtd_handle );
}

static inline int
parsec_dtd_not_sent_to_rank(parsec_dtd_task_t *this_task, int flow_index, int dst_rank)
{
    int array_index = dst_rank/(sizeof(int)*8);
    int rank_mask = 1U << (dst_rank % (sizeof(int)*8));

    if( !((FLOW_OF(this_task, flow_index))->rank_sent_to[array_index] & rank_mask) ) {
        (FLOW_OF(this_task, flow_index))->rank_sent_to[array_index] |= rank_mask;
        return 1;
    }
    return 0;
}

/* **************************************************************************** */
/**
 * This function checks the readyness of a task and if ready pushes it
 * in a list of ready task
 *
 * This function will have more functionality in the implementation
 * for distributed memory
 *
 * @param[in]   eu
 *                  Execution unit
 * @param[in]   newcontext
 *                  Pointer to DTD task we are trying to activate
 * @param[in]   oldcontext
 *                  Pointer to DTD task activating it's successor(newcontext)
 * @param       deps,data,src_rank,dst_rank,dst_vpid
 *                  Parameters we will use in distributed memory implementation
 * @param[out]  param
 *                  Pointer to list in which we will push if the task is ready
 * @return
 *              Instruction on how to iterate over the successors of oldcontext
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
parsec_ontask_iterate_t
dtd_release_dep_fct( parsec_execution_unit_t *eu,
                     const parsec_execution_context_t *newcontext,
                     const parsec_execution_context_t *oldcontext,
                     const dep_t *dep,
                     parsec_dep_data_description_t *data,
                     int src_rank, int dst_rank, int dst_vpid,
                     void *param )
{
    (void)eu; (void)data; (void)src_rank; (void)dst_rank; (void)oldcontext;
    parsec_release_dep_fct_arg_t *arg = (parsec_release_dep_fct_arg_t *)param;
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)newcontext;
    int not_ready = 1;

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank && src_rank == oldcontext->parsec_handle->context->my_rank) {
        assert( 0 == (arg->action_mask & PARSEC_ACTION_RECV_INIT_REMOTE_DEPS) );

        if( arg->action_mask & PARSEC_ACTION_SEND_INIT_REMOTE_DEPS ) {
            if( parsec_dtd_not_sent_to_rank((parsec_dtd_task_t *)oldcontext,
                                            dep->belongs_to->flow_index, dst_rank) ) {
                struct remote_dep_output_param_s* output;
                int _array_pos, _array_mask;

#if !defined(PARSEC_DIST_COLLECTIVES)
                assert(src_rank == eu->virtual_process->parsec_context->my_rank);
#endif
                _array_pos = dst_rank / (8 * sizeof(uint32_t));
                _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
                PARSEC_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
                output = &arg->remote_deps->output[dep->dep_datatype_index];
                assert( (-1 == arg->remote_deps->root) || (arg->remote_deps->root == src_rank) );
                arg->remote_deps->root = src_rank;
                arg->remote_deps->outgoing_mask |= (1 << dep->dep_datatype_index);
                if( !(output->rank_bits[_array_pos] & _array_mask) ) {
                    output->rank_bits[_array_pos] |= _array_mask;
                    output->deps_mask |= (1 << dep->dep_index);
                    if( 0 == output->count_bits ) {
                        output->data = *data;
                    } else {
                        assert(output->data.data == data->data);
                    }
                    output->count_bits++;
                    if(newcontext->priority > output->priority) {
                        output->priority = newcontext->priority;
                        if(newcontext->priority > arg->remote_deps->max_priority)
                            arg->remote_deps->max_priority = newcontext->priority;
                    }
                }  /* otherwise the bit is already flipped, the peer is already part of the propagation. */
            }
        }
    }
#else
    (void)src_rank;
    (void)data;
#endif

    if( parsec_dtd_task_is_local(current_task) ) {
        not_ready = parsec_atomic_add_32b((int *)&(current_task->flow_count), -1);

#if defined(PARSEC_PROF_GRAPHER)
        /* Check to not print stuff redundantly */
        parsec_flow_t *origin_flow = (parsec_flow_t*) calloc(1, sizeof(parsec_flow_t));
        parsec_flow_t *dest_flow = (parsec_flow_t*) calloc(1, sizeof(parsec_flow_t));

        origin_flow->name = "A";
        dest_flow->name = "A";
        dest_flow->flow_flags = FLOW_ACCESS_RW;

        parsec_prof_grapher_dep(oldcontext, newcontext, !not_ready, origin_flow, dest_flow);

        free(origin_flow);
        free(dest_flow);
#else
        (void)dep;
#endif
        if(!not_ready) {
            assert(parsec_dtd_task_is_local(current_task));
            if(dump_traversal_info) {
                parsec_output(parsec_debug_output, "------\ntask Ready: %s \t %" PRIu64 "\nTotal flow: %d  flow_count:"
                              "%d\n-----\n", current_task->super.function->name, current_task->ht_item.key,
                              current_task->super.function->nb_flows, current_task->flow_count);
            }

            arg->ready_lists[dst_vpid] = (parsec_execution_context_t*)
                parsec_list_item_ring_push_sorted( (parsec_list_item_t*)arg->ready_lists[dst_vpid],
                                                  &current_task->super.super,
                                                  parsec_execution_context_priority_comparator );
            return PARSEC_ITERATE_CONTINUE; /* Returns the status of the task being activated */
        } else {
            return PARSEC_ITERATE_STOP;
        }
    } else {
        return PARSEC_ITERATE_CONTINUE;
    }
}

/* **************************************************************************** */
/**
 * This function iterates over all the successors of a task and
 * activates them and builds a list of the ones that got ready
 * by this activation
 *
 * The actual implementation is done in ordering_correctly_2()
 *
 * @see     ordering_correctly_2()
 *
 * @param   eu,this_task,action_mask,ontask,ontask_arg
 *
 * @ingroup DTD_ITERFACE_INTERNAL
 */
static void
iterate_successors_of_dtd_task(parsec_execution_unit_t *eu,
                               const parsec_execution_context_t *this_task,
                               uint32_t action_mask,
                               parsec_ontask_function_t *ontask,
                               void *ontask_arg)
{
    parsec_dtd_task_t   *this_dtd_task;

    this_dtd_task = (parsec_dtd_task_t *)this_task;

    (void)eu; (void)this_task; (void)action_mask; (void)ontask; (void)ontask_arg;
    parsec_dtd_ordering_correctly( eu, (parsec_execution_context_t *)this_dtd_task,
                                   action_mask, ontask, ontask_arg );
}

/* **************************************************************************** */
/**
 * Release dependencies after a task is done
 *
 * Calls iterate successors function that returns a list of tasks that
 * are ready to go. Those ready tasks are scheduled in here.
 *
 * @param[in]   eu
 *                  Execution unit
 * @param[in]   this_task
 *                  Pointer to DTD task we are releasing deps of
 * @param       action_mask,deps
 * @return
 *              0
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static int
release_deps_of_dtd( parsec_execution_unit_t *eu,
                     parsec_execution_context_t *this_task,
                     uint32_t action_mask,
                     parsec_remote_deps_t *deps )
{
    (void)deps;
    parsec_release_dep_fct_arg_t arg;
    int __vp_id;

    assert(NULL != eu);

    PINS(eu, RELEASE_DEPS_BEGIN, this_task);
#if defined(DISTRIBUTED)
    arg.remote_deps = deps;
#endif /* defined(DISTRIBUTED) */

    arg.action_mask  = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
    arg.ready_lists  = alloca(sizeof(parsec_execution_context_t *) * eu->virtual_process->parsec_context->nb_vp);

    for (__vp_id = 0; __vp_id < eu->virtual_process->parsec_context->nb_vp; __vp_id++)
        arg.ready_lists[__vp_id] = NULL;

    parsec_dtd_task_t *this_dtd_task = NULL;
    const parsec_function_t  *function = this_task->function;
    parsec_dtd_handle_t *parsec_handle = (parsec_dtd_handle_t *)this_task->parsec_handle;

    if( (action_mask & PARSEC_ACTION_COMPLETE_LOCAL_TASK) ) {
        this_dtd_task = (parsec_dtd_task_t *)this_task;
    } else {
        /* passing the data from MPI to the local representation of the remote_task */
        parsec_dtd_two_hash_table_lock(parsec_handle->two_hash_table);
        int i, j, flow_index, track_flow = 0;
        for(i = 0; i < function->nb_flows; i++) {
            j = 0;
            if ( function->out[i] != NULL ) {
                while ( function->out[i]->dep_out[j] != NULL ) {
                    if ( ((1<<function->out[i]->dep_out[j]->dep_index) & action_mask) ) {
                        flow_index = function->out[i]->dep_out[j]->belongs_to->flow_index;

                        if( !(track_flow & (1U<<flow_index) ) ) {

                            this_dtd_task = parsec_dtd_find_task( parsec_handle, (uint64_t)this_task->locals[0].value<<32 | (1U<<flow_index) );
                            assert(this_dtd_task != NULL);

                            if ( this_task->data[flow_index].data_out != NULL ) {
                                assert(this_task->data[flow_index].data_out != NULL);
                                this_dtd_task->super.data[flow_index].data_in = this_task->data[flow_index].data_in;
                                this_dtd_task->super.data[flow_index].data_out = this_task->data[flow_index].data_out;
                                /* Push data in LRU cache */
                                parsec_dtd_retain_floating_data(this_task->data[flow_index].data_out);
                            }
                            track_flow |= (1U<<flow_index); /* to make sure we are retaining the data only once */
                        }
                    }
                    j++;
                }
            }
        }
        parsec_dtd_two_hash_table_unlock(parsec_handle->two_hash_table);
    }
    assert(NULL != this_dtd_task);
    iterate_successors_of_dtd_task(eu, (parsec_execution_context_t*)this_dtd_task, action_mask, dtd_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
    /* We perform this only for remote tasks that are being activated
     * from the comm engine. We remove the task from the hash table
     * for each flow a rank is concerned about.
     */
    if( parsec_dtd_task_is_remote(this_dtd_task) && !(action_mask & PARSEC_ACTION_COMPLETE_LOCAL_TASK) ) {
        int i, j, flow_index, track_flow = 0;
        parsec_dtd_two_hash_table_lock(parsec_handle->two_hash_table);
        for( i = 0; i < function->nb_flows; i++ ) {
            j = 0;
            if ( function->out[i] != NULL ) {
                while ( function->out[i]->dep_out[j] != NULL ) {
                    if ( ((1<<function->out[i]->dep_out[j]->dep_index) & action_mask) ) {
                        flow_index = function->out[i]->dep_out[j]->belongs_to->flow_index;
                        if( !(track_flow & (1U<<flow_index) ) ) {
                            if( NULL != parsec_dtd_find_and_remove_task( parsec_handle, ((uint64_t)this_task->locals[0].value<<32 | (1U<<flow_index) )) ) {
                                /* also releasing task */
                                parsec_dtd_remote_task_release( this_dtd_task );
                            }
                            track_flow |= (1U<<flow_index); /* to make sure we are releasing the data only once */
                        }
                    }
                    j++;
                }
            }
        }
        parsec_dtd_two_hash_table_unlock(parsec_handle->two_hash_table);
    }
#else
    (void)deps;
#endif

    /* Scheduling tasks */
    if (action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
        parsec_vp_t **vps = eu->virtual_process->parsec_context->virtual_processes;
        for (__vp_id = 0; __vp_id < eu->virtual_process->parsec_context->nb_vp; __vp_id++) {
            if (NULL == arg.ready_lists[__vp_id]) {
                continue;
            }
            if (__vp_id == eu->virtual_process->vp_id) {
                __parsec_schedule(eu, arg.ready_lists[__vp_id], 0);
            }else {
                __parsec_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id], 0);
            }
            arg.ready_lists[__vp_id] = NULL;
        }
    }

    PINS(eu, RELEASE_DEPS_END, this_task);
    return 0;
}

/* **************************************************************************** */
/**
 * This function is called internally by PaRSEC once a task is done
 *
 * @param[in]   context
 *                  Execution unit
 * @param[in]   this_task
 *                  Pointer to DTD task we just completed
 * @return
 *              0
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static int
complete_hook_of_dtd( parsec_execution_unit_t    *context,
                      parsec_execution_context_t *this_task )
{
    /* Assuming we only call this function for local tasks */
    parsec_dtd_task_t *this_dtd_task = (parsec_dtd_task_t *) this_task;

    int action_mask = 0;

    if (dump_traversal_info) {
        static int counter= 0;
        (void)parsec_atomic_add_32b(&counter,1);
        parsec_output(parsec_debug_output, "------------------------------------------------\n"
                      "execution done of task: %s \t %" PRIu64 "\n"
                      "task done %d rank --> %d\n",
                      this_task->function->name,
                      this_dtd_task->ht_item.key,
                      counter, this_task->parsec_handle->context->my_rank);
    }

#if defined(PARSEC_PROF_GRAPHER)
    parsec_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
                             this_task->function->key(this_task->parsec_handle, this_task->locals));
#endif /* defined(PARSEC_PROF_GRAPHER) */

    PARSEC_TASK_PROF_TRACE(context->eu_profile,
                           this_task->parsec_handle->profiling_array[2 * this_task->function->function_id + 1],
                           this_task);

    /* constructing action_mask for all flows of local task */
    int current_dep;
    for( current_dep = 0; current_dep < this_dtd_task->super.function->nb_flows; current_dep++ ) {
        action_mask |= (1U << current_dep);
    }

    release_deps_of_dtd( context, this_task, action_mask     |
                         PARSEC_ACTION_RELEASE_LOCAL_DEPS    |
                         PARSEC_ACTION_SEND_REMOTE_DEPS      |
                         PARSEC_ACTION_SEND_INIT_REMOTE_DEPS |
                         PARSEC_ACTION_RELEASE_REMOTE_DEPS   |
                         PARSEC_ACTION_COMPLETE_LOCAL_TASK,
                         NULL );

    return 0;
}

parsec_hook_return_t
parsec_dtd_release_local_task( parsec_dtd_task_t *this_task )
{
    parsec_object_t *object = (parsec_object_t *)this_task;
    assert(this_task->super.super.super.obj_reference_count > 1);
    if( 1 == parsec_atomic_add_32b( &object->obj_reference_count, -1 ) ) {
        int current_flow;
        for( current_flow = 0; current_flow < this_task->super.function->nb_flows; current_flow++ ) {
            parsec_dtd_tile_t *tile = (FLOW_OF(this_task, current_flow))->tile;
            if( tile == NULL ) continue;
            assert( NULL != this_task->super.data[current_flow].data_in );
            if( !((FLOW_OF(this_task, current_flow))->flags & DATA_RELEASED) ) {
                (FLOW_OF(this_task, current_flow))->flags |= DATA_RELEASED;
                parsec_dtd_release_floating_data(this_task->super.data[current_flow].data_in);
            }
            if( ((parsec_dtd_function_t *)this_task->super.function)->fpointer == parsec_dtd_copy_data_to_matrix ) {
                assert( current_flow == 0 );
                parsec_dtd_tile_release( tile );
            }
        }
        assert(this_task->super.super.super.obj_reference_count == 1);
        parsec_handle_t *handle = this_task->super.parsec_handle;

        parsec_thread_mempool_free( this_task->mempool_owner, this_task );
        parsec_handle_update_runtime_nbtask( handle, -1 );
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* Function to push back tasks in their mempool once the execution are done */
parsec_hook_return_t
parsec_release_dtd_task_to_mempool(parsec_execution_unit_t *eu,
                                  parsec_execution_context_t *this_task)
{
    (void)eu;
    (void)parsec_atomic_dec_32b( (uint32_t*)&this_task->parsec_handle->nb_tasks );
    return parsec_dtd_release_local_task( (parsec_dtd_task_t *)this_task );
}

void
parsec_dtd_remote_task_retain( parsec_dtd_task_t *this_task )
{
    parsec_object_t *object = (parsec_object_t *)this_task;
    (void)parsec_atomic_add_32b(&object->obj_reference_count, 1);
}

void
parsec_dtd_remote_task_release( parsec_dtd_task_t *this_task )
{
    parsec_object_t *object = (parsec_object_t *)this_task;
    assert(object->obj_reference_count > 1);
    if( 1 == parsec_atomic_add_32b( &object->obj_reference_count, -1 ) ){
        int current_flow;
        for( current_flow = 0; current_flow < this_task->super.function->nb_flows; current_flow++ ) {
            if( NULL != this_task->super.data[current_flow].data_out ) {
                parsec_dtd_release_floating_data(this_task->super.data[current_flow].data_out);
            }

            parsec_dtd_tile_t *tile = (FLOW_OF(this_task, current_flow))->tile;
            if( tile == NULL ) continue;
            if( ((parsec_dtd_function_t *)this_task->super.function)->fpointer == parsec_dtd_copy_data_to_matrix ) {
                assert( current_flow == 0 );
                parsec_dtd_tile_release( tile );
            }
        }
        assert(this_task->super.super.super.obj_reference_count == 1);
        parsec_handle_t *handle = this_task->super.parsec_handle;
        parsec_thread_mempool_free( this_task->mempool_owner, this_task );
        parsec_handle_update_runtime_nbtask( handle, -1 );
    }
    assert(object->obj_reference_count >= 1);
}

/* Prepare_input function */
int
data_lookup_of_dtd_task( parsec_execution_unit_t *context,
                         parsec_execution_context_t *this_task )
{
    (void)context;

    int current_dep, op_type_on_current_flow;
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;

    if( ((parsec_dtd_function_t *)this_task->function)->fpointer != parsec_dtd_copy_data_to_matrix ) {
        for( current_dep = 0; current_dep < current_task->super.function->nb_flows; current_dep++ ) {
            op_type_on_current_flow = ((FLOW_OF(current_task, current_dep))->op_type & GET_OP_TYPE);

            if( NULL == current_task->super.data[current_dep].data_in ) continue;

            if( INOUT == op_type_on_current_flow ||
                OUTPUT == op_type_on_current_flow ) {
                if( current_task->super.data[current_dep].data_in->readers > 0 ) {
                    return PARSEC_HOOK_RETURN_AGAIN;
                }
            }
        }
    }

    return PARSEC_HOOK_RETURN_DONE;
}

/* Prepare_output function */
int
output_data_of_dtd_task( parsec_execution_unit_t *context,
                         parsec_execution_context_t *this_task )
{
    (void)context;

    int current_dep;
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;

    for( current_dep = 0; current_dep < current_task->super.function->nb_flows; current_dep++ ) {
        /* The following check makes sure the behavior is correct when NULL is provided as input to a flow.
           Dependency tracking and building is completely ignored in flow = NULL case, and no expectation should exist.
        */
        if( NULL == current_task->super.data[current_dep].data_in &&
            NULL != current_task->super.data[current_dep].data_out ) {
            parsec_fatal( "Please make sure you are not assigning data in data_out when data_in is provided as NULL.\n"
                          "No dependency tracking is performed for a flow when NULL is passed as as INPUT for that flow.\n" );
        }

        if( NULL != current_task->super.data[current_dep].data_in ) {
            current_task->super.data[current_dep].data_out = current_task->super.data[current_dep].data_in;
        }
    }

    return PARSEC_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dtd_task(parsec_execution_unit_t *eu,
                                       const parsec_execution_context_t *this_task,
                                       uint32_t *flow_mask, parsec_dep_data_description_t *data)
{
    (void)eu;
    data->count = 1;
    data->displ = 0;

    int i;
    for( i = 0; i < this_task->function->nb_flows; i++) {
        if((*flow_mask) & (1U<<i)) {
            data->arena  = parsec_dtd_arenas[(FLOW_OF(((parsec_dtd_task_t *)this_task), i))->arena_index];
            data->layout = data->arena->opaque_dtt;
            (*flow_mask) &= ~(1U<<i);
            return PARSEC_HOOK_RETURN_NEXT;
        }
    }

    data->arena  = NULL;
    data->data   = NULL;
    data->layout = PARSEC_DATATYPE_NULL;
    data->count  = 0;
    data->displ  = 0;
    (*flow_mask) = 0;		/* nothing left */

    return PARSEC_HOOK_RETURN_DONE;
}


/***************************************************************************//**
 *
 * Function to find and return a dep between two tasks
 *
 * @param[in]   parent_task, desc_task, parent_flow_index, desc_flow_index
 * @return
 *              Dep found between two tasks
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
dep_t *
parsec_dtd_find_and_return_dep( parsec_dtd_task_t *parent_task, parsec_dtd_task_t *desc_task,
                                int parent_flow_index, int desc_flow_index )
{
    int out_index = parent_flow_index;
    parsec_flow_t *flow = (parsec_flow_t*)parent_task->super.function->out[out_index];
    int desc_function_id = desc_task->super.function->function_id, i;
    dep_t *dep = NULL;

    for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
        if( flow->dep_out[i]->function_id == desc_function_id &&
            flow->dep_out[i]->flow->flow_index == desc_flow_index ) {
            dep = (dep_t *) flow->dep_out[i];
            break;
        }
    }
    return dep;
}

/* This function creates relationship between two task function classes.
 * Arguments:   - parsec handle (parsec_handle_t *)
                - parent master structure (parsec_function_t *)
                - child master structure (parsec_function_t *)
                - flow index of task that belongs to the class of "parent master structure" (int)
                - flow index of task that belongs to the class of "child master structure" (int)
                - the type of data (the structure of the data like square,
                  triangular and etc) this dependency is about (int)
 * Returns:     - void
 */
void
set_dependencies_for_function(parsec_handle_t* parsec_handle,
                              parsec_function_t *parent_function,
                              parsec_function_t *desc_function,
                              uint8_t parent_flow_index,
                              uint8_t desc_flow_index)
{
    uint8_t i, dep_exists = 0, j;

    if (NULL == desc_function) {   /* Data is not going to any other task */
        int out_index = parent_flow_index;
        if(NULL != parent_function->out[out_index]) {
            parsec_flow_t *tmp_d_flow = (parsec_flow_t *)parent_function->out[out_index];
            for ( i = 0; i < MAX_DEP_IN_COUNT; i++ ) {
                if (NULL != tmp_d_flow->dep_out[i]) {
                    if (tmp_d_flow->dep_out[i]->function_id == LOCAL_DATA ) {
                        dep_exists = 1;
                        break;
                    }
                }
            }
        }
        if (!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            if (dump_function_info) {
                parsec_output(parsec_debug_output, "%s -> LOCAL\n", parent_function->name);
            }

            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = LOCAL_DATA; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = ((parsec_dtd_function_t*)parent_function)->dep_out_index++;
            desc_dep->belongs_to    = parent_function->out[out_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            /* specific for cholesky, will need to change */
            desc_dep->dep_datatype_index = parent_function->out[out_index]->flow_index;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == parent_function->out[out_index]->dep_out[i]) {
                    /* Bypassing constness in function structure */
                    parsec_flow_t *desc_in = (parsec_flow_t*)(parent_function->out[out_index]);
                    /* Setting dep in the next available dep_in array index */
                    desc_in->dep_out[i] = (dep_t *)desc_dep;
                    desc_in->flow_datatype_mask |= (1U<<desc_dep->dep_datatype_index);
                    break;
                }
            }

            if(MAX_DEP_IN_COUNT == i) {
                free(desc_dep);
                parsec_fatal("Fatal Error: user could try to insert more than MAX_DEP_IN_COUNT(%d) dependencies into a flow\n", MAX_DEP_IN_COUNT);
            }
        }
        return;
    }

    if (NULL == parent_function) {   /* Data is not coming from any other task */
        int in_index = desc_flow_index;
        if(NULL != desc_function->in[in_index]) {
            parsec_flow_t *tmp_d_flow = (parsec_flow_t *)desc_function->in[in_index];
            for (i = 0; i < MAX_DEP_IN_COUNT; i++) {
                if (NULL != tmp_d_flow->dep_in[i]) {
                    if (tmp_d_flow->dep_in[i]->function_id == LOCAL_DATA ) {
                        dep_exists = 1;
                        break;
                    }
                }
            }
        }
        if (!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            if(dump_function_info) {
                parsec_output(parsec_debug_output, "LOCAL -> %s\n", desc_function->name);
            }
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = LOCAL_DATA;
            desc_dep->dep_index     = ((parsec_dtd_function_t*)desc_function)->dep_in_index++;
            desc_dep->belongs_to    = desc_function->in[in_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = desc_function->in[in_index]->flow_index;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == desc_function->in[in_index]->dep_in[i]) {
                    /* Bypassing constness in function structure */
                    parsec_flow_t *desc_in = (parsec_flow_t*)(desc_function->in[in_index]);
                    /* Setting dep in the next available dep_in array index */
                    desc_in->dep_in[i]  = (dep_t *)desc_dep;
                    break;
                }
            }
            if( MAX_DEP_IN_COUNT == i ) {
                free(desc_dep);
                parsec_fatal("Fatal Error: user could try to insert more than MAX_DEP_IN_COUNT(%d) dependencies into a flow\n", MAX_DEP_IN_COUNT);
            }
        }
        return;
    } else {
        int out_index = parent_flow_index;
        parsec_flow_t *tmp_flow = (parsec_flow_t *) parent_function->out[out_index];

        assert(tmp_flow != NULL);

        if (tmp_flow->flow_flags == FLOW_ACCESS_READ) {
            int in_index = parent_flow_index;
            dep_t *tmp_dep = NULL;
            parsec_flow_t *tmp_p_flow = NULL;
            tmp_flow = (parsec_flow_t *) parent_function->in[in_index];
            for ( i = 0; i < MAX_DEP_IN_COUNT; i++ ) {
                if(NULL != tmp_flow->dep_in[i]) {
                    tmp_dep = (dep_t *) tmp_flow->dep_in[i];
                }
            }
            if(tmp_dep->function_id == LOCAL_DATA) {
                set_dependencies_for_function(parsec_handle,
                                              NULL, desc_function, 0,
                                              desc_flow_index);
                return;
            }
            tmp_p_flow = (parsec_flow_t *)tmp_dep->flow;
            parent_function = (parsec_function_t *)parsec_handle->functions_array[tmp_dep->function_id];

            for( j = 0; j < MAX_DEP_OUT_COUNT; j++ ) {
                if(NULL != tmp_p_flow->dep_out[j]) {
                    if((parsec_flow_t *)tmp_p_flow->dep_out[j]->flow == tmp_flow) {
                        parent_flow_index = tmp_p_flow->dep_out[j]->belongs_to->flow_index;
                        set_dependencies_for_function(parsec_handle,
                                                      parent_function,
                                                      desc_function,
                                                      parent_flow_index,
                                                      desc_flow_index);
                        return;
                    }
                }
            }
            dep_exists = 1;
        }

        int desc_in_index = desc_flow_index;
        for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
            if (NULL != tmp_flow->dep_out[i]) {
                if( tmp_flow->dep_out[i]->function_id == desc_function->function_id &&
                    tmp_flow->dep_out[i]->flow == desc_function->in[desc_in_index] ) {
                    dep_exists = 1;
                    break;
                }
            }
        }

        if(!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            dep_t *parent_dep = (dep_t *) malloc(sizeof(dep_t));

            if (dump_function_info) {
                parsec_output(parsec_debug_output, "%s -> %s\n", parent_function->name, desc_function->name);
            }

            /* setting out-dependency for parent */
            parent_dep->cond            = NULL;
            parent_dep->ctl_gather_nb   = NULL;
            parent_dep->function_id     = desc_function->function_id;
            parent_dep->flow            = desc_function->in[desc_in_index];
            parent_dep->dep_index       = ((parsec_dtd_function_t*)parent_function)->dep_out_index++;
            parent_dep->belongs_to      = parent_function->out[out_index];
            parent_dep->direct_data     = NULL;
            parent_dep->dep_datatype_index = parent_function->out[out_index]->flow_index;

            for(i=0; i<MAX_DEP_OUT_COUNT; i++) {
                if(NULL == parent_function->out[out_index]->dep_out[i]) {
                    /* to bypass constness in function structure */
                    parsec_flow_t **parent_out = (parsec_flow_t **)&(parent_function->out[out_index]);
                    (*parent_out)->dep_out[i] = (dep_t *)parent_dep;
                    (*parent_out)->flow_datatype_mask |= (1U << parent_dep->dep_datatype_index);
                    break;
                }
            }

            /* setting in-dependency for descendant */
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = parent_function->function_id;
            desc_dep->flow          = parent_function->out[out_index];
            desc_dep->dep_index     = ((parsec_dtd_function_t*)desc_function)->dep_in_index++;
            desc_dep->belongs_to    = desc_function->in[desc_in_index];
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = desc_function->in[desc_in_index]->flow_index;

            for(i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL == desc_function->in[desc_in_index]->dep_in[i]) {
                    /* Bypassing constness in function strucutre */
                    parsec_flow_t **desc_in = (parsec_flow_t **)&(desc_function->in[desc_in_index]);
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep;
                    break;
                }
            }
        }
    }
}

/* **************************************************************************** */
/**
 * This function creates and initializes members of master-structure
 * representing each task-class
 *
 * Most importantly mempool from which we will create DTD task of a
 * task class is created here. The size of each task is calculated
 * from the parameters passed to this function. A master-structure
 * is like a template of a task class. It store the common attributes
 * of all the tasks, belonging to a task class, will have.
 *
 * @param[in,out]   __parsec_handle
 *                      The DTD handle
 * @paramp[in]      fpointer
 *                      Function pointer that uniquely identifies a
 *                      task class, this is the pointer to the body
 *                      of the task
 * @param[in]       name
 *                      The name of the task class the master-structure
 *                      will represent
 * @param[in]       count_of_params
 *                      Total count of parameters each task of this task
 *                      class will have and work on
 * @param[in]       size_of_param
 *                      Total size of all params in bytes
 * @param[in]       flow_count
 *                      Total flow each task of this task class has
 * @return
 *                  The master-structure
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
parsec_function_t *
parsec_dtd_create_function( parsec_dtd_handle_t *__parsec_handle, parsec_dtd_funcptr_t* fpointer,
                            char* name, int count_of_params, long unsigned int size_of_param,
                            int flow_count )
{
    parsec_dtd_function_t *dtd_function = (parsec_dtd_function_t *) calloc(1, sizeof(parsec_dtd_function_t));
    parsec_function_t *function = (parsec_function_t *) dtd_function;

    dtd_function->dep_datatype_index = 0;
    dtd_function->dep_in_index       = 0;
    dtd_function->dep_out_index      = 0;
    dtd_function->count_of_params    = count_of_params;
    dtd_function->size_of_param      = size_of_param;
    dtd_function->fpointer           = fpointer;
    dtd_function->ref_count          = 1;

    /* Allocating mempool according to the size and param count */
    int total_size =  sizeof(parsec_dtd_task_t) +
                     (flow_count * sizeof(parsec_dtd_parent_info_t)) +
                     (flow_count * sizeof(parsec_dtd_descendant_info_t)) +
                     (flow_count * sizeof(parsec_dtd_flow_info_t)) +
                     (count_of_params * sizeof(parsec_dtd_task_param_t)) +
                      size_of_param;

    parsec_mempool_construct( &dtd_function->context_mempool,
                              OBJ_CLASS(parsec_dtd_task_t), total_size,
                              offsetof(parsec_dtd_task_t, mempool_owner),
                              1/* no. of threads*/ );

    int total_size_remote_task = sizeof(parsec_dtd_task_t) +
                     (flow_count * sizeof(parsec_dtd_parent_info_t)) +
                     (flow_count * sizeof(parsec_dtd_descendant_info_t)) +
                     (flow_count * sizeof(parsec_dtd_min_flow_info_t));

    parsec_mempool_construct( &dtd_function->remote_task_mempool,
                              OBJ_CLASS(parsec_dtd_task_t), total_size_remote_task,
                              offsetof(parsec_dtd_task_t, mempool_owner),
                              1/* no. of threads*/ );

    /*
     To bypass const in function structure.
     Getting address of the const members in local mutable pointers.
     */
    char **name_not_const = (char **)&(function->name);
    symbol_t **params     = (symbol_t **) &function->params;
    symbol_t **locals     = (symbol_t **) &function->locals;
    expr_t **priority     = (expr_t **)&function->priority;
    __parsec_chore_t **incarnations = (__parsec_chore_t **)&(function->incarnations);

    *name_not_const                 = name;
    function->function_id           = __parsec_handle->function_counter++;
    function->nb_flows              = flow_count;
    /* set to one so that prof_grpaher prints the task id properly */
    function->nb_parameters         = 1;
    function->nb_locals             = 1;
    params[0]                       = &symb_dtd_taskid;
    locals[0]                       = &symb_dtd_taskid;
    function->data_affinity         = NULL;
    function->initial_data          = NULL;
    function->final_data            = (parsec_data_ref_fn_t *) NULL;
    *priority                       = NULL;
    function->flags                 = 0x0 | PARSEC_HAS_IN_IN_DEPENDENCIES | PARSEC_USE_DEPS_MASK;
    function->dependencies_goal     = 0;
    function->key                   = (parsec_functionkey_fn_t *)DTD_identity_hash;
    function->fini                  = NULL;
    *incarnations                   = (__parsec_chore_t *)dtd_chore;
    function->find_deps             = NULL;
    function->iterate_successors    = iterate_successors_of_dtd_task;
    function->iterate_predecessors  = NULL;
    function->release_deps          = release_deps_of_dtd;
    function->prepare_input         = data_lookup_of_dtd_task;
    function->prepare_output        = output_data_of_dtd_task;
    function->get_datatype          = (parsec_datatype_lookup_t *)datatype_lookup_of_dtd_task,
    function->complete_execution    = complete_hook_of_dtd;
    function->release_task          = parsec_release_dtd_task_to_mempool;

    /* Inserting Function structure in the hash table to keep track for each class of task */
    uint64_t fkey = (uint64_t)(uintptr_t)fpointer + function->nb_flows;
    parsec_dtd_function_insert( __parsec_handle, fkey, dtd_function );
    __parsec_handle->super.functions_array[function->function_id] = (parsec_function_t *) function;
    __parsec_handle->super.nb_functions++;
    return function;
}

void
parsec_dtd_destroy_template( const parsec_function_t *func )
{
    int j, k;
    parsec_dtd_function_t *dtd_func = (parsec_dtd_function_t *)func;
    /* As we fill the flows and then deps in sequential order, we can bail out at the first NULL */
    for (j = 0; j < func->nb_flows; j++) {
        if( NULL == func->in[j] ) break;
        for(k = 0; k < MAX_DEP_IN_COUNT; k++) {
            if ( NULL == func->in[j]->dep_in[k] ) break;
            free((void*)func->in[j]->dep_in[k]);
        }
        for(k = 0; k < MAX_DEP_OUT_COUNT; k++) {
            if ( NULL == func->in[j]->dep_out[k] ) break;
            free((void*)func->in[j]->dep_out[k]);
        }
        free((void*)func->in[j]);
    }
    parsec_mempool_destruct(&dtd_func->context_mempool);
    parsec_mempool_destruct(&dtd_func->remote_task_mempool);
    free((void*)func);
}

void
parsec_dtd_template_retain( const parsec_function_t *function )
{
    ((parsec_dtd_function_t *)function)->ref_count++;
}

void
parsec_dtd_template_release( const parsec_function_t *function )
{
    ((parsec_dtd_function_t *)function)->ref_count = ((parsec_dtd_function_t *)function)->ref_count - 1;
    if( 0 == ((parsec_dtd_function_t *)function)->ref_count ) {
        parsec_dtd_destroy_template( function );
    }
}

/* **************************************************************************** */
/**
 * This function sets the flows in master-structure as we discover them
 *
 * @param[in,out]   __parsec_handle
 *                      DTD handle
 * @param[in]       this_task
 *                      Task to point to correct master-structure
 * @param[in]       tile_op_type
 *                      The operation type of the task on the flow
 *                      we are setting here
 * @param[in]       flow_index
 *                      The index of the flow we are setting
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
set_flow_in_function( parsec_dtd_handle_t *parsec_dtd_handle,
                      parsec_dtd_task_t *this_task, int tile_op_type,
                      int flow_index)
{
    (void)parsec_dtd_handle;
    parsec_flow_t* flow  = (parsec_flow_t *) calloc(1, sizeof(parsec_flow_t));
    flow->name          = "Random";
    flow->sym_type      = 0;
    flow->flow_index    = flow_index;
    flow->flow_datatype_mask = 0;

    int i;
    for (i=0; i<MAX_DEP_IN_COUNT; i++) {
        flow->dep_in[i] = NULL;
    }
    for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
        flow->dep_out[i] = NULL;
    }

    if ((tile_op_type & GET_OP_TYPE) == INPUT) {
        flow->flow_flags = FLOW_ACCESS_READ;
    } else if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
        flow->flow_flags = FLOW_ACCESS_WRITE;
    } else if ((tile_op_type & GET_OP_TYPE) == INOUT) {
        flow->flow_flags = FLOW_ACCESS_RW;
    }

    parsec_flow_t **in = (parsec_flow_t **)&(this_task->super.function->in[flow_index]);
    *in = flow;
    parsec_flow_t **out = (parsec_flow_t **)&(this_task->super.function->out[flow_index]);
    *out = flow;
}

/* **************************************************************************** */
/**
 * This function sets the parent of a task
 *
 * This function is called by the descendant and the descendant here
 * puts itself as the descendant of the parent.
 *
 * @param[out]  parent_task
 *                  Task we are setting descendant for
 * @param[in]   parent_flow_index
 *                  Flow index of parent for which the
 *                  descendant is being set
 * @param[in]   desc_task
 *                  The descendant
 * @param[in]   desc_flow_index
 *                  Flow index of descendant
 * @param[in]   parent_op_type
 *                  Operation type of parent task on its flow
 * @param[in]   desc_op_type
 *                  Operation type of descendant task on its flow
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
set_parent(parsec_dtd_task_t *parent_task, uint8_t parent_flow_index,
           parsec_dtd_task_t *desc_task, uint8_t desc_flow_index,
           int parent_op_type, int desc_op_type)
{
    parsec_dtd_parent_info_t *parent = PARENT_OF(desc_task, desc_flow_index);
    (void)desc_op_type;
    parent->task       = parent_task;
    parent->op_type    = parent_op_type;
    parent->flow_index = parent_flow_index;
}

/* **************************************************************************** */
/**
 * This function sets the descendant of a task
 *
 * This function is called by the descendant and the descendant here
 * puts itself as the descendant of the parent.
 *
 * @param[out]  parent_task
 *                  Task we are setting descendant for
 * @param[in]   parent_flow_index
 *                  Flow index of parent for which the
 *                  descendant is being set
 * @param[in]   desc_task
 *                  The descendant
 * @param[in]   desc_flow_index
 *                  Flow index of descendant
 * @param[in]   parent_op_type
 *                  Operation type of parent task on its flow
 * @param[in]   desc_op_type
 *                  Operation type of descendant task on its flow
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
set_descendant( parsec_dtd_task_t *parent_task, uint8_t parent_flow_index,
                parsec_dtd_task_t *desc_task, uint8_t desc_flow_index,
                int parent_op_type, int desc_op_type, int last_user_alive )
{
    parsec_dtd_handle_t *parsec_handle  = (parsec_dtd_handle_t *)parent_task->super.parsec_handle;
    parsec_dtd_task_t *real_parent_task = (PARENT_OF(desc_task, desc_flow_index))->task;
    int real_parent_flow_index          = (PARENT_OF(desc_task, desc_flow_index))->flow_index;

    (void)parent_op_type;
    parsec_dtd_descendant_info_t *desc = DESC_OF(parent_task, parent_flow_index);
    desc->flow_index = desc_flow_index;
    desc->op_type    = desc_op_type;
    parsec_mfence();
    desc->task       = desc_task;

#if defined(DISTRIBUTED)
    /* only do if parent is remote and desc is local and parent has not been activated by remote node */
    if( parsec_dtd_task_is_remote(real_parent_task) && parsec_dtd_task_is_local(desc_task) && last_user_alive == TASK_IS_ALIVE) {
        parsec_dtd_flow_info_t *flow = FLOW_OF(real_parent_task, real_parent_flow_index);

        /* Marking we need the parent task */
        if( !(flow->flags & TASK_INSERTED) ) {
            parsec_dtd_remote_task_retain( real_parent_task );
        }

        /* Locking the two hash table */
        parsec_dtd_two_hash_table_lock(parsec_handle->two_hash_table);

        uint64_t key = (uint64_t)(real_parent_task->ht_item.key<<32) | (1U<<real_parent_flow_index);
        parsec_remote_deps_t *dep = parsec_dtd_find_remote_dep( parsec_handle, key );
        if( NULL == dep ) {
            if( !(flow->flags & TASK_INSERTED) ) {
                flow->flags |= TASK_INSERTED;
                parsec_dtd_insert_task( parsec_handle, key, real_parent_task );
            }
        } else {
            if( !(flow->flags & TASK_INSERTED) ) {
                assert(dep->from == real_parent_task->rank);
                flow->flags |= TASK_INSERTED;
                parsec_dtd_find_and_remove_remote_dep( parsec_handle, key );
#if defined(PARSEC_PROF_TRACE)
                parsec_profiling_trace(parsec_handle->super.context->virtual_processes[0]->execution_units[0]->eu_profile, hashtable_trace_keyin, 0, parsec_handle->super.handle_id, NULL );
#endif
                parsec_dtd_insert_task( parsec_handle, key, real_parent_task );
                remote_dep_dequeue_delayed_dep_release(dep);
            }
        }

        /* Unlocking the two hash table */
        parsec_dtd_two_hash_table_unlock(parsec_handle->two_hash_table);
    }
#endif
}

/* **************************************************************************** */
/**
 * Function to push ready task in PaRSEC's scheduler
 *
 * @param[in,out]   __parsec_handle
 *                      DTD handle
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
parsec_dtd_schedule_tasks( parsec_dtd_handle_t *__parsec_handle )
{
    parsec_execution_context_t **startup_list = __parsec_handle->startup_list;
    parsec_list_t temp;

    OBJ_CONSTRUCT( &temp, parsec_list_t );
    for(int p = 0; p < vpmap_get_nb_vp(); p++) {
        if( NULL == startup_list[p] ) continue;

        /* Order the tasks by priority */
        parsec_list_chain_sorted(&temp, (parsec_list_item_t*)startup_list[p],
                                parsec_execution_context_priority_comparator);
        startup_list[p] = (parsec_execution_context_t*)parsec_list_nolock_unchain(&temp);
        /* We should add these tasks on the system queue when there is one */
        __parsec_schedule( __parsec_handle->super.context->virtual_processes[p]->execution_units[0],
                          startup_list[p], 0 );
        startup_list[p] = NULL;
    }
    OBJ_DESTRUCT(&temp);
}

/* **************************************************************************** */
/**
 * Create and initialize a dtd task
 *
 */
parsec_dtd_task_t *
parsec_dtd_create_and_initialize_task( parsec_dtd_handle_t *parsec_dtd_handle,
                                       parsec_function_t   *function,
                                       int rank )
{
    int i;
    parsec_dtd_task_t *this_task;
    assert( NULL != parsec_dtd_handle );
    assert( NULL != function );

    parsec_mempool_t *dtd_task_mempool;
    /* Creating Task object */
    if(parsec_dtd_handle->super.context->my_rank == rank) {
        dtd_task_mempool = &((parsec_dtd_function_t*)function)->context_mempool;
    } else {
        dtd_task_mempool = &((parsec_dtd_function_t*)function)->remote_task_mempool;
    }
    this_task = (parsec_dtd_task_t *)parsec_thread_mempool_allocate(dtd_task_mempool->thread_mempools);

    assert(this_task->super.super.super.obj_reference_count == 1);

    this_task->orig_task = NULL;
    this_task->super.parsec_handle   = (parsec_handle_t*)parsec_dtd_handle;
    this_task->ht_item.key           = parsec_dtd_handle->task_id++;
    /* this is needed for grapher to work properly */
    this_task->super.locals[0].value = (int)this_task->ht_item.key;
    this_task->super.function        = function;
    /**
     * +1 to make sure the task cannot be completed by the potential predecessors,
     * before we are completely done with it here. As we have an atomic operation
     * in all cases, increasing the expected flows by one will have no impact on
     * the performance.
     * */
    this_task->flow_count     = this_task->super.function->nb_flows + 1;
    this_task->rank           = rank;
    this_task->super.priority = 0;
    this_task->super.chore_id = 0;
    this_task->super.status   = PARSEC_TASK_STATUS_NONE;

    int j;
    parsec_dtd_flow_info_t *flow;
    parsec_dtd_descendant_info_t *desc;
    for( i = 0; i < function->nb_flows; i++ ) {
        flow = FLOW_OF(this_task, i);
        if( parsec_dtd_task_is_local(this_task) ) {
            for( j = 0; j < (int)( parsec_dtd_handle->super.context->nb_nodes / (sizeof(int)*8) ) + 1; j++) {
                flow->rank_sent_to[j] = 0;
            }
        }
        flow->op_type = 0;
        flow->tile    = NULL;

        desc = DESC_OF(this_task, i);
        desc->op_type    = 0;
        desc->flow_index = -1;
        desc->task       = NULL;
    }

    return this_task;
}

/* **************************************************************************** */
/**
 * Function to set parameters of a dtd task
 *
 */
void
parsec_dtd_set_params_of_task( parsec_dtd_task_t *this_task, parsec_dtd_tile_t *tile,
                               int tile_op_type, int *flow_index, void **current_val,
                               parsec_dtd_task_param_t *current_param, int arg_size )
{
    if( (tile_op_type & GET_OP_TYPE) == INPUT  ||
        (tile_op_type & GET_OP_TYPE) == OUTPUT ||
        (tile_op_type & GET_OP_TYPE) == INOUT  ||
        (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE)
    {
        this_task->super.data[*flow_index].data_in   = NULL;
        this_task->super.data[*flow_index].data_out  = NULL;
        this_task->super.data[*flow_index].data_repo = NULL;

        parsec_dtd_flow_info_t *flow = FLOW_OF(this_task, *flow_index);
        /* Saving tile pointer for each flow in a task */
        flow->tile    = tile;
        flow->flags   = 0;
        flow->arena_index = -1;
        flow->op_type = tile_op_type;

        *flow_index += 1;
    } else if ((tile_op_type & GET_OP_TYPE) == SCRATCH) {
        assert(parsec_dtd_task_is_local(this_task));
        if(NULL == tile) {
            current_param->pointer_to_tile = *current_val;
           *current_val = ((char*)*current_val) + arg_size;
        } else {
            current_param->pointer_to_tile = (void *)tile;
        }
    } else {  /* We have a VALUE */
        assert(parsec_dtd_task_is_local(this_task));
        memcpy(*current_val, (void *)tile, arg_size);
        current_param->pointer_to_tile = *current_val;
       *current_val = ((char*)*current_val) + arg_size;
    }
}

/* **************************************************************************** */
/**
 * Body of fake task we insert before every INPUT task that reads
 * from memory or remote tasks that reads from local memory
 *
 * @param   context, this_task
 *
 * @ingroup DTD_INTERFACE_INTERNAL
 */
int
fake_first_out_body( parsec_execution_unit_t *context, parsec_execution_context_t *this_task)
{
    (void)context; (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

/* **************************************************************************** */
/**
 * Function to insert dtd task in PaRSEC
 *
 * In this function we track all the dependencies and create the DAG
 *
 */
void
parsec_insert_dtd_task( parsec_dtd_task_t *this_task )
{
    const parsec_function_t *function     =  this_task->super.function;
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)this_task->super.parsec_handle;

    int flow_index, satisfied_flow = 0, tile_op_type = 0, put_in_chain = 1;
    static int vpid = 0;
    parsec_dtd_tile_t *tile = NULL;

    /* Retaining runtime_task */
    parsec_handle_update_runtime_nbtask( this_task->super.parsec_handle, 1 );

    /* Retaining every remote_task */
    if( parsec_dtd_task_is_remote( this_task ) ) {
        parsec_dtd_remote_task_retain( this_task );
    }

    /* In the next segment we resolve the dependencies of each flow */
    for( flow_index = 0, tile = NULL, tile_op_type = 0; flow_index < function->nb_flows; flow_index ++ ) {
        parsec_dtd_tile_user_t last_user, last_writer;
        tile = (FLOW_OF(this_task, flow_index))->tile;
        tile_op_type = (FLOW_OF(this_task, flow_index))->op_type;
        put_in_chain = 1;

        if(0 == parsec_dtd_handle->flow_set_flag[function->function_id]) {
            /* Setting flow in function structure */
            set_flow_in_function( parsec_dtd_handle, this_task, tile_op_type, flow_index);
        }

        if( NULL == tile ) {
            satisfied_flow++;
            continue;
        }

        /* User has instructed us not to track this data */
        if( tile_op_type & DONT_TRACK ) {
            this_task->super.data[flow_index].data_in = tile->data_copy;
            satisfied_flow++;
            continue;
        }

        if( tile->arena_index == -1 ) {
            tile->arena_index = (tile_op_type & GET_REGION_INFO);
        }
        (FLOW_OF(this_task, flow_index))->arena_index = tile->arena_index;

        /* Locking the last_user of the tile */
        parsec_dtd_last_user_lock( &(tile->last_user) );

        READ_FROM_TILE(last_user, tile->last_user);
        READ_FROM_TILE(last_writer, tile->last_writer);

        if( NULL == last_user.task && (this_task->rank != tile->rank || (tile_op_type & GET_OP_TYPE) == INPUT) ) {
            parsec_dtd_last_user_unlock( &(tile->last_user) );

            //OBJ_RETAIN(tile); /* Recreating the effect of inserting a real task using the tile */
            /* parentless */
            /* Create Fake output_task */
            parsec_insert_task( this_task->super.parsec_handle,
                                &fake_first_out_body,   0, "Fake_FIRST_OUT",
                                PASSED_BY_REF,         tile,       INOUT | AFFINITY,
                                0 );

            parsec_dtd_last_user_lock( &(tile->last_user) );

            READ_FROM_TILE(last_user, tile->last_user);
            READ_FROM_TILE(last_writer, tile->last_writer);

            assert( (last_user.task == NULL) || ((FLOW_OF(last_writer.task, last_writer.flow_index))->tile == tile) );
        }

        if( INOUT == (tile_op_type & GET_OP_TYPE) || OUTPUT == (tile_op_type & GET_OP_TYPE) ) {
#if defined(PARSEC_PROF_TRACE)
            this_task->super.prof_info.id = tile->key;
#endif

            /* Setting the last_user info with info of this_task */
            tile->last_writer.task        = this_task;
            tile->last_writer.flow_index  = flow_index;
            tile->last_writer.op_type     = tile_op_type;
            tile->last_writer.alive       = TASK_IS_ALIVE;

            /* retaining every remote_write task */
            if( parsec_dtd_task_is_remote( this_task ) ) {
                parsec_dtd_remote_task_retain( this_task ); /* for every write remote_task */
                if( NULL != last_writer.task ) {
                    if( parsec_dtd_task_is_local( last_writer.task ) ) {
                        parsec_dtd_remote_task_retain( this_task ); /* everytime we have a remote_task as descendant of a local task */
                    }
                }
            }
        } else { /* considering everything else in INPUT */
            if( parsec_dtd_task_is_remote( this_task ) ) {
                if( NULL != last_writer.task ) {
                    if( parsec_dtd_task_is_remote( last_writer.task ) ) {
                        /* if both last writer and this task(read) is remote
                           we do not put them in the chain
                        */
                        put_in_chain = 0;
                    } else {
                        parsec_dtd_remote_task_retain( this_task );
                    }
                } else {
                    assert(0);
                }
            }
        }

        if( put_in_chain ) {
            /* Setting the last_user info with info of this_task */
            tile->last_user.task       = this_task;
            tile->last_user.flow_index = flow_index;
            tile->last_user.op_type    = tile_op_type;
            tile->last_user.alive      = TASK_IS_ALIVE;
        }

        /* Unlocking the last_user of the tile */
        parsec_dtd_last_user_unlock( &(tile->last_user) );

        /* TASK_IS_ALIVE indicates we have a parent */
        if(TASK_IS_ALIVE == last_user.alive) {
            set_parent( last_writer.task, last_writer.flow_index,
                        this_task, flow_index, last_writer.op_type,
                        tile_op_type );

            set_dependencies_for_function( (parsec_handle_t *)parsec_dtd_handle,
                                           (parsec_function_t *)(PARENT_OF(this_task, flow_index))->task->super.function,
                                           (parsec_function_t *)this_task->super.function,
                                            (PARENT_OF(this_task, flow_index))->flow_index, flow_index );

            if( put_in_chain ) {
                assert(NULL != last_user.task);
                set_descendant(last_user.task, last_user.flow_index,
                               this_task, flow_index, last_user.op_type,
                               tile_op_type, last_user.alive);
            }

            /* Are we using the same data multiple times for the same task? */
            if(last_user.task == this_task) {
                satisfied_flow += 1;
                this_task->super.data[flow_index].data_in = tile->data_copy;
                /* We retain data for each flow of a task */
                if( tile->data_copy != NULL ) {
                    parsec_dtd_retain_data_copy(tile->data_copy);
                }

                /* What if we have the same task using the same data in different flows
                 * with the corresponding  operation type on the data : R then W, we are
                 * doomed and this is to not get doomed
                 */
                if(parsec_dtd_task_is_local(this_task)) {
                    /* Checking if a task uses same data on different
                     * flows in the order W, R ...
                     * If yes, we MARK the later flow, as the last flow
                     * needs to release ownership and it is not released
                     * in the case where the last flow is a R.
                     */
                    if( ( (last_user.op_type  & GET_OP_TYPE) == INOUT || (last_user.op_type  & GET_OP_TYPE) == OUTPUT ) && ((tile_op_type & GET_OP_TYPE) == INPUT) ) {
                        FLOW_OF(this_task, flow_index)->flags |= RELEASE_OWNERSHIP_SPECIAL;
                    } else if( (last_user.op_type  & GET_OP_TYPE) == INPUT && (tile_op_type & GET_OP_TYPE) == INPUT ) {
                        /* we unset flag for previous flow and set it for last one */
                        FLOW_OF(last_user.task, last_user.flow_index)->flags &= ~RELEASE_OWNERSHIP_SPECIAL;
                        FLOW_OF(this_task, flow_index)->flags |= RELEASE_OWNERSHIP_SPECIAL;
                    }

                    if( ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT)
                        && (last_user.op_type & GET_OP_TYPE) == INPUT ) {

                        /* clearing bit set to track special release of ownership */
                        FLOW_OF(last_user.task, last_user.flow_index)->flags &= ~RELEASE_OWNERSHIP_SPECIAL;

                        (void)parsec_atomic_add_32b( (int *)&(this_task->super.data[flow_index].data_in->readers) , -1 );

                    }
                }

                if( ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT) ) {
                    if( parsec_dtd_task_is_local(this_task) ) {
                        parsec_dtd_release_local_task( this_task );
                    }
                }
            }
            assert( NULL != last_user.task );
            if( parsec_dtd_task_is_remote(last_user.task) && parsec_dtd_task_is_remote(this_task) ) {
                if( ((parsec_dtd_function_t *)this_task->super.function)->fpointer == parsec_dtd_copy_data_to_matrix ) {
                    if( last_writer.task == last_user.task ) {
                        parsec_dtd_remote_task_release( this_task );
                    }
                }
            }
        } else {  /* Have parent, but parent is not alive
                     We have to call iterate successor on the parent to activate this task
                   */
            if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
                set_dependencies_for_function( (parsec_handle_t *)parsec_dtd_handle, NULL,
                                               (parsec_function_t *)this_task->super.function,
                                                0, flow_index );
            }

            if( last_user.task != NULL ) {
                set_parent( last_writer.task, last_writer.flow_index,
                            this_task, flow_index, last_writer.op_type,
                            tile_op_type );

                set_dependencies_for_function( (parsec_handle_t *)parsec_dtd_handle,
                                               (parsec_function_t *)(PARENT_OF(this_task, flow_index))->task->super.function,
                                               (parsec_function_t *)this_task->super.function,
                                               (PARENT_OF(this_task, flow_index))->flow_index, flow_index );

                /* we can avoid all the hash table crap if the last_writer is not alive */
                if( put_in_chain ) {
                    set_descendant((PARENT_OF(this_task, flow_index))->task, (PARENT_OF(this_task, flow_index))->flow_index,
                                   this_task, flow_index, (PARENT_OF(this_task, flow_index))->op_type,
                                   tile_op_type, last_user.alive);

                    parsec_dtd_task_t *parent_task = (PARENT_OF(this_task, flow_index))->task;
                    if( parsec_dtd_task_is_local(parent_task) || parsec_dtd_task_is_local(this_task) ) {
                        int action_mask = 0;
                        action_mask |= (1<<(PARENT_OF(this_task, flow_index))->flow_index);

                        parsec_execution_unit_t *eu = parsec_dtd_handle->super.context->virtual_processes[0]->execution_units[0];

                        if( parsec_dtd_task_is_local(parent_task) && parsec_dtd_task_is_remote(this_task) ) {
                            /* To make sure we do not release any remote data held by this task */
                            OBJ_RETAIN(parent_task);
                        }
                        release_deps_of_dtd(eu, (parsec_execution_context_t *)(PARENT_OF(this_task, flow_index))->task,
                                            action_mask |
                                            PARSEC_ACTION_SEND_REMOTE_DEPS      |
                                            PARSEC_ACTION_SEND_INIT_REMOTE_DEPS |
                                            PARSEC_ACTION_RELEASE_REMOTE_DEPS   |
                                            PARSEC_ACTION_COMPLETE_LOCAL_TASK   |
                                            PARSEC_ACTION_RELEASE_LOCAL_DEPS , NULL);
                        if( parsec_dtd_task_is_local(parent_task) && parsec_dtd_task_is_remote(this_task) ) {
                            parsec_dtd_release_local_task( parent_task );
                        }
                    } else {
                        if((tile_op_type & GET_OP_TYPE) == INPUT) {
                            parsec_dtd_last_user_lock( &(tile->last_user) );
                            tile->last_user.alive = TASK_IS_NOT_ALIVE;
                            parsec_dtd_last_user_unlock( &(tile->last_user) );
                        }
                        if( ((parsec_dtd_function_t *)this_task->super.function)->fpointer == parsec_dtd_copy_data_to_matrix ) {
                            if( parsec_dtd_task_is_remote(this_task) ) {
                                parsec_dtd_remote_task_release( this_task );
                            }
                        }
                    }
                }
            } else {
                if( ((parsec_dtd_function_t *)this_task->super.function)->fpointer == parsec_dtd_copy_data_to_matrix ) {
                    if( parsec_dtd_task_is_remote(this_task) ) {
                        parsec_dtd_remote_task_release( this_task );
                    }
                }
                this_task->super.data[flow_index].data_in = tile->data_copy;
                satisfied_flow += 1;
                if( tile->data_copy != NULL ) {
                    /* We are using this local data for the first time, let's retain it */
                    parsec_dtd_retain_data_copy(tile->data_copy);
                }
            }
        }

        if( INOUT == (tile_op_type & GET_OP_TYPE) || OUTPUT == (tile_op_type & GET_OP_TYPE) ) {
            if( NULL != last_writer.task ) {
                if( parsec_dtd_task_is_remote( last_writer.task ) ) {
                    /* releasing last writer every time writer is changed */
                    parsec_dtd_remote_task_release( last_writer.task );
                }
            }
        }
    }

    parsec_dtd_handle->flow_set_flag[function->function_id] = 1;

    if( parsec_dtd_task_is_local(this_task) ) {/* Task is local */
        (void)parsec_atomic_add_32b((int *)&(parsec_dtd_handle->super.nb_tasks), 1);
        parsec_dtd_handle->local_task_inserted++;
        if(dump_traversal_info) {
            parsec_output(parsec_debug_output, "Task generated -> %s %d rank %d\n", this_task->super.function->name, this_task->ht_item.key, this_task->rank);
        }
    }

    /* Releasing every remote_task */
    if( parsec_dtd_task_is_remote( this_task ) ) {
        parsec_dtd_remote_task_release( this_task );
    }

    /* Increase the count of satisfied flows to counter-balance the increase in the
     * number of expected flows done during the task creation.  */
    satisfied_flow++;

    if( parsec_dtd_task_is_local(this_task) ) {
        /* Building list of initial ready task */
        if ( 0 == parsec_atomic_add_32b((int *)&(this_task->flow_count), -satisfied_flow) ) {
            if(dump_traversal_info) {
                parsec_output(parsec_debug_output, "------\ntask Ready: %s \t %lld\nTotal flow: %d  flow_count:"
                              "%d\n-----\n", this_task->super.function->name, this_task->ht_item.key,
                              this_task->super.function->nb_flows, this_task->flow_count);
            }

            PARSEC_LIST_ITEM_SINGLETON(this_task);
            __parsec_schedule( parsec_dtd_handle->super.context->virtual_processes[vpid]->execution_units[0],
                               (parsec_execution_context_t *)this_task, 0 );
            vpid = (vpid+1)%parsec_dtd_handle->super.context->nb_vp;
        }
    }

    if( NULL != parsec_dtd_handle->super.context ) {
        if( (parsec_dtd_handle->local_task_inserted % parsec_dtd_handle->task_window_size) == 0 ) {
            if( parsec_dtd_handle->task_window_size < dtd_window_size ) {
                parsec_dtd_handle->task_window_size *= 2;
            } else {
                int task_threshold_count = dtd_threshold_size;
                parsec_execute_and_come_back( parsec_dtd_handle->super.context, &parsec_dtd_handle->super,
                                              task_threshold_count );
            }
        }
    }
}

/* **************************************************************************** */
/**
 * Function to insert task in PaRSEC
 *
 * Each time the user calls it a task is created with the
 * respective parameters the user has passed. For each task
 * class a structure known as "function" is created as well.
 * (e.g. for Cholesky 4 function structures are created for
 * each task class). The flow of data from each task to others
 * and all other dependencies are tracked from this function.
 *
 * @param[in,out]   __parsec_handle
 *                      DTD handle
 * @param[in]       fpointer
 *                      The pointer to the body of the task
 * @param[in]       name
 *                      The name of the task
 * @param[in]       ...
 *                      Variadic parameters of the task
 *
 * @ingroup         DTD_INTERFACE
 */
void
parsec_insert_task( parsec_handle_t  *parsec_handle,
                    parsec_dtd_funcptr_t *fpointer, int priority,
                    char *name_of_kernel, ... )
{
    parsec_dtd_handle_t *parsec_dtd_handle = (parsec_dtd_handle_t *)parsec_handle;

    if( parsec_handle->context == NULL ) {
        parsec_fatal( "Sorry! You can not insert task wihtout enqueuing the handle to parsec_context"
                      " first. Please make sure you call parsec_enqueue(parsec_context, handle) before"
                      " you try inserting task in PaRSEC\n" );
    }

    if( parsec_dtd_handle == NULL ) {
        parsec_fatal( "Wait! You need to pass a correct parsec handle in order to insert task. "
                      "Please use \"parsec_dtd_handle_new()\" to create new handle "
                      "and then try to insert task. Thank you\n" );
    }

    va_list args, args_for_size, args_for_rank;
    int next_arg, tile_op_type, flow_index = 0, this_task_rank = -1;
    void *tile;

    va_start(args, name_of_kernel);

#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_trace(parsec_dtd_handle->super.context->virtual_processes[0]->execution_units[0]->eu_profile, insert_task_trace_keyin, 0, parsec_dtd_handle->super.handle_id, NULL );
#endif

    /* extracting the rank of the task */
    va_copy(args_for_rank, args);
    int write_flow_count = 1;
    int flow_count_of_template = 0;
    while( 0 != (next_arg = va_arg(args_for_rank, int)) ) {
        tile         = va_arg(args_for_rank, void *);
        tile_op_type = va_arg(args_for_rank, int);

        if( !((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) ) {
            if( NULL != tile ) {
                if( (tile_op_type & AFFINITY) ) {
                    this_task_rank = ((parsec_dtd_tile_t *)tile)->rank;
                }
                if( !((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) ) {
                    flow_count_of_template++;
                    if( INOUT == (tile_op_type & GET_OP_TYPE) || OUTPUT == (tile_op_type & GET_OP_TYPE) ) {
                        if( !(tile_op_type & DONT_TRACK) ) {
                            write_flow_count++;
                        }
                    }
                }
            }
        }
    }

    uint64_t fkey = (uint64_t)(uintptr_t)fpointer + flow_count_of_template;
    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    parsec_function_t *function = (parsec_function_t *) parsec_dtd_function_find
                                                       (parsec_dtd_handle, fkey);

    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        flow_count_of_template = 0;
        int count_of_params_sent_by_user = 0;
        long unsigned int size_of_params = 0;

        va_copy(args_for_size, args);

        while( 0 != (next_arg = va_arg(args_for_size, int)) ) {
            tile         = va_arg(args_for_size, void *);
            tile_op_type = va_arg(args_for_size, int);
            count_of_params_sent_by_user++;

            if( (tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH ) {
                size_of_params += next_arg;
            } else {
                flow_count_of_template++;
            }
        }

        va_end(args_for_size);

#if 0
        if (dump_function_info) {
            parsec_output(parsec_debug_output, "Function Created for task Class: %s\n Has %d parameters\n"
                         "Total Size: %lu\n", name_of_kernel, count_of_params_sent_by_user, size_of_params);
        }
#endif

        function = parsec_dtd_create_function( parsec_dtd_handle, fpointer, name_of_kernel,
                                               count_of_params_sent_by_user,
                                               size_of_params, flow_count_of_template );

#if defined(PARSEC_PROF_TRACE)
        parsec_dtd_add_profiling_info((parsec_handle_t *)parsec_dtd_handle, function->function_id, name_of_kernel);
#endif /* defined(PARSEC_PROF_TRACE) */
    }

#if defined(DISTRIBUTED)
    if( parsec_handle->context->nb_nodes > 1 ) {
        if( (-1 == this_task_rank) && (write_flow_count > 1) ) {
            parsec_fatal( "You inserted a task with out indicating where the task should be executed(using AFFINITY flag)."
                          "This will result in executing this task on all nodes and the outcome might be not be what you want."
                          "So we are exiting for now. Please see the usage of AFFINITY flag.\n" );
        } else if( this_task_rank == -1 && write_flow_count == 1 ) {
            /* we have tasks with no real data as parameter so we are safe to execute it in each mpi process */
            this_task_rank = parsec_handle->context->my_rank;
        }
    } else {
        this_task_rank = 0;
    }
#else
    this_task_rank = 0;
#endif

    va_end(args_for_rank);

    parsec_dtd_task_t *this_task = parsec_dtd_create_and_initialize_task(parsec_dtd_handle, function, this_task_rank);
    this_task->super.priority = priority;

    parsec_object_t *object = (parsec_object_t *)this_task;
    if( parsec_dtd_task_is_local(this_task) ) {
        /* retaining the local task as many write flows as it has and one to indicate when we have executed the task */
        (void)parsec_atomic_add_32b( &object->obj_reference_count, (write_flow_count) );

        /* Iterating through the parameters of the task */
        parsec_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param = NULL;
        void *value_block, *current_val;

        /* Getting the pointer to allocated memory by mempool */
        head_of_param_list = GET_HEAD_OF_PARAM_LIST(this_task);
        current_param      = head_of_param_list;
        value_block        = GET_VALUE_BLOCK(head_of_param_list, ((parsec_dtd_function_t*)function)->count_of_params);
        current_val        = value_block;

        while( 0 != (next_arg = va_arg(args, int)) ) {
            tile         = (parsec_dtd_tile_t *)va_arg(args, void *);
            tile_op_type = va_arg(args, int);

            parsec_dtd_set_params_of_task( this_task, tile, tile_op_type,
                                           &flow_index, &current_val,
                                           current_param, next_arg );

            tmp_param = current_param;
            current_param = current_param + 1;
            tmp_param->next = current_param;
        }

        if( tmp_param != NULL )
            tmp_param->next = NULL;
    } else {
        while( 0 != (next_arg = va_arg(args, int)) ) {
            tile         = (parsec_dtd_tile_t *)va_arg(args, void *);
            tile_op_type = va_arg(args, int);

            if( !((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) ) {
                parsec_dtd_set_params_of_task( this_task, tile, tile_op_type,
                                               &flow_index, NULL,
                                               NULL, next_arg );
            }
        }
    }
    va_end(args);

#if defined(DISTRIBUTED)
    assert(this_task->rank != -1);
#endif

    parsec_insert_dtd_task( this_task );

#if defined(PARSEC_PROF_TRACE)
    /* try using PARSEC_PROFILING_TRACE */
    parsec_profiling_trace(parsec_dtd_handle->super.context->virtual_processes[0]->execution_units[0]->eu_profile, insert_task_trace_keyout, 0, parsec_dtd_handle->super.handle_id, NULL );
#endif

}
