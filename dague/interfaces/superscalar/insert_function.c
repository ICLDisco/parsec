/**
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file insert_function.c
 *
 * @version 2.0.0
 * @author Reazul Hoque
 *
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup DTD_INTERFACE DTD_INTERFACE: Dynamic Task Discovery interface for PaRSEC
 *
 * These functions are available from the PaRSEC library for the
 * scheduling of kernel routines.
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup DTD_INTERFACE_INTERNAL DTD_INTERFACE_INTERNAL: Internal Dynamic Task Discovery functions for PaRSEC
 *
 * These functions are not available from the PaRSEC library for the
 * scheduling of kernel routines.
 */

#include <stdlib.h>
#include <sys/time.h>

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "dague/scheduling.h"
#include "dague/remote_dep.h"
#include "dague/devices/device.h"
#include "dague/constants.h"
#include "dague/vpmap.h"
#include "dague/utils/mca_param.h"
#include "dague/mca/sched/sched.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"
#include "dague/dague_prof_grapher.h"

int dtd_window_size             =  2048; /**< Default window size */
uint32_t dtd_threshold_size     =  2048; /**< Default threshold size of tasks for master thread to wait on */
static int task_hash_table_size = (10+1); /**< Default task hash table size */
static int tile_hash_table_size =  104729; /**< Default tile hash table size */

int my_rank = -1;
int dump_traversal_info; /**< For printing traversal info */
int dump_function_info; /**< For printing function_structure info */

/* Master structures for Fake_writer tasks */
static dague_dtd_function_t *__dague_dtd_master_fake_function;

extern dague_sched_module_t *current_scheduler;

/* Global mempool for all the dague handles that will be created for a run */
dague_mempool_t *handle_mempool = NULL;

/**
 * All the static functions should be declared before being defined.
 */
static int
call_to_fake_writer( dague_execution_unit_t *context, dague_execution_context_t *this_task);

static int
hook_of_dtd_task(dague_execution_unit_t *context,
                      dague_execution_context_t *this_task);

static int
dtd_is_ready(const dague_dtd_task_t *dest);

static void
iterate_successors_of_dtd_task(dague_execution_unit_t *eu,
                               const dague_execution_context_t *this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t *ontask,
                               void *ontask_arg);
static int
release_deps_of_dtd(dague_execution_unit_t *,
                    dague_execution_context_t *,
                    uint32_t, dague_remote_deps_t *);

static dague_hook_return_t
complete_hook_of_dtd(dague_execution_unit_t *,
                     dague_execution_context_t *);

/* Copied from dague/scheduling.c, will need to be exposed */
#define TIME_STEP 5410
#define MIN(x, y) ( (x)<(y)?(x):(y) )
static inline unsigned long exponential_backoff(uint64_t k)
{
    unsigned int n = MIN( 64, k );
    unsigned int r = (unsigned int) ((double)n * ((double)rand()/(double)RAND_MAX));
    return r * TIME_STEP;
}

/* To create object of class dague_dtd_task_t that inherits dague_execution_context_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_task_t, dague_execution_context_t,
                   NULL, NULL);

/* To create object of class .list_itemdtd_tile_t that inherits dague_list_item_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_tile_t, dague_hashtable_item_t,
                   NULL, NULL);

/* To create object of class dague_handle_t that inherits dague_list_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_handle_t, dague_list_item_t,
                   NULL, NULL);

/***************************************************************************//**
 *
 * Constructor of PaRSEC's DTD handle.
 *
 * @param[in,out]   dague_handle
 *                      Pointer to handle which will be constructed
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
void
dague_dtd_handle_constructor
(dague_dtd_handle_t *dague_handle)
{
    int i;

    dague_handle->startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*));

    dague_handle->function_counter          = 0;

    dague_handle->task_h_table              = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->task_h_table,
                    task_hash_table_size,
                    &hash_key);
    dague_handle->tile_h_table              = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->tile_h_table,
                    tile_hash_table_size,
                    &hash_key);
    dague_handle->function_h_table          = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->function_h_table,
                    DAGUE_dtd_NB_FUNCTIONS,
                    &hash_key);

    dague_handle->super.startup_hook        = dtd_startup;
    dague_handle->super.destructor          = (dague_destruct_fn_t) dague_dtd_handle_destruct;
    dague_handle->super.functions_array     = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));

    for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_handle->super.functions_array[i] = NULL;
    }

    dague_handle->super.dependencies_array  = calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    dague_handle->arenas_size               = 1;
    dague_handle->arenas = (dague_arena_t **) malloc(dague_handle->arenas_size * sizeof(dague_arena_t *));

    for (i = 0; i < dague_handle->arenas_size; i++) {
        dague_handle->arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

#if defined(DAGUE_PROF_TRACE)
    dague_handle->super.profiling_array     = calloc (2 * DAGUE_dtd_NB_FUNCTIONS , sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */

    /* Initializing the tile mempool and attaching it to the dague_handle */
    dague_dtd_tile_t fake_tile;
    dague_handle->tile_mempool          = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( dague_handle->tile_mempool,
                             OBJ_CLASS(dague_dtd_tile_t), sizeof(dague_dtd_tile_t),
                             ((char*)&fake_tile.super.mempool_owner) - ((char*)&fake_tile),
                             1/* no. of threads*/ );

    /* Initializing hash_table_bucket mempool and attaching it to the dague_handle */
    dague_generic_bucket_t fake_bucket;
    dague_handle->hash_table_bucket_mempool = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( dague_handle->hash_table_bucket_mempool,
                             OBJ_CLASS(dague_generic_bucket_t), sizeof(dague_generic_bucket_t),
                             ((char*)&fake_bucket.super.mempool_owner) - ((char*)&fake_bucket),
                             1/* no. of threads*/ );

}

/***************************************************************************//**
 *
 * Destructor of PaRSEC's DTD handle.
 *
 * @param[in,out]   dague_handle
 *                      Pointer to handle which will be destroyed
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 *
 ******************************************************************************/
void
dague_dtd_handle_destructor
(dague_dtd_handle_t *dague_handle)
{
    int i;
#if defined(DAGUE_PROF_TRACE)
    free((void *)dague_handle->super.profiling_array);
#endif /* defined(DAGUE_PROF_TRACE) */

    free(dague_handle->super.functions_array);
    dague_handle->super.nb_functions = 0;

    for (i = 0; i < dague_handle->arenas_size; i++) {
        if (dague_handle->arenas[i] != NULL) {
            free(dague_handle->arenas[i]);
            dague_handle->arenas[i] = NULL;
        }
    }

    free(dague_handle->arenas);
    dague_handle->arenas      = NULL;
    dague_handle->arenas_size = 0;

    /* Destroy the data repositories for this object */
    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_destruct_dependencies(dague_handle->super.dependencies_array[i]);
        dague_handle->super.dependencies_array[i] = NULL;
    }

    free(dague_handle->super.dependencies_array);
    dague_handle->super.dependencies_array = NULL;

    /* Unregister the handle from the devices */
    for (i = 0; i < (int)dague_nb_devices; i++) {
        if (!(dague_handle->super.devices_mask & (1 << i)))
            continue;
        dague_handle->super.devices_mask ^= (1 << i);
        dague_device_t *device = dague_devices_get(i);
        if ((NULL == device) || (NULL == device->device_handle_unregister))
            continue;
        if (DAGUE_SUCCESS != device->device_handle_unregister(device, &dague_handle->super))
            continue;
    }

    /* dtd_handle specific */
    dague_mempool_destruct(dague_handle->tile_mempool);
    free (dague_handle->tile_mempool);
    dague_mempool_destruct(dague_handle->hash_table_bucket_mempool);
    free (dague_handle->hash_table_bucket_mempool);
    free(dague_handle->startup_list);

    hash_table_fini(dague_handle->task_h_table, dague_handle->task_h_table->size);
    hash_table_fini(dague_handle->tile_h_table, dague_handle->tile_h_table->size);
    hash_table_fini(dague_handle->function_h_table, dague_handle->function_h_table->size);
}

/* To create object of class dague_dtd_handle_t that inherits dague_handle_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_handle_t, dague_handle_t,
                   dague_dtd_handle_constructor, dague_dtd_handle_destructor);


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
dague_dtd_init()
{
    dague_dtd_handle_t  fake_handle, *dague_handle;

    handle_mempool          = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( handle_mempool,
                             OBJ_CLASS(dague_dtd_handle_t), sizeof(dague_dtd_handle_t),
                             ((char*)&fake_handle.mempool_owner) - ((char*)&fake_handle),
                             1/* no. of threads*/ );

    dague_handle = (dague_dtd_handle_t *)dague_thread_mempool_allocate(handle_mempool->thread_mempools);
    dague_thread_mempool_free( handle_mempool->thread_mempools, dague_handle );

    /* Registering mca param for printing out traversal info */
    (void)dague_mca_param_reg_int_name("dtd", "traversal_info",
                                       "Show graph traversal info",
                                       false, false, 0, &dump_traversal_info);

    /* Registering mca param for printing out function_structure info */
    (void)dague_mca_param_reg_int_name("dtd", "function_info",
                                       "Show master structure info",
                                       false, false, 0, &dump_function_info);

    /* Registering mca param for tile hash table size */
    (void)dague_mca_param_reg_int_name("dtd", "tile_hash_size",
                                       "Registers the supplied size overriding the default size of tile hash table",
                                       false, false, tile_hash_table_size, &tile_hash_table_size);

    /* Registering mca param for task hash table size */
    (void)dague_mca_param_reg_int_name("dtd", "task_hash_size",
                                       "Registers the supplied size overriding the default size of task hash table",
                                       false, false, task_hash_table_size, &task_hash_table_size);

    /* Registering mca param for window size */
    (void)dague_mca_param_reg_int_name("dtd", "window_size",
                                       "Registers the supplied size overriding the default size of window size",
                                       false, false, dtd_window_size, &dtd_window_size);

    /* Registering mca param for threshold size */
    (void)dague_mca_param_reg_int_name("dtd", "threshold_size",
                                       "Registers the supplied size overriding the default size of threshold size",
                                       false, false, dtd_threshold_size, (int *)&dtd_threshold_size);
}

/* **************************************************************************** */
/**
 * Fini function of Dynamic Task Discovery Interface.
 *
 * The global mempool of dague_dtd_handle is destroyed here.
 *
 * @ingroup DTD_INTERFACE
 */
void
dague_dtd_fini()
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert(handle_mempool != NULL);
#endif

    dague_mempool_destruct(handle_mempool);
    free (handle_mempool);
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
 * @param[in]   dague_handle
 *                  PaRSEC dtd handle
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
dague_execute_and_come_back(dague_context_t *context,
                            dague_handle_t *dague_handle)
{
    dague_dtd_handle_t *dtd_handle = (dague_dtd_handle_t *)dague_handle;
    uint64_t misses_in_a_row;
    dague_execution_unit_t* eu_context = context->virtual_processes[0]->execution_units[0];
    dague_execution_context_t* exec_context;
    int nbiterations = 0;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    /* Checking if the context has been started or not */
    /* The master thread might not have to trigger the barrier if the other
     * threads have been activated by a previous start.
     */
    if( DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags ) {

    } else {
        (void)dague_remote_dep_on(context);
        /* Mark the context so that we will skip the initial barrier during the _wait */
        context->flags |= DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE;
        /* Wake up the other threads */
        dague_barrier_wait( &(context->barrier) );
    }

    while(dague_handle->nb_tasks > dtd_handle->task_threshold_size ) {
        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }

        exec_context = current_scheduler->module.select(eu_context);

        if( exec_context != NULL ) {
            PINS(eu_context, SELECT_END, exec_context);
            misses_in_a_row = 0;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
            {
                uint32_t my_idx = dague_atomic_inc_32b(&sched_priority_trace_counter);
                if(my_idx < DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
                    sched_priority_trace[my_idx].step = eu_context->sched_nb_tasks_done++;
                    sched_priority_trace[my_idx].thread_id = eu_context->th_id;
                    sched_priority_trace[my_idx].vp_id     = eu_context->virtual_process->vp_id;
                    sched_priority_trace[my_idx].priority  = exec_context->priority;
                }
            }
#endif

            PINS(eu_context, PREPARE_INPUT_BEGIN, exec_context);
            switch( exec_context->function->prepare_input(eu_context, exec_context) ) {
            case DAGUE_HOOK_RETURN_DONE:
            {
                PINS(eu_context, PREPARE_INPUT_END, exec_context);
                int rv = 0;
                /* We're good to go ... */
                rv = __dague_execute( eu_context, exec_context );
                if( 0 == rv ) {
                    __dague_complete_execution( eu_context, exec_context );
                }
                nbiterations++;
                break;
            }
            default:
                assert( 0 ); /* Internal error: invalid return value for data_lookup function */
            }

            PINS(eu_context, SELECT_BEGIN, NULL);
        } else {
            misses_in_a_row++;
        }
    }
}

/* **************************************************************************** */
/**
 * Function to end the execution of a DTD handle
 *
 * This function should be called exactly once for each handle to
 * detach the handle from the PaRSEC context. This should be called
 * after the user do not want to attach any task colection to the
 * handle any more(done with this handle). After this function is called the handle can
 * not be used any more. This function should be called for exactly
 * as many times as PaRSEC DTD handles are created.
 *
 * @param[in]       dague
 *                      The PaRSEC context
 * @param[in,out]   dague_handle
 *                      PaRSEC dtd handle
 *
 * @ingroup         DTD_INTERFACE
 */
void
dague_dtd_context_wait_on_handle( dague_context_t     *dague,
                                  dague_dtd_handle_t  *dague_handle )
{
    (void)dague;
    /* decrementing the extra task we initialized the handle with */
    dague_handle_update_nbtask( &(dague_handle->super), -1);

    /* We are checking if we have any handle still waiting to
     * be comepleted, if not we call the final function to
     * finish the run */
    if (dague->active_objects == 0)
        dague_context_wait(dague);
}

/* **************************************************************************** */
/**
 * Function to call when PaRSEC context should wait on a specific handle.
 *
 * This function is called to execute a task collection attached to the
 * handle by the user. This function will schedule all the initially ready
 * tasks in the engine and return when all the pending tasks are executed.
 * Users should call this function everytime they insert a bunch of tasks.
 * Calling this function on a DTD handle for which
 * dague_dtd_context_wait_on_handle() has been already called will result
 * in undefined behavior. Users can call this function any number of times
 * on a DTD handle before dague_dtd_context_wait_on_handle() is called for
 * that DTD handle.
 *
 * @param[in]       dague
 *                      The PaRSEC context
 * @param[in,out]   dague_handle
 *                      PaRSEC dtd handle
 *
 * @ingroup         DTD_INTERFACE
 */
void
dague_dtd_handle_wait( dague_context_t     *dague,
                       dague_dtd_handle_t  *dague_handle )
{
    (void)dague;
    /* Scheduling all the remaining tasks */
    schedule_tasks (dague_handle);

    uint32_t tmp_threshold = dague_handle->task_threshold_size;
    dague_handle->task_threshold_size = 1;
    dague_execute_and_come_back (dague_handle->super.context, &dague_handle->super);
    dague_handle->task_threshold_size = tmp_threshold;
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
dague_dtd_unpack_args(dague_execution_context_t *this_task, ...)
{
    dague_dtd_task_t *current_task = (dague_dtd_task_t *)this_task;
    dague_dtd_task_param_t *current_param = current_task->param_list;
    int next_arg;
    int i = 0;
    void **tmp;
    va_list arguments;
    va_start(arguments, this_task);
    next_arg = va_arg(arguments, int);

    while (current_param != NULL) {
        tmp = va_arg(arguments, void**);
        if(UNPACK_VALUE == next_arg) {
            *tmp = current_param->pointer_to_tile;
        }else if (UNPACK_DATA == next_arg) {
            /* Let's return directly the usable pointer to the user */
            *tmp = DAGUE_DATA_COPY_GET_PTR(this_task->data[i].data_out);
            i++;
        }else if (UNPACK_SCRATCH == next_arg) {
            *tmp = current_param->pointer_to_tile;
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}

#if defined(DAGUE_PROF_TRACE)
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
    snprintf(str,12,"fill:%s",color);
    free(color);
    return str;
}

/* **************************************************************************** */
/**
 * This function adds info about a task class into a global dictionary
 * used for profiling.
 *
 * @param[in]   __dague_handle
 *                  The pointer to the DTD handle
 * @param[in]   function
 *                  Pointer to the master structure representing a
 *                  task class
 * @param[in]   name
 *                  Name of the task class
 * @param[in]   flow_count
 *                  Total number of flows of the task class
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
add_profiling_info( dague_dtd_handle_t *__dague_handle,
                    dague_function_t *function, char* name,
                    int flow_count )
{
    char *str = fill_color(function->function_id, DAGUE_dtd_NB_FUNCTIONS);
    dague_profiling_add_dictionary_keyword(name, str,
                                           sizeof(dague_profile_ddesc_info_t) + flow_count * sizeof(assignment_t),
                                           DAGUE_PROFILE_DDESC_INFO_CONVERTOR,
                                           (int *) &__dague_handle->super.profiling_array[0 +
                                                                                          2 *
                                                                                          function->function_id
                                                                                          /* start key */
                                                                                          ],
                                           (int *) &__dague_handle->super.profiling_array[1 +
                                                                                          2 *
                                                                                          function->function_id
                                                                                          /*  end key */
                                                                                          ]);
    free(str);

}
#endif /* defined(DAGUE_PROF_TRACE) */

/* **************************************************************************** */
/**
 * This function produces a hash from a key and a size
 *
 * This function returns a hash for a key. The hash is produced
 * by the following operation key % size
 *
 * @param[in]   key
 *                  The key to be hashed
 * @param[in]   size
 *                  Size of the hash table
 * @return
 *              The hash for the key provided
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
uint32_t
hash_key (uintptr_t key, int size)
{
    uint32_t hash_val = key % size;
    return hash_val;
}

/* **************************************************************************** */
/**
 * This function inserts a DTD task into task hash table(for distributed)
 *
 * @param[in,out]  dague_handle
 *                      Pointer to DTD handle, the task hash table
 *                      is attached to the handle
 * @param[in]       value
 *                      The task to be inserted
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
dague_dtd_task_insert( dague_dtd_handle_t   *dague_handle,
                       dague_dtd_task_t     *value )
{
    uint64_t    key         = value->super.super.key;
    hash_table *hash_table  =  dague_handle->task_h_table;
    uint32_t    hash        =  hash_table->hash ( key, hash_table->size );

    hash_table_insert ( hash_table, (dague_hashtable_item_t *)value, hash );
}

/* **************************************************************************** */
/**
 * This function removes a DTD task from a task hash table(for distributed)
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the task hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The key of the task to be removed
 * @return
 *                  The pointer to the task removed from the hash table
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void *
dague_dtd_task_remove( dague_dtd_handle_t  *dague_handle,
                       uint32_t             key )
{
    hash_table *hash_table      =  dague_handle->task_h_table;
    uint32_t    hash            =  hash_table->hash ( key, hash_table->size );

    return hash_table_remove ( hash_table, key, hash );
}

/* **************************************************************************** */
/**
 * This function searches for a DTD task in task hash table(for distributed)
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the task hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The key of the task to be removed
 * @return
 *                  The pointer to the task if found, NULL otherwise
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
dague_hashtable_item_t *
dague_dtd_task_find_internal( dague_dtd_handle_t  *dague_handle,
                              uint32_t             key )
{
    hash_table *hash_table      =  dague_handle->task_h_table;
    uint32_t    hash            =  hash_table->hash( key, hash_table->size );

    return hash_table_find_no_lock ( hash_table, key, hash );
}

/* **************************************************************************** */
/**
 * This function calls API to find a DTD task in task hash table(for distributed)
 *
 * @see             dague_dtd_task_find_internal()
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the task hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The key of the task to be removed
 * @return
 *                  The pointer to the task if found, NULL otherwise
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
dague_dtd_task_t *
dague_dtd_task_find( dague_dtd_handle_t  *dague_handle,
                     uint32_t             key )
{
    dague_dtd_task_t *task = (dague_dtd_task_t *)dague_dtd_task_find_internal ( dague_handle, key );
    return task;
}

/* **************************************************************************** */
/**
 * This function inserts master structure in hash table
 *
 * @param[in,out]   dague_handle
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
dague_dtd_function_insert( dague_dtd_handle_t   *dague_handle,
                           dague_dtd_funcptr_t  *key,
                           dague_dtd_function_t *value )
{
    dague_generic_bucket_t *bucket  =  (dague_generic_bucket_t *)dague_thread_mempool_allocate(dague_handle->hash_table_bucket_mempool->thread_mempools);

    hash_table *hash_table          =  dague_handle->function_h_table;
    uint32_t    hash                =  hash_table->hash ( (uint64_t)key, hash_table->size );

    bucket->super.key   = (uint64_t)key;
    bucket->value       = (void *)value;

    hash_table_insert ( hash_table, (dague_hashtable_item_t *)bucket, hash );
}

/* **************************************************************************** */
/**
 * This function removes master structure from hash table
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the hash table
 *                      is attached to the handle
 * @param[in]       key
 *                      The function-pointer to the body of task-class
 *                      is treated as the key
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
dague_dtd_function_remove( dague_dtd_handle_t  *dague_handle,
                           dague_dtd_funcptr_t *key )
{
    hash_table *hash_table      =  dague_handle->function_h_table;
    uint32_t    hash            =  hash_table->hash ( (uint64_t)key, hash_table->size );

    hash_table_remove ( hash_table, (uint64_t)key, hash );
}

/* **************************************************************************** */
/**
 * This function searches for master-structure in hash table
 *
 * @param[in,out]   dague_handle
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
dague_generic_bucket_t *
dague_dtd_function_find_internal( dague_dtd_handle_t  *dague_handle,
                                  dague_dtd_funcptr_t *key )
{
    hash_table *hash_table      =  dague_handle->function_h_table;
    uint32_t    hash            =  hash_table->hash( (uint64_t)key, hash_table->size );

    return (dague_generic_bucket_t *)hash_table_find_no_lock ( hash_table, (uint64_t)key, hash );
}

/* **************************************************************************** */
/**
 * This function internal API to search for master-structure in hash table
 *
 * @see             dague_dtd_function_find_internal()
 *
 * @param[in,out]   dague_handle
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
dague_dtd_function_t *
dague_dtd_function_find( dague_dtd_handle_t  *dague_handle,
                         dague_dtd_funcptr_t *key )
{
    dague_generic_bucket_t *bucket = dague_dtd_function_find_internal ( dague_handle, key );
    if( bucket != NULL ) {
        return (dague_dtd_function_t *)bucket->value;
    } else {
        return NULL;
    }
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
 * @param[in,out]   dague_handle
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
dague_dtd_tile_insert( dague_dtd_handle_t *dague_handle, uint32_t key,
                       dague_dtd_tile_t   *tile,
                       dague_ddesc_t      *ddesc )
{
    hash_table *hash_table   =  dague_handle->tile_h_table;
    uint64_t    combined_key = (uint64_t)ddesc << 32 | (uint64_t)key;
    uint32_t    hash         =  hash_table->hash ( combined_key, hash_table->size );

    tile->super.key = combined_key;

    hash_table_insert ( hash_table, (dague_hashtable_item_t *)tile, hash );
}

/* **************************************************************************** */
/**
 * This function removes DTD tile from tile hash table
 *
 * @param[in,out]   dague_handle
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
dague_dtd_tile_remove( dague_dtd_handle_t *dague_handle, uint32_t key,
                       dague_ddesc_t      *ddesc )
{
    hash_table *hash_table   =  dague_handle->tile_h_table;
    uint64_t    combined_key = (uint64_t)ddesc << 32 | (uint64_t)key;
    uint32_t    hash         =  hash_table->hash ( combined_key, hash_table->size );

    hash_table_remove ( hash_table, combined_key, hash );
}

/* **************************************************************************** */
/**
 * This function searches a DTD tile in the tile hash table
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The key of the tile
 * @param[in]       ddesc
 *                      Pointer to the ddesc the tile belongs to
 *
 * @ingroup         DTD_ITERFACE_INTERNAL
 */
dague_dtd_tile_t *
dague_dtd_tile_find( dague_dtd_handle_t *dague_handle, uint32_t key,
                     dague_ddesc_t      *ddesc )
{
    hash_table *hash_table   =  dague_handle->tile_h_table;
    uint64_t    combined_key = (uint64_t)ddesc << 32 | (uint64_t)key;
    uint32_t    hash         =  hash_table->hash ( combined_key, hash_table->size );

    dague_list_t *list = hash_table->item_list[hash];

    dague_list_lock( list );
    dague_dtd_tile_t *tile = hash_table_find_no_lock ( hash_table, combined_key, hash );
    if( NULL != tile ) {
        OBJ_RETAIN(tile);
    }
    dague_list_unlock( list );
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
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       tile
 *                      Tile to be released
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
dague_dtd_tile_release(dague_dtd_handle_t *dague_handle, dague_dtd_tile_t *tile)
{
    assert(tile->super.list_item.super.obj_reference_count>1);
    dague_dtd_tile_remove ( dague_handle, tile->key, tile->ddesc );
    if( tile->super.list_item.super.obj_reference_count == 1 ) {
#if defined(DAGUE_DEBUG_PARANOID)
        assert(tile->super.list_item.refcount == 0);
#endif
        dague_thread_mempool_free( dague_handle->tile_mempool->thread_mempools, tile );
    }
}

/* **************************************************************************** */
/**
 * This function releases the master-structure and pushes them back in mempool
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The function pointer to the body of the task class
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
dague_dtd_function_release( dague_dtd_handle_t  *dague_handle,
                            dague_dtd_funcptr_t *key )
{
    dague_generic_bucket_t *bucket = dague_dtd_function_find_internal ( dague_handle, key );
#if defined(DAGUE_DEBUG_PARANOID)
    assert (bucket != NULL);
#endif
    dague_dtd_function_remove ( dague_handle, key );
    dague_thread_mempool_free( dague_handle->hash_table_bucket_mempool->thread_mempools, bucket );
}

/* **************************************************************************** */
/**
 * Function to release the task structures inserted into the hash table
 *
 * @param[in,out]   dague_handle
 *                      Pointer to DTD handle, the tile hash table
 *                      is attached to this handle
 * @param[in]       key
 *                      The unique key of the task
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
dague_dtd_task_release( dague_dtd_handle_t  *dague_handle,
                        uint32_t             key )
{
    dague_dtd_task_remove ( dague_handle, key );
}

/* **************************************************************************** */
/**
 * Function to recover tiles inserted by insert_task()
 *
 * This function search for a tile if already inserted in the system,
 * and if not returns the freshly created tile.
 *
 * @param[in,out]   dague handle
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
dague_dtd_tile_t*
dague_dtd_tile_of(dague_dtd_handle_t *dague_dtd_handle,
                  dague_ddesc_t *ddesc, int i, int j)
{
    dague_dtd_tile_t *tmp = dague_dtd_tile_find ( dague_dtd_handle, ddesc->data_key(ddesc, i, j),
                                                  ddesc );
    if( NULL == tmp ) {
        /* Creating Tile object */
        dague_dtd_tile_t *temp_tile = (dague_dtd_tile_t *) dague_thread_mempool_allocate
                                                          (dague_dtd_handle->tile_mempool->thread_mempools);
#if defined(DAGUE_DEBUG_PARANOID)
        assert(temp_tile->super.list_item.refcount == 0);
#endif
        temp_tile->key                  = ddesc->data_key(ddesc, i, j);
        temp_tile->rank                 = ddesc->rank_of_key(ddesc, temp_tile->key);
        temp_tile->vp_id                = ddesc->vpid_of_key(ddesc, temp_tile->key);
        temp_tile->data                 = ddesc->data_of_key(ddesc, temp_tile->key);
        temp_tile->data_copy            = temp_tile->data->device_copies[0];
        temp_tile->ddesc                = ddesc;
        temp_tile->last_user.flow_index = -1;
        temp_tile->last_user.op_type    = -1;
        temp_tile->last_user.task       = NULL;

        dague_dtd_tile_insert ( dague_dtd_handle, temp_tile->key,
                                temp_tile, ddesc );
        return temp_tile;
    }else {
#if defined(DAGUE_DEBUG_PARANOID)
        assert(tmp->super.list_item.super.obj_reference_count > 0);
#endif
        return tmp;
    }
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
 *              Returns DAGUE_HOOK_RETURN_DONE if the task was
 *              successfully executed, anything else otherwise
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
static int
hook_of_dtd_task( dague_execution_unit_t    *context,
                  dague_execution_context_t *this_task )
{
    dague_dtd_task_t   *dtd_task   = (dague_dtd_task_t*)this_task;
    int rc = 0;

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id],
                          this_task);

    rc = dtd_task->fpointer(context, this_task);
#if defined(DAGUE_DEBUG_PARANOID)
    assert( rc == DAGUE_HOOK_RETURN_DONE );
#endif

    return rc;
}

/* chores and dague_function_t structure initialization */
static const __dague_chore_t dtd_chore[] = {
    {.type      = DAGUE_DEV_CPU,
     .evaluate  = NULL,
     .hook      = hook_of_dtd_task },
    {.type      = DAGUE_DEV_NONE,
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
static inline uint64_t DTD_identity_hash(const dague_dtd_handle_t * __dague_handle,
                                         const assignment_t * assignments)
{
    (void)__dague_handle;
    return (uint64_t)assignments[0].value;
}

/* **************************************************************************** */
/**
 * Intializes all the needed members and returns the DTD handle
 *
 * For correct profiling the task_class_counter should be correct
 *
 * @param[in]   context
 *                  The DAGUE context
 * @param[in]   arena_count
 *                  The count of the task class DTD handle will deal with
 * @return
 *              The DAGUE DTD handle
 *
 * @ingroup     DTD_INTERFACE
 */
dague_dtd_handle_t *
dague_dtd_handle_new( dague_context_t *context)
{
    if (dump_traversal_info) {
        printf("\n\n------ New Handle -----\n\n\n");
    }

    my_rank = context->my_rank;
    dague_dtd_handle_t *__dague_handle;
    int i;

#if defined(DAGUE_DEBUG_PARANOID)
    assert( handle_mempool != NULL );
#endif
    __dague_handle = (dague_dtd_handle_t *)dague_thread_mempool_allocate(handle_mempool->thread_mempools);

    __dague_handle->super.context             = context;
    __dague_handle->super.on_enqueue          = NULL;
    __dague_handle->super.on_enqueue_data     = NULL;
    __dague_handle->super.on_complete         = NULL;
    __dague_handle->super.on_complete_data    = NULL;
    __dague_handle->super.devices_mask        = DAGUE_DEVICES_ALL;
    __dague_handle->super.nb_functions        = DAGUE_dtd_NB_FUNCTIONS;
    __dague_handle->super.nb_tasks            = 1;  /* For the bounded window, starting with +1 task */
    __dague_handle->super.nb_pending_actions  = 1;  /* For the future tasks that will be inserted */

    for(i = 0; i < vpmap_get_nb_vp(); i++) {
        __dague_handle->startup_list[i] = NULL;
    }

    /* Keeping track of total tasks to be executed per handle for the window */
    for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        __dague_handle->flow_set_flag[i]  = 0;
        /* Added new */
        __dague_handle->super.functions_array[i] = NULL;
    }

    __dague_handle->task_id               = 0;
    __dague_handle->task_window_size      = 1;
    __dague_handle->task_threshold_size   = dtd_threshold_size;
    __dague_handle->function_counter      = 0;

    (void)dague_handle_reserve_id((dague_handle_t *) __dague_handle);
    __dague_dtd_master_fake_function = (dague_dtd_function_t *)create_function(__dague_handle, call_to_fake_writer, "Fake_writer", 1,
                                                                               sizeof(int), 1);
    (void)dague_handle_enable((dague_handle_t *)__dague_handle, NULL, NULL, NULL, __dague_handle->super.nb_pending_actions);

    return (dague_dtd_handle_t*) __dague_handle;
}

/* **************************************************************************** */
/**
 * Clean up function to clean memory allocated dynamically for the run
 *
 * @param[in,out]   dague_handle
 *                      Pointer to the DTD handle
 *
 * @ingroup         DTD_INTERFACE
 */
void
dague_dtd_handle_destruct(dague_dtd_handle_t *dague_handle)
{
    int i;
    for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        const dague_function_t   *func = dague_handle->super.functions_array[i];
        dague_dtd_function_t *dtd_func = (dague_dtd_function_t *)func;

        if( func != NULL ) {
            int j, k;

            dague_dtd_function_release( dague_handle, dtd_func->fpointer );

            for (j=0; j< func->nb_flows; j++) {
                if(func->in[j] != NULL && func->in[j]->flow_flags == FLOW_ACCESS_READ) {
                    for(k=0; k<MAX_DEP_IN_COUNT; k++) {
                        if (func->in[j]->dep_in[k] != NULL) {
                            free((void*)func->in[j]->dep_in[k]);
                        }
                    }
                    for(k=0; k<MAX_DEP_OUT_COUNT; k++) {
                        if (func->in[j]->dep_out[k] != NULL) {
                            free((void*)func->in[j]->dep_out[k]);
                        }
                    }
                    free((void*)func->in[j]);
                }
                if(func->out[j] != NULL) {
                    for(k=0; k<MAX_DEP_IN_COUNT; k++) {
                        if (func->out[j]->dep_in[k] != NULL) {
                            free((void*)func->out[j]->dep_in[k]);
                        }
                    }
                    for(k=0; k<MAX_DEP_OUT_COUNT; k++) {
                        if (func->out[j]->dep_out[k] != NULL) {
                            free((void*)func->out[j]->dep_out[k]);
                        }
                    }
                    free((void*)func->out[j]);
                }
            }
            dague_mempool_destruct(dtd_func->context_mempool);
            free(dtd_func->context_mempool);
            free((void*)func);
        }
    }

    dague_handle_unregister( &dague_handle->super );
    dague_thread_mempool_free( handle_mempool->thread_mempools, dague_handle );
}

/* **************************************************************************** */
/**
 * This is the hook that connects the function to start initial ready
 * tasks with the context. Called internally by PaRSEC.
 *
 * @param[in]   context
 *                  DAGUE context
 * @param[in]   dague_handle
 *                  Pointer to DTD handle
 * @param[in]   pready_list
 *                  Lists of ready tasks for each core
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
void
dtd_startup( dague_context_t            *context,
             dague_handle_t             *dague_handle,
             dague_execution_context_t **pready_list )
{
    uint32_t supported_dev = 0;
    dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) dague_handle;

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    __dague_handle->super.context = context;
#endif /* defined(PINS_ENABLE) */

    uint32_t wanted_devices = dague_handle->devices_mask;
    dague_handle->devices_mask = 0;

    for (uint32_t _i = 0; _i < dague_nb_devices; _i++) {
        if (!(wanted_devices & (1 << _i)))
            continue;
        dague_device_t *device = dague_devices_get(_i);

        if (NULL == device)
            continue;
        if (NULL != device->device_handle_register)
            if (DAGUE_SUCCESS != device->device_handle_register(device, (dague_handle_t *) dague_handle))
                continue;

        supported_dev |= device->type;
        dague_handle->devices_mask |= (1 << _i);
    }
    (void)pready_list;
}

/* **************************************************************************** */
/**
 * Function that checks if a task is ready or not
 *
 * @param[in,out]   dest
 *                      The task we are performing the ready check for
 * @return
 *                  If the task is ready we return 1, 0 otherwise
 *
 * @ingroup         DTD_ITERFACE_INTERNAL
 */
static int
dtd_is_ready(const dague_dtd_task_t *dest)
{
    dague_dtd_task_t *dest_task = (dague_dtd_task_t*)dest;
    if ( 0 == dague_atomic_add_32b((int *)&(dest_task->flow_count), -1) ) {
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
 * @param[in]   new_context
 *                  Pointer to DTD task we are trying to activate
 * @param[in]   old_context
 *                  Pointer to DTD task activating it's successor(new_context)
 * @param       deps,data,src_rank,dst_rank,dst_vpid
 *                  Parameters we will use in distributed memory implementation
 * @param[out]  param
 *                  Pointer to list in which we will push if the task is ready
 * @return
 *              Instruction on how to iterate over the successors of old_context
 *
 * @ingroup     DTD_INTERFACE_INTERNAL
 */
dague_ontask_iterate_t
dtd_release_dep_fct( dague_execution_unit_t *eu,
                     const dague_execution_context_t *new_context,
                     const dague_execution_context_t *old_context,
                     const dep_t *deps,
                     dague_dep_data_description_t *data,
                     int src_rank, int dst_rank, int dst_vpid,
                     void *param )
{
    (void)eu; (void)data; (void)src_rank; (void)dst_rank; (void)old_context;
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    dague_dtd_task_t *current_task = (dague_dtd_task_t*) new_context;
    int is_ready = 0;

    is_ready = dtd_is_ready(current_task);

#if defined(DAGUE_PROF_GRAPHER)
    dague_dtd_task_t *parent_task  = (dague_dtd_task_t*)old_context;
    /* Check to not print stuff redundantly */
    if(!parent_task->dont_skip_releasing_data[deps->dep_index]) {
        dague_flow_t * origin_flow = (dague_flow_t*) calloc(1, sizeof(dague_flow_t));
        dague_flow_t * dest_flow = (dague_flow_t*) calloc(1, sizeof(dague_flow_t));

        char aa ='A';
        origin_flow->name = &aa;
        dest_flow->name = &aa;
        dest_flow->flow_flags = FLOW_ACCESS_RW;

        dague_prof_grapher_dep(old_context, new_context, is_ready, origin_flow, dest_flow);

        free(origin_flow);
        free(dest_flow);
    }
#else
    (void)deps;
#endif

    if(is_ready) {
        if(dump_traversal_info) {
            printf("------\ntask Ready: %s \t %" PRIu64 "\nTotal flow: %d  flow_count:"
                   "%d\n-----\n", current_task->super.function->name, current_task->super.super.key,
                   current_task->super.function->nb_flows, current_task->flow_count);
        }

#if defined(DEBUG_HEAVY)
            dague_dtd_task_release( (dague_dtd_handle_t *)current_task->super.dague_handle,
                                    current_task->super.super.key );
#endif

            arg->ready_lists[dst_vpid] = (dague_execution_context_t*)
                dague_list_item_ring_push_sorted( (dague_list_item_t*)arg->ready_lists[dst_vpid],
                                                  &current_task->super.super.list_item,
                                                  dague_execution_context_priority_comparator );
            return DAGUE_ITERATE_CONTINUE; /* Returns the status of the task being activated */
    } else {
        return DAGUE_ITERATE_STOP;
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
iterate_successors_of_dtd_task(dague_execution_unit_t *eu,
                               const dague_execution_context_t *this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t *ontask,
                               void *ontask_arg)
{
    (void)eu; (void)this_task; (void)action_mask; (void)ontask; (void)ontask_arg;
    ordering_correctly_2(eu, this_task, action_mask, ontask, ontask_arg);
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
release_deps_of_dtd( dague_execution_unit_t *eu,
                     dague_execution_context_t *this_task,
                     uint32_t action_mask,
                     dague_remote_deps_t *deps )
{
    (void)deps;
    dague_release_dep_fct_arg_t arg;
    int __vp_id;

    arg.action_mask  = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
    arg.ready_lists  = (NULL != eu) ? alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp) : NULL;

    if (NULL != eu)
        for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);

    iterate_successors_of_dtd_task(eu, (dague_execution_context_t*)this_task, action_mask, dtd_release_dep_fct, &arg);

    dague_vp_t **vps = eu->virtual_process->dague_context->virtual_processes;
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
        if (NULL == arg.ready_lists[__vp_id]) {
            continue;
        }
        if (__vp_id == eu->virtual_process->vp_id) {
            __dague_schedule(eu, arg.ready_lists[__vp_id]);
        }else {
            __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
        }
        arg.ready_lists[__vp_id] = NULL;
    }

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
complete_hook_of_dtd( dague_execution_unit_t    *context,
                      dague_execution_context_t *this_task )
{
    dague_dtd_task_t *task = (dague_dtd_task_t*) this_task;

    if (dump_traversal_info) {
        static int counter= 0;
        dague_atomic_add_32b(&counter,1);
        printf("------------------------------------------------\n"
               "execution done of task: %s \t %" PRIu64 "\n"
               "task done %d \n",
               this_task->function->name,
               task->super.super.key,
               counter);
    }

#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
                            task->super.super.key);
#endif /* defined(DAGUE_PROF_GRAPHER) */

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
                          this_task);

    release_deps_of_dtd(context, (dague_execution_context_t*)this_task, 0xFFFF, NULL);
    return 0;
}

/* prepare_input function, to be consistent with PaRSEC */
int
data_lookup_of_dtd_task(dague_execution_unit_t *context,
                        dague_execution_context_t *this_task)
{
    (void)context; (void)this_task;
    return DAGUE_HOOK_RETURN_DONE;
}

#if defined (WILL_USE_IN_DISTRIBUTED)
/* This function creates relationship between different types of task classes.
 * Arguments:   - dague handle (dague_handle_t *)
                - parent master structure (dague_function_t *)
                - child master structure (dague_function_t *)
                - flow index of task that belongs to the class of "parent master structure" (int)
                - flow index of task that belongs to the class of "child master structure" (int)
                - the type of data (the structure of the data like square,
                  triangular and etc) this dependency is about (int)
 * Returns:     - void
 */
void
set_dependencies_for_function(dague_handle_t* dague_handle,
                              dague_function_t *parent_function,
                              dague_function_t *desc_function,
                              uint8_t parent_flow_index,
                              uint8_t desc_flow_index,
                              int tile_type_index)
{
    uint8_t i, dep_exists = 0, j;

    if (NULL == desc_function) {   /* Data is not going to any other task */
        if(parent_function->out[parent_flow_index]) {
            dague_flow_t *tmp_d_flow = (dague_flow_t *)parent_function->out[parent_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
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
                printf("%s -> LOCAL\n", parent_function->name);
            }

            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = LOCAL_DATA; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = parent_flow_index;
            desc_dep->belongs_to    = parent_function->out[parent_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            /* specific for cholesky, will need to change */
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = 0;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(parent_function->out[parent_flow_index]);
                    /* Setting dep in the next available dep_in array index */
                    (*desc_in)->dep_out[i] = (dep_t *)desc_dep;
                    break;
                }
            }
        }
        return;
    }

    if (NULL == parent_function) {   /* Data is not coming from any other task */
        if(desc_function->in[desc_flow_index]) {
            dague_flow_t *tmp_d_flow = (dague_flow_t *)desc_function->in[desc_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
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
                printf("LOCAL -> %s\n", desc_function->name);
            }
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = LOCAL_DATA; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = 0;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(desc_function->in[desc_flow_index]);
                    /* Setting dep in the next available dep_in array index */
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep;
                    break;
                }
            }
        }
        return;
    } else {
        dague_flow_t *tmp_flow = (dague_flow_t *) parent_function->out[parent_flow_index];

        if (NULL == tmp_flow) {
            dague_flow_t *tmp_p_flow = NULL;
            tmp_flow =(dague_flow_t *) parent_function->in[parent_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL != tmp_flow->dep_in[i]) {
                    if(tmp_flow->dep_in[i]->dep_index == parent_flow_index &&
                       tmp_flow->dep_in[i]->dep_datatype_index == tile_type_index) {
                        if(tmp_flow->dep_in[i]->function_id == LOCAL_DATA) {
                            set_dependencies_for_function(dague_handle,
                                                          NULL, desc_function, 0,
                                                          desc_flow_index, tile_type_index);
                            return;
                        }
                        tmp_p_flow = (dague_flow_t *)tmp_flow->dep_in[i]->flow;
                        parent_function =(dague_function_t *) dague_handle->functions_array[tmp_flow->dep_in[i]->function_id];
                        for(j=0; j<MAX_DEP_OUT_COUNT; j++) {
                            if(NULL != tmp_p_flow->dep_out[j]) {
                                if((dague_flow_t *)tmp_p_flow->dep_out[j]->flow == tmp_flow) {
                                    parent_flow_index = tmp_p_flow->dep_out[j]->dep_index;
                                    set_dependencies_for_function(dague_handle,
                                                                  parent_function,
                                                                  desc_function,
                                                                  parent_flow_index,
                                                                  desc_flow_index,
                                                                  tile_type_index);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            dep_exists = 1;
        }

        for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
            if (NULL != tmp_flow->dep_out[i]) {
                if (tmp_flow->dep_out[i]->function_id == desc_function->function_id &&
                    tmp_flow->dep_out[i]->flow == desc_function->in[desc_flow_index] &&
                    tmp_flow->dep_out[i]->dep_datatype_index == tile_type_index) {
                    dep_exists = 1;
                    break;
                }
            }
        }

        if(!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            dep_t *parent_dep = (dep_t *) malloc(sizeof(dep_t));

            if (dump_function_info) {
                printf("%s -> %s\n", parent_function->name, desc_function->name);
            }

            /* setting out-dependency for parent */
            parent_dep->cond            = NULL;
            parent_dep->ctl_gather_nb   = NULL;
            parent_dep->function_id     = desc_function->function_id;
            parent_dep->flow            = desc_function->in[desc_flow_index];
            parent_dep->dep_index       = parent_flow_index;
            parent_dep->belongs_to      = parent_function->out[parent_flow_index];
            parent_dep->direct_data     = NULL;
            parent_dep->dep_datatype_index = tile_type_index;
            parent_dep->datatype.type.cst     = 0;
            parent_dep->datatype.layout.cst   = 0;
            parent_dep->datatype.count.cst    = 0;
            parent_dep->datatype.displ.cst    = 0;

            for(i=0; i<MAX_DEP_OUT_COUNT; i++) {
                if(NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* to bypass constness in function structure */
                    dague_flow_t **parent_out = (dague_flow_t **)&(parent_function->out[parent_flow_index]);
                    (*parent_out)->dep_out[i] = (dep_t *)parent_dep;
                    break;
                }
            }

            /* setting in-dependency for descendant */
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = parent_function->function_id;
            desc_dep->flow          = parent_function->out[parent_flow_index];
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = 0;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for(i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function strucutre */
                    dague_flow_t **desc_in = (dague_flow_t **)&(desc_function->in[desc_flow_index]);
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep;
                    break;
                }
            }
        }
    }
    return;
}
#endif

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
 * @param[in,out]   __dague_handle
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
dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, dague_dtd_funcptr_t* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param, int flow_count)
{
    dague_dtd_function_t *dtd_function = (dague_dtd_function_t *) calloc(1, sizeof(dague_dtd_function_t));
    dague_function_t *function = (dague_function_t *) dtd_function;

    dtd_function->count_of_params   = count_of_params;
    dtd_function->size_of_param     = size_of_param;
    dtd_function->fpointer          = fpointer;

    /* Allocating mempool according to the size and param count */
    dtd_function->context_mempool = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_dtd_task_t fake_task;

    /*int total_size = sizeof(dague_dtd_task_t) + count_of_params * sizeof(dague_dtd_task_param_t)
     + size_of_param + 2; */ /* this is for memory alignment */

    int total_size = sizeof(dague_dtd_task_t) + count_of_params * sizeof(dague_dtd_task_param_t) + size_of_param;
    dague_mempool_construct( dtd_function->context_mempool,
                             OBJ_CLASS(dague_dtd_task_t), total_size,
                             ((char*)&fake_task.super.super.mempool_owner) - ((char*)&fake_task),
                             1/* no. of threads*/ );

    /*
     To bypass const in function structure.
     Getting address of the const members in local mutable pointers.
     */
    char **name_not_const = (char **)&(function->name);
    symbol_t **params     = (symbol_t **) &function->params;
    symbol_t **locals     = (symbol_t **) &function->locals;
    expr_t **priority     = (expr_t **)&function->priority;
    __dague_chore_t **incarnations = (__dague_chore_t **)&(function->incarnations);

    *name_not_const                 = name;
    function->function_id           = __dague_handle->function_counter++;
    function->nb_flows              = flow_count;
    /* set to one so that prof_grpaher prints the task id properly */
    function->nb_parameters         = 1;
    function->nb_locals             = 1;
    params[0]                       = &symb_dtd_taskid;
    locals[0]                       = &symb_dtd_taskid;
    function->data_affinity         = NULL;
    function->initial_data          = NULL;
    *priority                       = NULL;
    function->flags                 = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK;
    function->dependencies_goal     = 0;
    function->key                   = (dague_functionkey_fn_t *)DTD_identity_hash;
    *incarnations                   = (__dague_chore_t *)dtd_chore;
    function->iterate_successors    = iterate_successors_of_dtd_task;
    function->iterate_predecessors  = NULL;
    function->release_deps          = release_deps_of_dtd;
    function->prepare_input         = data_lookup_of_dtd_task;
    function->prepare_output        = NULL;
    function->complete_execution    = complete_hook_of_dtd;
    function->release_task          = dague_release_task_to_mempool;
    function->fini                  = NULL;

    /* Inserting Function structure in the hash table to keep track for each class of task */
    dague_dtd_function_insert( __dague_handle, fpointer, dtd_function );
    __dague_handle->super.functions_array[function->function_id] = (dague_function_t *) function;
    return function;
}

/* **************************************************************************** */
/**
 * This function sets the flows in master-structure as we discover them
 *
 * @param[in,out]   __dague_handle
 *                      DTD handle
 * @param[in]       this_task
 *                      Task to point to correct master-structure
 * @param[in]       tile_op_type
 *                      The operation type of the task on the flow
 *                      we are setting here
 * @param[in]       flow_index
 *                      The index of the flow we are setting
 * @param[in]       tile_type_index
 *                      Tells the region type the flow is on
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
set_flow_in_function( dague_dtd_handle_t *__dague_handle,
                      dague_dtd_task_t *this_task, int tile_op_type,
                      int flow_index, int tile_type_index )
{
    dague_flow_t* flow  = (dague_flow_t *) calloc(1, sizeof(dague_flow_t));
    flow->name          = "Random";
    flow->sym_type      = 0;
    flow->flow_index    = flow_index;
    flow->flow_datatype_mask = 1<<tile_type_index;

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

    /*
     cannot pack the flows like PTG as it creates
     a lot more complicated dependency building
     between master structures.
     */
    if ((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
        dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[this_task->belongs_to_function]->in[flow_index]);
        *in = flow;
    }
    if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE || (tile_op_type & GET_OP_TYPE) == INOUT) {
        dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[this_task->belongs_to_function]->out[flow_index]);
        *out = flow;
    }
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
set_descendant(dague_dtd_task_t *parent_task, uint8_t parent_flow_index,
               dague_dtd_task_t *desc_task, uint8_t desc_flow_index,
               int parent_op_type, int desc_op_type)
{
    (void)parent_op_type;
    parent_task->desc[parent_flow_index].flow_index = desc_flow_index;
    parent_task->desc[parent_flow_index].op_type    = desc_op_type;
    dague_mfence();
    parent_task->desc[parent_flow_index].task       = desc_task;
}

/* **************************************************************************** */
/**
 * This function is called once for each parameter of a task
 *
 * Parameter of a task is treated differently depending on the type. If
 * the parameter is of type tile then dependencies are created. If parameter
 * is of type VALUE then it is copied. If parameter is of type SCRATCH
 * then memory of specified size if allocated to be used by the DTD task.
 * We use a list to attach the parameters of the task to it. Each parameter
 * is pushed in that list after it has been dealt with.
 *
 * @param[in,out]   this_task
 *                      Task whose paramter is being treated
 * @param[in]       tmp
 *                      Pointer to actual parameter
 * @param[in,out]   tile
 *                      Pointer to tile if parameter is data
 * @param[in]       tile_op_type
 *                      Operation type on data(tile)
 * @param[out]      current_param
 *                      Member of a list of parameters of this task
 * @param[out]      flow_set_flag
 *                      Array of flags to indicate whether a flow has been
 *                      set for this task or not
 * @param[in]       currrent_val
 *                      Offset used to manage the copies of a parameter
 * @param[in]       __dague_handle
 *                      DTD handle
 * @param[in]       flow_index
 *                      The current flow index of this task
 * @param[in]       next_arg
 *                      The size in bytes to copy a paramter of type
 *                      VALUE and SCRATCH
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
set_task(dague_dtd_task_t *this_task, void *tmp, dague_dtd_tile_t *tile, int *satisfied_flow,
         int tile_op_type, dague_dtd_task_param_t *current_param,
         uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS], void **current_val,
         dague_dtd_handle_t *__dague_handle, int *flow_index, int *next_arg)
{
    /* We pack the task pointer and flow information together to avoid updating multiple fields
     * atomically. Currently the last 4 bits are available and hence we can not deal with flow exceeding 16
     */
    assert( *flow_index < 16 );

    int tile_type_index;
    if( (tile_op_type & GET_OP_TYPE) == INPUT  ||
        (tile_op_type & GET_OP_TYPE) == OUTPUT ||
        (tile_op_type & GET_OP_TYPE) == INOUT  ||
        (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE)
    {
        struct user last_user;
        tile_type_index = tile_op_type & GET_REGION_INFO;
        current_param->tile_type_index = tile_type_index;
        current_param->pointer_to_tile = tmp;

        this_task->super.data[*flow_index].data_in   = tile->data_copy;
        this_task->super.data[*flow_index].data_out  = tile->data_copy;
        this_task->super.data[*flow_index].data_repo = NULL;

        if(NULL != tile) {
            if(0 == flow_set_flag[this_task->belongs_to_function]) {
                /* Setting flow in function structure */
                set_flow_in_function(__dague_handle, this_task, tile_op_type, *flow_index, tile_type_index);
            }

            this_task->desc[*flow_index].op_type_parent = tile_op_type;
            /* Saving tile pointer foreach flow in a task */
            this_task->desc[*flow_index].tile = tile;

            dague_dtd_task_t *parent = NULL;
            dague_dtd_task_t *task_pointer = (dague_dtd_task_t *)((uintptr_t)this_task|*flow_index);

            do {
                parent = tile->last_user.task;
                last_user.flow_index   = GET_FLOW_IND(parent);
                last_user.task = GET_TASK_PTR(parent);
                dague_mfence();  /* write */
            } while (!dague_atomic_cas(&(tile->last_user.task), parent, task_pointer));
            last_user.op_type      = tile->last_user.op_type;
            tile->last_user.flow_index  = *flow_index;
            tile->last_user.op_type     = tile_op_type;

            if(NULL != last_user.task) {
                set_descendant(last_user.task, last_user.flow_index,
                               this_task, *flow_index, last_user.op_type,
                               tile_op_type);
                /* Are we using the same data multiple times for the same task? */
                if(last_user.task == this_task) {
                    *satisfied_flow += 1;
                }
#if defined (WILL_USE_IN_DISTRIBUTED)
                if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                    if (!testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)this_task->super.function, NULL,
                                                      *flow_index, 0, tile_type_index);
                    }

                } else {
                    if (!testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)last_user.task->super.function,
                                                      (dague_function_t *)this_task->super.function,
                                                      last_user.flow_index, *flow_index, tile_type_index);
                    }
                }
#endif
            } else {  /* parentless */
                *satisfied_flow += 1;

                if(INPUT == (tile_op_type & GET_OP_TYPE) || ATOMIC_WRITE == (tile_op_type & GET_OP_TYPE)) {
                    /* Saving the Flow for which a Task is the first one to
                     use the data and the operation is INPUT or ATOMIC_WRITE
                     */
                    this_task->dont_skip_releasing_data[*flow_index] = 1;
                }

#if defined (WILL_USE_IN_DISTRIBUTED)
                if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
                    if (!testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                                      (dague_function_t *)this_task->super.function,
                                                      0, *flow_index, tile_type_index);
                    }
                }
                if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                    if (!testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)this_task->super.function, NULL,
                                                      *flow_index, 0, tile_type_index);
                    }
                }
#endif

            }
            *flow_index += 1;
        }
    } else if ((tile_op_type & GET_OP_TYPE) == SCRATCH){
        if(NULL == tmp) {
            current_param->pointer_to_tile = *current_val;
            *current_val = ((char*)*current_val) + *next_arg;
        }else {
            current_param->pointer_to_tile = tmp;
        }
    } else {
        memcpy(*current_val, tmp, *next_arg);
        current_param->pointer_to_tile = *current_val;
        *current_val = ((char*)*current_val) + *next_arg;
    }
    current_param->operation_type = tile_op_type;
}

/* **************************************************************************** */
/**
 * Function to push ready tasks in PaRSEC's scheduler
 *
 * We only push when we reach a window size. Window size start from 1
 * and keeps on doubling until a threshold. We also call this function
 * once at the end to schedule remainder tasks, that did not fit in
 * the window.
 *
 * @param[in,out]   __dague_handle
 *                      DTD handle
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
void
schedule_tasks (dague_dtd_handle_t *__dague_handle)
{
    dague_execution_context_t **startup_list = __dague_handle->startup_list;

    int p;
    for(p = 0; p < vpmap_get_nb_vp(); p++) {
        if( NULL != startup_list[p] ) {
            dague_list_t temp;

            OBJ_CONSTRUCT( &temp, dague_list_t );
            /* Order the tasks by priority */
            dague_list_chain_sorted(&temp, (dague_list_item_t*)startup_list[p],
                                    dague_execution_context_priority_comparator);
            startup_list[p] = (dague_execution_context_t*)dague_list_nolock_unchain(&temp);
            OBJ_DESTRUCT(&temp);
            /* We should add these tasks on the system queue when there is one */
            __dague_schedule( __dague_handle->super.context->virtual_processes[p]->execution_units[0],
                              startup_list[p] );
            startup_list[p] = NULL;
        }
    }
}

/* **************************************************************************** */
/**
 * This is the body of the fake task we insert to resolve WAR
 *
 * @param   context,this_task
 *
 * @ingroup DTD_INTERFACE_INTERNAL
 */
static int
call_to_fake_writer( dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
    (void)context; (void)this_task;
    return DAGUE_HOOK_RETURN_DONE;
}

/* **************************************************************************** */
/**
 * We create a fake task to resolve WAR, here
 *
 * This task is not finalized so we do not use the same insert-task
 * interface for this fake task. We may use or may not use this task.
 *
 * @param[in,out]   __dague_handle
 *                      DTD handle
 * @param[in]       tile
 *                      Tile this fake task is faking to write on
 * @return
 *                  Fake DTD task
 *
 * @ingroup         DTD_INTERFACE_INTERNAL
 */
dague_dtd_task_t *
create_fake_writer_task( dague_dtd_handle_t  *__dague_handle, dague_dtd_tile_t *tile )
{
    dague_dtd_funcptr_t *fpointer = call_to_fake_writer;

    dague_function_t *function = (dague_function_t *)__dague_dtd_master_fake_function;
    dague_mempool_t *context_mempool_in_function = ((dague_dtd_function_t *)function)->context_mempool;

    /* Creating Task object */
    dague_dtd_task_t *this_task = (dague_dtd_task_t *)
                                   dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools);

    for(int i = 0;i < MAX_DESC; i++ ) {
        this_task->desc[i].op_type_parent = OUTPUT;
        this_task->desc[i].op_type        = 0;
        this_task->desc[i].flow_index     = -1;
        this_task->desc[i].task           = NULL;
        this_task->desc[i].tile           = tile;
        this_task->dont_skip_releasing_data[i] = 0;
    }

    this_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    this_task->super.super.key = dague_atomic_add_32b((int *)&(__dague_handle->task_id),1);
    this_task->belongs_to_function = function->function_id;
    this_task->super.function = __dague_handle->super.functions_array[(this_task->belongs_to_function)];
    this_task->orig_task = NULL;
    this_task->flow_count = this_task->super.function->nb_flows;
    this_task->fpointer = fpointer;
    this_task->super.priority = 0;
    this_task->super.chore_id = 0;
    this_task->super.status = DAGUE_TASK_STATUS_NONE;

    return this_task;
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
 * @param[in,out]   __dague_handle
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
insert_task_generic_fptr(dague_dtd_handle_t *__dague_handle,
                         dague_dtd_funcptr_t* fpointer,
                         char* name, ...)
{
    va_list args, args_for_size;
    int next_arg, i, flow_index=0;
    int tile_op_type;
#if defined(DAGUE_PROF_TRACE)
    int track_function_created_or_not=0;
#endif
    dague_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param = NULL;
    void *tmp, *value_block, *current_val;
    static int vpid = 0;

    va_start(args, name);

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    dague_function_t *function = (dague_function_t *)dague_dtd_function_find
                                                    ( __dague_handle, fpointer );

    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        int flow_count_master=0;
        int count_of_params = 0;
        long unsigned int size_of_param = 0;
        va_copy(args_for_size, args);
        next_arg = va_arg(args_for_size, int);
        while(next_arg != 0) {
            count_of_params++;
            tmp = va_arg(args_for_size, void *);
            tile_op_type = va_arg(args_for_size, int);

            if((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) {
                size_of_param += next_arg;
            } else {
                flow_count_master++;
            }
            next_arg = va_arg(args_for_size, int);
        }

        va_end(args_for_size);

        if (dump_function_info) {
            printf("Function Created for task Class: %s\n Has %d parameters\n"
                   "Total Size: %lu\n", name, count_of_params, size_of_param);
        }

        function = create_function(__dague_handle, fpointer, name, count_of_params,
                                   size_of_param, flow_count_master);
#if defined(DAGUE_PROF_TRACE)
        track_function_created_or_not = 1;
#endif
    }

    dague_mempool_t *context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dague_dtd_tile_t *tile;
    dague_dtd_task_t *this_task;

    /* Creating Task object */
    this_task = (dague_dtd_task_t *)dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools);

    for( i = 0; i < function->nb_flows; i++ ) {
        this_task->desc[i].op_type_parent = 0;
        this_task->desc[i].op_type        = 0;
        this_task->desc[i].flow_index     = -1;
        this_task->desc[i].task           = NULL;
        this_task->desc[i].tile           = NULL;
        this_task->dont_skip_releasing_data[i] = 0;
    }
    /* for(i=0;i<MAX_PARAM_COUNT;i++) { */
    /*     this_task->super.data[i].data_repo = NULL; */
    /*     this_task->super.data[i].data_in   = NULL; */
    /*     this_task->super.data[i].data_out  = NULL; */
    /* } */

    this_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    this_task->super.super.key = dague_atomic_add_32b((int *)&(__dague_handle->task_id),1);
    this_task->belongs_to_function = function->function_id;
    this_task->super.function = __dague_handle->super.functions_array[(this_task->belongs_to_function)];
    this_task->orig_task = NULL;
    /**
     * +1 to make sure the task cannot be completed by the potential predecessors,
     * before we are completely done with it here. As we have an atomic operation
     * in all cases, increasing the expected flows by one will have no impact on
     * the performance.
     * */
    this_task->flow_count = this_task->super.function->nb_flows+1;

    this_task->fpointer = fpointer;
    this_task->super.priority = 0;
    this_task->super.chore_id = 0;
    this_task->super.status = DAGUE_TASK_STATUS_NONE;

    /* Getting the pointer to allocated memory by mempool */
    head_of_param_list = (dague_dtd_task_param_t *) (((char *)this_task) + sizeof(dague_dtd_task_t));
    current_param = head_of_param_list;
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(dague_dtd_task_param_t);
    current_val = value_block;

    next_arg = va_arg(args, int);

    int satisfied_flow = 0;
    while(next_arg != 0) {
        tmp = va_arg(args, void *);
        tile = (dague_dtd_tile_t *) tmp;
        tile_op_type = va_arg(args, int);
        current_param->tile_type_index = REGION_FULL;

        set_task(this_task, tmp, tile, &satisfied_flow,
                 tile_op_type, current_param,
                 __dague_handle->flow_set_flag, &current_val,
                 __dague_handle, &flow_index, &next_arg);

        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;

        next_arg = va_arg(args, int);
    }

    if(tmp_param != NULL) {
        tmp_param->next = NULL;
    }
    va_end(args);

    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[this_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[this_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    __dague_handle->flow_set_flag[this_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    this_task->param_list = head_of_param_list;

    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_tasks), 1);

#if defined(DEBUG_HEAVY)
    dague_dtd_task_insert( __dague_handle, this_task );
#endif

    /* Increase the count of satisfied flows to counter-balance the increase in the
     * number of expected flows done during the task creation.  */
    satisfied_flow++;

    if(!__dague_handle->super.context->active_objects) {
        assert(0);
    }

    /* Building list of initial ready task */
    if ( 0 == dague_atomic_add_32b((int *)&(this_task->flow_count), -satisfied_flow) ) {
#if defined(DEBUG_HEAVY)
            dague_dtd_task_release( __dague_handle, this_task->super.super.key );
#endif
            DAGUE_LIST_ITEM_SINGLETON(this_task);
            if(NULL != __dague_handle->startup_list[vpid]) {
                dague_list_item_ring_merge((dague_list_item_t *)this_task,
                                           (dague_list_item_t *) (__dague_handle->startup_list[vpid]));
            }
            __dague_handle->startup_list[vpid] = (dague_execution_context_t*)this_task;
            vpid = (vpid+1)%__dague_handle->super.context->nb_vp;
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not) {
        add_profiling_info(__dague_handle, function, name, flow_index);
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    if((this_task->super.super.key % __dague_handle->task_window_size) == 0 ) {
        schedule_tasks (__dague_handle);
        if ( __dague_handle->task_window_size <= dtd_window_size ) {
            __dague_handle->task_window_size *= 2;
        } else {
#if defined (OVERLAP)
            dague_execute_and_come_back (__dague_handle->super.context, &__dague_handle->super);
#endif
        }
    }
}
