/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague_config.h"
#include "dague/dague_internal.h"

#include <stdlib.h>
/* #include <stdarg.h> */
/* #include "dague/data_distribution.h" */
/* #include "data_dist/matrix/precision.h" */
/* #include "data_dist/matrix/matrix.h" */
/* #include "dplasma/lib/memory_pool.h" */
/* #include "dague/data.h" */
/* #include "dague/data_internal.h" */
/* #include "dague/debug.h" */
#include "dague/scheduling.h"
/* #include "dague/mca/pins/pins.h" */
#include "dague/remote_dep.h"
/* #include "dague/datarepo.h" */
/* #include "dague/dague_prof_grapher.h" */
/* #include "dague/mempool.h" */
#include "dague/devices/device.h"
#include "dague/constants.h"
#include "dague/vpmap.h"
#include "dague/utils/mca_param.h"
/* #include "dplasma/testing/common_timing.h" */
#include "dague/mca/sched/sched.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

int window_size = 2048;
int threshold_size = 204;
double time_double = 0;

extern dague_sched_module_t *current_scheduler;


dague_mempool_t *handle_mempool = NULL;

/* Copied from dague/scheduling.c, will need to be exposed */
#define TIME_STEP 5410
#define MIN(x, y) ( (x)<(y)?(x):(y) )
static inline unsigned long exponential_backoff(uint64_t k)
{
    unsigned int n = MIN( 64, k );
    unsigned int r = (unsigned int) ((double)n * ((double)rand()/(double)RAND_MAX));
    return r * TIME_STEP;
}

static int task_hash_table_size = (10+1);
static int tile_hash_table_size = (1000*100+1);

/* To create object of class dague_dtd_task_t that inherits dague_execution_context_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_task_t, dague_execution_context_t,
                   NULL, NULL);

/* To create object of class dague_dtd_tile_t that inherits dague_list_item_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_tile_t, dague_generic_bucket_t,
                   NULL, NULL);

/* To create object of class bucket_element_tile_t that inherits dague_list_item_t
 * class
 */
OBJ_CLASS_INSTANCE(bucket_element_tile_t, dague_generic_bucket_t,
                   NULL, NULL);

/* To create object of class dague_handle_t that inherits dague_object_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_handle_t, dague_list_item_t,
                   NULL, NULL);

/* constructor for dague_handle_t
 */
void
dague_dtd_handle_constructor
(dague_dtd_handle_t *dague_handle)
{
    int i;

    dague_handle->startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*));

    dague_handle->tile_hash_table_size      = tile_hash_table_size;
    dague_handle->task_hash_table_size      = task_hash_table_size;
    dague_handle->function_hash_table_size  = DAGUE_dtd_NB_FUNCTIONS;

    dague_handle->task_h_table              = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->task_h_table,
                    dague_handle->task_hash_table_size,
                    sizeof(bucket_element_task_t*), &hash_key);
    dague_handle->tile_h_table              = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->tile_h_table,
                    dague_handle->tile_hash_table_size,
                    sizeof(bucket_element_tile_t*), &hash_key);
    dague_handle->function_h_table          = OBJ_NEW(hash_table);
    hash_table_init(dague_handle->function_h_table,
                    dague_handle->function_hash_table_size,
                    sizeof(bucket_element_f_t *), &hash_key);

    dague_handle->super.functions_array     = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));

    for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_handle->super.functions_array[i] = NULL;
    }

    dague_handle->super.dependencies_array  = (dague_dependencies_t **) calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    dague_handle->arenas_size               = 1;
    dague_handle->arenas = (dague_arena_t **) malloc(dague_handle->arenas_size * sizeof(dague_arena_t *));

    for (i = 0; i < dague_handle->arenas_size; i++) {
        dague_handle->arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

#if defined(DAGUE_PROF_TRACE)
    dague_handle->super.profiling_array     = calloc (2 * DAGUE_dtd_NB_FUNCTIONS , sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */

    dague_dtd_tile_t fake_tile;
    dague_handle->tile_mempool          = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( dague_handle->tile_mempool,
                             OBJ_CLASS(dague_dtd_tile_t), sizeof(dague_dtd_tile_t),
                             ((char*)&fake_tile.super.mempool_owner) - ((char*)&fake_tile),
                             1/* no. of threads*/ );

    dague_generic_bucket_t fake_bucket;
    dague_handle->hash_table_bucket_mempool = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( dague_handle->hash_table_bucket_mempool,
                             OBJ_CLASS(dague_generic_bucket_t), sizeof(dague_generic_bucket_t),
                             ((char*)&fake_bucket.mempool_owner) - ((char*)&fake_bucket),
                             1/* no. of threads*/ );

}

/* desctructor for dague_handle_t
 */
void
dague_dtd_handle_destructor
(dague_dtd_handle_t *dague_handle)
{
    int i;
#if defined(DAGUE_PROF_TRACE)
    free((void *)dague_handle->super.profiling_array);
#endif /* defined(DAGUE_PROF_TRACE) */

#if 0
    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_function_t *func = (dague_function_t *) dague_handle->super.functions_array[i];

        dague_dtd_function_t *func_parent = (dague_dtd_function_t *)func;
        if (func != NULL) {
            int j, k;
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
            dague_mempool_destruct(func_parent->context_mempool);
            free (func_parent->context_mempool);
            free(func);
        }
    }
#endif
    free(dague_handle->super.functions_array);
    dague_handle->super.functions_array = NULL;
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

    /* dtd handle specific */
    dague_mempool_destruct(dague_handle->tile_mempool);
    free (dague_handle->tile_mempool);
    dague_mempool_destruct(dague_handle->hash_table_bucket_mempool);
    free (dague_handle->hash_table_bucket_mempool);
    free(dague_handle->startup_list);

#if 0
    for (i=0;i<task_hash_table_size;i++) {
        free((bucket_element_task_t *)dague_handle->task_h_table->buckets[i]);
    }
    for (i=0;i<tile_hash_table_size;i++) {
        bucket_element_tile_t *bucket = handle->super.tile_h_table->buckets[i];
        bucket_element_tile_t *tmp_bucket;
        if( bucket != NULL) {
            /* cleaning chains */
            while (bucket != NULL) {
                tmp_bucket = bucket;
                free((dague_dtd_tile_t *)bucket->tile);
                bucket = bucket->next;
                free(tmp_bucket);
            }
        }
    }
    for (i=0;i<DAGUE_dtd_NB_FUNCTIONS;i++){
        free((bucket_element_f_t *)handle->super.function_h_table->buckets[i]);
    }
#endif
    hash_table_fini(dague_handle->task_h_table, dague_handle->task_hash_table_size);
    hash_table_fini(dague_handle->tile_h_table, dague_handle->tile_hash_table_size);
    hash_table_fini(dague_handle->function_h_table, dague_handle->function_hash_table_size);
}

/* To create object of class dague_dtd_handle_t that inherits dague_handle_t
 * class
 */
OBJ_CLASS_INSTANCE(dague_dtd_handle_t, dague_handle_t,
                   dague_dtd_handle_constructor, dague_dtd_handle_destructor);

/**
 * All the static functions should be declared before being defined.
 */
static int
test_hook_of_dtd_task(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task);
static int
dtd_startup_tasks(dague_context_t * context,
                  __dague_dtd_internal_handle_t * __dague_handle,
                  dague_execution_context_t ** pready_list);
static int
dtd_is_ready(const dague_dtd_task_t *dest, int dest_flow_index);

static void
iterate_successors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg);
static int
release_deps_of_dtd(dague_execution_unit_t *,
                    dague_execution_context_t *,
                    uint32_t, dague_remote_deps_t *);

static dague_hook_return_t
complete_hook_of_dtd(dague_execution_unit_t *,
                     dague_execution_context_t *);

static dague_hook_return_t
push_tasks_back_in_mempool(dague_execution_unit_t *,
                           dague_execution_context_t *);

/* Function to initialize dague_handle_mempool
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

    if (testing_ptg_to_dtd == 99) {
        testing_ptg_to_dtd = 0;
    } else {
        testing_ptg_to_dtd = 1;
    }

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
                                       false, false, window_size, &window_size);

    /* Registering mca param for threshold size */
    (void)dague_mca_param_reg_int_name("dtd", "threshold_size",
                                       "Registers the supplied size overriding the default size of threshold size",
                                       false, false, threshold_size, &threshold_size);

}

/* Function to return the handle to the mempool */
void
dague_dtd_fini()
{
    assert(handle_mempool != NULL);

    /*dague_dtd_handle_t *ret = NULL;
    while ( ret = (dague_dtd_handle_t *)dague_lifo_pop( &handle_mempool->thread_mempools->mempool ) != NULL ) {
        OBJ_RELEASE(ret);
    }*/

    dague_mempool_destruct(handle_mempool);
    free (handle_mempool);
#if 0
    int i, j, k;
#if defined(DAGUE_PROF_TRACE)
    free((void *)handle->super.super.profiling_array);
#endif /* defined(DAGUE_PROF_TRACE) */

    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_function_t *func = (dague_function_t *) handle->super.super.functions_array[i];

        dague_dtd_function_t *func_parent = (dague_dtd_function_t *)func;
        if (func != NULL) {
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
            dague_mempool_destruct(func_parent->context_mempool);
            free (func_parent->context_mempool);
            free(func);
        }
    }
    free(handle->super.super.functions_array);
    handle->super.super.functions_array = NULL;
    handle->super.super.nb_functions = 0;

    for (i = 0; i < (uint32_t) handle->super.arenas_size; i++) {
        if (handle->super.arenas[i] != NULL) {
            free(handle->super.arenas[i]);
            handle->super.arenas[i] = NULL;
        }
    }

    free(handle->super.arenas);
    handle->super.arenas      = NULL;
    handle->super.arenas_size = 0;

    /* Destroy the data repositories for this object */
    data_repo_destroy_nothreadsafe(handle->dtd_data_repository);
    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_destruct_dependencies(handle->super.super.dependencies_array[i]);
        handle->super.super.dependencies_array[i] = NULL;
    }

    free(handle->super.super.dependencies_array);
    handle->super.super.dependencies_array = NULL;

    /* Unregister the handle from the devices */
    for (i = 0; i < dague_nb_devices; i++) {
        if (!(handle->super.super.devices_mask & (1 << i)))
            continue;
        handle->super.super.devices_mask ^= (1 << i);
        dague_device_t *device = dague_devices_get(i);
        if ((NULL == device) || (NULL == device->device_handle_unregister))
            continue;
        if (DAGUE_SUCCESS != device->device_handle_unregister(device, &handle->super.super))
            continue;
    }

    /* dtd handle specific */
    dague_mempool_destruct(handle->super.tile_mempool);
    free (handle->super.tile_mempool);
    free(handle->super.startup_list);
    for (i=0;i<task_hash_table_size;i++) {
        free((bucket_element_task_t *)handle->super.task_h_table->buckets[i]);
    }
    for (i=0;i<tile_hash_table_size;i++) {
        bucket_element_tile_t *bucket = handle->super.tile_h_table->buckets[i];
        bucket_element_tile_t *tmp_bucket;
        if( bucket != NULL) {
            /* cleaning chains */
            while (bucket != NULL) {
                tmp_bucket = bucket;
                free((dague_dtd_tile_t *)bucket->tile);
                bucket = bucket->next;
                free(tmp_bucket);
            }
        }
    }
    for (i=0;i<DAGUE_dtd_NB_FUNCTIONS;i++){
        free((bucket_element_f_t *)handle->super.function_h_table->buckets[i]);
    }
    hash_table_fini(handle->super.task_h_table, handle->super.task_hash_table_size);
    hash_table_fini(handle->super.tile_h_table, handle->super.tile_hash_table_size);
    hash_table_fini(handle->super.function_h_table, handle->super.function_hash_table_size);

    dague_handle_unregister(&handle->super.super);
    free(handle);
#endif
}

/* Function tp push back tasks in their mempool once the execution are done */
static dague_hook_return_t
push_tasks_back_in_mempool(dague_execution_unit_t *eu,
                           dague_execution_context_t *this_task)
{
    dague_dtd_function_t *function = (dague_dtd_function_t *)this_task->function;
    (void)eu;
    dague_thread_mempool_free( function->context_mempool->thread_mempools, this_task );
    return DAGUE_HOOK_RETURN_DONE;
}

/*  Function that is supposed to take the main thread into executing tasks and coming back to
 building the DAG
 Check if the engine is started or not
 Check if anybody is left at the end or not
 Assuming only Master thread will be calling this function
 */
void
dague_execute_and_come_back(dague_context_t *context,
                            dague_handle_t *dague_handle)
{
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

    /* Change it to some threshold */
    while(dague_handle->nb_local_tasks > threshold_size ) {
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

            // subsequent select begins
            PINS(eu_context, SELECT_BEGIN, NULL);
        } else {
            misses_in_a_row++;
        }
    }

#if 0
#if defined(DAGUE_PROF_RUSAGE_EU)
    dague_statistics_per_eu("EU ", eu_context);
#endif

    /* We're all done ? */
    dague_barrier_wait( &(context->barrier) );

#if defined(DAGUE_SIM)
    if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
        dague_vp_t *vp;
        int32_t my_vpid, my_idx;
        int largest_date = 0;
        for(my_vpid = 0; my_vpid < dague_context->nb_vp; my_vpid++) {
            vp = dague_context->virtual_processes[my_vpid];
            for(my_idx = 0; my_idx < vp->nb_cores; my_idx++) {
                if( vp->execution_units[my_idx]->largest_simulation_date > largest_date )
                    largest_date = vp->execution_units[my_idx]->largest_simulation_date;
            }
        }
        dague_context->largest_simulation_date = largest_date;
    }
    dague_barrier_wait( &(context->barrier) );
    eu_context->largest_simulation_date = 0;
#endif

    // finalize_progress:
    // final select end - can we mark this as special somehow?
    // actually, it will already be obviously special, since it will be the only select
    // that has no context
    PINS(eu_context, SELECT_END, NULL);

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    STATUS(("#Scheduling: th <%3d/%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu\n",
            eu_context->th_id, eu_context->virtual_process->vp_id, nbiterations, (long long unsigned int)found_local,
            (long long unsigned int)found_remote,
            (long long unsigned int)found_victim,
            (long long unsigned int)miss_local,
            (long long unsigned int)miss_victim ));

    if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
        char  priority_trace_fname[64];
        FILE *priority_trace = NULL;
        sprintf(priority_trace_fname, "priority_trace-%d.dat", eu_context->virtual_process->dague_context->my_rank);
        priority_trace = fopen(priority_trace_fname, "w");
        if( NULL != priority_trace ) {
            uint32_t my_idx;
            fprintf(priority_trace,
                    "#Step\tPriority\tThread\tVP\n"
                    "#Tasks are ordered in execution order\n");
            for(my_idx = 0; my_idx < MIN(sched_priority_trace_counter, DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER); my_idx++) {
                fprintf(priority_trace, "%d\t%d\t%d\t%d\n",
                        sched_priority_trace[my_idx].step, sched_priority_trace[my_idx].priority,
                        sched_priority_trace[my_idx].thread_id, sched_priority_trace[my_idx].vp_id);
            }
            fclose(priority_trace);
        }
    }
#endif  /* DAGUE_REPORT_STATISTICS */

    if( context->__dague_internal_finalization_in_progress ) {
        PINS_THREAD_FINI(eu_context);
    }
#endif
}

/* This function infact decrements the dague handles task counter by executing
 * one fake task after all the real tasks has been inserted in the dague context.
 * Arguments:   - handle (dague_dtd_handle_t *)
 * Returns:     - void
 */
void
increment_task_counter(dague_dtd_handle_t *__dague_handle)
{
    /* Scheduling all the remaining tasks */
    schedule_tasks (__dague_handle);

    /* decrementing the extra task we initialized the handle with */
    __dague_complete_task( &(__dague_handle->super), __dague_handle->super.context);
}

/* Unpacks all the arguments of a task, the variables(in which the actual values
 * will be copied) are passed from the body of this task and the parameters of each
 * task is copied back on the passed variables * Arguments:   - this_task
 * (dague_execution_context_t *).
 - variadic arguments (the number of arguments depends on the
 arguments supplied while inserting this task)
 * Returns:     - void
 */
void
dague_dtd_unpack_args(dague_execution_context_t *this_task, ...)
{
    dague_dtd_task_t *current_task = (dague_dtd_task_t *)this_task;
    dague_dtd_task_param_t *current_param = current_task->param_list;
    int next_arg;
    int i = 0;
    void **tmp;
    dague_data_copy_t *tmp_data;
    va_list arguments;
    va_start(arguments, this_task);
    next_arg = va_arg(arguments, int);

    while (current_param != NULL) {
        tmp = va_arg(arguments, void**);
        if(UNPACK_VALUE == next_arg) {
            memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }else if (UNPACK_DATA == next_arg) {
            //tmp_data = ((dague_dtd_tile_t*)(current_param->pointer_to_tile))->data_copy;
            tmp_data = this_task->data[i].data_out;
            memcpy(tmp, &tmp_data, sizeof(dague_data_copy_t *));
            i++;
        }else if (UNPACK_SCRATCH == next_arg) {
            memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}

/* For generating color code required for profiling and dot generation
 * Keeping this function as GET_UNIQUE_RGB_COLOR (PaRSECs default)
 * has worse API and according to me this is better[period]
 */
static inline char*
color_hash(char *name)
{
    int c, r1, r2, g1, g2, b1, b2;
    uint32_t i;
    char *color=(char *)calloc(7,sizeof(char));

    r1 = 0xA3;
    r2 = 7;
    g1 = 0x2C;
    g2 = 135;
    b1 = 0x97;
    b2 = 49;

    for(i=0; i<strlen(name); i++) {
        c = name[i];
        c &= 0xFF; // Make sure we don't get a Unicode or something.

        r1 ^= c;
        r2 *= c;
        r2 %= 1<<24;

        g1 ^= c;
        g2 *= c;
        g2 %= 1<<24;

        b1 ^= c;
        b2 *= c;
        b2 %= 1<<24;
    }
    r1 ^= (r2)&0xFF;
    r1 ^= (r2>>8)&0xFF;
    r1 ^= (r2>>16)&0xFF;

    g1 ^= (g2)&0xFF;
    g1 ^= (g2>>8)&0xFF;
    g1 ^= (g2>>16)&0xFF;

    b1 ^= (b2)&0xFF;
    b1 ^= (b2>>8)&0xFF;
    b1 ^= (b2>>16)&0xFF;

    snprintf(color,7,"%02X%02X%02X",r1,g1,b1);
    return(color);
}

#if defined(DAGUE_PROF_GRAPHER)
static inline void
print_color_graph(char* name)
{
    char *color = color_hash(name);
    /*fprintf(grapher_file,"#%s",color); */
    free(color);
}
char *
print_color_graph_str(char* name)
{
    char *color = color_hash(name);
    return color;
}
#endif

#if defined(DAGUE_PROF_TRACE)
static inline char*
fill_color(char *name)
{
    char *str, *color;
    str = (char *)calloc(12,sizeof(char));
    color = color_hash(name);
    snprintf(str,12,"fill:%s",color);
    free(color);
    return str;
}

void
profiling_trace(dague_dtd_handle_t *__dague_handle,
                dague_function_t *function, char* name,
                int flow_count)
{
    char *str = fill_color(name);
    dague_profiling_add_dictionary_keyword(name, str,
                                           sizeof(dague_profile_ddesc_info_t) + flow_count * sizeof(assignment_t),
                                           //dague_profile_ddesc_key_to_string,
                                           DAGUE_PROFILE_DDESC_INFO_CONVERTOR,
                                           //"ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};m{int32_t};n{int32_t}",
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

/* Generic function to produce hash from any key
 * Arguments:   - the kay to be hashed (uintptr_t)
                - the size of the hash table (int)
 * Returns:     - the hash value (uint32_t)
 */
uint32_t
hash_key (uintptr_t key, int size)
{
    uint32_t hash_val = key % size;
    return hash_val;
}

/* Function to search for a specific master_structure (named as Function in
 * PaRSEC) from the hash table that stores that structures
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to search the hash table with (dague_dtd_funcptr_t *)
                - size of the hash table (int)
 * Returns:     - the function structure (dague_function_t *) if found / Null if not
 */
dague_function_t *
find_function(hash_table *hash_table,
              dague_dtd_funcptr_t *key, int h_size)
{
    uint32_t hash_val = hash_table->hash((uintptr_t)key, h_size);
    bucket_element_f_t *current;

    current = hash_table->buckets[hash_val];

    /* Finding the elememnt, the pointer to the tile in the bucket of Hash table
     * is returned if found, else NULL is returned
     */
    if(current != NULL) {
        while(current != NULL) {
            if(current->key == key) {
                break;
            }
            current = current->next;
        }
        if(NULL != current) {
            return current->dtd_function;
        }else {
            return NULL;
        }
    }else {
        return NULL;
    }
}

/* Function to insert master structures in hash_table
 */
void
dague_dtd_function_insert( dague_dtd_handle_t   *dague_handle,
                           dague_dtd_funcptr_t  *key,
                           dague_dtd_function_t *value )
{
    dague_generic_bucket_t *bucket  =  (dague_generic_bucket_t *)dague_thread_mempool_allocate(dague_handle->hash_table_bucket_mempool->thread_mempools);

    hash_table *hash_table          =  dague_handle->function_h_table;
    uint32_t    hash                =  hash_table->hash ( (uint64_t)key, hash_table->size );

    hash_table_insert ( hash_table, (dague_generic_bucket_t *)bucket, (uint64_t)key, (void *)value, hash );
}

/* Function to remove master structure from hash_table
 */
void
dague_dtd_function_remove( dague_dtd_handle_t  *dague_handle,
                           dague_dtd_funcptr_t *key )
{
    hash_table *hash_table      =  dague_handle->function_h_table;
    uint32_t    hash            =  hash_table->hash ( (uint64_t)key, hash_table->size );

    hash_table_remove ( hash_table, (uint64_t)key, hash );
}

/* Function to find master structure in hash_table
 */
dague_generic_bucket_t *
dague_dtd_function_find_internal( dague_dtd_handle_t  *dague_handle,
                                  dague_dtd_funcptr_t *key )
{
    hash_table *hash_table      =  dague_handle->function_h_table;
    uint32_t    hash            =  hash_table->hash( (uint64_t)key, hash_table->size );

    return hash_table_find ( hash_table, (uint64_t)key, hash );
}

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

void
dague_dtd_function_release
( dague_dtd_handle_t *dague_handle, dague_dtd_funcptr_t *key )
{
    dague_generic_bucket_t *bucket = dague_dtd_function_find_internal ( dague_handle, key );
    assert (bucket != NULL);
    bucket->super.super.obj_reference_count = 1;
    dague_dtd_function_remove ( dague_handle, key );
    dague_thread_mempool_free( dague_handle->hash_table_bucket_mempool->thread_mempools, bucket );
}

/* Function to insert master structure in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (dague_dtd_funcptr_t *)
                - the function structure to be stored (dague_function_t *)
                - the size of the hash table (int)
 * Returns:     - void
 */
void
function_insert_h_t(hash_table *hash_table,
                    dague_dtd_funcptr_t *key, dague_function_t *dtd_function,
                    int h_size)
{
    uint32_t hash_val = hash_table->hash((uintptr_t)key, h_size);
    bucket_element_f_t *new_list, *current_table_element;

    /** Assigning values to new element **/
    new_list                = (bucket_element_f_t *) malloc(sizeof(bucket_element_f_t));
    new_list->next          = NULL;
    new_list->key           = key;
    new_list->dtd_function  = dtd_function;

    current_table_element   = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        /* Finding the last element of the list */
        while(current_table_element->next != NULL) {
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to remove tile from hash_table
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

/* Function to insert tile in hash_table
 */
dague_dtd_tile_t *
dague_dtd_tile_find( dague_dtd_handle_t *dague_handle, uint32_t key,
                     dague_ddesc_t      *ddesc )
{
    hash_table *hash_table   =  dague_handle->tile_h_table;
    uint64_t    combined_key = (uint64_t)ddesc << 32 | (uint64_t)key;
    uint32_t    hash         =  hash_table->hash ( combined_key, hash_table->size );

    return (dague_dtd_tile_t *) hash_table_find ( hash_table, combined_key, hash );
}

/* Function to insert tile in hash_table
 */
void
dague_dtd_tile_insert( dague_dtd_handle_t *dague_handle, uint32_t key,
                       dague_dtd_tile_t   *tile,
                       dague_ddesc_t      *ddesc )
{
    hash_table *hash_table   =  dague_handle->tile_h_table;
    uint64_t    combined_key = (uint64_t)ddesc << 32 | (uint64_t)key;
    uint32_t    hash         =  hash_table->hash ( combined_key, hash_table->size );

    hash_table_insert ( hash_table, (dague_generic_bucket_t *)tile, combined_key, (void *)tile, hash );
}

/* Function to insert tiles in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (uint32_t)
                - the tile to be stored (dague_dtd_tile_t *)
                - the size of the hash table (int)
                - data descriptor used along key to uniqely identify a tile (dague_ddesc_t *)
 * Returns:     - void
 */
void
tile_insert_h_t(hash_table *hash_table,
                uint32_t key, dague_dtd_tile_t *tile,
                int h_size, dague_ddesc_t* belongs_to)
{
    uint32_t hash_val = hash_table->hash(key, h_size);
    bucket_element_tile_t *new_list, *current_table_element;

    /** Assigning values to new element **/
    new_list             = (bucket_element_tile_t *) malloc(sizeof(bucket_element_tile_t));
    new_list->next       = NULL;
    new_list->key        = key;
    new_list->tile       = tile;
    new_list->belongs_to = belongs_to;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        /* Finding the last element of the list */
        while(current_table_element->next != NULL) {
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to search for a specific Task in the hash table that stores the tasks
 * Arguments:   - hash table that stores the Tiles (hash_table *)
                - key to search the hash table with (uint32_t)
                - size of the hash table (int)
 * Returns:     - the task (dague_dtd_task_t *) if found / Null if not
 */
dague_dtd_task_t*
find_task(hash_table* hash_table,
          int32_t key, int task_h_size)
{
    uint32_t hash_val = hash_table->hash(key, task_h_size);
    bucket_element_task_t *current;

    current = hash_table->buckets[hash_val];

    /* Finding the elememnt, the pointer to the list in the Hash table is returned
     * if found, else NILL is returned
     */
    if(current != NULL) {
        while(current!=NULL) {
            if(current->key == key) {
                break;
            }
            current = current->next;
        }
        return current->task;
    }else {
        return NULL;
    }
}

/* Function to insert tasks in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (uint32_t)
                - the task to be stored (dague_dtd_task_t *)
                - the size of the hash table (int)
 * Returns:     - void
 */
void
task_insert_h_t(hash_table* hash_table, uint32_t key,
                dague_dtd_task_t *task, int task_h_size)
{
    uint32_t hash_val = hash_table->hash(key, task_h_size);
    bucket_element_task_t *new_list, *current_table_element;

    /* Assigning values to new element */
    new_list       = (bucket_element_task_t *) malloc(sizeof(bucket_element_task_t));
    new_list->next = NULL;
    new_list->key  = key;
    new_list->task = task;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        /* Finding the last element of the list */
        while( current_table_element->next != NULL) {
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to manage tiles once insert_task() is called, this functions is to
 * generate tasks from PTG and insert it using insert task interface.
 * This function checks if the tile structure(dague_dtd_tile_t) is created for the data
 * already or not.
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - data descriptor (dague_ddesc_t *)
                - key of this data (dague_data_key_t)
 * Returns:     - tile, creates one if not already created, and returns that
                  tile, (dague_dtd_tile_t *)
 */
dague_dtd_tile_t*
tile_manage_for_testing(dague_dtd_handle_t *dague_dtd_handle,
                        dague_ddesc_t *ddesc, dague_data_key_t key)
{
    dague_dtd_tile_t *tmp = NULL;
    /* tmp = find_tile(dague_dtd_handle->tile_h_table, key,
                       dague_dtd_handle->tile_hash_table_size, ddesc); */
    (void)dague_dtd_handle;

    if( NULL == tmp) {
        dague_dtd_tile_t *temp_tile     = (dague_dtd_tile_t*) malloc(sizeof(dague_dtd_tile_t));
        temp_tile->key                  = key;
        temp_tile->rank                 = 0;
        temp_tile->vp_id                = 0;
        temp_tile->data                 = (dague_data_t*)ddesc;
        temp_tile->data_copy            = temp_tile->data->device_copies[0];
        temp_tile->ddesc                = NULL;
        temp_tile->last_user.flow_index = -1;
        temp_tile->last_user.op_type    = -1;
        temp_tile->last_user.task       = NULL;
        /*tile_insert_h_t(dague_dtd_handle->tile_h_table,
                        temp_tile->key,
                        temp_tile,
                        dague_dtd_handle->tile_hash_table_size,
                        ddesc);*/
        return temp_tile;
    } else {
        return tmp;
    }
}

/* Function to manage tiles once insert_task() is called
 * This function checks if the tile structure(dague_dtd_tile_t) is created for the
 * data already or not
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - data descriptor (dague_ddesc_t *)
                - key of this data (dague_data_key_t)
 * Returns:     - tile, creates one if not already created, and returns that
                  tile, (dague_dtd_tile_t *)
 */
dague_dtd_tile_t*
tile_manage(dague_dtd_handle_t *dague_dtd_handle,
            dague_ddesc_t *ddesc, int i, int j)
{
    /*dague_dtd_tile_t *tmp = find_tile(dague_dtd_handle->tile_h_table,
                                ddesc->data_key(ddesc, i, j),
                                dague_dtd_handle->tile_hash_table_size,
                                ddesc); */
    dague_dtd_tile_t *tmp = dague_dtd_tile_find ( dague_dtd_handle, ddesc->data_key(ddesc, i, j),
                                                  ddesc );
    if( NULL == tmp ) {
        /* Creating Task object */
        dague_dtd_tile_t *temp_tile = (dague_dtd_tile_t *) dague_thread_mempool_allocate
                                                          (dague_dtd_handle->tile_mempool->thread_mempools);
        //printf("allocated : %p \n", temp_tile);
        //dague_dtd_tile_t *temp_tile           = (dague_dtd_tile_t*) malloc(sizeof(dague_dtd_tile_t));
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
        /*tile_insert_h_t(dague_dtd_handle->tile_h_table,
                        temp_tile->key,
                        temp_tile,
                        dague_dtd_handle->tile_hash_table_size,
                        ddesc); */
        return temp_tile;
    }else {
        assert(tmp->super.super.super.obj_reference_count > 0);
        return tmp;
    }
}

void
tile_release(dague_dtd_handle_t *dague_handle, dague_dtd_tile_t *tile)
{
    OBJ_RELEASE(tile);
    if( tile->super.super.super.obj_reference_count == 1 ) {
        dague_dtd_tile_remove ( dague_handle, tile->key, tile->ddesc );
        if( tile->super.super.super.obj_reference_count == 1 ) {
            //printf("freed : %p \n", tile);
            dague_thread_mempool_free( dague_handle->tile_mempool->thread_mempools, tile );
        }
    }
}

/* This function sets the descendant of a task, the descendant checks the "last
 * user" of a tile to see if there's a last user.
 * If there is, the descsendant calls this function and sets itself as the
 * descendant of the parent task * Arguments:
                - parent task (dague_dtd_task_t *)
                - flow index of parent for which we are setting the descendant (uint8_t)
                - the descendant task (dague_dtd_task_t *)
                - flow index of descendant task (uint8_t)
                - operation type of parent on the data (int)
                - operation type of descendant on the data (int)
 * Returns:     - void
 */
void
set_descendant(dague_dtd_task_t *parent_task, uint8_t parent_flow_index,
               dague_dtd_task_t *desc_task, uint8_t desc_flow_index,
               int parent_op_type, int desc_op_type)
{
    (void)parent_op_type;
    parent_task->desc[parent_flow_index].flow_index = desc_flow_index;
    parent_task->desc[parent_flow_index].op_type    = desc_op_type;
    parent_task->desc[parent_flow_index].task       = desc_task;
}

/**
 * This function acts as the hook to connect the PaRSEC task with the actual task.
 * The function users passed while inserting task in PaRSEC is called in this procedure.
 * Called internally by the scheduler
 * Arguments:
 *   - the execution unit (dague_execution_unit_t *)
 *   - the PaRSEC task (dague_execution_context_t *)
 */
static int
test_hook_of_dtd_task(dague_execution_unit_t    *context,
                      dague_execution_context_t *this_task)
{
    dague_dtd_task_t   *dtd_task   = (dague_dtd_task_t*)this_task;
    dague_dtd_handle_t *dtd_handle = (dague_dtd_handle_t*)(dtd_task->super.dague_handle);
    dague_execution_context_t *orig_task = dtd_task->orig_task;
    int rc = 0;

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id],
                          this_task);

    /**
     * Check to see which interface, if it is the PTG inserting task in DTD then
     * this condition will be true
     */
    if(orig_task != NULL) {
        dague_handle_t *orig_handle = orig_task->dague_handle;

        rc = dtd_task->fpointer(context, orig_task);
        if(rc == DAGUE_HOOK_RETURN_DONE) {
            dague_atomic_add_32b(&(dtd_handle->tasks_scheduled), 1);
        }

        if(dtd_handle->total_tasks_to_be_exec == dtd_handle->tasks_scheduled)
        {
            dague_handle_update_nbtask(orig_handle, -1);
        }
        else if ((orig_handle->nb_local_tasks == 1) &&
                 (dtd_handle->tasks_created == dtd_handle->tasks_scheduled))
        {
            dague_handle_update_nbtask(orig_handle, -1);
        }
    }
    else { /* This is the default behavior */
        rc = dtd_task->fpointer(context, this_task);
        assert( rc == DAGUE_HOOK_RETURN_DONE );
    }

    return rc;
}

/* chores and dague_function_t structure initialization */
static const __dague_chore_t dtd_chore[] = {
    {.type      = DAGUE_DEV_CPU,
     .evaluate  = NULL,
     .hook      = test_hook_of_dtd_task },
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

/* To make it consistent with PaRSEC we need to intialize and have function we do not use at this point */
static inline uint64_t DTD_identity_hash(const __dague_dtd_internal_handle_t * __dague_handle,
                                         const assignment_t * assignments)
{
    (void)__dague_handle;
    return (uint64_t)assignments[0].value;
}

/* This function is called when the handle is enqueued in the context.
 * This function checks if there is any initial ready task to be scheduled and schedules if any
 * Arguments:   - the dague context (dague_context_t *)
                - dague handle (dague_internal_handle_t *)
                - list of PaRSEC tasks (dague_execution_context_t **)
 * Returns:     - 0 (int)
 */
static int
dtd_startup_tasks(dague_context_t * context,
                  __dague_dtd_internal_handle_t * __dague_handle,
                  dague_execution_context_t ** pready_list)
{
    dague_dtd_handle_t* dague_dtd_handle = (dague_dtd_handle_t*)__dague_handle;
    (void)context; (void)pready_list;
    //#error "Affectation of pready_list is not returned (Mathieu)"
    pready_list = dague_dtd_handle->startup_list;

    /* It doesn't do what is said in the comment. Fix that or remove the function!*/
    return 0;
}

/* Clean up function to clean memory allocated dynamically for the run
 * Arguments:   - the dague handle (dague_dtd_internal_handle_t *)
 * Returns:     - void
 */
void
dtd_destructor(dague_dtd_handle_t *dague_handle)
{
    int i;
    for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_function_t     *func = dague_handle->super.functions_array[i];
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
            free(func);
        }
    }
    dague_handle_unregister( &dague_handle->super );
    dague_thread_mempool_free( handle_mempool->thread_mempools, dague_handle );
}

/* This is the hook that connects the function to start initial ready tasks with the context.
 * Called internally by PaRSEC
 * Arguments:   - dague context (dague_context_t *)
                - dague handle (dague_handle_t *)
                - list of ready tasks (dague_execution_context_t **)
 * Returns:     - void
 */
void
dtd_startup(dague_context_t * context,
            dague_handle_t * dague_handle,
            dague_execution_context_t ** pready_list)
{
    uint32_t supported_dev = 0;
    __dague_dtd_internal_handle_t *__dague_handle = (__dague_dtd_internal_handle_t *) dague_handle;

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    __dague_handle->super.super.context = context;
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

        supported_dev |= (1 << device->type);
        dague_handle->devices_mask |= (1 << _i);
    }

    dtd_startup_tasks(context, (__dague_dtd_internal_handle_t *) dague_handle, pready_list);
}

/* dague_dtd_new()
 * Intializes all the needed members and returns the DAGUE handle
 * Arguments:   - Dague context
 - Total number of task CLASSES
 - Total number of arenas
 - An integer for checking some of the Kernels (TODO: remove it)
 * Returns:     - DAGUE handle
 * For correct profiling the task_class_counter should be correct
 */
dague_dtd_handle_t *
dague_dtd_new(dague_context_t* context,
              int arena_count)
{
    dague_dtd_handle_t *__dague_handle;
    int i;

    //__dague_dtd_internal_handle_t *__dague_handle   = (__dague_dtd_internal_handle_t *) calloc (1, sizeof(__dague_dtd_internal_handle_t) );
    assert( handle_mempool != NULL );
    __dague_handle = (dague_dtd_handle_t *)dague_thread_mempool_allocate(handle_mempool->thread_mempools);

    /*__dague_handle->super.tile_hash_table_size      = tile_hash_table_size;
     __dague_handle->super.task_hash_table_size      = task_hash_table_size; */
    //__dague_handle->task_id                   = 0;
    //__dague_handle->super.function_hash_table_size  = DAGUE_dtd_NB_FUNCTIONS;
    //__dague_handle->super.startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*));

    /*__dague_handle->super.task_h_table              = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.task_h_table,
                    __dague_handle->super.task_hash_table_size,
                    sizeof(bucket_element_task_t*), &hash_key);
    __dague_handle->super.tile_h_table              = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.tile_h_table,
                    __dague_handle->super.tile_hash_table_size,
                    sizeof(bucket_element_tile_t*), &hash_key);
    __dague_handle->super.function_h_table          = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.function_h_table,
                    __dague_handle->super.function_hash_table_size,
                    sizeof(bucket_element_f_t *), &hash_key); */
    __dague_handle->super.context             = context;
    __dague_handle->super.on_enqueue          = NULL;
    __dague_handle->super.on_enqueue_data     = NULL;
    __dague_handle->super.on_complete         = NULL;
    __dague_handle->super.on_complete_data    = NULL;
    __dague_handle->super.devices_mask        = DAGUE_DEVICES_ALL;
    __dague_handle->super.nb_functions        = DAGUE_dtd_NB_FUNCTIONS;
    //__dague_handle->super.super.functions_array     = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));

    /*for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
     __dague_handle->super.super.functions_array[i] = NULL;
     }*/

#if 0
    __dague_handle->super.super.dependencies_array  = (dague_dependencies_t **) calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    __dague_handle->super.arenas_size               = arena_count/arena_count;
    __dague_handle->super.arenas = (dague_arena_t **) malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t *));

    for (i = 0; i < __dague_handle->super.arenas_size; i++) {
        __dague_handle->super.arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

    __dague_handle->dtd_data_repository             = data_repo_create_nothreadsafe(DTD_TASK_COUNT, MAX_DEP_OUT_COUNT);
#if defined(DAGUE_PROF_TRACE)
    __dague_handle->super.super.profiling_array     = calloc (2 * DAGUE_dtd_NB_FUNCTIONS , sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */
#endif

    /* Keeping track of total tasks to be executed per handle for the window */
    for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        __dague_handle->flow_set_flag[i]  = 0;
        /* Added new */
        __dague_handle->super.functions_array[i] = NULL;
    }
    __dague_handle->tasks_created         = 0;
    __dague_handle->task_window_size      = 1;
    __dague_handle->tasks_scheduled       = 0; /* For the testing of PTG inserting in DTD */

#if 0
    /* allocating tile mempool */
    dague_dtd_tile_t fake_tile;

    __dague_handle->super.tile_mempool          = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dague_mempool_construct( __dague_handle->super.tile_mempool,
                             OBJ_CLASS(dague_dtd_tile_t), sizeof(dague_dtd_tile_t),
                             ((char*)&fake_tile.super.mempool_owner) - ((char*)&fake_tile),
                             1/* no. of threads*/ );
#endif
    __dague_handle->super.nb_local_tasks  = 1; /* For the bounded window, starting with +1 task */
    __dague_handle->super.startup_hook    = dtd_startup;
    __dague_handle->super.destructor      = (dague_destruct_fn_t) dtd_destructor;

    /* for testing interface*/
    __dague_handle->total_tasks_to_be_exec = arena_count;

    (void) dague_handle_reserve_id((dague_handle_t *) __dague_handle);
    return (dague_dtd_handle_t*) __dague_handle;
}

/* DTD version of is_completed()
 * Input:   - dtd task (dague_dtd_task_t *)
            - flow index (int)
 * Return:  - 1 - indicating task in ready / 0 - indicating task is not ready
 */
static int
dtd_is_ready(const dague_dtd_task_t *dest,
             int dest_flow_index)
{
    dague_dtd_task_t *dest_task = (dague_dtd_task_t*)dest;
    if ( dest_task->flow_count == dague_atomic_inc_32b(&(dest_task->flow_satisfied))) {
        return 1;
    }
    return 0;
}

/* Checks whether the task is ready or not and packs the ready tasks in a list
 *
 */
dague_ontask_iterate_t
dtd_release_dep_fct( dague_execution_unit_t *eu,
                     const dague_execution_context_t* new_context,
                     const dague_execution_context_t * old_context,
                     const dep_t * deps,
                     dague_dep_data_description_t * data,
                     int src_rank, int dst_rank, int dst_vpid,
                     void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    int is_ready = 0;
    dague_dtd_task_t *current_task = (dague_dtd_task_t*) new_context;
    dague_dtd_task_t *parent_task  = (dague_dtd_task_t*)old_context;

    is_ready = dtd_is_ready(current_task, deps->flow->flow_index);

#if defined(DAGUE_PROF_GRAPHER)
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
#endif

    if(is_ready) {
        if(dump_traversal_info) {
            printf("------\ntask Ready: %s \t %d\nTotal flow: %d  flow_count:"
                   "%d\n-----\n", current_task->super.function->name, current_task->task_id,
                   current_task->super.function->nb_flows, current_task->flow_count);
        }

#if defined (OVERLAP)
        int ii = dague_atomic_cas(&(current_task->ready_mask), 0, 1);
        if(ii) {
#endif
            arg->ready_lists[dst_vpid] = (dague_execution_context_t*)
                dague_list_item_ring_push_sorted( (dague_list_item_t*)arg->ready_lists[dst_vpid],
                                                  &current_task->super.list_item,
                                                  dague_execution_context_priority_comparator );
            return DAGUE_ITERATE_CONTINUE; /* Returns the status of the task being activated */
#if defined (OVERLAP)
        }else {
            return DAGUE_ITERATE_STOP;
        }
#endif
    } else {
        return DAGUE_ITERATE_STOP;
    }

}

/* This function iterates over all the successors of a task and activates them
 * and builds a list of the ones that got ready by this activation
 *
 */
static void
iterate_successors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg)
{
    ordering_correctly_1(eu, this_task, action_mask, ontask, ontask_arg);
}

/* To be consistent with PaRSECs structures, is not used or implemented */
static void
iterate_predecessors_of_dtd_task(dague_execution_unit_t * eu,
                                 const dague_execution_context_t * this_task,
                                 uint32_t action_mask,
                                 dague_ontask_function_t * ontask,
                                 void *ontask_arg)
{
}

/* Release dependencies after a task is done.
 * Calls iterate successors function that returns a list of tasks that are ready to go.
 * Those ready tasks are scheduled in here
 */
static int
release_deps_of_dtd(dague_execution_unit_t* eu,
                    dague_execution_context_t* this_task,
                    uint32_t action_mask,
                    dague_remote_deps_t* deps)
{
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

/* This function is called internally by PaRSEC once a task is done
 *
 */
static int
complete_hook_of_dtd(dague_execution_unit_t* context,
                     dague_execution_context_t* this_task)
{
    dague_dtd_task_t *task = (dague_dtd_task_t*) this_task;
    if (dump_traversal_info) {
        static int counter= 0;
        dague_atomic_add_32b(&counter,1);
        printf("------------------------------------------------\nexecution done"
               "of task: %s \t %d\ntask done %d \n", this_task->function->name, task->task_id,
               counter);
    }

#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id,
                            task->task_id);
#endif /* defined(DAGUE_PROF_GRAPHER) */

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
                          this_task);

    release_deps_of_dtd(context, (dague_execution_context_t*)this_task, 0xFFFF, NULL);
    return 0;
}

/* prepare_input function, to be consistent with PaRSEC */
int
data_lookup_of_dtd_task(dague_execution_unit_t * context,
                        dague_execution_context_t * this_task)
{
    return DAGUE_HOOK_RETURN_DONE;
}

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
                    if (tmp_d_flow->dep_out[i]->function_id == 100 ) {
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
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = parent_flow_index;
            desc_dep->belongs_to    = parent_function->out[parent_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            /* specific for cholesky, will need to change */
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = NULL;
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
                    if (tmp_d_flow->dep_in[i]->function_id == 100 ) {
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
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = NULL;
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
                        if(tmp_flow->dep_in[i]->function_id == 100) {
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
            parent_dep->datatype.layout.cst   = NULL;
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
            desc_dep->datatype.layout.cst   = NULL;
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

/* Function structure declaration and initializing
 * Also creates the mempool_context for each task class
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - function pointer to the actual task (dague_dtd_funcptr_t *)
                - name of the task class (char *)
                - count of parameter each task of this class has, to estimate the memory we need
                  to allocate for the mempool (int)
                - total size of memory required in bytes to hold the values of those paramters (int)
                - flow count of the tasks belonging to a particular class (int)
 * Returns:     - the master structure (dague_function_t *)
 */
dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, dague_dtd_funcptr_t* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param, int flow_count)
{
    static int handle_id = 0;
    static uint8_t function_counter = 0;

    /* TODO: Instead of resetting counter we need to keep track of which handles
             we have already encountered already */
     if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        function_counter = 0;
    }

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
                             ((char*)&fake_task.super.mempool_owner) - ((char*)&fake_task),
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
    function->function_id           = function_counter;
    function->nb_flows              = flow_count;
    /* set to one so that prof_grpaher prints the task id properly */
    function->nb_parameters         = 1;
    function->nb_locals             = 0;
    params[0]                       = &symb_dtd_taskid;
    locals[0]                       = &symb_dtd_taskid;
    function->data_affinity         = NULL;
    function->initial_data          = NULL;
    *priority                       = NULL;
    function->flags                 = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK;
    function->dependencies_goal     = 0;
    function->key                   = (dague_functionkey_fn_t *)DTD_identity_hash;
    function->fini                  = NULL;
    *incarnations                   = (__dague_chore_t *)dtd_chore;
    function->iterate_successors    = iterate_successors_of_dtd_task;
    function->iterate_predecessors  = iterate_predecessors_of_dtd_task;
    function->release_deps          = release_deps_of_dtd;
    function->prepare_input         = data_lookup_of_dtd_task;
    function->prepare_output        = NULL;
    function->complete_execution    = complete_hook_of_dtd;
    function->pushback              = push_tasks_back_in_mempool;

    /* Inserting Fucntion structure in the hash table to keep track for each class of task */
#if 0
    function_insert_h_t(__dague_handle->function_h_table, fpointer,
                       (dague_function_t *)function, __dague_handle->function_hash_table_size);
#endif
    dague_dtd_function_insert( __dague_handle, fpointer, dtd_function );
    __dague_handle->super.functions_array[function_counter] = (dague_function_t *) function;
    function_counter++;
    return function;
}

/* For each flow in the task class we call this function to set up a flow
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - the task to extract the class this task belongs to (dague_dtd_task_t *)
                - the operation this flow does on the data (int)
                - the index of this flow for this task class (int)
                - the data type(triangular, square and etc) this flow works on (int)
 * Returns:     - void
 */
void
set_flow_in_function(dague_dtd_handle_t *__dague_handle,
                 dague_dtd_task_t *temp_task, int tile_op_type,
                 int flow_index, int tile_type_index)
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
        dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
        *in = flow;
    }
    if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE || (tile_op_type & GET_OP_TYPE) == INOUT) {
        dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
        *out = flow;
    }
}

/*
 * INSERT Task Function.
 * Each time the user calls it a task is created with the respective parameters
   the user has passed.
 * For each task class a structure known as "function" is created as well.
   (e.g. for Cholesky 4 function structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are
   tracked from this function.
 */
void
insert_task_generic_fptr(dague_dtd_handle_t *__dague_handle,
                         dague_dtd_funcptr_t* fpointer,
                         char* name, ...)
{
    va_list args, args_for_size;
    static int handle_id=0;
    int next_arg, i, flow_index=0;
    int tile_op_type;
    int track_function_created_or_not=0;
    dague_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param = NULL;
    void *tmp, *value_block, *current_val;
    static int vpid = 0;

    va_start(args, name);

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
#if 0
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size);
#endif

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
            count_of_params ++;
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
        track_function_created_or_not = 1;
    }

    dague_mempool_t *context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dague_dtd_tile_t *tile;
    dague_dtd_task_t *temp_task;

    /* Creating Task object */
    temp_task = (dague_dtd_task_t *)dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools);

    /*printf("Orignal Address : %p\t", temp_task);
     int n = ((uintptr_t)temp_task) & 0xF;
     printf("n is : %lx\n", n);

     uintptr_t ptrr =  ((((uintptr_t)temp_task)+16)/16)*16;
     printf("New Address :%lx \t", ptrr);
     n = ((uintptr_t)ptrr) & 0xF;
     printf("n is : %lx\n", n);*/

    for(i=0;i<MAX_DESC;i++) {
        // temp_task->desc[i].op_type_parent = -1;
        //temp_task->desc[i].op_type        = -1;
        //temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;
        temp_task->dont_skip_releasing_data[i] = 0;
    }
    /*for(i=0;i<MAX_PARAM_COUNT;i++) {
     temp_task->super.data[i].data_repo = NULL;
     temp_task->super.data[i].data_in   = NULL;
     temp_task->super.data[i].data_out  = NULL;
     }*/

    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];
    temp_task->flow_satisfied = 0;
    temp_task->orig_task = NULL;
    temp_task->ready_mask = 0;
    temp_task->task_id = __dague_handle->task_id;
#if defined(OVERLAP)
    /* +1 to make sure the task is completely ready before it gets executed */
    temp_task->flow_count = temp_task->super.function->nb_flows+1;
#else
    temp_task->flow_count = temp_task->super.function->nb_flows;
#endif
    temp_task->fpointer = fpointer;
    temp_task->super.locals[0].value = __dague_handle->task_id;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    /* Getting the pointer to allocated memory by mempool */
    head_of_param_list = (dague_dtd_task_param_t *) (((char *)temp_task) + sizeof(dague_dtd_task_t));
    current_param = head_of_param_list;
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(dague_dtd_task_param_t);
    current_val = value_block;

    next_arg = va_arg(args, int);

    //double time__ = get_cur_time();
    while(next_arg != 0) {
        tmp = va_arg(args, void *);
        tile = (dague_dtd_tile_t *) tmp;
        tile_op_type = va_arg(args, int);
        current_param->tile_type_index = REGION_FULL;

        set_task(temp_task, tmp, tile,
                 tile_op_type, current_param,
                 __dague_handle->flow_set_flag, &current_val,
                 __dague_handle, &flow_index, &next_arg);

        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;

        next_arg = va_arg(args, int);
    }
    //time__ = get_cur_time() - time__;
    //time_double += time__;

    if(tmp_param != NULL) {
        tmp_param->next = NULL;
    }
    va_end(args);

    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    __dague_handle->flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->param_list = head_of_param_list;

    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_local_tasks),1);

#if defined (OVERLAP)
    /* in attempt to make the task not ready till the whole body is constructed */
    dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1);
#endif

    if(!__dague_handle->super.context->active_objects) {
        assert(0);
        __dague_handle->task_id++;
        /* executing the tasks as soon as we find it, if no engine is attached */
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0],
                        (dague_execution_context_t *)temp_task);
        return;
    }

    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied) {
#if defined (OVERLAP)
        int ii = dague_atomic_cas(&(temp_task->ready_mask), 0, 1);
        if(ii) {
#endif
            DAGUE_LIST_ITEM_SINGLETON(temp_task);
            if (NULL != __dague_handle->startup_list[vpid]) {
                dague_list_item_ring_merge((dague_list_item_t *)temp_task,
                                           (dague_list_item_t *) (__dague_handle->startup_list[vpid]));
            }
            __dague_handle->startup_list[vpid] = (dague_execution_context_t*)temp_task;
            vpid = (vpid+1)%__dague_handle->super.context->nb_vp;
#if defined (OVERLAP)
        }
#endif
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not) {
        profiling_trace(__dague_handle, function, name, flow_index);
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    __dague_handle->task_id++;
    __dague_handle->tasks_created++;

    if((__dague_handle->tasks_created % __dague_handle->task_window_size) == 0 ) {
        schedule_tasks (__dague_handle);
        if ( __dague_handle->task_window_size <= window_size ) {
            __dague_handle->task_window_size *= 2;
        } else {
#if defined (OVERLAP)
            dague_execute_and_come_back (__dague_handle->super.context, &__dague_handle->super);
#endif
        }
    }
}

/* Function that sets all dependencies between tasks according to the operation type of that task
 * on the data and also creates relationship between master structures.
 * Arguments:   - the current task (dague_dtd_task_t *)
                - pointer to the tile/data (void *)
                - pointer to the tile/data (tile *)
                - operation type on the data (int)
                - task parameters (dague_dtd_task_param_t *)
                - structure to hold information about the last user of the data,
                  if any (struct user)
                - array of int to indicate whether we have set a flow for this
                  task class in the master structure or not (uint8_t [])
                - pointer to the memory allocated by mempool for holding the
                  parameter of this task (void **)
                - dague handle (dague_dtd_handle_t)
                - current flow index (int *)
                - next argument sent to insert task (int *)
 * Returns:     - void
 */
void
set_task(dague_dtd_task_t *temp_task, void *tmp, dague_dtd_tile_t *tile,
         int tile_op_type, dague_dtd_task_param_t *current_param,
         uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS], void **current_val,
         dague_dtd_handle_t *__dague_handle, int *flow_index, int *next_arg)
{
    int tile_type_index;
    if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
        struct user last_user;
        tile_type_index = tile_op_type & GET_REGION_INFO;
        current_param->tile_type_index = tile_type_index;
        current_param->pointer_to_tile = tmp;

        temp_task->super.data[*flow_index].data_in = tile->data_copy;
        temp_task->super.data[*flow_index].data_out = tile->data_copy;
        temp_task->super.data[*flow_index].data_repo = NULL;

        if(tile !=NULL) {
            if(0 == flow_set_flag[temp_task->belongs_to_function]) {
                /*setting flow in function structure */
                set_flow_in_function(__dague_handle, temp_task, tile_op_type, *flow_index, tile_type_index);
            }

            last_user.flow_index   = tile->last_user.flow_index;
            last_user.op_type      = tile->last_user.op_type;
            last_user.task         = tile->last_user.task;

            tile->last_user.flow_index       = *flow_index;
            tile->last_user.op_type          = tile_op_type;
            temp_task->desc[*flow_index].op_type_parent = tile_op_type;
            /* Saving tile pointer foreach flow in a task*/
            temp_task->desc[*flow_index].tile = tile;

            dague_dtd_task_t *parent;
            int no_parent = 1;
            if(NULL != (parent = last_user.task)) {
#if defined (OVERLAP)
                int ii = dague_atomic_cas(&(tile->last_user.task), parent, temp_task);
                if(ii) {
#endif
                    no_parent = 0;
                    set_descendant(parent, last_user.flow_index,
                                   temp_task, *flow_index, last_user.op_type,
                                   tile_op_type);
                    if (parent == temp_task) {
#if defined (OVERLAP)
                        dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1);
#else
                        temp_task->flow_satisfied++;
#endif
                    }
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                        if (testing_ptg_to_dtd) {
                            set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                          (dague_function_t *)temp_task->super.function, NULL,
                                                          *flow_index, 0, tile_type_index);
                        }

                    } else {
                        if (testing_ptg_to_dtd) {
                            set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                          (dague_function_t *)parent->super.function,
                                                          (dague_function_t *)temp_task->super.function,
                                                          last_user.flow_index, *flow_index, tile_type_index);
                        }
                    }
#if defined (OVERLAP)
                }
#endif
            }
            if (no_parent) {
#if defined (OVERLAP)
                dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1);
#else
                temp_task->flow_satisfied++;
#endif

                if(INPUT == (tile_op_type & GET_OP_TYPE) || ATOMIC_WRITE == (tile_op_type & GET_OP_TYPE)) {
                    /* Saving the Flow for which a Task is the first one to
                     use the data and the operation is INPUT or ATOMIC_WRITE
                     */
                    temp_task->dont_skip_releasing_data[*flow_index] = 1;
                }

                if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
                    if (testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                                      (dague_function_t *)temp_task->super.function,
                                                      0, *flow_index, tile_type_index);
                    }
                }
                if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                    if (testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)temp_task->super.function, NULL,
                                                      *flow_index, 0, tile_type_index);
                    }
                }

            }

            tile->last_user.task = temp_task; /* at the end to maintain order */
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

/* Function to schedule tasks in PaRSEC's scheduler
 * Arguments:   - Dague handle that has the list of ready tasks (dague_dtd_handle_t *)
 * Returns:     - void
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
#if 0
/* ------------ */

/*  Everything under this is for testing the insert task interface by using the
 existing PTG tests so not to be counted as a part of insert task interface
 */
/* ------------ */
/*
 * INSERT Task Function.
 * Each time the user calls it a task is created with the respective parameters the user has passed.
 * For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
 structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are tracked from this function.
 */
void
insert_task_generic_fptr_for_testing(dague_dtd_handle_t *__dague_handle,
                                     dague_dtd_funcptr_t* fpointer, dague_execution_context_t *orig_task,
                                     char* name, dague_dtd_task_param_t *head_paramm)
{
    dague_dtd_task_param_t *current_paramm;
    static int handle_id = 0;
    static uint32_t task_id = 0, _internal_task_counter=0;
    static uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS];
    int next_arg=-1, i, flow_index = 0;
    int tile_op_type;
    int track_function_created_or_not = 0;
    dague_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param;
    void *tmp, *value_block, *current_val;
    static int vpid = 0;

    if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
            flow_set_flag[i] = 0;
        }
    }

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size);

    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        int count_of_params = 0;
        long unsigned int size_of_param = 0;

        current_paramm = head_paramm;

        while(current_paramm != NULL) {
            count_of_params++;
            current_paramm = current_paramm->next;
        }

        if (dump_function_info) {
            printf("Function Created for task Class: %s\n Has %d parameters\n Total Size: %lu\n", name, count_of_params, size_of_param);
        }

        function = create_function(__dague_handle, fpointer, name, count_of_params, size_of_param, count_of_params);
        track_function_created_or_not = 1;
    } else { /* Because I am tracking things that does not make a lot of sense hence */
        int count_of_params = 0;

        current_paramm = head_paramm;

        while(current_paramm != NULL) {
            count_of_params++;
            current_paramm = current_paramm->next;
        }

        if(function->nb_flows < count_of_params) {
            function->nb_flows = count_of_params;
            flow_set_flag[function->function_id] = 0;
        }
        else if(function->nb_flows > count_of_params) {
            function->nb_flows = count_of_params;
        }
    }

    dague_mempool_t * context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dague_dtd_tile_t *tile;
    dague_dtd_task_t *temp_task;

    temp_task = (dague_dtd_task_t *) dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools); /* Creating Task object */
    for(i=0;i<MAX_DESC;i++) {
        temp_task->desc[i].op_type_parent = -1;
        temp_task->desc[i].op_type        = -1;
        temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;

        temp_task->dont_skip_releasing_data[i] = 0;
    }
    for(i=0;i<MAX_PARAM_COUNT;i++) {
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in   = NULL;
        temp_task->super.data[i].data_out  = NULL;
    }


    dague_execution_context_t *orig_task_copy =(dague_execution_context_t *) malloc(sizeof(dague_execution_context_t));
    memcpy(orig_task_copy, orig_task, sizeof(dague_execution_context_t));
    temp_task->orig_task = orig_task_copy;

    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];
    temp_task->flow_satisfied = 0;
    temp_task->ready_mask = 0;
    temp_task->task_id = task_id;
    temp_task->flow_count = temp_task->super.function->nb_flows+1; /* +1 to make sure the task is completely ready before it gets executed */
    temp_task->fpointer = fpointer;
    temp_task->super.locals[0].value = task_id;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    head_of_param_list = (dague_dtd_task_param_t *) (((char *)temp_task) + sizeof(dague_dtd_task_t)); /* Getting the pointer allocated from mempool */
    current_param = head_of_param_list;
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(dague_dtd_task_param_t);
    current_val = value_block;

    current_paramm = head_paramm;

    struct user *last_user = (struct user *) malloc(sizeof(struct user));
    while(current_paramm != NULL) {
        tmp = current_paramm->pointer_to_tile;
        tile = (dague_dtd_tile_t *) tmp;
        tile_op_type = current_paramm->operation_type;
        current_param->tile_type_index = REGION_FULL;

        set_task(temp_task, tmp, tile,
                 tile_op_type, current_param,
                 flow_set_flag, &current_val,
                 __dague_handle, &flow_index, &next_arg);

        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;

        current_paramm = current_paramm->next;
    }

    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->param_list = head_of_param_list;

    /* Atomically increasing the nb_local_tasks_counter */
    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_local_tasks),1);
    dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1); /* in attempt to make the task not ready till the whole body is constructed */

    if(!__dague_handle->super.context->active_objects) {
        task_id++;
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0], (dague_execution_context_t *)temp_task);  /* executing the tasks as soon as we find it if no engine is attached */
        return;
    }

    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied && !temp_task->ready_mask) {
        DAGUE_LIST_ITEM_SINGLETON(temp_task);
        if (NULL != __dague_handle->startup_list[vpid]) {
            dague_list_item_ring_merge((dague_list_item_t *)temp_task,
                                       (dague_list_item_t *) (__dague_handle->startup_list[vpid]));
        }
        __dague_handle->startup_list[vpid] = (dague_execution_context_t*)temp_task;
        vpid = (vpid+1)%__dague_handle->super.context->nb_vp;
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not) {
        profiling_trace(__dague_handle, function, name, flow_index);
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    __dague_handle->tasks_created = _internal_task_counter;

    static int task_window_size = 1;

    if((__dague_handle->tasks_created % task_window_size) == 0 ) {
        schedule_tasks (__dague_handle);
    }
}

#endif
/* To copy the dague_context_t of the predecessor needed for tracking control flow
 *
 */
static dague_ontask_iterate_t copy_content(dague_execution_unit_t *eu,
                const dague_execution_context_t *newcontext,
                const dague_execution_context_t *oldcontext,
                const dep_t *dep, dague_dep_data_description_t *data,
                int src_rank, int dst_rank, int dst_vpid, void *param)
{
    /* assinging 1 to "unused" field in dague_context_t of the successor to indicate we found a predecesor */
    uint8_t *val = (uint8_t *) &(oldcontext->unused);
    *val += 1;

    /* Saving the flow index of the parent in the "unused" field of the predecessor */
    uint8_t *val1 = (uint8_t *) &(newcontext->unused);
    dague_flow_t* parent_outflow = (dague_flow_t*)(dep->flow);
    *val1 = parent_outflow->flow_index;

    memcpy(param, newcontext, sizeof(dague_execution_context_t));
    return DAGUE_ITERATE_STOP;
}

static int
fake_hook_for_testing(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    int count = 0;
    dague_dtd_handle_t *dtd_handle = __dtd_handle;
    const char *name = this_task->function->name;
    dague_dtd_task_param_t *head_param = NULL, *current_param = NULL, *tmp_param = NULL;
    dague_ddesc_t *ddesc;
    dague_data_key_t key;
    int tmp_op_type;

    int i;

    data_repo_entry_t *entry;
    dague_execution_context_t *T1;

    for (i=0; this_task->function->in[i] != NULL ; i++) {
        tmp_param = (dague_dtd_task_param_t *) malloc(sizeof(dague_dtd_task_param_t));

        dague_data_copy_t* copy;
        tmp_op_type = this_task->function->in[i]->flow_flags;
        int op_type;
        int mask, pred_found = 0;

        if ((tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_RW) {
            op_type = INOUT | REGION_FULL;
        } else if((tmp_op_type & FLOW_ACCESS_READ) == FLOW_ACCESS_READ) {
            op_type = INPUT | REGION_FULL;
        } else if((tmp_op_type & FLOW_ACCESS_WRITE) == FLOW_ACCESS_WRITE) {
            op_type = OUTPUT | REGION_FULL;
        } else if((tmp_op_type) == FLOW_ACCESS_NONE || tmp_op_type == FLOW_HAS_IN_DEPS) {
            op_type = INOUT | REGION_FULL;

            this_task->unused = 0;
            T1 = malloc (sizeof(dague_execution_context_t));
            mask = 1 << i;
            this_task->function->iterate_predecessors(context, this_task,  mask, copy_content, (void*)T1);
            if (this_task->unused != 0) {
                pred_found = 1;
            } else {
                pred_found = 2;
                continue;
            }

            if (pred_found == 1) {
                uint64_t id = T1->function->key(T1->dague_handle, T1->locals);
                entry = data_repo_lookup_entry(T1->dague_handle->repo_array[T1->function->function_id], id);
                copy = entry->data[T1->unused];
            } else {
            }
        }else {
            continue;
        }

        if (pred_found == 0) {
            ddesc = (dague_ddesc_t *)this_task->data[i].data_in->original;
            key = this_task->data[i].data_in->original->key;
            OBJ_RETAIN(this_task->data[i].data_in);
        } else if (pred_found == 1) {
            ddesc = (dague_ddesc_t *)copy->original;
            key   =  copy->original->key;
        }
        dague_dtd_tile_t *tile = tile_manage_for_testing(dtd_handle, ddesc, key);

        tmp_param->pointer_to_tile = (void *)tile;
        tmp_param->operation_type = op_type;
        tmp_param->tile_type_index = REGION_FULL;
        tmp_param->next = NULL;

        if(head_param == NULL) {
            head_param = tmp_param;
        } else {
            current_param->next = tmp_param;
        }
        count ++;
        current_param = tmp_param;
    }

    for (i=0; this_task->function->out[i] != NULL; i++) {
        int op_type;
        tmp_op_type = this_task->function->out[i]->flow_flags;
        dague_data_copy_t* copy;
        tmp_param = (dague_dtd_task_param_t *) malloc(sizeof(dague_dtd_task_param_t));
        int pred_found = 0;
        if((tmp_op_type & FLOW_ACCESS_WRITE) == FLOW_ACCESS_WRITE) {
            op_type = OUTPUT | REGION_FULL;
        } else if((tmp_op_type) == FLOW_ACCESS_NONE || tmp_op_type == FLOW_HAS_IN_DEPS) {
            pred_found = 1;
            op_type = INOUT | REGION_FULL;

            dague_data_t *fake_data = OBJ_NEW(dague_data_t);
            fake_data->key = rand();
            dague_data_copy_t *fake_data_copy = OBJ_NEW(dague_data_copy_t);
            copy = fake_data_copy;
            fake_data_copy->original = fake_data;
            this_task->data[this_task->function->out[i]->flow_index].data_out = fake_data_copy;

            ddesc = (dague_ddesc_t *)fake_data;
            key   =  fake_data->key;
        } else {
            continue;
        }

        if (pred_found == 0) {
            ddesc = (dague_ddesc_t *)this_task->data[i].data_out->original;
            key = this_task->data[i].data_out->original->key;
            OBJ_RETAIN(this_task->data[i].data_out);
        } else if (pred_found == 1) {
            ddesc = (dague_ddesc_t *)copy->original;
            key   = copy->original->key;
        }
        dague_dtd_tile_t *tile = tile_manage_for_testing(dtd_handle, ddesc, key);

        tmp_param->pointer_to_tile = (void *)tile;
        tmp_param->operation_type = op_type;
        tmp_param->tile_type_index = REGION_FULL;
        tmp_param->next = NULL;

        if(head_param == NULL) {
            head_param = tmp_param;
        } else {
            current_param->next = tmp_param;
        }
        count ++;
        current_param = tmp_param;

    }

    /* testing Insert Task */
    //insert_task_generic_fptr_for_testing(dtd_handle, __dtd_handle->actual_hook[this_task->function->function_id].hook,
     //                                    this_task, (char *)name, head_param);

    return DAGUE_HOOK_RETURN_DONE;
}

void
copy_chores(dague_handle_t *handle, dague_dtd_handle_t *dtd_handle)
{
    int total_functions = handle->nb_functions;
    int i, j;
    for (i=0; i<total_functions; i++) {
        for (j =0; handle->functions_array[i]->incarnations[j].hook != NULL; j++) {
            /* saving the CPU hook only */
            if (handle->functions_array[i]->incarnations[j].type == DAGUE_DEV_CPU) {
                dtd_handle->actual_hook[i].hook = handle->functions_array[i]->incarnations[j].hook;
            }
        }
        for (j =0; handle->functions_array[i]->incarnations[j].hook != NULL; j++) {
            /* copying ther dake hook in all the hooks (CPU, GPU etc) */
            dague_hook_t **hook_not_const = (dague_hook_t **)&(handle->functions_array[i]->incarnations[j].hook);
            *hook_not_const = &fake_hook_for_testing;
        }
    }
}
