/*
 * Copyright (c) 2016-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec.h"
#include "parsec/constants.h"
#include "parsec/data_internal.h"
#include "parsec/mca/device/device_gpu.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/utils/output.h"
#include "parsec/scheduling.h"

#if !defined(PARSEC_HAVE_CUDA) && !defined(PARSEC_HAVE_HIP) && !defined(PARSEC_HAVE_LEVEL_ZERO)
#error This file should not be included in a non-GPU (CUDA/HIP/LEVEL_ZERO) build
#endif  /* !defined(PARSEC_HAVE_CUDA) && !defined(PARSEC_HAVE_HIP) && !defined(PARSEC_HAVE_LEVEL_ZERO) */

/**
 * Entirely local tasks that should only be used to move data between a device and the main memory. Such
 * tasks would be generated by the GPU engine when a lack of data for future tasks is detected, in which
 * case the least recently used data owned by the device will be marked for transfer back to the main
 * memory. Upon successful transfer the data will become shared, and future tasks would then be able to
 * acquire the GPU space for their own usage. A second usage will be to forcefully move a data back to the
 * main memory (or any other location in fact), when a task that will execute in another context requires
 * one of the data from the GPU.
 */
typedef parsec_task_t parsec_gpu_d2h_task_t;

static int
hook_of_gpu_d2h_task( parsec_execution_stream_t* es,
                       parsec_gpu_d2h_task_t* this_task )
{
    (void)es; (void)this_task;
    return PARSEC_SUCCESS;
}

static int
affinity_of_gpu_d2h_task( parsec_gpu_d2h_task_t* this_task,
                           parsec_data_ref_t* ref )
{
    (void)this_task; (void)ref;
    return PARSEC_SUCCESS;
}

static void
iterate_successors_of_gpu_d2h_task( parsec_execution_stream_t* es,
                                     const parsec_gpu_d2h_task_t* this_task,
                                     uint32_t action_mask,
                                     parsec_ontask_function_t * ontask, void *ontask_arg )
{
    (void)es; (void)this_task; (void)action_mask; (void)ontask; (void)ontask_arg;
}

static void
iterate_predecessors_of_gpu_d2h_task( parsec_execution_stream_t* es,
                                       const parsec_gpu_d2h_task_t* this_task,
                                       uint32_t action_mask,
                                       parsec_ontask_function_t * ontask, void *ontask_arg )
{
    (void)es; (void)this_task; (void)action_mask; (void)ontask; (void)ontask_arg;
}

static int
release_deps_of_gpu_d2h_task( parsec_execution_stream_t* es,
                               parsec_gpu_d2h_task_t* this_task,
                               uint32_t action_mask,
                               parsec_remote_deps_t* deps )
{
    (void)es; (void)this_task; (void)action_mask; (void)deps;
    return PARSEC_SUCCESS;
}

static int
data_lookup_of_gpu_d2h_task( parsec_execution_stream_t* es,
                              parsec_gpu_d2h_task_t* this_task )
{
    (void)es; (void)this_task;
    return PARSEC_SUCCESS;
}

static int
complete_hook_of_gpu_d2h_task( parsec_execution_stream_t* es,
                                parsec_gpu_d2h_task_t* this_task )
{
    (void)es; (void)this_task;
    return PARSEC_SUCCESS;
}

static parsec_hook_return_t
release_task_of_gpu_d2h_task(parsec_execution_stream_t* es,
                              parsec_gpu_d2h_task_t* this_task )
{
    (void)es; (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

static int
datatype_lookup_of_gpu_d2h_task( parsec_execution_stream_t * es,
                                  const parsec_gpu_d2h_task_t* this_task,
                                  uint32_t * flow_mask, parsec_dep_data_description_t * data)
{
    (void)es; (void)this_task; (void)flow_mask; (void)data;
    return PARSEC_SUCCESS;
}

static int32_t parsec_gpu_d2h_counter = 0;
static parsec_key_t
key_of_gpu_d2h_task(const parsec_taskpool_t *tp,
                    const parsec_assignment_t *assignments)
{
    (void)tp; (void)assignments;
    return (parsec_key_t)(uint64_t)(1 + parsec_atomic_fetch_inc_int32(&parsec_gpu_d2h_counter));
}

static parsec_data_t*
flow_of_gpu_d2h_task_direct_access( const parsec_gpu_d2h_task_t* this_task,
                                     const parsec_assignment_t *assignments )
{
    (void)this_task; (void)assignments;
    return NULL;
}

static const __parsec_chore_t __gpu_d2h_task_chores[] = {
#if defined(PARSEC_HAVE_CUDA)
    {.type = PARSEC_DEV_CUDA,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) hook_of_gpu_d2h_task},
#endif
#if defined(PARSEC_HAVE_LEVEL_ZERO)
    {.type = PARSEC_DEV_LEVEL_ZERO,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) hook_of_gpu_d2h_task},
#endif
#if defined(PARSEC_HAVE_HIP)
    {.type = PARSEC_DEV_HIP,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) hook_of_gpu_d2h_task},
#endif

    {.type = PARSEC_DEV_NONE,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) NULL},   /* End marker */
};

static const parsec_flow_t flow_of_gpu_d2h_task;
static const parsec_dep_t flow_of_gpu_d2h_task_dep = {
    .cond = NULL,
    .ctl_gather_nb = NULL,
    .task_class_id = -1,
    .direct_data = (parsec_data_lookup_func_t)flow_of_gpu_d2h_task_direct_access,
    .dep_index = 1,
    .dep_datatype_index = 0,
    .belongs_to = &flow_of_gpu_d2h_task,
};

static const parsec_flow_t flow_of_gpu_d2h_task = {
    .name = "Generic flow for d2h tasks",
    .sym_type = PARSEC_SYM_OUT,
    .flow_flags = PARSEC_FLOW_ACCESS_RW | PARSEC_FLOW_HAS_IN_DEPS,
    .flow_index = 0,
    .flow_datatype_mask = 0x1,
    .dep_in = {},
    .dep_out = {&flow_of_gpu_d2h_task_dep}
};

static const parsec_symbol_t symb_gpu_d2h_task_param = {
    .name = "unnamed",
    .min = NULL,
    .max = NULL,
    .context_index = 0,
    .cst_inc = 1,
    .expr_inc = NULL,
    .flags = 0x0
};

int32_t parsec_gpu_d2h_max_flows = 0;

static const parsec_task_class_t parsec_gpu_d2h_task_class = {
    .name = "GPU D2H data transfer",
    .task_class_id = 0,
    .nb_flows = MAX_PARAM_COUNT,  /* This value will have an impact on the duration of the
                                   * search for additional data to move. As this search is linear
                                   * we need to keep this upper bound set to a reasonable value. */
    .nb_parameters = 1,
    .nb_locals = 0,
    .params = {&symb_gpu_d2h_task_param},
    .locals = {NULL},
    .data_affinity = (parsec_data_ref_fn_t *) affinity_of_gpu_d2h_task,
    .initial_data = (parsec_data_ref_fn_t *) NULL,
    .final_data = (parsec_data_ref_fn_t *) NULL,
    .priority = NULL,
    .in = {&flow_of_gpu_d2h_task, NULL},
    .out = {&flow_of_gpu_d2h_task, NULL},
    .flags = 0x0 | PARSEC_HAS_IN_IN_DEPENDENCIES | PARSEC_USE_DEPS_MASK,
    .dependencies_goal = 0x1,  /* we generate then when needed so the dependencies_goal is useless */
    .make_key = key_of_gpu_d2h_task,
    .fini = (parsec_hook_t *) NULL,
    .incarnations = __gpu_d2h_task_chores,
    .find_deps = parsec_hash_find_deps,
    .iterate_successors = (parsec_traverse_function_t *) iterate_successors_of_gpu_d2h_task,
    .iterate_predecessors = (parsec_traverse_function_t *) iterate_predecessors_of_gpu_d2h_task,
    .release_deps = (parsec_release_deps_t *) release_deps_of_gpu_d2h_task,
    .prepare_input = (parsec_hook_t *) data_lookup_of_gpu_d2h_task,
    .prepare_output = (parsec_hook_t *) NULL,
    .get_datatype = (parsec_datatype_lookup_t *) datatype_lookup_of_gpu_d2h_task,
    .complete_execution = (parsec_hook_t *) complete_hook_of_gpu_d2h_task,
    .release_task = (parsec_hook_t *) release_task_of_gpu_d2h_task,
#if defined(PARSEC_SIM)
    .sim_cost_fct = (parsec_sim_cost_fct_t *) NULL,
#endif
};


/**
 * Transfer at most the MAX_PARAM_COUNT oldest data from the GPU back
 * to main memory. Create a single task to move them all out, then switch the
 * GPU data copy in shared mode.
 */
parsec_gpu_task_t*
parsec_gpu_create_w2r_task(parsec_device_gpu_module_t *gpu_device,
                           parsec_execution_stream_t *es)
{
    parsec_gpu_task_t *w2r_task = NULL;
    parsec_gpu_d2h_task_t *d2h_task = NULL;
    parsec_gpu_data_copy_t *gpu_copy;
    parsec_list_item_t* item = (parsec_list_item_t*)gpu_device->gpu_mem_owned_lru.ghost_element.list_next;
    int nb_cleaned = 0;

    /* Find a data copy that has no pending users on the GPU, and can be
     * safely moved back on the main memory */
    while(nb_cleaned < parsec_gpu_d2h_max_flows) {
        /* Break at the end of the list */
        if( item == &(gpu_device->gpu_mem_owned_lru.ghost_element) ) {
            break;
        }
        gpu_copy = (parsec_gpu_data_copy_t*)item;
        parsec_atomic_lock( &gpu_copy->original->lock );
        /* get the next item before altering the next pointer */
        item = (parsec_list_item_t*)item->list_next;  /* conversion needed for volatile */
        if( 0 == gpu_copy->readers ) {
            if( PARSEC_UNLIKELY(NULL == d2h_task) ) {  /* allocate on-demand */
                d2h_task = (parsec_gpu_d2h_task_t*)parsec_thread_mempool_allocate(es->context_mempool);
                if( PARSEC_UNLIKELY(NULL == d2h_task) ) { /* we're running out of memory. Bail out. */
                    parsec_atomic_unlock( &gpu_copy->original->lock );
                    return NULL;
                }
            }
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            gpu_copy->readers++;
            d2h_task->data[nb_cleaned].data_out = gpu_copy;
            gpu_copy->data_transfer_status = PARSEC_DATA_STATUS_UNDER_TRANSFER;  /* mark the copy as in transfer */
            parsec_atomic_unlock( &gpu_copy->original->lock );
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "D2H[%s] task %p:\tdata %d -> %p [%p] readers %d",
                                 gpu_device->super.name, (void*)d2h_task,
                                 nb_cleaned, gpu_copy, gpu_copy->original, gpu_copy->readers);
            nb_cleaned++;
            if (MAX_PARAM_COUNT == nb_cleaned)
                break;
        }
    }

    if( 0 == nb_cleaned )
        return NULL;

    d2h_task->priority        = INT32_MAX;
    d2h_task->task_class      = &parsec_gpu_d2h_task_class;
    d2h_task->status          = PARSEC_TASK_STATUS_NONE;
    d2h_task->taskpool        = NULL;
    d2h_task->locals[0].value = nb_cleaned;

    w2r_task = (parsec_gpu_task_t *)malloc(sizeof(parsec_gpu_task_t));
    PARSEC_OBJ_CONSTRUCT(w2r_task, parsec_list_item_t);
    w2r_task->ec               = (parsec_task_t*)d2h_task;
    w2r_task->task_type        = PARSEC_GPU_TASK_TYPE_D2HTRANSFER;
    w2r_task->last_data_check_epoch = gpu_device->data_avail_epoch - 1;
    w2r_task->stage_in         = NULL;
    w2r_task->stage_out        = NULL;
    w2r_task->complete_stage   = NULL;

    (void)es;
    return w2r_task;
}

/**
 * Complete a data copy transfer originated from the engine.
 */
int parsec_gpu_complete_w2r_task(parsec_device_gpu_module_t *gpu_device,
                             parsec_gpu_task_t *gpu_task,
                             parsec_execution_stream_t *es)
{
    parsec_gpu_data_copy_t *gpu_copy, *cpu_copy;
    parsec_gpu_d2h_task_t* task = (parsec_gpu_d2h_task_t*)gpu_task->ec;
    parsec_data_t* original;

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "D2H[%s] task %p: %d data transferred to host",
                         gpu_device->super.name, (void*)task, task->locals[0].value);
    assert(gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER);
    for( int i = 0; i < task->locals[0].value; i++ ) {
        gpu_copy = task->data[i].data_out;
        parsec_atomic_lock(&gpu_copy->original->lock);
        gpu_copy->readers--;
        gpu_copy->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;
        gpu_device->super.data_out_to_host += gpu_copy->original->nb_elts; /* TODO: not hardcoded, use datatype size */
        assert(gpu_copy->readers >= 0);

        original = gpu_copy->original;

        cpu_copy = original->device_copies[0];

        if( cpu_copy->version < gpu_copy->version ) {
            /* the GPU version has been acquired by a new task that is waiting for submission */
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "D2H[%s] task %p:%i GPU data copy %p [%p] has a backup in memory",
                                 gpu_device->super.name, (void*)task, i, gpu_copy, gpu_copy->original);
        } else {
            gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->coherency_state =  PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->version = gpu_copy->version;
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%s]: CPU copy %p gets the same version %d as GPU copy %p at %s:%d",
                                 gpu_device->super.name,
                                 cpu_copy, cpu_copy->version, gpu_copy,
                                 __FILE__, __LINE__);
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "D2H[%s] task %p:%i GPU data copy %p [%p] now available",
                                 gpu_device->super.name, (void*)task, i, gpu_copy, gpu_copy->original);
            parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
        parsec_atomic_unlock(&gpu_copy->original->lock);
    }
    parsec_thread_mempool_free(es->context_mempool, task);
    free(gpu_task);
    gpu_device->data_avail_epoch++;
    return 0;
}
