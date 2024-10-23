/*
 *
 * Copyright (c) 2021-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/device_gpu.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/argv.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"

#include <limits.h>

#define PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL 1024

#if defined(PARSEC_PROF_TRACE)
static int parsec_gpu_movein_key_start;
static int parsec_gpu_movein_key_end;
static int parsec_gpu_moveout_key_start;
static int parsec_gpu_moveout_key_end;
static int parsec_gpu_own_GPU_key_start;
static int parsec_gpu_own_GPU_key_end;
static int parsec_gpu_allocate_memory_key;
static int parsec_gpu_free_memory_key;
static int parsec_gpu_use_memory_key_start;
static int parsec_gpu_use_memory_key_end;
static int parsec_gpu_prefetch_key_start;
static int parsec_gpu_prefetch_key_end;
static int parsec_gpu_profiling_initiated = 0;
#endif  /* defined(PROFILING) */
int parsec_gpu_output_stream = -1;
int parsec_gpu_verbosity;

static inline int
parsec_device_check_space_needed(parsec_device_gpu_module_t *gpu_device,
                                 parsec_gpu_task_t *gpu_task)
{
    int i;
    int space_needed = 0;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_data_t *original;
    parsec_data_copy_t *data;
    const parsec_flow_t *flow;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        data = this_task->data[i].data_in;
        if (data == NULL) continue;

        original = data->original;
        if( NULL != PARSEC_DATA_GET_COPY(original, gpu_device->super.device_index) ) {
            continue;
        }
        if(flow->flow_flags & PARSEC_FLOW_ACCESS_READ)
            space_needed++;
    }
    return space_needed;
}

#if defined(PARSEC_PROF_TRACE)
void parsec_device_init_profiling(void)
{
    if(parsec_gpu_profiling_initiated == 0) {
        parsec_profiling_add_dictionary_keyword("gpu", "fill:#66ff66",
                                                0, NULL,
                                                &parsec_gpu_own_GPU_key_start, &parsec_gpu_own_GPU_key_end);
        parsec_profiling_add_dictionary_keyword("movein", "fill:#33FF33",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_movein_key_start, &parsec_gpu_movein_key_end);
        parsec_profiling_add_dictionary_keyword("moveout", "fill:#ffff66",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_moveout_key_start, &parsec_gpu_moveout_key_end);
        parsec_profiling_add_dictionary_keyword("prefetch", "fill:#66ff66",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_prefetch_key_start, &parsec_gpu_prefetch_key_end);
        parsec_profiling_add_dictionary_keyword("gpu_mem_alloc", "fill:#FF66FF",
#if (PARSEC_SIZEOF_SIZE_T == 4)
                                                sizeof(uint32_t), "size{uint32_t}",
#elif (PARSEC_SIZEOF_SIZE_T == 8)
                                                sizeof(uint64_t), "size{uint64_t}",
#else
#error "Unsupported case: sizeof(size_t) is neither 8 nor 4"
#endif // PARSEC_SIZEOF_SIZE_T
                                                &parsec_gpu_allocate_memory_key, &parsec_gpu_free_memory_key);
        parsec_profiling_add_dictionary_keyword("gpu_mem_use", "fill:#FF66FF",
                                                sizeof(parsec_device_gpu_memory_prof_info_t),
                                                PARSEC_DEVICE_GPU_MEMORY_PROF_INFO_CONVERTER,
                                                &parsec_gpu_use_memory_key_start, &parsec_gpu_use_memory_key_end);
        parsec_gpu_profiling_initiated = 1;
    }
}
#endif

void parsec_device_enable_debug(void)
{
    if(parsec_gpu_output_stream == -1) {
        parsec_gpu_output_stream = parsec_device_output;
        if( parsec_gpu_verbosity >= 0 ) {
            parsec_gpu_output_stream = parsec_output_open(NULL);
            parsec_output_set_verbosity(parsec_gpu_output_stream, parsec_gpu_verbosity);
        }
    }
}

int parsec_device_sort_pending_list(parsec_device_module_t *device)
{
    if( !PARSEC_DEV_IS_GPU(device->type) )
        return 0;

    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t *)device;
    parsec_list_t *sort_list = gpu_device->exec_stream[0]->fifo_pending;

    if (parsec_list_is_empty(sort_list) ) { /* list is empty */
        return 0;
    }

    if (gpu_device->sort_starting_p == NULL || !parsec_list_nolock_contains(sort_list, gpu_device->sort_starting_p) ) {
        gpu_device->sort_starting_p = (parsec_list_item_t*)sort_list->ghost_element.list_next;
    }

    /* p is head */
    parsec_list_item_t *p = gpu_device->sort_starting_p;
    int i, j, NB_SORT = 10, space_q, space_min;

    parsec_list_item_t *q, *prev_p, *min_p;
    for (i = 0; i < NB_SORT; i++) {
        if ( p == &(sort_list->ghost_element) ) {
            break;
        }
        min_p = p; /* assume the minimum one is the first one p */
        q = (parsec_list_item_t*)min_p->list_next;
        space_min = parsec_device_check_space_needed(gpu_device, (parsec_gpu_task_t*)min_p);
        for (j = i+1; j < NB_SORT; j++) {
            if ( q == &(sort_list->ghost_element) ) {
                break;
            }
            space_q = parsec_device_check_space_needed(gpu_device, (parsec_gpu_task_t*)q);
            if ( space_min > space_q ) {
                min_p = q;
                space_min = space_q;
            }
            q = (parsec_list_item_t*)q->list_next;

        }
        if (min_p != p) { /* minimum is not the first one, let's insert min_p before p */
            /* take min_p out */
            parsec_list_item_ring_chop(min_p);
            PARSEC_LIST_ITEM_SINGLETON(min_p);
            prev_p = (parsec_list_item_t*)p->list_prev;

            /* insert min_p after prev_p */
            parsec_list_add_after( sort_list, prev_p, min_p);
        }
        p = (parsec_list_item_t*)min_p->list_next;
    }

    return 0;
}

void* parsec_device_pop_workspace(parsec_device_gpu_module_t* gpu_device,
                                  parsec_gpu_exec_stream_t* gpu_stream, size_t size)
{
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (parsec_gpu_workspace_t *)malloc(sizeof(parsec_gpu_workspace_t));
        gpu_stream->workspace->total_workspace = PARSEC_GPU_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = PARSEC_GPU_MAX_WORKSPACE - 1;

        for( int i = 0; i < PARSEC_GPU_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                 "GPU[%d:%s] Succeeded Allocating workspace %d (device_ptr %p)",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 i, gpu_stream->workspace->workspace[i]);
#if defined(PARSEC_PROF_TRACE)
            if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
               (gpu_device->exec_stream[0]->prof_event_track_enable ||
                gpu_device->exec_stream[1]->prof_event_track_enable)) {
                parsec_profiling_trace_flags(gpu_stream->profiling,
                                             parsec_gpu_allocate_memory_key, (int64_t)
                                             gpu_stream->workspace->workspace[i], gpu_device->super.device_index,
                                             &size, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
            }
#endif
        }
    }
    if (gpu_stream->workspace->stack_head < 0) {
        parsec_fatal("parsec_device_pop_workspace: user requested more than %d GPU workspaces which is the current hard-coded limit per GPU stream\n", PARSEC_GPU_MAX_WORKSPACE);
        return NULL;
    }
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return work;
}

int parsec_device_push_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_GPU_MAX_WORKSPACE);
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return 0;
}

int parsec_device_free_workspace(parsec_device_gpu_module_t * gpu_device)
{
    (void)gpu_device;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *gpu_stream = gpu_device->exec_stream[i];
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
#if defined(PARSEC_PROF_TRACE)
                if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
                   (gpu_device->exec_stream[0]->prof_event_track_enable ||
                    gpu_device->exec_stream[1]->prof_event_track_enable)) {
                    parsec_profiling_trace_flags(gpu_stream->profiling,
                                                 parsec_gpu_allocate_memory_key, (int64_t)
                                                 gpu_stream->workspace->workspace[i], gpu_device->super.device_index,
                                                 NULL, PARSEC_PROFILING_EVENT_COUNTER);
                }
#endif
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s] Release workspace %d (device_ptr %p)",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     j, gpu_stream->workspace->workspace[j]);
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return 0;
}

#if defined(PARSEC_DEBUG_NOISIER)
char *parsec_device_describe_gpu_task( char *tmp, size_t len, parsec_gpu_task_t *gpu_task )
{
    char buffer[64];
    parsec_data_t *data;
    switch( gpu_task->task_type ) {
        case PARSEC_GPU_TASK_TYPE_KERNEL:
            return parsec_task_snprintf(tmp, len, gpu_task->ec);
        case PARSEC_GPU_TASK_TYPE_PREFETCH:
            assert(NULL != gpu_task->ec);
            assert(NULL != gpu_task->ec->data[0].data_in );
            data = gpu_task->ec->data[0].data_in->original;
            if( NULL == data || NULL == data->dc )
                snprintf(tmp, len, "PREFETCH for unbound data %p", data);
            else {
                data->dc->key_to_string(data->dc, data->key, buffer, 64);
                snprintf(tmp, len, "PREFETCH for %s (data %p)", buffer, data);
            }
            return tmp;
        case PARSEC_GPU_TASK_TYPE_WARMUP:
            assert(NULL != gpu_task->copy->original && NULL != gpu_task->copy->original->dc);
            gpu_task->copy->original->dc->key_to_string(gpu_task->copy->original->dc, gpu_task->copy->original->key, buffer, 64);
            snprintf(tmp, len, "WARMUP %s on device %d",
                     buffer, gpu_task->copy->device_index);
            return tmp;
        case PARSEC_GPU_TASK_TYPE_D2HTRANSFER:
            snprintf(tmp, len, "Device to Host Transfer");
            return tmp;
        case PARSEC_GPU_TASK_TYPE_D2D_COMPLETE:
            snprintf(tmp, len, "D2D Transfer Complete for data copy %p [ref_count %d]",
                     gpu_task->ec->data[0].data_out, gpu_task->ec->data[0].data_out->super.super.obj_reference_count);
            return tmp;
        default:
            snprintf(tmp, len, "*** Internal Error: unknown gpu task type %d ***", gpu_task->task_type);
            return tmp;
    }
}
#endif

void parsec_device_dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    parsec_debug_verbose(0, parsec_gpu_output_stream,
                         "Dev: GPU stream %d{%p} [events = %d, start = %d, end = %d, executed = %d]",
                         exec_stream->name, exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
                         exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        parsec_debug_verbose(0, parsec_gpu_output_stream,
                             "    %d: %s", i, parsec_task_snprintf(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void parsec_device_dump_gpu_state(parsec_device_gpu_module_t* gpu_device)
{
    int i;
    uint64_t data_in_host, data_in_dev = 0;

    data_in_host = gpu_device->super.data_in_from_device[0];
    for(int i = 1; i < gpu_device->super.data_in_array_size; i++) {
        data_in_dev += gpu_device->super.data_in_from_device[i];
    }

    parsec_output(parsec_gpu_output_stream, "\n\n");
    parsec_output(parsec_gpu_output_stream, "Device %d:%d (%p) epoch\n", gpu_device->super.device_index,
                  gpu_device->super.device_index, gpu_device, gpu_device->data_avail_epoch);
    parsec_output(parsec_gpu_output_stream, "\tpeer mask %x executed tasks with %llu streams %d\n",
                  gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->num_exec_streams);
    parsec_output(parsec_gpu_output_stream, "\tstats transferred [in: %llu from host %llu from other device out: %llu] required [in: %llu out: %llu]\n",
                  (unsigned long long)data_in_host, (unsigned long long)data_in_dev,
                  (unsigned long long)gpu_device->super.data_out_to_host,
                  (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        parsec_device_dump_exec_stream(gpu_device->exec_stream[i]);
    }
    if( !parsec_list_is_empty(&gpu_device->gpu_mem_lru) ) {
        parsec_output(parsec_gpu_output_stream, "#\n# LRU list\n#\n");
        i = 0;
        PARSEC_LIST_ITERATOR(&gpu_device->gpu_mem_lru, item,
                             {
                                 parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                 parsec_output(parsec_gpu_output_stream, "  %d. elem %p flags 0x%x GPU mem %p\n",
                                               i, gpu_copy, gpu_copy->flags, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    }
    if( !parsec_list_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        parsec_output(parsec_gpu_output_stream, "#\n# Owned LRU list\n#\n");
        i = 0;
        PARSEC_LIST_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                             {
                                 parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                 parsec_output(parsec_gpu_output_stream, "  %d. elem %p flags 0x%x GPU mem %p\n",
                                               i, gpu_copy, gpu_copy->flags, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    }
    parsec_output(parsec_gpu_output_stream, "\n\n");
}


static parsec_flow_t parsec_device_data_prefetch_flow = {
    .name = "PREFETCH FLOW",
    .flow_flags = PARSEC_FLOW_ACCESS_READ,
    .flow_index = 0,
};

static parsec_task_class_t parsec_device_data_prefetch_tc = {
    .name = "DEVICE PREFETCH",
    .flags = 0,
    .task_class_id = 0,
    .nb_flows = 1,
    .nb_parameters = 0,
    .nb_locals = 0,
    .dependencies_goal = 0,
    .params = { NULL, },
    .in = { &parsec_device_data_prefetch_flow, NULL },
    .out = { NULL, },
    .priority = NULL,
    .properties = NULL,
    .initial_data = NULL,
    .final_data = NULL,
    .data_affinity = NULL,
    .key_functions = NULL,
    .make_key = NULL,
    .get_datatype = NULL,
    .prepare_input = NULL,
    .incarnations = NULL,
    .prepare_output = NULL,
    .find_deps = NULL,
    .iterate_successors = NULL,
    .iterate_predecessors = NULL,
    .release_deps = NULL,
    .complete_execution = NULL,
    .new_task = NULL,
    .release_task = NULL,
    .fini = NULL
};

static int
parsec_device_release_resources_prefetch_task(parsec_device_gpu_module_t* gpu_device,
                        parsec_gpu_task_t** out_task)
{
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    parsec_gpu_task_t *gpu_task = *out_task;
    (void)gpu_device;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]: Releasing resources for task %s (%p with ec %p)",
                         gpu_device->super.device_index, gpu_device->super.name, parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task),
                         gpu_task, gpu_task->ec);
    assert( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type );
    PARSEC_DATA_COPY_RELEASE( gpu_task->ec->data[0].data_in);
    free( gpu_task->ec );
    gpu_task->ec = NULL;
    return 0;
}

#if defined(PARSEC_DEBUG_NOISIER)
static char *parsec_device_debug_advice_to_string(int advice)
{
    switch(advice) {
    case PARSEC_DEV_DATA_ADVICE_PREFETCH:
        return "Prefetch";
    case PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE:
        return "Set Preferred Device";
    case PARSEC_DEV_DATA_ADVICE_WARMUP:
        return "Mark data as recently used";
    default:
        assert(0);
        return "Undefined advice";
    }
}
#endif

int
parsec_device_data_advise(parsec_device_module_t *dev, parsec_data_t *data, int advice)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)dev;
#if defined(PARSEC_DEBUG_NOISIER)
    char buffer[64];
    if(NULL != data->dc) {
        data->dc->key_to_string(data->dc, data->key, buffer, 64);
    } else {
        snprintf(buffer, 64, "unbound data");
    }
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]: User provides advice %s of %s (%p)",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_device_debug_advice_to_string(advice),
                         buffer,
                         data);

    switch(advice) {
    case PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE:
        data->preferred_device = dev->device_index;
        break;
    case PARSEC_DEV_DATA_ADVICE_PREFETCH:
        {
            if( parsec_type_contiguous(data->device_copies[ data->owner_device ]->dtt) != PARSEC_SUCCESS){
                parsec_warning( "GPU[%d:%s]: PARSEC_DEV_DATA_ADVICE_PREFETCH cannot be applied to non contiguous types @%s:%d",
                                gpu_device->super.device_index, gpu_device->super.name, __func__, __LINE__);
                return PARSEC_ERROR;
            }
            parsec_gpu_task_t* gpu_task = NULL;
            gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
            gpu_task->task_type = PARSEC_GPU_TASK_TYPE_PREFETCH;
            gpu_task->ec = calloc(1, sizeof(parsec_task_t));
            PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
            gpu_task->ec->task_class = &parsec_device_data_prefetch_tc;
            gpu_task->flow[0] = &parsec_device_data_prefetch_flow;
            gpu_task->flow_nb_elts[0] = data->device_copies[ data->owner_device ]->original->nb_elts;
            gpu_task->stage_in  = parsec_default_gpu_stage_in;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Retain data copy %p [ref_count %d]",
                                 data->device_copies[ data->owner_device ],
                                 data->device_copies[ data->owner_device ]->super.super.obj_reference_count);
            PARSEC_OBJ_RETAIN(data->device_copies[ data->owner_device ]);
            gpu_task->ec->data[0].data_in = data->device_copies[ data->owner_device ];
            gpu_task->ec->data[0].data_out = NULL;
            gpu_task->ec->data[0].source_repo_entry = NULL;
            gpu_task->ec->data[0].source_repo = NULL;
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: data copy %p [ref_count %d] linked to prefetch gpu task %p on GPU copy %p [ref_count %d]",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_task->ec->data[0].data_in, gpu_task->ec->data[0].data_in->super.super.obj_reference_count,
                                 gpu_task, gpu_task->ec->data[0].data_out, gpu_task->ec->data[0].data_out->super.super.obj_reference_count);
            parsec_fifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
            return PARSEC_SUCCESS;
        }
        break;
    case PARSEC_DEV_DATA_ADVICE_WARMUP:
        return PARSEC_ERR_NOT_IMPLEMENTED;
        break;
    default:
        assert(0);
        return PARSEC_ERR_NOT_FOUND;
    }
    return PARSEC_SUCCESS;
}

/**
 * Register a taskpool with a device by checking that the device
 * supports the dynamic function required by the different incarnations.
 * If multiple devices of the same type exists we assume that all have
 * the same capabilities.
 */
int
parsec_device_taskpool_register(parsec_device_module_t* device,
                                parsec_taskpool_t* tp)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)device;
    int32_t rc = PARSEC_ERR_NOT_FOUND;
    uint32_t i, j;

    /**
     * Detect if a particular chore has a dynamic load dependency and if yes
     * load the corresponding module and find the function.
     */
    assert(PARSEC_DEV_IS_GPU(device->type));
    assert(tp->devices_index_mask & (1 << device->device_index));

    for( i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type != device->type )
                continue;
            if( NULL != chores[j].dyld_fn ) {
                /* the function has been set for another device of the same type */
                return PARSEC_SUCCESS;
            }
            if ( NULL == chores[j].dyld ) {
                chores[j].dyld_fn = NULL;  /* No dynamic support required for this kernel */
                rc = PARSEC_SUCCESS;
            } else {
                void* devf = gpu_device->find_incarnation(gpu_device, chores[j].dyld);
                if( NULL != devf ) {
                    chores[j].dyld_fn = devf;
                    rc = PARSEC_SUCCESS;
                }
            }
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        tp->devices_index_mask &= ~(1 << device->device_index);  /* drop support for this device */
        parsec_debug_verbose(10, parsec_gpu_output_stream,
                             "Device %d:%s disabled for taskpool %d:%s (%p)", device->device_index, device->name,
                             tp->taskpool_id, tp->taskpool_name, tp);
    }
    return rc;
}

int
parsec_device_taskpool_unregister(parsec_device_module_t* device, parsec_taskpool_t* tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

/**
 * Attach a device to a PaRSEC context. A device can only be attached to
 * a single context at the time.
 */
int
parsec_device_attach( parsec_device_module_t* device, parsec_context_t* context )
{
    return parsec_mca_device_add(context, device);
}

/**
 * Detach a device from a context. Both the context and the device remain
 * valid, they are simply disconnected.
 * This function should only be called once all tasks and all data related to the
 * context has been removed from the device.
 */
int
parsec_device_detach( parsec_device_module_t* device, parsec_context_t* context )
{
    (void)context;
    return parsec_mca_device_remove(device);
}

/**
 * This function reserve the memory_percentage of the total device memory for PaRSEC.
 * This memory will be managed in chunks of size eltsize. However, multiple chunks
 * can be reserved in a single allocation.
 */
int
parsec_device_memory_reserve( parsec_device_gpu_module_t* gpu_device,
                              int           memory_percentage,
                              int           number_blocks,
                              size_t        eltsize )
{
    int rc;

    size_t alloc_size;
    size_t total_mem, initial_free_mem;
    size_t mem_elem_per_gpu = 0;

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return rc;

    /* Determine how much memory we can allocate */
    rc = gpu_device->memory_info( gpu_device, &initial_free_mem, &total_mem );
    if(PARSEC_SUCCESS != rc)
        return rc;

    if( number_blocks != -1 ) {
        if( number_blocks == 0 ) {
            parsec_warning("GPU[%d:%s] Invalid argument: requesting 0 bytes of memory",
                           gpu_device->super.device_index, gpu_device->super.name);
            return PARSEC_ERROR;
        } else {
            alloc_size = number_blocks * eltsize;
        }
    } else {
        /* number_blocks == -1 means memory_percentage is used */
        alloc_size = (memory_percentage * initial_free_mem) / 100;
        /* round-up in eltsize */
        alloc_size = eltsize * ((alloc_size + eltsize - 1 ) / eltsize);
    }
    if( alloc_size >= initial_free_mem ) {
        /* Mapping more than 100% of GPU memory is obviously wrong
         * Mapping exactly 100% of the GPU memory ends up producing errors about __global__ function call is not configured
         * Mapping 95% works with low-end GPUs like 1060, how much to let available for gpu runtime, I don't know how to calculate */
        parsec_warning("GPU[%d:%s] Requested %zd bytes on GPU device, but only %zd bytes are available -- reducing allocation to 95%% of max available",
                       gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem);
        alloc_size = (95 * initial_free_mem) / 100;
        /* round-up in eltsize */
        alloc_size = eltsize * ((alloc_size + eltsize - 1 ) / eltsize);
    }
    if( alloc_size < eltsize ) {
        /* Handle another kind of jokers entirely, and cases of
         * not enough memory on the device */
        parsec_warning("GPU[%d:%s] Cannot allocate at least one element",
                       gpu_device->super.device_index, gpu_device->super.name);
        return PARSEC_ERROR;
    }

#if defined(PARSEC_GPU_ALLOC_PER_TILE)
    size_t free_mem = initial_free_mem;
    /*
     * We allocate a bunch of tiles that will be used
     * during the computations
     */
    while( (free_mem > eltsize )
           && ((total_mem - free_mem) < alloc_size) ) {
        parsec_gpu_data_copy_t* gpu_elem;
        void *device_ptr;

        rc = gpu_device->memory_allocate(gpu_device, eltsize, &device_ptr);
        if(PARSEC_SUCCESS != rc) {
            size_t _free_mem, _total_mem;
            gpu_device->memory_info(gpu_device, &_free_mem, &_total_mem );
            parsec_inform("GPU[%d:%s] Per context: free mem %zu total mem %zu (allocated tiles %zu)",
                          gpu_device->super.device_index, gpu_device->super.name,_free_mem, _total_mem, mem_elem_per_gpu);
            break;
        }
        gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Allocate GPU copy %p [ref_count %d] for data [%p]",
                            gpu_device->super.device_index, gpu_device->super.name,gpu_elem, gpu_elem->super.obj_reference_count, NULL);
        gpu_elem->device_private = (void*)(long)device_ptr;
        gpu_elem->flags |= PARSEC_DATA_FLAG_PARSEC_OWNED;
        gpu_elem->device_index = gpu_device->super.device_index;
        mem_elem_per_gpu++;
        PARSEC_OBJ_RETAIN(gpu_elem);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Retain and insert GPU copy %p [ref_count %d] in LRU",
                            gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.obj_reference_count);
        parsec_list_push_back( &gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem );
        gpu_device->memory_info( gpu_device, &free_mem, &total_mem );
    }
    if( 0 == mem_elem_per_gpu && parsec_list_is_empty( &gpu_device->gpu_mem_lru ) ) {
        parsec_warning("GPU[%d:%s] Cannot allocate memory on GPU %s. Skip it!", gpu_device->super.device_index, gpu_device->super.name, gpu_device->super.name);
    }
    else {
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s] Allocate %zu tiles on the GPU memory",
                             gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu );
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                         "GPU[%d:%s] Allocate %zu tiles on the GPU memory", gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu);
#else
    if( NULL == gpu_device->memory ) {
        void* base_ptr;

        rc = gpu_device->memory_allocate(gpu_device, alloc_size, &base_ptr);
        if(PARSEC_SUCCESS != rc) {
            parsec_warning("GPU[%d:%s] Allocating %zu bytes of memory on the GPU device failed (initial_free_mem was %zu)",
                           gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem);
            gpu_device->memory = NULL;
            return PARSEC_ERROR;
        }

        assert(alloc_size % eltsize == 0); /* we rounded up earlier... */
        mem_elem_per_gpu = alloc_size / eltsize;
        gpu_device->memory = zone_malloc_init( base_ptr, mem_elem_per_gpu, eltsize );
        if( gpu_device->memory == NULL ) {
            parsec_warning("GPU[%d:%s] Failed trying to allocate %zu bytes. We tried to do so based on an initial_free_mem of %zu bytes and elt_size of %zu bytes",
                           gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem, eltsize);
            return PARSEC_ERROR;
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Allocate %zu segments of size %zu on the GPU memory",
                            gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu, eltsize );
    }
#endif
    gpu_device->mem_block_size = eltsize;
    gpu_device->mem_nb_blocks = mem_elem_per_gpu;

    return PARSEC_SUCCESS;
}

static void parsec_device_memory_release_list(parsec_device_gpu_module_t* gpu_device,
                                              parsec_list_t* list)
{
    parsec_list_item_t* item;

    while(NULL != (item = parsec_list_pop_front(list)) ) {
        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
        parsec_data_t* original = gpu_copy->original;

        PARSEC_DEBUG_VERBOSE(35, parsec_gpu_output_stream,
                            "GPU[%d:%s] Release GPU copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p, in map %p",
                            gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                             original, (NULL != original ? original->dc : NULL));
        assert( gpu_copy->device_index == gpu_device->super.device_index );

        if( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state ) {
            parsec_warning("GPU[%d:%s] still OWNS the master memory copy for data %d and it is discarding it!",
                           gpu_device->super.device_index, gpu_device->super.name, original->key);
        }
        assert(0 != (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );

#if defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_device->memory_free( gpu_copy->device_private );
#else

#if defined(PARSEC_PROF_TRACE)
        if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
           (gpu_device->exec_stream[0]->prof_event_track_enable ||
            gpu_device->exec_stream[1]->prof_event_track_enable)) {
            parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                         parsec_gpu_free_memory_key, (int64_t)gpu_copy->device_private,
                                         gpu_device->super.device_index,
                                         NULL, PARSEC_PROFILING_EVENT_COUNTER);
            parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                         parsec_gpu_use_memory_key_end,
                                         (uint64_t)gpu_copy->device_private,
                                         gpu_device->super.device_index, NULL, 0);
        }
#endif
        zone_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
        gpu_copy->device_private = NULL;

        /* At this point the data copies should have no attachment to a data_t. Thus,
         * before we get here (aka below parsec_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        PARSEC_OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
    }
}

/**
 * This function only flushes the data copies pending in LRU, and checks
 * (in debug mode) that the entire allocated memory is free to use */
int
parsec_device_flush_lru( parsec_device_module_t *device )
{
    size_t in_use;
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)device;
    /* Free all memory on GPU */
    parsec_device_memory_release_list(gpu_device, &gpu_device->gpu_mem_lru);
    parsec_device_memory_release_list(gpu_device, &gpu_device->gpu_mem_owned_lru);
    parsec_device_free_workspace(gpu_device);
#if !defined(PARSEC_GPU_ALLOC_PER_TILE) && !defined(_NDEBUG)
    if( (in_use = zone_in_use(gpu_device->memory)) != 0 ) {
        parsec_warning("GPU[%d:%s] memory leak detected: %lu bytes still allocated on GPU",
                       device->device_index, device->name, in_use);
        zone_debug(gpu_device->memory, 0, parsec_gpu_output_stream, "flush_lru: ");
        assert(!in_use);
    }
#endif
    return PARSEC_SUCCESS;
}

/**
 * This function release the GPU memory reserved for this device.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
int
parsec_device_memory_release( parsec_device_gpu_module_t* gpu_device )
{
    int rc;

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return rc;

    parsec_device_flush_lru(&gpu_device->super);

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    assert( NULL != gpu_device->memory );
    void* ptr = zone_malloc_fini(&gpu_device->memory);
    rc = gpu_device->memory_free(gpu_device, ptr);
    if(PARSEC_SUCCESS != rc) {
        parsec_warning("GPU[%d:%s]: Failed to free the GPU backend memory.",
                       gpu_device->super.device_index, gpu_device->super.name);
        return rc;
    }
#endif

    return PARSEC_SUCCESS;
}

/**
 * Try to find memory space to move all data on the GPU. We attach a device_elem to
 * a memory_elem as soon as a device_elem is available. If we fail to find enough
 * available elements, we push all the elements handled during this allocation
 * back into the pool of available device_elem, to be picked up by another call
 * (this call will remove them from the current task).
 * Returns:
 *   PARSEC_HOOK_RETURN_DONE:  All gpu_mem/mem_elem have been initialized
 *   PARSEC_HOOK_RETURN_AGAIN: At least one flow is marked under transfer, task cannot be scheduled yet
 *   PARSEC_HOOK_RETURN_NEXT:  The task needs to rescheduled
 */
static inline int
parsec_device_data_reserve_space( parsec_device_gpu_module_t* gpu_device,
                                  parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    parsec_data_t* master, *oldmaster;
    const parsec_flow_t *flow;
    int i, j, data_avail_epoch = 0, copy_readers_update = 0;
    parsec_gpu_data_copy_t *gpu_mem_lru_cycling = NULL;

#if defined(PARSEC_DEBUG_NOISIER)
    char task_name[MAX_TASK_STRLEN];
    parsec_task_snprintf(task_name, MAX_TASK_STRLEN, this_task);
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    (void)copy_readers_update; // potentially unused

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Investigating flow %s:%d",
                             gpu_device->super.device_index, gpu_device->super.name, task_name, flow->name, i);
        temp_loc[i] = NULL;
        if (this_task->data[i].data_in == NULL)
            continue;

        master   = this_task->data[i].data_in->original;
        parsec_atomic_lock(&master->lock);
        gpu_elem = PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;

        /* There is already a copy on the device */
        if( NULL != gpu_elem ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Flow %s:%i has a copy on the device %p%s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 flow->name, i, gpu_elem,
                                 gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ? " [in transfer]" : "");
            if ( gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {
              /* The data is indeed under transfer, but as we always force an event at the end of this
                 * step, we do not need to have a special case for this, because the forced event will
                 * ensure the data will be available on the GPU by the time this task will move to the
                 * next step.
                 */
            }
            parsec_atomic_unlock(&master->lock);
            continue;
        }

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Allocate GPU copy %p sz %zu [ref_count %d] for data %p",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             gpu_elem, gpu_task->flow_nb_elts[i], gpu_elem->super.super.obj_reference_count, master);
        gpu_elem->flags = PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED;
    malloc_data:
        copy_readers_update = 0;
        assert(0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        gpu_elem->device_private = zone_malloc(gpu_device->memory, gpu_task->flow_nb_elts[i]);
        if( NULL == gpu_elem->device_private ) {
#endif

        find_another_data:
            temp_loc[i] = NULL;
            /* Look for a data_copy to free */
            lru_gpu_elem = (parsec_gpu_data_copy_t*)parsec_list_pop_front(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the begining of
                 * the LRU (in order to be reused asap) and return with error.
                 */
            release_temp_and_return:
#if defined(PARSEC_DEBUG_NOISIER)
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\tRequest space on GPU failed for flow %s index %d/%d for task %s",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     flow->name, i, this_task->task_class->nb_flows, task_name );
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
                for( j = 0; j <= i; j++ ) {
                    /* This flow could be a control flow */
                    if( NULL == temp_loc[j] ) continue;
                    /* This flow could be non-parsec-owned, in which case we can't reclaim it */
                    if( 0 == (temp_loc[j]->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:%s:\tAdd copy %p [ref_count %d] back to the LRU list",
                                         gpu_device->super.device_index, gpu_device->super.name, task_name,
                                         temp_loc[j], temp_loc[j]->super.super.obj_reference_count);
                    /* push them at the head to reach them again at the next iteration */
                    parsec_list_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
                }
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
                PARSEC_OBJ_RELEASE(gpu_elem);
#endif
                parsec_atomic_unlock(&master->lock);
                return PARSEC_HOOK_RETURN_AGAIN;
            }

            PARSEC_LIST_ITEM_SINGLETON(lru_gpu_elem);
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Evaluate LRU-retrieved GPU copy %p [ref_count %d] original %p",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count,
                                 lru_gpu_elem->original);

            if( gpu_mem_lru_cycling == lru_gpu_elem ) {
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: Cycle detected on allocating memory for %s",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name);
                temp_loc[i] = lru_gpu_elem;  /* save it such that it gets pushed back into the LRU */
                goto release_temp_and_return;
            }

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the parsec_device_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s: Drop LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count, lru_gpu_elem->original);
                /* We do not add the copy back into the LRU. This means that for now this copy is not
                 * tracked via the LRU (despite being only used in read mode) and instead is dangling
                 * on other tasks. Thus, it will eventually need to be added back into the LRU when
                 * current task using it completes.
                 */
                goto find_another_data;
            }
            /* It's also possible that the ref_count of that element is bigger than 1
             * In that case, it's because some task completion did not execute yet, and
             * we need to keep it in the list until it reaches 1.
             */
            if( lru_gpu_elem->super.super.obj_reference_count > 1 ) {
                /* It's also possible (although unlikely) that we livelock here:
                 * if gpu_mem_lru has *only* elements with readers == 0 but
                 * ref_count > 1, then we might pop/push forever. We save the
                 * earliest element found and if we see it again it means we
                 * run over the entire list without finding a suitable replacement.
                 * We need to make progress on something else. This remains safe for as long as the
                 * LRU is only modified by a single thread (in this case the current thread).
                 */
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s: Push back LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count, lru_gpu_elem->original);
                assert(0 != (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
                parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                goto find_another_data;
            }

            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
            if ( NULL != lru_gpu_elem->original ) {
                /* Let's check we're not trying to steal one of our own data */
                oldmaster = lru_gpu_elem->original;
                if( !parsec_atomic_trylock( &oldmaster->lock ) ) {
                    /* Even if we have the lock on oldmaster, any other thread
                     * might be adding/removing other elements to the list, so we
                     * need to protect all accesses to gpu_mem_lru with the locked version */
                    assert(0 != (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
                    parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                    gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                    goto find_another_data;
                }
                for( j = 0; j < i; j++ ) {
                    if( NULL == this_task->data[j].data_in ) continue;
                    if( this_task->data[j].data_in->original == oldmaster ) {
                        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                             "GPU[%d:%s]:%s: Drop LRU-retrieved GPU copy %p [ref_count %d] already in use by same task %d:%d original %p",
                                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                                             lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count, i, j, lru_gpu_elem->original);
                        /* If we are the owner of this tile we need to make sure it remains available for
                         * other tasks or we run in deadlock situations.
                         */
                        parsec_atomic_unlock( &oldmaster->lock );
                        goto find_another_data;
                    }
                }
                /* There is still one last thing to ensure: if another accelerator uses this copy as a source
                 * for a d2d transfer it will mark it by atomically increasing the readers. So, we need to
                 * avoid altering the copy while they are using it, by protecting the access to the readers
                 * with a cas.
                 */
                if( !parsec_atomic_cas_int32(&lru_gpu_elem->readers, 0, -PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL) ) {
                    assert(lru_gpu_elem->readers > 0);
                    /* we can't use this copy, push it back */
                    parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                    gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:%s: Push back LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p : Concurrent accesses",
                                         gpu_device->super.device_index, gpu_device->super.name, task_name,
                                         lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count,
                                         lru_gpu_elem->original);
                    parsec_atomic_unlock( &oldmaster->lock );
                    goto find_another_data;
                }
                copy_readers_update = PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL;
                /* Check if this copy is the last dangling reference to the oldmaster. This is safe to do as we own one of the data refcounts. */
                int do_unlock = oldmaster->super.obj_reference_count != 1;
                parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
                parsec_atomic_wmb();
                /* detach could have released the oldmaster if it only had a single refcount */
                if( do_unlock )
                    parsec_atomic_unlock( &oldmaster->lock );

                /* The data is not used, it's not one of ours, and it has been detached from the device
                 * so no other device can use it as a source for their copy : we can free it or reuse it */
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\ttask %s:%d repurpose copy %p [ref_count %d] to data %p instead of %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name, this_task->task_class->name, i, lru_gpu_elem,
                                     lru_gpu_elem->super.super.obj_reference_count, master, oldmaster);
            }
            else {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\ttask %s:%d found detached memory from previously destructed data %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name, this_task->task_class->name, i, lru_gpu_elem);
                oldmaster = NULL;
            }
            gpu_device->super.nb_evictions++;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
            /* Let's free this space, and try again to malloc some space */
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s] Release GPU copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 lru_gpu_elem, lru_gpu_elem->device_private, lru_gpu_elem->super.super.obj_reference_count,
                                 oldmaster);
#if defined(PARSEC_PROF_TRACE)
            if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
               (gpu_device->exec_stream[0]->prof_event_track_enable ||
                gpu_device->exec_stream[1]->prof_event_track_enable)) {
                parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                             parsec_gpu_free_memory_key, (int64_t)lru_gpu_elem->device_private,
                                             gpu_device->super.device_index,
                                             NULL, PARSEC_PROFILING_EVENT_COUNTER);
                parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                             parsec_gpu_use_memory_key_end,
                                             (uint64_t)lru_gpu_elem->device_private,
                                             gpu_device->super.device_index, NULL, 0);
            }
#endif
            assert( 0 != (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
            zone_free( gpu_device->memory, (void*)(lru_gpu_elem->device_private) );
            lru_gpu_elem->device_private = NULL;
            data_avail_epoch++;
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Release LRU-retrieved GPU copy %p [ref_count %d: must be 1]",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            PARSEC_OBJ_RELEASE(lru_gpu_elem);
            assert( NULL == lru_gpu_elem );
            goto malloc_data;
        }
        PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                             "GPU[%d:%s] Succeeded Allocating GPU copy %p at real address %p [ref_count %d] for data %p",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->device_private, gpu_elem->super.super.obj_reference_count, master);
#if defined(PARSEC_PROF_TRACE)
        if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
                        (gpu_device->exec_stream[0]->prof_event_track_enable ||
                         gpu_device->exec_stream[1]->prof_event_track_enable)) {
            parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                         parsec_gpu_allocate_memory_key, (int64_t)gpu_elem->device_private,
                                         gpu_device->super.device_index,
                                         &gpu_task->flow_nb_elts[i], PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
        }
#endif
#else
        gpu_elem = lru_gpu_elem;
        /* The readers must be manipulated via atomic operations to avoid race conditions
         * with threads that would use them as candidate for updating their own copies.
         */
        if (copy_readers_update != 0) {
            parsec_atomic_fetch_add_int32(&gpu_elem->readers, copy_readers_update);
        }
#endif

        /* Do not push it back into the LRU for now to prevent others from discovering
         * this copy and trying to acquire it. If we fail to find all the copies we need
         * we will push it back in the release_temp_and_return, otherwise they will become
         * available once properly updated.
         */
        gpu_elem->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
        gpu_elem->version = UINT_MAX;  /* scrap value for now */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]: GPU copy %p [ref_count %d] gets created with version 0",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count);
        parsec_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;
        /* set the new datacopy type to the correct one */
        this_task->data[i].data_out->dtt = this_task->data[i].data_in->dtt;
        temp_loc[i] = gpu_elem;
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Retain and insert GPU copy %p [ref_count %d] in LRU",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count);
        assert(0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        parsec_atomic_unlock(&master->lock);
    }
    if( data_avail_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* Default stage_in function to transfer data to the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_in(parsec_gpu_task_t        *gtask,
                            uint32_t                  flow_mask,
                            parsec_gpu_exec_stream_t *gpu_stream)
{
    int ret;
    parsec_data_copy_t * source;
    parsec_data_copy_t * dest;
    parsec_device_gpu_module_t *src_dev;
    parsec_device_gpu_module_t *dst_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;

    for(int i = 0; i < task->task_class->nb_flows; i++) {
        if( !(flow_mask & (1U << i)) ) continue;
        source = gtask->sources[i];
        dest = task->data[i].data_out;
        src_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(source->device_index);
        dst_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(dest->device_index);

        if(src_dev->super.type == dst_dev->super.type) {
            assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
            dir = parsec_device_gpu_transfer_direction_d2d;
        } else {
            dir = parsec_device_gpu_transfer_direction_h2d;
        }

        count = (source->original->nb_elts <= dest->original->nb_elts) ?
            source->original->nb_elts : dest->original->nb_elts;
        ret = dst_dev->memcpy_async( dst_dev, gpu_stream,
                                     dest->device_private,
                                     source->device_private,
                                     count,
                                     dir );
        if(PARSEC_SUCCESS != ret)
            return PARSEC_HOOK_RETURN_ERROR;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* Default stage_out function to transfer data from the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_out(parsec_gpu_task_t        *gtask,
                             uint32_t                  flow_mask,
                             parsec_gpu_exec_stream_t *gpu_stream)
{
    int ret;
    parsec_data_copy_t * source;
    parsec_data_copy_t * dest;
    parsec_device_gpu_module_t *dst_dev, *src_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;
    int i;
    for(i = 0; i < task->task_class->nb_flows; i++){
        if(flow_mask & (1U << i)){
            source = task->data[i].data_out;
            dest = source->original->device_copies[0];
            assert(NULL != dest);
            dst_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(dest->device_index);
            src_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(source->device_index);

            count = (source->original->nb_elts <= dest->original->nb_elts) ? source->original->nb_elts :
                        dest->original->nb_elts;
            if( src_dev->super.type == dst_dev->super.type ) {
                assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
                dir = parsec_device_gpu_transfer_direction_d2d;
            } else {
                dir = parsec_device_gpu_transfer_direction_d2h;
            }
            ret = src_dev->memcpy_async( src_dev, gpu_stream,
                                         dest->device_private,
                                         source->device_private,
                                         count,
                                         dir );
            if(PARSEC_SUCCESS != ret) {
                return PARSEC_HOOK_RETURN_ERROR;
            }
        }
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns hook special return codes or a positive number:
 *    HOOK_DONE: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   HOOK_ERROR: A copy cannot be issued due to GPU.
 */
static inline int
parsec_device_data_stage_in( parsec_device_gpu_module_t* gpu_device,
                             const parsec_flow_t *flow,
                             parsec_data_pair_t* task_data,
                             parsec_gpu_task_t *gpu_task,
                             parsec_gpu_exec_stream_t *gpu_stream )
{
    int32_t type = flow->flow_flags;
    parsec_data_copy_t *candidate = task_data->data_in;  /* best candidate for now */
    parsec_data_t* original = candidate->original;
    parsec_gpu_data_copy_t* gpu_elem = task_data->data_out;
    size_t nb_elts = gpu_task->flow_nb_elts[flow->flow_index];
    int transfer_from = -1;

    if( gpu_task->task_type == PARSEC_GPU_TASK_TYPE_PREFETCH ) {
        PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream,
                             "GPU[%d:%s]: Prefetch task %p is staging in",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_task);
    }

    parsec_atomic_lock( &original->lock );

    gpu_task->sources[flow->flow_index] = candidate;  /* default source for the transfer */
    /**
     * If the data will be accessed in write mode, remove it from any GPU data management
     * lists until the task is completed.
     */
    if( PARSEC_FLOW_ACCESS_WRITE & type ) {
        if (gpu_elem->readers > 0 ) {
            if( !((1 == gpu_elem->readers) && (PARSEC_FLOW_ACCESS_READ & type)) ) {
                parsec_warning("GPU[%d:%s]:\tWrite access to data copy %p [ref_count %d] with existing readers [%d]\n"
                               "\tPossible anti-dependency, or concurrent accesses: please prevent that with CTL dependencies\n",
                               gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->readers);
            }
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tDetach writable GPU copy %p [ref_count %d] from any lists",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.super.obj_reference_count);
        /* make sure the element is not in any tracking lists */
        parsec_list_item_ring_chop((parsec_list_item_t*)gpu_elem);
        PARSEC_LIST_ITEM_SINGLETON(gpu_elem);
    }

    transfer_from = parsec_data_start_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

    /* If data is from NEW (it doesn't have a source_repo_entry and is not a direct data collection reference),
     * and nobody has touched it yet, then we don't need to pull it in, we have created it already, that's enough. */
    /*
     * TODO: this test is not correct for anything but PTG
     */
    if( (NULL == task_data->source_repo_entry) &&
        (NULL == task_data->data_in->original->dc) &&
        (0 == task_data->data_in->version) )
        transfer_from = -1;

    /* Update the transferred required_data_in size */
    gpu_device->super.required_data_in += original->nb_elts;

    if( -1 == transfer_from ) {  /* Do not need to be transferred */
        gpu_elem->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;

        parsec_data_end_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

        if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) ) {
            gpu_elem->version = candidate->version + 1;
        }

        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tNO Move for data copy %p v%d [ref_count %d, key %x]",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->version, gpu_elem->super.super.obj_reference_count, original->key);
        parsec_atomic_unlock( &original->lock );
        /* TODO: data keeps the same coherence flags as before */
        return PARSEC_HOOK_RETURN_DONE;
    }
    /* If it is already under transfer, don't schedule the transfer again.
     * This happens if the task refers twice (or more) to the same input flow */
    if( gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tMove data copy %p [ref_count %d, key %x] of %zu bytes: data copy is already under transfer, ignoring double request",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, nb_elts);
        parsec_atomic_unlock( &original->lock );
        return 1;  /* positive returns have special meaning and are used for optimizations */
    }

    /* Try to find an alternate source, to avoid always transferring from the host to the device.
     * Current limitations: only for read-only data used read-only on the hosting GPU. */
    parsec_device_gpu_module_t *candidate_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get( candidate->device_index );
    if( (PARSEC_FLOW_ACCESS_READ & type) && !(PARSEC_FLOW_ACCESS_WRITE & type) ) {
        int potential_alt_src = 0;
        PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tSelecting candidate data copy %p [ref_count %d] on data %p",
                             gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original);
        if( gpu_device->super.type == candidate_dev->super.type ) {
            if( gpu_device->peer_access_mask & (1 << candidate_dev->super.device_index) ) {
                /* We can directly do D2D, so let's skip the selection */
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tskipping candidate lookup: data_in copy %p on %s has PEER ACCESS",
                                     gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, candidate_dev->super.name);
                goto src_selected;
            }
        }

        /* If gpu_elem is not invalid, then it is already there and the right version,
         * and we're not going to transfer from another source, skip the selection */
        if( gpu_elem->coherency_state != PARSEC_DATA_COHERENCY_INVALID ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tskipping candidate lookup: VALID COPY for %p already on this GPU at %p",
                                 gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, gpu_elem);
            goto src_selected;
        }

        for(int t = 1; t < (int)parsec_nb_devices; t++) {
            parsec_device_gpu_module_t *target = (parsec_device_gpu_module_t*)parsec_mca_device_get(t);
            if( !(gpu_device->peer_access_mask & (1 << target->super.device_index)) ) {
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tskipping device: %s has NO PEER ACCESS",
                                     gpu_device->super.device_index, gpu_device->super.name, target->super.name);
                continue;
            }
            assert( PARSEC_DEV_IS_GPU(target->super.type) );

            candidate = original->device_copies[t];
            if( (NULL == candidate) || (candidate->version != task_data->data_in->version) ) {
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tcopy %p:%d cannot be a candidate VERSION MISMATCH with %p:%d",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     candidate, candidate?(int)candidate->version:-1, task_data->data_in, task_data->data_in->version);
                continue;
            }

            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tData copy %p [ref_count %d] on GPU device %d is a potential alternative source for data_in %p on data %p",
                                 gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.device_index, task_data->data_in, original);
            if(PARSEC_DATA_COHERENCY_INVALID == candidate->coherency_state) {
                /* We're already pulling this data on candidate...
                 * If there is another candidate that already has it, we'll use
                 * that one; otherwise, we'll fall back on the CPU version. */
                potential_alt_src = 1;
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tData copy %p [ref_count %d] on GPU device %d is invalid, continuing to look for alternatives",
                                     gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.device_index);
                continue;
            }
            /* We have a candidate for the d2d transfer. */
            int readers = parsec_atomic_fetch_inc_int32(&candidate->readers);
            if( readers >= 0 ) {
                parsec_atomic_rmb();
                /* Coordination protocol with the owner of the candidate. If the owner had repurposed the copy, by the
                 * time we succesfully increase the readers, the device copy will be associated with a different data.
                 */
                if( (candidate->original == original) && (candidate->version == task_data->data_in->version) ) {
                    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tData copy %p [ref_count %d] on PaRSEC device %s is the best candidate to do Device to Device copy, increasing its readers to %d",
                                         gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name, candidate->readers+1);
                    candidate_dev = target;
                    goto src_selected;
                }
            }
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tCandidate %p [ref_count %d] on PaRSEC device %s is being repurposed by owner device. Looking for another candidate",
                                 gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name);
            /* We are trying to use a candidate that is repurposed by the owner device. Let's find another one */
            parsec_atomic_fetch_add_int32(&candidate->readers, -1);
        }
        if( potential_alt_src ) {
            /* We found a potential alternative source, but it's not ready now,
             * we delay the scheduling of this task. */
            /** TODO: when considering RW accesses, don't forget to chop gpu_elem
             *        from its queue... */
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tThere is a potential alternative source for data_in %p [ref_count %d] in original %p to go in copy %p [ref_count %d], but it is not ready, falling back on CPU source",
                                 gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original, gpu_elem, gpu_elem->super.super.obj_reference_count);
            //return PARSEC_HOOK_RETURN_NEXT;
        }

        /* We fall back on the CPU copy */
        candidate = task_data->data_in;
    }

 src_selected:
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]:\t\tMove %s data copy %p [ref_count %d, key %x] of %zu bytes\t(src dev: %d, v:%d, ptr:%p, copy:%p [ref_count %d, under_transfer: %d, coherency_state: %d] / dst dev: %d, v:%d, ptr:%p)",
                         gpu_device->super.device_index, gpu_device->super.name,
                         PARSEC_DEV_IS_GPU(candidate_dev->super.type) ? "D2D": "H2D",
                         gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, nb_elts,
                         candidate_dev->super.device_index, candidate->version, (void*)candidate->device_private,
                         candidate, candidate->super.super.obj_reference_count, candidate->data_transfer_status, candidate->coherency_state,
                         gpu_device->super.device_index, gpu_elem->version, (void*)gpu_elem->device_private);

#if defined(PARSEC_PROF_TRACE)
    if( gpu_stream->prof_event_track_enable  ) {
        parsec_profile_data_collection_info_t info;

        if( NULL != original->dc ) {
            info.desc    = original->dc;
            info.data_id = original->key;
        } else {
            assert( PARSEC_GPU_TASK_TYPE_PREFETCH != gpu_task->task_type );
            info.desc    = (parsec_dc_t*)original;
            info.data_id = -1;
        }
        gpu_task->prof_key_end = -1;

        if( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type && (gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_PREFETCH) ) {
            gpu_task->prof_key_end = parsec_gpu_prefetch_key_end;
            gpu_task->prof_event_id = (int64_t)gpu_elem->device_private;
            gpu_task->prof_tp_id = gpu_device->super.device_index;
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   parsec_gpu_prefetch_key_start,
                                   gpu_task->prof_event_id,
                                   gpu_task->prof_tp_id,
                                   &info);
        }
        if(PARSEC_GPU_TASK_TYPE_PREFETCH != gpu_task->task_type && (gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_IN) ) {
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   parsec_gpu_movein_key_start,
                                   (int64_t)gpu_elem->device_private,
                                   gpu_device->super.device_index,
                                   &info);
        }
        if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) {
            parsec_device_gpu_memory_prof_info_t _info;
            _info.size = (uint64_t)nb_elts;
            _info.data_key = gpu_elem->original->key;
            _info.dc_id = (uint64_t)(gpu_elem->original->dc);
            parsec_profiling_trace_flags(gpu_stream->profiling,
                                         parsec_gpu_use_memory_key_start, (uint64_t)
                                         gpu_elem->device_private,
                                         gpu_device->super.device_index, &_info,
                                         PARSEC_PROFILING_EVENT_HAS_INFO);
        }
    }
#endif
    gpu_task->sources[flow->flow_index] = candidate;  /* save the candidate for release on transfer completion */
    /* Push data into the GPU from the source device */
    int rc = gpu_task->stage_in ? gpu_task->stage_in(gpu_task, (1U << flow->flow_index), gpu_stream): PARSEC_SUCCESS;
    if(PARSEC_SUCCESS != rc) {
        parsec_warning( "GPU[%d:%s]: gpu_task->stage_in to device rc=%d @%s:%d\n"
                        "\t<<%p on device %d:%s>> -> <<%p on device %d:%s>> [%zu, %s]",
                        gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                        candidate->device_private, candidate_dev->super.device_index, candidate_dev->super.name,
                        gpu_elem->device_private, gpu_device->super.device_index, gpu_device->super.name,
                        nb_elts, (candidate_dev->super.type != gpu_device->super.type)? "H2D": "D2D");
        parsec_atomic_unlock( &original->lock );
        assert(0);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    assert(candidate_dev->super.device_index < gpu_device->super.data_in_array_size);
    gpu_device->super.data_in_from_device[candidate_dev->super.device_index] += nb_elts;
    if( PARSEC_GPU_TASK_TYPE_KERNEL == gpu_task->task_type )
        gpu_device->super.nb_data_faults += nb_elts;

    /* We assign the version of the data preemptively (i.e. before the task is executing)
     * For read-only data, the GPU copy will get the same version as the source
     * For write-only or read-write data, we increment the version number.
     * The copy is still invalid & marked to be under transfer until the transfer_ownership is ended */
    assert((gpu_elem->version != candidate->version) || (gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_NOT_TRANSFER));
    if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) )
        gpu_elem->version = candidate->version + 1;
    else
        gpu_elem->version = candidate->version;
    gpu_elem->data_transfer_status = PARSEC_DATA_STATUS_UNDER_TRANSFER;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: GPU copy %p [ref_count %d] gets the version %d from copy %p version %d [ref_count %d]",
                         gpu_device->super.device_index, gpu_device->super.name,
                         gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->version, candidate, candidate->version, candidate->super.super.obj_reference_count);

    parsec_atomic_unlock( &original->lock );
    return 1;  /* positive returns have special meaning and are used for optimizations */
}

#if PARSEC_GPU_USE_PRIORITIES

static inline parsec_list_item_t* parsec_device_push_task_ordered( parsec_list_t* list,
                                                                   parsec_list_item_t* elem )
{
    parsec_list_push_sorted(list, elem, parsec_execution_context_priority_comparator);
    return elem;
}
#define PARSEC_PUSH_TASK parsec_device_push_task_ordered
#else
#define PARSEC_PUSH_TASK parsec_list_push_back
#endif

static parsec_flow_t parsec_device_d2d_complete_flow = {
    .name = "D2D FLOW",
    .flow_flags = PARSEC_FLOW_ACCESS_READ,
    .flow_index = 0,
};

static parsec_task_class_t parsec_device_d2d_complete_tc = {
    .name = "D2D TRANSFER COMPLETE",
    .flags = 0,
    .task_class_id = 0,
    .nb_flows = 1,
    .nb_parameters = 0,
    .nb_locals = 0,
    .dependencies_goal = 0,
    .params = { NULL, },
    .in = { &parsec_device_d2d_complete_flow, NULL },
    .out = { NULL, },
    .priority = NULL,
    .properties = NULL,
    .initial_data = NULL,
    .final_data = NULL,
    .data_affinity = NULL,
    .key_functions = NULL,
    .make_key = NULL,
    .get_datatype = NULL,
    .prepare_input = NULL,
    .incarnations = NULL,
    .prepare_output = NULL,
    .find_deps = NULL,
    .iterate_successors = NULL,
    .iterate_predecessors = NULL,
    .release_deps = NULL,
    .complete_execution = NULL,
    .new_task = NULL,
    .release_task = NULL,
    .fini = NULL
};

static void
parsec_device_send_transfercomplete_cmd_to_device(parsec_data_copy_t *copy,
                                                  parsec_device_module_t *current_dev,
                                                  parsec_device_module_t *dst_dev)
{
    parsec_gpu_task_t* gpu_task = NULL;
    gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
    gpu_task->task_type = PARSEC_GPU_TASK_TYPE_D2D_COMPLETE;
    gpu_task->ec = calloc(1, sizeof(parsec_task_t));
    PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
    gpu_task->ec->task_class = &parsec_device_d2d_complete_tc;
    gpu_task->flow[0] = &parsec_device_d2d_complete_flow;
    gpu_task->flow_nb_elts[0] = copy->original->nb_elts;
    gpu_task->stage_in  = parsec_default_gpu_stage_in;
    gpu_task->stage_out = parsec_default_gpu_stage_out;
    gpu_task->ec->data[0].data_in = copy;  /* We need to set not-null in data_in, so that the fake flow is
                                            * not ignored when poping the data from the fake task */
    gpu_task->ec->data[0].data_out = copy; /* We "free" data[i].data_out if its readers reaches 0 */
    gpu_task->ec->data[0].source_repo_entry = NULL;
    gpu_task->ec->data[0].source_repo = NULL;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_key_end = -1; /* D2D complete tasks are pure internal management, we do not trace them */
#endif
    (void)current_dev;
    PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                         "GPU[%d:%s]: data copy %p [ref_count %d] D2D transfer is complete, sending order to count it "
                         "to GPU Device %d:%s",
                         current_dev->device_index, current_dev->name, gpu_task->ec->data[0].data_out,
                         gpu_task->ec->data[0].data_out->super.super.obj_reference_count,
                         dst_dev->device_index, dst_dev->name);
    parsec_fifo_push( &(((parsec_device_gpu_module_t*)dst_dev)->pending), (parsec_list_item_t*)gpu_task );
}

static int
parsec_device_callback_complete_push(parsec_device_gpu_module_t   *gpu_device,
                                     parsec_gpu_task_t           **gpu_task,
                                     parsec_gpu_exec_stream_t     *gpu_stream)
{
    (void)gpu_stream;

    parsec_gpu_task_t *gtask = *gpu_task;
    parsec_task_t *task;
    int32_t i;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
#endif
    const parsec_flow_t        *flow;
    /**
     * Even though gpu event return success, the PUSH may not be
     * completed if no PUSH is required by this task and the PUSH is
     * actually done by another task, so we need to check if the data is
     * actually ready to use
     */
    assert(gpu_stream == gpu_device->exec_stream[0]);
    task = gtask->ec;
    PARSEC_DEBUG_VERBOSE(19, parsec_gpu_output_stream,
                         "GPU[%d:%s]: parsec_device_callback_complete_push, PUSH of %s",
                         gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task));

    for( i = 0; i < task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == task->data[i].data_in ) continue;
        /* We also don't push back non-parsec-owned copies */
        if(NULL != task->data[i].data_out &&
           0 == (task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)) continue;

        flow = gtask->flow[i];
        assert( flow );
        assert( flow->flow_index == i );
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if(PARSEC_DATA_STATUS_UNDER_TRANSFER == task->data[i].data_out->data_transfer_status ) {
            /* only the task who did the PUSH can modify the status */
            parsec_atomic_lock(&task->data[i].data_out->original->lock);
            task->data[i].data_out->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;
            parsec_data_end_transfer_ownership_to_copy(task->data[i].data_out->original,
                                                       gpu_device->super.device_index,
                                                       flow->flow_flags);
#if defined(PARSEC_PROF_TRACE)
            if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_IN) {
                PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                       parsec_gpu_movein_key_end,
                                       (int64_t)(int64_t)task->data[i].data_out->device_private,
                                       gpu_device->super.device_index,
                                       NULL);
            }
#endif
            parsec_atomic_unlock(&task->data[i].data_out->original->lock);
            parsec_data_copy_t* source = gtask->sources[i];
            parsec_device_gpu_module_t *src_device =
                    (parsec_device_gpu_module_t*)parsec_mca_device_get( source->device_index );
            if( PARSEC_DEV_IS_GPU(src_device->super.type) ) {
                int om;
                while(1) {
                    /* There are two ways out:
                     *   either we exit with om = 0, and then nobody was managing src_device,
                     *   and nobody can start managing src_device until we make it change from -1 to 0
                     *   (but anybody who has work to do will wait until that happens), or
                     *   we exit with om > 0, then there is a manager for that thread, and we have
                     *   increased mutex to warn the manager that there is another task for it to do.
                     */
                    om = src_device->mutex;
                    if(om == 0) {
                        /* Nobody at the door, let's try to lock the door */
                        if( parsec_atomic_cas_int32(&src_device->mutex, 0, -1) )
                            break;
                        continue;
                    }
                    if(om < 0 ) {
                        /* Damn, another thread is also trying to do an atomic operation on src_device,
                         * we give it some time and try again */
                        struct timespec delay;
                        delay.tv_nsec = 100;
                        delay.tv_sec = 0;
                        nanosleep(&delay, NULL);
                        continue;
                    }
                    /* There is a manager, let's try to reserve another task to do.
                     * If that fails, the manager may have leaved, try a gain. */
                    if( parsec_atomic_cas_int32(&src_device->mutex, om, om+1) )
                        break;
                }
                if( 0 == om ) {
                    int rc;
                    /* Nobody is at the door to handle that event on the source of that data...
                     * we do the command directly */
                    parsec_atomic_lock( &source->original->lock );
                    int readers = parsec_atomic_fetch_sub_int32(&source->readers, 1) - 1;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tExecuting D2D transfer complete for copy %p [ref_count %d] for "
                                         "device %s -- readers now %d",
                                         gpu_device->super.device_index, gpu_device->super.name, source,
                                         source->super.super.obj_reference_count, src_device->super.name,
                                         readers);
                    assert(readers >= 0);
                    if(0 == readers) {
                        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                             "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available",
                                             gpu_device->super.device_index, gpu_device->super.name, source,
                                             source->super.super.obj_reference_count);
                        parsec_list_item_ring_chop((parsec_list_item_t*)source);
                        PARSEC_LIST_ITEM_SINGLETON(source);
                        parsec_list_push_back(&src_device->gpu_mem_lru, (parsec_list_item_t*)source);
                        src_device->data_avail_epoch++;
                    }
                    parsec_atomic_unlock( &source->original->lock );
                    /* Notify any waiting thread that we're done messing with that device structure */
                    rc = parsec_atomic_cas_int32(&src_device->mutex, -1, 0); (void)rc;
                    assert(rc);
                } else {
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tSending D2D transfer complete command to %s for copy %p "
                                         "[ref_count %d] -- readers is still %d",
                                         gpu_device->super.device_index, gpu_device->super.name, src_device->super.name, source,
                                         source->super.super.obj_reference_count, source->readers);
                    parsec_device_send_transfercomplete_cmd_to_device(source,
                                                                      (parsec_device_module_t*)gpu_device,
                                                                      (parsec_device_module_t*)src_device);
                }
            }
            continue;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tparsec_device_callback_complete_push, PUSH of %s: task->data[%d].data_out = %p [ref_count = %d], and %s because transfer_status is %d",
                             gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task),
                             i, task->data[i].data_out, task->data[i].data_out->super.super.obj_reference_count,
                             (task->data[i].data_out->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER) ? "all is good" : "Assertion",
                             task->data[i].data_out->data_transfer_status);
        if( task->data[i].data_out->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {  /* data is not ready */
            /**
             * As long as we have only one stream to push the data on the GPU we should never
             * end up in this case. Remove previous assert if changed.
             */
            return PARSEC_HOOK_RETURN_ERROR;
        }
    }
    gtask->complete_stage = NULL;

    if( PARSEC_GPU_TASK_TYPE_PREFETCH == gtask->task_type ) {
        parsec_data_copy_t *gpu_copy = task->data[0].data_out;
#if defined(PARSEC_DEBUG_NOISIER)
        char tmp[MAX_TASK_STRLEN];
        assert(NULL != gpu_copy);
        if( NULL != gpu_copy->original->dc )
            gpu_copy->original->dc->key_to_string(gpu_copy->original->dc, gpu_copy->original->key, tmp, MAX_TASK_STRLEN);
        else
            snprintf(tmp, MAX_TASK_STRLEN, "unbound data");
#endif
        PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tPrefetch for data copy %p [ref_count %d] (%s) done. readers = %d, device_index = %d, version = %d, flags = %d, state = %d, data_transfer_status = %d",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count,
                             tmp,
                             gpu_copy->readers, gpu_copy->device_index, gpu_copy->version,
                             gpu_copy->flags, gpu_copy->coherency_state, gpu_copy->data_transfer_status);
        int readers = parsec_atomic_fetch_sub_int32(&gpu_copy->readers, 1);
        if( 0 == readers ) {
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tMake copy %p [ref_count %d] available after prefetch from gpu_task %p, ec %p",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, gtask, gtask->ec);
            parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
        (void)parsec_device_release_resources_prefetch_task(gpu_device, gpu_task);
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * This function tries to progress a stream, by picking up a ready task
 * and applying the progress function. The task to be progresses is
 * always the highest priority in the waiting queue, even when a task
 * has been specified as an input argument.
 * The progress function is either specified by the caller via the
 * upstream_progress_fct input argument or by the next task to be progresses
 * via the submit function associated with the task. In any case, this
 * function progresses a single task, which is then returned as the
 * out_task parameter.
 *
 * Beware: this function does not generate errors by itself, instead
 * it propagates upward the return code of the progress function.
 * However, by convention the error code follows the parsec_hook_return_e
 * enum.
 */
static inline int
parsec_device_progress_stream( parsec_device_gpu_module_t* gpu_device,
                               parsec_gpu_exec_stream_t* stream,
                               parsec_advance_task_function_t progress_fct,
                               parsec_gpu_task_t* task,
                               parsec_gpu_task_t** out_task )
{
    int rc;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
#endif

    /* We always handle the tasks in order. Thus if we got a new task, add it to the
     * local list (possibly by reordering the list). Also, as we can return a single
     * task first try to see if anything completed. */
    if( NULL != task ) {
        PARSEC_PUSH_TASK(stream->fifo_pending, (parsec_list_item_t*)task);
        task = NULL;
    }
    *out_task = NULL;

    if( NULL != stream->tasks[stream->end] ) {
        rc = gpu_device->event_query(gpu_device, stream, stream->end);
        if( 1 == rc ) {
            /* Save the task for the next step */
            task = *out_task = stream->tasks[stream->end];
            PARSEC_DEBUG_VERBOSE(19, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: Completed %s on stream %s{%p}",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task->ec),
                                 stream->name, (void*)stream);
            stream->tasks[stream->end]    = NULL;
            stream->end = (stream->end + 1) % stream->max_events;

#if defined(PARSEC_PROF_TRACE)
            if( stream->prof_event_track_enable ) {
                if( task->prof_key_end != -1 ) {
                    PARSEC_PROFILING_TRACE(stream->profiling, task->prof_key_end, task->prof_event_id, task->prof_tp_id, NULL);
                }
            }
#endif /* (PARSEC_PROF_TRACE) */
            if( PARSEC_HOOK_RETURN_AGAIN == task->last_status ) {
                /* we can now reschedule the task on the same execution stream */
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: GPU task %p[%p] is ready to be rescheduled on the same GPU device and same stream",
                                     gpu_device->super.device_index, gpu_device->super.name, (void*)task, (void*)task->ec);
                *out_task = NULL;
                goto schedule_task;
            }
            assert( PARSEC_HOOK_RETURN_ASYNC != task->last_status );
            rc = PARSEC_HOOK_RETURN_DONE;
            if (task->complete_stage)
                rc = task->complete_stage(gpu_device, out_task, stream);
            /* the task can be withdrawn by the system */
            return rc;
        }
        if( 0 != rc ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }
    }

 grab_a_task:
    if( NULL == stream->tasks[stream->start] ) {  /* there is room on the stream */
        task = (parsec_gpu_task_t*)parsec_list_pop_front(stream->fifo_pending);  /* get the best task */
    }
    if( NULL == task ) {  /* No tasks, we're done */
        return PARSEC_HOOK_RETURN_DONE;
    }
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)task);

    assert( NULL == stream->tasks[stream->start] );

  schedule_task:
    rc = progress_fct( gpu_device, task, stream );
    if( 0 > rc ) {
        if( PARSEC_HOOK_RETURN_AGAIN != rc ) {
           *out_task = task;
            return rc;
        }

        *out_task = NULL;
        /**
         * The task requested to be rescheduled but it might have added some kernels on the
         * stream and we need to wait for their completion. Thus, treat the task as usual,
         * create and event and upon completion of this event add the task back into the
         * execution stream pending list (to be executed again).
         */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]: GPU task %p has returned with ASYNC or AGAIN. Once the event "
                             "trigger the task will be handled accordingly",
                             gpu_device->super.device_index, gpu_device->super.name, (void*)task);
    }
    task->last_status = rc;
    /**
     * Do not skip the gpu event generation. The problem is that some of the inputs
     * might be in the pipe of being transferred to the GPU. If we activate this task
     * too early, it might get executed before the data is available on the GPU.
     * Obviously, this lead to incorrect results.
     */
    rc = gpu_device->event_record(gpu_device, stream, stream->start);
    assert(PARSEC_SUCCESS == rc);
    stream->tasks[stream->start] = task;
    stream->start = (stream->start + 1) % stream->max_events;
    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Submitted %s(task %p) on stream %s{%p}",
                         gpu_device->super.device_index, gpu_device->super.name,
                         task->ec->task_class->name, (void*)task->ec,
                         stream->name, (void*)stream);

    task = NULL;
    goto grab_a_task;
}

/**
 *  @brief This function prepare memory on the target device for all the inputs and output
 *  of the task, and then initiate the necessary copies from the best location of the input
 *  data. The best location is defined as any other accelerator that has the same version
 *  of the data (taking advantage of faster accelerator-to-accelerator connectors, such as
 *  NVLink), or from the CPU memory if no other candidate is found.
 *
 *  @returns
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
static int
parsec_device_kernel_push( parsec_device_gpu_module_t      *gpu_device,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t *flow;
    int i, ret = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    /* if no changes were made to the available memory dont waste time */
    if( gpu_task->last_data_check_epoch == gpu_device->data_avail_epoch )
        return PARSEC_HOOK_RETURN_AGAIN;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Try to Push %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task) );

    if( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type ) {
        if( NULL == gpu_task->ec->data[0].data_in->original ) {
            /* The PREFETCH order comes after the copy was detached and released, ignore it */
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s has been released already, destroying prefetch request",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
            parsec_device_release_resources_prefetch_task(gpu_device, &gpu_task);
            return PARSEC_HOOK_RETURN_ASYNC;
        }
        if( NULL != gpu_task->ec->data[0].data_in->original->device_copies[gpu_device->super.device_index] &&
            gpu_task->ec->data[0].data_in->original->owner_device == gpu_device->super.device_index ) {
            /* There is already a copy of this data in the GPU */
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s data_copy at index %d is %p, destroying prefetch request",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task),
                                 gpu_device->super.device_index,
                                 gpu_task->ec->data[0].data_in->original->device_copies[gpu_device->super.device_index]);
            parsec_device_release_resources_prefetch_task(gpu_device, &gpu_task);
            return PARSEC_HOOK_RETURN_ASYNC;
        }
    }

    /* Do we have enough available memory on the GPU to hold the input and output data ? */
    ret = parsec_device_data_reserve_space( gpu_device, gpu_task );
    if( ret < 0 ) {
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch;
        return ret;
    }

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {

        flow = gpu_task->flow[i];
        /* Skip CTL flows */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* If there is already a GPU data copy (set by reserve_device_space), and this copy
         * is not parsec-owned, don't stage in */
        if( NULL != this_task->data[i].data_out &&
            (0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) ) continue;

        assert( NULL != parsec_data_copy_get_ptr(this_task->data[i].data_in) );

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tIN  Data of %s <%x> on GPU",
                             gpu_device->super.device_index, gpu_device->super.name, flow->name,
                             this_task->data[i].data_out->original->key);
        ret = parsec_device_data_stage_in( gpu_device, flow,
                                           &(this_task->data[i]), gpu_task, gpu_stream );
        if( ret < 0 ) {
            gpu_task->last_status = ret;
            return ret;
        }
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Push task %s DONE",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
    gpu_task->complete_stage = parsec_device_callback_complete_push;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_key_end = -1; /* We do not log that event as the completion of this task */
#endif
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Prepare a task for execution on the GPU. Basically, does some upstream initialization,
 * setup the profiling information and then calls directly into the task submission body. Upon
 * return from the body handle the state machine of the task, taking care of the special cases
 * such as AGAIN and ASYNC.
 * @returns An error if anything unexpected came out of the task submission body, otherwise
 */
static int
parsec_device_kernel_exec( parsec_device_gpu_module_t      *gpu_device,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_advance_task_function_t progress_fct = gpu_task->submit;
    parsec_task_t* this_task = gpu_task->ec;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]:\tEnqueue on device %s stream %s"     ,
                         gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(tmp, MAX_TASK_STRLEN,
                         (parsec_task_t *) this_task), gpu_stream->name);
#endif /* defined(PARSEC_DEBUG_NOISIER) */
#if defined(PARSEC_PROF_TRACE)
    if (gpu_stream->prof_event_track_enable &&
        (0 == gpu_task->prof_key_end)) {
        parsec_task_class_t* tc = (parsec_task_class_t*)this_task->task_class;
        PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,
                               PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                          tc->task_class_id),
                               (parsec_task_t *) this_task, 1);
        gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool, tc->task_class_id);
        gpu_task->prof_event_id = tc->key_functions->key_hash(
                                        tc->make_key(this_task->taskpool, ((parsec_task_t *) this_task)->locals), NULL);
        gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;
    }
#endif /* PARSEC_PROF_TRACE */

#if defined(PARSEC_DEBUG_PARANOID)
    const parsec_flow_t *flow;
    for( int i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if( 0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
        assert(this_task->data[i].data_out->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER);
    }
#endif /* defined(PARSEC_DEBUG_PARANOID) */

    (void)this_task;
    return progress_fct( gpu_device, gpu_task, gpu_stream );
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: HOOK_ERROR if any error occurred.
 *           positive: the number of data to be moved.
 */
static int
parsec_device_kernel_pop( parsec_device_gpu_module_t   *gpu_device,
                          parsec_gpu_task_t            *gpu_task,
                          parsec_gpu_exec_stream_t     *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy;
    parsec_data_t              *original;
    size_t                      nb_elts;
    const parsec_flow_t        *flow;
    int return_code = 0, rc, how_many = 0, i, update_data_epoch = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    if (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < this_task->locals[0].value; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            /* If the gpu copy is not owned by parsec, we don't manage it at all */
            if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
            original = gpu_copy->original;
            rc = gpu_task->stage_out? gpu_task->stage_out(gpu_task, (1U << i), gpu_stream): PARSEC_SUCCESS;
            if(PARSEC_SUCCESS != rc) {
                parsec_warning( "GPU[%d:%s]: gpu_task->stage_out from device rc=%d @%s:%d\n"
                                "\tdata %s <<%p>> -> <<%p>>\n",
                                gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                                this_task->task_class->out[i]->name,
                                gpu_copy->device_private, original->device_copies[0]->device_private);
                return_code = PARSEC_HOOK_RETURN_DISABLE;
                goto release_and_return_error;
            }
        }
        return PARSEC_HOOK_RETURN_DONE;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                        "GPU[%d:%s]: Try to Pop %s",
                        gpu_device->super.device_index, gpu_device->super.name,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* We need to manage all data that has been used as input, even if they were read only */

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if( PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags) )  continue;  /* control flow */

        gpu_copy = this_task->data[i].data_out;

        /* If the gpu copy is not owned by parsec, we don't manage it at all */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;

        original = gpu_copy->original;
        nb_elts = gpu_task->flow_nb_elts[i];

        assert( this_task->data[i].data_in == NULL || original == this_task->data[i].data_in->original );

        if( (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_D2D_COMPLETE) && !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[i].data_out = original->device_copies[0];
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: pop %s swap %d GPU read-only data_out %p [ref_count %d] with the corresponding CPU copy %p [ref_count %d] original %p",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), i,
                                     gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     this_task->data[i].data_out, this_task->data[i].data_out->super.super.obj_reference_count,
                                     original);
        }
        parsec_atomic_lock(&original->lock);
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_READ ) {
            int current_readers = parsec_atomic_fetch_sub_int32(&gpu_copy->readers, 1) - 1;
            if( current_readers < 0 ) {
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: While trying to Pop %s, gpu_copy %p [ref_count %d] on flow %d with original %p had a negative number of readers (%d)",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task),
                                     gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     i, original, current_readers);
            }
            assert(current_readers >= 0);
            if( (0 == current_readers) && !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
                 PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available on flow %s",
                                     gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);
                parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
                PARSEC_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
                update_data_epoch = 1;
                parsec_atomic_unlock(&original->lock);
                continue;  /* done with this element, go for the next one */
            }
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tread copy %p [ref_count %d] on flow %s has readers (%i)",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name, current_readers);
        }
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == parsec_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                "GPU[%d:%s]:\tOUT Data copy %p [ref_count %d] for flow %s",
                                gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += nb_elts;
            assert( ((parsec_list_item_t*)gpu_copy)->list_next == (parsec_list_item_t*)gpu_copy );
            assert( ((parsec_list_item_t*)gpu_copy)->list_prev == (parsec_list_item_t*)gpu_copy );

            assert( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
            if( gpu_task->pushout & (1 << i) ) {
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                    "GPU[%d:%s]:\tMove D2H data <%s:%x> copy %p [ref_count %d] -- D:%p -> H:%p requested",
                                    gpu_device->super.device_index, gpu_device->super.name, flow->name, original->key, gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     (void*)gpu_copy->device_private, original->device_copies[0]->device_private);
#if defined(PARSEC_PROF_TRACE)
                if( gpu_stream->prof_event_track_enable ) {
                    if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_OUT) {
                        parsec_profile_data_collection_info_t info;
                        if( NULL != original->dc ) {
                            info.desc    = original->dc;
                            info.data_id = original->key;
                        } else {
                            info.desc    = (parsec_dc_t*)original;
                            info.data_id = -1;
                        }
                        gpu_task->prof_key_end = parsec_gpu_moveout_key_end;
                        gpu_task->prof_tp_id   = this_task->taskpool->taskpool_id;
                        gpu_task->prof_event_id = this_task->task_class->key_functions->key_hash(this_task->task_class->make_key(this_task->taskpool, this_task->locals), NULL);
                        PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                               parsec_gpu_moveout_key_start,
                                               gpu_task->prof_event_id,
                                               gpu_task->prof_tp_id,
                                               &info);
                    } else {
                        gpu_task->prof_key_end = -1;
                    }
                }
#endif
                /* Move the data back into main memory */
                rc = gpu_task->stage_out? gpu_task->stage_out(gpu_task, (1U << flow->flow_index), gpu_stream): PARSEC_SUCCESS;
                if(PARSEC_SUCCESS != rc) {
                    parsec_warning( "GPU[%d:%s]: gpu_task->stage_out from device rc=%d @%s:%d\n"
                                    "\tdata %s <<%p>> -> <<%p>>\n",
                                    gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                                    this_task->task_class->out[i]->name,
                                    gpu_copy->device_private, original->device_copies[0]->device_private);
                    return_code = PARSEC_HOOK_RETURN_DISABLE;
                    parsec_atomic_unlock(&original->lock);
                    goto release_and_return_error;
                }
                gpu_device->super.data_out_to_host += nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            } else {
                assert( 0 == gpu_copy->readers );
            }
        }
        parsec_atomic_unlock(&original->lock);
    }

  release_and_return_error:
    if( update_data_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Pop %s DONE (return %d data epoch %"PRIu64")",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), return_code, gpu_device->data_avail_epoch );

    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
static int
parsec_device_kernel_epilog( parsec_device_gpu_module_t *gpu_device,
                             parsec_gpu_task_t          *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Epilog of %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;


        if( !(gpu_task->flow[i]->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        /* If it is a copy managed by the user, don't bother either */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;

        /**
         * There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         *
         * WARNING: For now we always forward the cpu_copy to the next task, to
         * do that, we lie to the engine by updating the CPU copy to the same
         * status than the GPU copy without updating the data itself. Thus, the
         * cpu copy is really invalid. this is related to Issue #88, and the
         * fact that:
         *      - we don't forward the gpu copy as output
         *      - we always take a cpu copy as input, so it has to be in the
         *        same state as the GPU to prevent an extra data movement.
         */
        assert( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;

        cpu_copy->version = gpu_copy->version;
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]: CPU copy %p [ref_count %d] gets the same version %d as GPU copy %p [ref_count %d]",
                             gpu_device->super.device_index, gpu_device->super.name,
                             cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->version, gpu_copy, gpu_copy->super.super.obj_reference_count);

        /**
         * Let's lie to the engine by reporting that working version of this
         * data is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;

        assert( 0 <= gpu_copy->readers );

        if( gpu_task->pushout & (1 << i) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU copy %p [ref_count %d] moved to the read LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        } else {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU copy %p [ref_count %d] moved to the owned LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_push_back(&gpu_device->gpu_mem_owned_lru, (parsec_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

/** @brief Release the GPU copies of the data used in WRITE mode.
 *
 * @details This function can be used when the GPU task didn't run
 *          to completion on the device (either due to an error, or
 *          simply because the body requested a reexecution on a
 *          different location). It releases the GPU copies of the
 *          output data, allowing them to be reused by the runtime.
 *          This function has the drawback of kicking in too late,
 *          after all data transfers have been completed toward the
 *          device.
 *
 * @param [IN] gpu_device, the GPU device the the task has been
 *             supposed to execute.
 * @param [IN] gpu_task, the task that has been cancelled, and which
 *             needs it's data returned to the runtime.
 * @return Currently only success.
 */
static int
parsec_device_kernel_cleanout( parsec_device_gpu_module_t *gpu_device,
                               parsec_gpu_task_t          *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i, data_avail_epoch = 0;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Cleanup of %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        if( !(gpu_task->flow[i]->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        parsec_atomic_lock(&original->lock);
        assert(gpu_copy->super.super.obj_reference_count > 1);
        /* Issue #134 */
        parsec_data_copy_detach(original, gpu_copy, gpu_device->super.device_index);
        gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
        cpu_copy = original->device_copies[0];

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;
        if( 0 != (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) {
            parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
        parsec_atomic_unlock(&original->lock);
        data_avail_epoch++;
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU copy %p [ref_count %d] moved to the read LRU in %s\n",
                             gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
    }
    if( data_avail_epoch )  /* Update data availability epoch */
        gpu_device->data_avail_epoch++;
    return 0;
}

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on GPU events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_device_kernel_scheduler( parsec_device_module_t *module,
                                parsec_execution_stream_t *es,
                                void *_gpu_task )
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t *)module;
    int rc, exec_stream = 0;
    parsec_gpu_task_t *progress_task, *out_task_submit = NULL, *out_task_pop = NULL;
    parsec_gpu_task_t *gpu_task = (parsec_gpu_task_t*)_gpu_task;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    int pop_null = 0;

#if defined(PARSEC_PROF_TRACE)
    PARSEC_PROFILING_TRACE_FLAGS( es->es_profile,
                                  PARSEC_PROF_FUNC_KEY_END(gpu_task->ec->taskpool,
                                                           gpu_task->ec->task_class->task_class_id),
                                  gpu_task->ec->task_class->key_functions->key_hash(gpu_task->ec->task_class->make_key(gpu_task->ec->taskpool, gpu_task->ec->locals), NULL),
                                  gpu_task->ec->taskpool->taskpool_id, NULL,
                                  PARSEC_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(PARSEC_PROF_TRACE) */

    /* Check the GPU status -- three kinds of values for rc:
     *   - rc < 0: somebody is doing a short atomic operation while there is no manager,
     *             so wait.
     *   - rc == 0: there is no manager, and at the exit of the while, this thread
     *             made rc go from 0 to 1, so it is the new manager of the GPU and
     *             needs to deal with gpu_task
     *   - rc > 0: there is a manager, and at the exit of the while, this thread has
     *             committed new work that the manager will need to do, but the work is
     *             not in the queue yet.
     */
    while(1) {
        rc = gpu_device->mutex;
        struct timespec delay;
        if( rc >= 0 ) {
            if( parsec_atomic_cas_int32( &gpu_device->mutex, rc, rc+1 ) ) {
                break;
            }
        } else {
            delay.tv_nsec = 100;
            delay.tv_sec = 0;
            nanosleep(&delay, NULL);
        }
    }
    if( 0 < rc ) {
        parsec_fifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream, "GPU[%d:%s]: Entering GPU management",
                         gpu_device->super.device_index, gpu_device->super.name);

#if defined(PARSEC_PROF_TRACE)
    if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_start,
                                (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return PARSEC_HOOK_RETURN_DISABLE;

 check_in_deps:
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tUpload data (if any) for %s",
                             gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
    }
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[0],
                                        parsec_device_kernel_push,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {  /* In case of error progress_task is the task that raised it */
        if( PARSEC_HOOK_RETURN_ERROR == rc )
            goto disable_gpu;
        /* We are in the early stages, and if there no room on the GPU for a task we need to
         * delay all retries for the same task for a little while. Meanwhile, put the task back
         * trigger a device flush, and keep executing tasks that have their data on the device.
         */
        if( PARSEC_HOOK_RETURN_ASYNC == rc ) {
            gpu_task = progress_task;
            progress_task = NULL;
            goto remove_gpu_task;
        }
        assert(NULL == progress_task);

        /* TODO: check this */
        /* If we can extract data go for it, otherwise try to drain the pending tasks */
        gpu_task = parsec_gpu_create_w2r_task(gpu_device, es);
        if( NULL != gpu_task )
            goto get_data_out_of_device;
    }
    gpu_task = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->num_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tExecute %s", gpu_device->super.device_index, gpu_device->super.name,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    }
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[2+exec_stream],
                                        parsec_device_kernel_exec,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {
        if( (PARSEC_HOOK_RETURN_DISABLE == rc) || (PARSEC_HOOK_RETURN_ERROR == rc) )
            goto disable_gpu;
        if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
            /* Reschedule the task. As the chore_id has been modified,
               another incarnation of the task will be executed. */
            if( NULL != progress_task ) {
                assert(PARSEC_HOOK_RETURN_NEXT == rc);
                parsec_device_kernel_cleanout(gpu_device, progress_task);
                __parsec_reschedule(es, progress_task->ec);
                gpu_task = progress_task;
                progress_task = NULL;
                goto remove_gpu_task;
            }
            gpu_task = NULL;
            goto fetch_task_from_shared_queue;
        }
        gpu_task = progress_task;
        progress_task = NULL;
        goto remove_gpu_task;
    }
    gpu_task = progress_task;
    out_task_submit = progress_task;

 get_data_out_of_device:
    if( NULL != gpu_task ) {  /* This task has completed its execution */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tRetrieve data (if any) for %s", gpu_device->super.device_index, gpu_device->super.name,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    }
    /* Task is ready to move the data back to main memory */
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[1],
                                        parsec_device_kernel_pop,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {
        if( (PARSEC_HOOK_RETURN_ERROR == rc) || (PARSEC_HOOK_RETURN_DISABLE == rc) )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a successfully completed task. However, it is not gpu_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        gpu_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    gpu_task = progress_task;
    out_task_pop = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    if (NULL != gpu_device->super.sort_pending_list && out_task_submit == NULL && out_task_pop == NULL) {
        gpu_device->super.sort_pending_list(&gpu_device->super);
    }
    gpu_task = (parsec_gpu_task_t*)parsec_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != gpu_task ) {
        pop_null = 0;
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch - 1;  /* force at least one tour */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tGet from shared queue %s", gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
        if( PARSEC_GPU_TASK_TYPE_D2D_COMPLETE == gpu_task->task_type ) {
            goto get_data_out_of_device;
        }
    } else {
        pop_null++;
        if( pop_null % 1024 == 1023 ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,  "GPU[%d:%s]:\tStill waiting for %d tasks to execute, but poped NULL the last %d times I tried to pop something...",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_device->mutex, pop_null);
        }
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != gpu_task );
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tComplete %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    /* Everything went fine so far, the result is correct and back in the main memory */
    PARSEC_LIST_ITEM_SINGLETON(gpu_task);
    if (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER) {
        parsec_gpu_complete_w2r_task(gpu_device, gpu_task, es);
        gpu_task = progress_task;
        goto fetch_task_from_shared_queue;
    }
    if (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2D_COMPLETE) {
        free( gpu_task->ec );
        gpu_task->ec = NULL;
        goto remove_gpu_task;
    }
    parsec_device_kernel_epilog( gpu_device, gpu_task );
    __parsec_complete_execution( es, gpu_task->ec );
    gpu_device->super.executed_tasks++;
 remove_gpu_task:
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]: gpu_task %p freed",
                         gpu_device->super.device_index, gpu_device->super.name,
                         gpu_task);
    free( gpu_task );
    rc = parsec_atomic_fetch_dec_int32( &(gpu_device->mutex) );
    if( 1 == rc ) {  /* I was the last one */
#if defined(PARSEC_PROF_TRACE)
        if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
            PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_end,
                                    (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */
        PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream, "GPU[%d:%s]: Leaving GPU management",
                             gpu_device->super.device_index, gpu_device->super.name);
        /* inform the upper layer not to use the task argument, it has been long gone */
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    gpu_task = progress_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    parsec_warning("GPU[%d:%s]: Critical issue related to the GPU discovered. Giving up",
                   gpu_device->super.device_index, gpu_device->super.name);
    return PARSEC_HOOK_RETURN_DISABLE;
}
