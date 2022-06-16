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

#if defined(PARSEC_PROF_TRACE)
/* Accepted values are: PARSEC_PROFILE_GPU_TRACK_DATA_IN | PARSEC_PROFILE_GPU_TRACK_DATA_OUT |
 *                      PARSEC_PROFILE_GPU_TRACK_OWN | PARSEC_PROFILE_GPU_TRACK_EXEC |
 *                      PARSEC_PROFILE_GPU_TRACK_MEM_USE | PARSEC_PROFILE_GPU_TRACK_PREFETCH
 */
int parsec_gpu_trackable_events = PARSEC_PROFILE_GPU_TRACK_EXEC | PARSEC_PROFILE_GPU_TRACK_DATA_OUT
                                  | PARSEC_PROFILE_GPU_TRACK_DATA_IN | PARSEC_PROFILE_GPU_TRACK_OWN | PARSEC_PROFILE_GPU_TRACK_MEM_USE
                                  | PARSEC_PROFILE_GPU_TRACK_PREFETCH;
int parsec_gpu_movein_key_start;
int parsec_gpu_movein_key_end;
int parsec_gpu_moveout_key_start;
int parsec_gpu_moveout_key_end;
int parsec_gpu_own_GPU_key_start;
int parsec_gpu_own_GPU_key_end;
int parsec_gpu_allocate_memory_key;
int parsec_gpu_free_memory_key;
int parsec_gpu_use_memory_key_start;
int parsec_gpu_use_memory_key_end;
int parsec_gpu_prefetch_key_start;
int parsec_gpu_prefetch_key_end;
int parsec_device_gpu_one_profiling_stream_per_gpu_stream = 0;
static int parsec_gpu_profiling_initiated = 0;
#endif  /* defined(PROFILING) */
int parsec_gpu_output_stream = -1;
int parsec_gpu_verbosity;

static inline int
parsec_gpu_check_space_needed(parsec_device_gpu_module_t *gpu_device,
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
void parsec_gpu_init_profiling(void)
{
    if(parsec_gpu_profiling_initiated == 0) {
        parsec_profiling_add_dictionary_keyword("cuda", "fill:#66ff66",
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
                                                sizeof(int64_t), "size{int64_t}",
                                                &parsec_gpu_allocate_memory_key, &parsec_gpu_free_memory_key);
        parsec_profiling_add_dictionary_keyword("gpu_mem_use", "fill:#FF66FF",
                                                sizeof(parsec_device_gpu_memory_prof_info_t),
                                                PARSEC_DEVICE_GPU_MEMORY_PROF_INFO_CONVERTER,
                                                &parsec_gpu_use_memory_key_start, &parsec_gpu_use_memory_key_end);
        parsec_gpu_profiling_initiated = 1;
    }
}
#endif

void parsec_gpu_enable_debug(void)
{
    if(parsec_gpu_output_stream == -1) {
        parsec_gpu_output_stream = parsec_device_output;
        if( parsec_gpu_verbosity >= 0 ) {
            parsec_gpu_output_stream = parsec_output_open(NULL);
            parsec_output_set_verbosity(parsec_gpu_output_stream, parsec_gpu_verbosity);
        }
    }
}

int parsec_gpu_sort_pending_list(parsec_device_gpu_module_t *gpu_device)
{
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
        space_min = parsec_gpu_check_space_needed(gpu_device, (parsec_gpu_task_t*)min_p);
        for (j = i+1; j < NB_SORT; j++) {
            if ( q == &(sort_list->ghost_element) ) {
                break;
            }
            space_q = parsec_gpu_check_space_needed(gpu_device, (parsec_gpu_task_t*)q);
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

void* parsec_gpu_pop_workspace(parsec_device_gpu_module_t* gpu_device,
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
            PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                 "GPU[%s] Succeeded Allocating workspace %d (device_ptr %p)",
                                 gpu_device->super.name,
                                 i, gpu_stream->workspace->workspace[i]);
#if defined(PARSEC_PROF_TRACE)
            if((parsec_gpu_trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
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
    assert (gpu_stream->workspace->stack_head >= 0);
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return work;
}

int parsec_gpu_push_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_GPU_MAX_WORKSPACE);
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return 0;
}

int parsec_gpu_free_workspace(parsec_device_gpu_module_t * gpu_device)
{
    (void)gpu_device;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *gpu_stream = gpu_device->exec_stream[i];
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
#if defined(PARSEC_PROF_TRACE)
                if((parsec_gpu_trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
                   (gpu_device->exec_stream[0]->prof_event_track_enable ||
                    gpu_device->exec_stream[1]->prof_event_track_enable)) {
                    parsec_profiling_trace_flags(gpu_stream->profiling,
                                                 parsec_gpu_allocate_memory_key, (int64_t)
                                                 gpu_stream->workspace->workspace[i], gpu_device->super.device_index,
                                                 NULL, PARSEC_PROFILING_EVENT_COUNTER);
                }
#endif
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%s] Release workspace %d (device_ptr %p)",
                                     gpu_device->super.name,
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
char *parsec_gpu_describe_gpu_task( char *tmp, size_t len, parsec_gpu_task_t *gpu_task )
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

void dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    parsec_debug_verbose(0, parsec_gpu_output_stream,
                         "Dev: CUDA stream %d{%p} [events = %d, start = %d, end = %d, executed = %d]",
                         exec_stream->name, exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
                         exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        parsec_debug_verbose(0, parsec_gpu_output_stream,
                             "    %d: %s", i, parsec_task_snprintf(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void dump_GPU_state(parsec_device_gpu_module_t* gpu_device)
{
    int i;

    parsec_output(parsec_gpu_output_stream, "\n\n");
    parsec_output(parsec_gpu_output_stream, "Device %d:%d (%p) epoch\n", gpu_device->super.device_index,
                  gpu_device->super.device_index, gpu_device, gpu_device->data_avail_epoch);
    parsec_output(parsec_gpu_output_stream, "\tpeer mask %x executed tasks with %llu streams %d\n",
                  gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->num_exec_streams);
    parsec_output(parsec_gpu_output_stream, "\tstats transferred [in: %llu from host %llu from other device out: %llu] required [in: %llu out: %llu]\n",
                  (unsigned long long)gpu_device->super.transferred_data_in, (unsigned long long)gpu_device->super.d2d_transfer,
                  (unsigned long long)gpu_device->super.transferred_data_out,
                  (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        dump_exec_stream(gpu_device->exec_stream[i]);
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
