/*
 * Copyright (c) 2021-2026 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024-2026 NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      Stony Brook University. All rights reserved.
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
#include "parsec/remote_dep.h"
#include "parsec/scheduling.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct parsec_gpu_host_mirror_lookup_s {
    const parsec_flow_t            *producer_flow;
    parsec_dep_data_description_t   datatype;
    int                             found;
} parsec_gpu_host_mirror_lookup_t;

static parsec_ontask_iterate_t
parsec_device_gpu_retrieve_host_mirror_datatype(parsec_execution_stream_t *es,
                                                const parsec_task_t *newcontext,
                                                const parsec_task_t *oldcontext,
                                                const parsec_dep_t *dep,
                                                parsec_dep_data_description_t *data,
                                                int rank_src, int rank_dst, int vpid_dst,
                                                data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                                                void *param)
{
    parsec_gpu_host_mirror_lookup_t *lookup = (parsec_gpu_host_mirror_lookup_t*)param;
    uint32_t flow_mask;
    int rc;

    (void)data; (void)rank_src; (void)rank_dst; (void)vpid_dst;
    (void)successor_repo; (void)successor_repo_key;

    if( dep->belongs_to->flow_index != lookup->producer_flow->flow_index ) {
        return PARSEC_ITERATE_CONTINUE;
    }

    /* This mirrors the receive side of the communication engine: ask the
     * successor task which datatype it expects on the input flow that receives
     * this producer output, then use the returned remote descriptor as the
     * layout for the temporary host copy.
     */
    flow_mask = (1U << dep->flow->flow_index) | 0x80000000U;
    rc = newcontext->task_class->get_datatype(es, newcontext, oldcontext,
                                              &flow_mask, &lookup->datatype);
    if( PARSEC_HOOK_RETURN_NEXT != rc ) {
        return PARSEC_ITERATE_CONTINUE;
    }

    lookup->found = 1;
    return PARSEC_ITERATE_STOP;
}

static parsec_data_copy_t *
parsec_device_gpu_ensure_host_mirror(parsec_execution_stream_t *es,
                                     const parsec_task_t *producer,
                                     const parsec_flow_t *producer_flow,
                                     parsec_data_copy_t *gpu_copy)
{
    parsec_data_t *original = gpu_copy->original;
    parsec_data_copy_t *cpu_copy = original->device_copies[0];
    parsec_gpu_host_mirror_lookup_t lookup;
    uint32_t action_mask = 0;
    int rc;

    if( NULL != cpu_copy ) {
        return cpu_copy;
    }

    /* A self-contained NEW data item can start on the CPU, move to the GPU,
     * and then legitimately lose its CPU copy once the CPU-side reference
     * count reaches zero. If a later GPU task must push that data to a CPU
     * successor, or to a communication backend that cannot send directly from
     * device memory, the GPU copy is still the authoritative copy but there is
     * no host mirror left to receive the D2H transfer.
     *
     * Recreate the missing host copy as if this data had just arrived from a
     * remote dependency: find the successor flow that receives this producer
     * flow, call that task class get_datatype() hook, and allocate the CPU
     * buffer from the returned arena. This preserves the DSL-selected arena,
     * count, and datatype instead of inventing an anonymous host allocation.
     *
     * The caller holds original->lock while installing the mirror.
     */
    memset(&lookup, 0, sizeof(lookup));
    lookup.producer_flow = producer_flow;

    if( PARSEC_TASKPOOL_TYPE_DTD == producer->taskpool->taskpool_type ) {
        action_mask = 1U << producer_flow->flow_index;
    } else {
        for( int j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != producer_flow->dep_out[j]); j++ ) {
            action_mask |= 1U << producer_flow->dep_out[j]->dep_index;
        }
    }
    producer->task_class->iterate_successors(es, producer,
                                             action_mask,
                                             parsec_device_gpu_retrieve_host_mirror_datatype,
                                             &lookup);
    if( !lookup.found ) {
        return NULL;
    }

    cpu_copy = parsec_data_copy_new(original, 0, lookup.datatype.remote.dst_datatype,
                                    PARSEC_DATA_FLAG_PARSEC_OWNED |
                                    PARSEC_DATA_FLAG_PARSEC_MANAGED |
                                    PARSEC_DATA_FLAG_ARENA);
    if( NULL == cpu_copy ) {
        return NULL;
    }
    rc = parsec_arena_allocate_device_private(cpu_copy, lookup.datatype.remote.arena,
                                              lookup.datatype.remote.dst_count, 0,
                                              lookup.datatype.remote.dst_datatype);
    if( PARSEC_SUCCESS != rc ) {
        PARSEC_DATA_COPY_RELEASE(cpu_copy);
        return NULL;
    }
    cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
    cpu_copy->version = (0 < gpu_copy->version) ? gpu_copy->version - 1 : 0;
    return cpu_copy;
}

/* The return value of these functions is either a parsec_hook_return_t for <= 0 values,
 * or a positive number which represents that something has been scheduled on the gpu_stream
 */
typedef int(*parsec_gpu_step_function_t)(parsec_device_gpu_module_t  *gpu_device,
                                         parsec_execution_stream_t    *es,
                                         parsec_gpu_task_t           *gpu_task,
                                         parsec_gpu_exec_stream_t    *gpu_stream);
/**
 * This is a special function to release standard device tasks instead of calling
 * PARSEC_OBJ_RELEASE on them. If we use the PARSEC_OBJ_RELEASE route the memory pointed
 * by the task will be free. In some cases, we don't want that to happen, but we still
 * want to inform the DSL that the task has been done with.
 */
static void parsec_device_release_gpu_task(parsec_gpu_task_t *gpu_task)
{
    PARSEC_OBJ_RELEASE(gpu_task);
}

static void parsec_device_task_t_constructor(parsec_gpu_task_t *gpu_task)
{
    gpu_task->task_type = PARSEC_GPU_TASK_TYPE_INVALID; /* need to be set later */
    gpu_task->pushout = 0;
    gpu_task->last_status = 0;
    gpu_task->submit = NULL;
    gpu_task->complete_stage = NULL;
    gpu_task->stage_in = NULL;
    gpu_task->stage_out = NULL;
    gpu_task->release_device_task = NULL;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_event_id = 0;
    gpu_task->prof_stage_key_end = -1;
    gpu_task->prof_stage_object_id = 0;
    gpu_task->prof_exec_state = PARSEC_GPU_TASK_PROF_EXEC_PENDING;
#endif
    gpu_task->ec = NULL;
    gpu_task->last_data_check_epoch = UINT64_MAX; /* force at least one validation for the task */
    gpu_task->nb_flows = 0;
    gpu_task->flow_info = NULL;
    gpu_task->priority = -1; // priority is inherited from the task
    /* Default release mechanism, can be replaced by the DSL */
    gpu_task->release_device_task = parsec_device_release_gpu_task;
}
PARSEC_OBJ_CLASS_INSTANCE(parsec_gpu_task_t, parsec_list_item_t,
                          parsec_device_task_t_constructor, NULL);

static void parsec_device_dsl_task_t_constructor(parsec_gpu_dsl_task_t *gpu_dsl_task)
{
    memset(gpu_dsl_task->flows, 0, sizeof(gpu_dsl_task->flows));
    gpu_dsl_task->super.flow_info = gpu_dsl_task->flows;
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_gpu_dsl_task_t, parsec_gpu_task_t,
                          parsec_device_dsl_task_t_constructor, NULL);

typedef struct parsec_gpu_pushout_plan_s {
    parsec_gpu_task_t *gpu_task;
    uint32_t remaining_flows;
    int      send_from_gpu_denied;  /**< the engine cannot send from accelerator memory */
    int      peers_incomplete;      /**< some accelerator cannot read another one */
} parsec_gpu_pushout_plan_t;

void parsec_device_gpu_discover_peer_mesh(void)
{
    int incomplete = 0;

    /* Peer access is not required to be symmetric, and it is the consumer that
     * decides whether it can read another device, so every device is asked
     * about all the others. */
    for( uint32_t consumer = 0; (consumer < parsec_nb_devices) && !incomplete; consumer++ ) {
        parsec_device_module_t *reader = parsec_mca_device_get(consumer);
        if( (NULL == reader) || !PARSEC_DEV_IS_GPU(reader->type) ) continue;
        for( uint32_t producer = 0; producer < parsec_nb_devices; producer++ ) {
            parsec_device_module_t *holder = parsec_mca_device_get(producer);
            if( (NULL == holder) || (consumer == producer) ) continue;
            if( !PARSEC_DEV_IS_GPU(holder->type) ) continue;
            if( !(((parsec_device_gpu_module_t*)reader)->peer_access_mask & (1 << producer)) ) {
                incomplete = 1;
                break;
            }
        }
    }
    parsec_device_peer_mesh_incomplete = incomplete;
}

static parsec_ontask_iterate_t
parsec_gpu_pushout_successor(parsec_execution_stream_t *es,
                             const parsec_task_t *newcontext,
                             const parsec_task_t *oldcontext,
                             const parsec_dep_t *dep,
                             parsec_dep_data_description_t *data,
                             int rank_src, int rank_dst, int vpid_dst,
                             data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                             void *param)
{
    parsec_gpu_pushout_plan_t *plan = (parsec_gpu_pushout_plan_t*)param;
    uint32_t flow_bit;
    int needs_host_copy;

    (void)es; (void)newcontext; (void)oldcontext; (void)data; (void)vpid_dst;
    (void)successor_repo; (void)successor_repo_key;

    if( (NULL == dep) || (NULL == dep->belongs_to) ) {
        return PARSEC_ITERATE_CONTINUE;
    }
    flow_bit = (1U << dep->belongs_to->flow_index);
    if( 0 == (plan->remaining_flows & flow_bit) ) {
        return (0 == plan->remaining_flows) ? PARSEC_ITERATE_STOP : PARSEC_ITERATE_CONTINUE;
    }
    if( rank_src != rank_dst ) {
        /* The communication engine is not allowed to send from accelerator
         * memory, so a remote successor must observe the CPU copy.
         */
        needs_host_copy = plan->send_from_gpu_denied;
    } else {
        /* Where a local successor will be scheduled is not known here, so any
         * accelerator is a possible destination. When one of them cannot read
         * the memory this flow is being written to, the host copy is the only
         * medium that reaches it.
         */
        needs_host_copy = plan->peers_incomplete;
    }
    if( needs_host_copy ) {
        plan->gpu_task->pushout |= flow_bit;
        plan->remaining_flows &= ~flow_bit;
    }
    return (0 == plan->remaining_flows) ? PARSEC_ITERATE_STOP : PARSEC_ITERATE_CONTINUE;
}

static void
parsec_gpu_task_update_pushout(parsec_execution_stream_t *es,
                               parsec_gpu_task_t *gpu_task)
{
    const parsec_task_t *this_task = gpu_task->ec;
    const parsec_task_class_t *tc = this_task->task_class;
    parsec_gpu_pushout_plan_t plan;
    uint32_t action_mask = 0;
    int i, j;

#if defined(DISTRIBUTED)
    plan.send_from_gpu_denied = !(parsec_mpi_allow_gpu_memory_communications & PARSEC_RUNTIME_SEND_GPU_MEMORY);
#else
    plan.send_from_gpu_denied = 0;  /* there are no remote successors to begin with */
#endif  /* defined(DISTRIBUTED) */
    plan.peers_incomplete = parsec_device_peer_mesh_incomplete;

    /* The walk only has something to discover when a successor could be unable
     * to read this flow where it is being written: a remote one in the degraded
     * GPU-aware modes, mpi_gpu_aware=0 or 1, or a local one on a machine whose
     * accelerators are not all able to read each other.
     */
    if( (!plan.send_from_gpu_denied && !plan.peers_incomplete) ||
        (NULL == tc->iterate_successors) ) {
        return;
    }

    plan.gpu_task = gpu_task;
    plan.remaining_flows = 0;
    /* Keep pushout bits already set by upper layers, notably final writes back
     * to data collections. This pass only discovers task successors that cannot
     * consume an accelerator pointer.
     */
    for( i = 0; i < tc->nb_flows; i++ ) {
        const parsec_flow_t *flow = gpu_task->flow_info[i].flow;
        if( NULL == flow ) {
            continue;
        }
        if( !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            continue;
        }
        if( gpu_task->pushout & (1U << flow->flow_index) ) {
            continue;
        }
        plan.remaining_flows |= (1U << flow->flow_index);
        if( PARSEC_TASKPOOL_TYPE_DTD == this_task->taskpool->taskpool_type ) {
            action_mask |= (1U << flow->flow_index);
        } else {
            for( j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != flow->dep_out[j]); j++ ) {
                action_mask |= (1U << flow->dep_out[j]->dep_index);
            }
        }
    }

    if( (0 == action_mask) || (0 == plan.remaining_flows) ) {
        return;
    }
    tc->iterate_successors(es, this_task, action_mask,
                           parsec_gpu_pushout_successor, &plan);
}

static inline int
parsec_gpu_has_protected_cpu_mirror(parsec_data_t *data)
{
    parsec_data_copy_t *cpu_copy;

    if( (NULL == data) || (NULL != data->dc) ) {
        return 0;
    }
    cpu_copy = data->device_copies[0];
    return (NULL != cpu_copy) &&
           (0 != (cpu_copy->flags & PARSEC_DATA_FLAG_CPU_MIRROR_PROTECTED));
}

static inline int
parsec_device_check_space_needed(parsec_device_gpu_module_t *gpu_device,
                                 parsec_gpu_task_t *gpu_task)
{
    int space_needed = 0;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_data_t *original;
    parsec_data_copy_t *data;
    const parsec_flow_t *flow;

    /* would have been this_task->task_class->nb_flows for classical DSL */
    for( uint32_t i = 0; i < gpu_task->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow_info[i].flow;
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
                         "Dev: GPU stream %s{%p} [events = %d, start = %d, end = %d, executed = %d]",
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

    parsec_output(parsec_gpu_output_stream,
                  "\n\nDevice %s:%d (%p) epoch %zu\n"
                  "\tpeer mask %x executed tasks %llu streams %d\n"
                  "\tstats transferred [in: %llu from host %llu from other device out: %llu] required [in: %llu out: %llu]\n",
                  gpu_device->super.name, gpu_device->super.device_index, gpu_device, gpu_device->data_avail_epoch,
                  gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->num_exec_streams,
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
                                 parsec_data_copy_dump(gpu_copy);
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
                                 parsec_data_copy_dump(gpu_copy);
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
            parsec_gpu_task_t* gpu_task = (parsec_gpu_task_t*)PARSEC_OBJ_NEW(parsec_gpu_dsl_task_t);
            gpu_task->task_type = PARSEC_GPU_TASK_TYPE_PREFETCH;
            gpu_task->ec = calloc(1, sizeof(parsec_task_t));
            PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
            gpu_task->ec->task_class = &parsec_device_data_prefetch_tc;
            /* The pending heap compares gpu_task->priority, not gpu_task->ec->priority:
             * priority inheritance from ec normally happens in
             * parsec_device_kernel_scheduler(), which this direct-enqueue path bypasses.
             * Without this, gpu_task->priority keeps its constructor default of -1 and
             * this prefetch task is treated as lower priority than any normal task,
             * even ones with the same (default 0) ec->priority. */
            gpu_task->priority = gpu_task->ec->priority;
            gpu_task->nb_flows = 1;
            gpu_task->flow_info[0].flow = &parsec_device_data_prefetch_flow;
            gpu_task->flow_info[0].flow_span = data->device_copies[ data->owner_device ]->original->span;
            gpu_task->stage_in  = parsec_default_gpu_stage_in;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Retain data copy %p [ref_count %d]",
                                 data->device_copies[ data->owner_device ],
                                 data->device_copies[ data->owner_device ]->super.super.obj_reference_count);
            PARSEC_DATA_COPY_RETAIN(data->device_copies[ data->owner_device ]);
            gpu_task->ec->data[0].data_in = data->device_copies[ data->owner_device ];
            gpu_task->ec->data[0].data_out = NULL;
            gpu_task->ec->data[0].source_repo_entry = NULL;
            gpu_task->ec->data[0].source_repo = NULL;
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: data copy %p [ref_count %d] linked to prefetch gpu task %p on GPU copy %p",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_task->ec->data[0].data_in, gpu_task->ec->data[0].data_in->super.super.obj_reference_count,
                                 gpu_task, gpu_task->ec->data[0].data_out);
            parsec_lifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
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
            if( !(chores[j].type & device->type) )
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
        }
        alloc_size = number_blocks * eltsize;
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
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Insert GPU copy %p [ref_count %d] in LRU",
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
        if( NULL == gpu_copy->device_private ) {
            PARSEC_DEBUG_VERBOSE(35, parsec_gpu_output_stream,
                                 "GPU[%d:%s] copy %p is dangling without private data. This is OK.",
                                 gpu_device->super.device_index, gpu_device->super.name, (void*)gpu_copy);
            goto release_and_continue;
        }
        if( NULL == gpu_copy->original ) {
            PARSEC_DEBUG_VERBOSE(35, parsec_gpu_output_stream,
                                 "GPU[%d:%s] copy %p detached from a data but not yet reclaimed!",
                                 gpu_device->super.device_index, gpu_device->super.name, (void*)gpu_copy);
        }
        if (PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state &&
            NULL != original) {
            parsec_warning("GPU[%d:%s] still OWNS the master memory copy for data %d (%p) and it is discarding it!",
                           gpu_device->super.device_index, gpu_device->super.name, original->key, (void*)gpu_copy->device_private);
        }
        assert(0 != (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED));

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

      release_and_continue:
        /* At this point the data copies should have no attachment to a data_t. Thus,
         * before we get here (aka below parsec_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        PARSEC_DATA_COPY_RELEASE(gpu_copy); assert(NULL == gpu_copy);
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

int
parsec_device_get_copy( parsec_device_gpu_module_t* gpu_device, parsec_data_copy_t** dc )
{
#if defined(PARSEC_DEBUG_NOISIER)
    char task_name[] = "unknown";
#endif /* defined(PARSEC_DEBUG_NOISIER) */
    parsec_gpu_data_copy_t *gpu_mem_lru_cycling = NULL, *lru_gpu_elem;
    /* Get the head of the LRU, assuming it has no readers and mark it as used, using the same mechanism as
     * the GPU to GPU tranfers. Once the communication into this copy completes, the task will get into
     * the GPU queues, and the data will be reattributed accordingly to this GPU.
     */
  find_another_data:
    lru_gpu_elem = (parsec_gpu_data_copy_t*)parsec_list_pop_front(&gpu_device->gpu_mem_lru);
    if( NULL == lru_gpu_elem ) {
        /* nothing available on the GPU. Let the upper level know about this */
        *dc = NULL;
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
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
        goto find_another_data;
    }
    if( gpu_mem_lru_cycling == lru_gpu_elem ) {
        PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                             "GPU[%d:%s]: Cycle detected on allocating memory for %s",
                             gpu_device->super.device_index, gpu_device->super.name, task_name);
        *dc = NULL;  /* did our best but failed to find a data. Return and allocate it onto another device. */
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    /* detect cycles to have an opportunity to stop */
    gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */

    parsec_data_t* master = lru_gpu_elem->original;
    if (NULL == master ) {
        /* This copy has been detached by the CPU once it has been consumed (by the communication engine),
         * there is no device memory associated with it, we can safely release the CPU copy.
         */
        assert(1 == lru_gpu_elem->super.super.obj_reference_count);
        PARSEC_OBJ_RELEASE(lru_gpu_elem);
        goto find_another_data;
    }
    parsec_atomic_lock(&master->lock);
    if ( lru_gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {
        /* can't reuse, it is drained right now by another device */
        parsec_atomic_unlock(&master->lock);
        goto find_another_data;
    }
    int release_protected_cpu_mirror = parsec_gpu_has_protected_cpu_mirror(master);
    parsec_data_copy_detach(master, lru_gpu_elem, gpu_device->super.device_index);
    parsec_atomic_wmb();
    *dc = lru_gpu_elem;
    parsec_atomic_unlock(&master->lock);
    if( release_protected_cpu_mirror ) {
        parsec_data_release_self_contained_data(master);
    }
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
 *   PARSEC_HOOK_RETURN_NEXT:  The task needs to rescheduled
 */
static inline int
parsec_device_data_reserve_space( parsec_device_gpu_module_t* gpu_device,
                                  parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT] = { NULL }, *gpu_elem, *lru_gpu_elem;
    parsec_data_t* master, *oldmaster;
    const parsec_flow_t *flow;
    int data_avail_epoch = 0, copy_readers_update = 0;
    /* A writable flow may already have the right GPU copy selected while that
     * copy is still being filled by the input stream. In that case all data
     * dependencies are located, but the task must not execute until stream
     * ordering makes the transfer complete. Track this separately from memory
     * allocation failures so we can preserve existing bindings and only undo
     * temporary reservations made during this reserve_space() pass.
     */
    int pending_write_transfer = 0;
    parsec_gpu_data_copy_t *gpu_mem_lru_cycling = NULL;
    uint32_t i, j;

#if defined(PARSEC_DEBUG_NOISIER)
    char task_name[MAX_TASK_STRLEN];
    parsec_task_snprintf(task_name, MAX_TASK_STRLEN, this_task);
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    (void)copy_readers_update; // potentially unused

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for (i = 0; i < gpu_task->nb_flows /* not this_task->task_class->nb_flows */; i++) {
        flow = gpu_task->flow_info[i].flow;
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) {
            gpu_task->flow_info[i].flow_span = 0;  /* assume there is nothing to transfer to the GPU */
            continue;
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Investigating flow %s:%d",
                             gpu_device->super.device_index, gpu_device->super.name, task_name, flow->name, i);
        temp_loc[i] = NULL;
        if (this_task->data[i].data_in == NULL)
            continue;
        /* if the input data is already on this device there is nothing else to do */
        if( gpu_device->super.device_index == this_task->data[i].data_in->device_index ) {
            parsec_data_copy_t *input_copy = this_task->data[i].data_in;
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Flow %s:%i was already on the device %p%s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 flow->name, i, input_copy,
                                 input_copy->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ? " [in transfer]" : "");
            this_task->data[i].data_out = input_copy;
            if( (PARSEC_DATA_STATUS_UNDER_TRANSFER == input_copy->data_transfer_status) &&
                (0 != (PARSEC_FLOW_ACCESS_WRITE & flow->flow_flags)) ) {
                /* The selected input/output copy is already the device-local
                 * one. Keep it attached to the task, but postpone execution
                 * until the input stream records completion and the next
                 * reserve attempt can transfer write ownership safely.
                 */
                pending_write_transfer = 1;
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\tLocal write flow under transfer for copy %p; deferring task readiness",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     input_copy);
            }
            continue;
        }
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
                /* The data is under transfer, which is fine for RO data: the
                 * stage-in path reports the already-queued transfer as work so
                 * the input stream event is still recorded. For WRITE flows,
                 * the copy already satisfies the placement requirement but
                 * cannot be written yet. Keep data_out set to the selected GPU
                 * copy and retry later instead of forgetting the binding; the
                 * completion callback bumps data_avail_epoch so the task will
                 * re-enter reserve_space() after the stream-ordered transfer
                 * is visible.
                 */
                if (0 != (PARSEC_FLOW_ACCESS_WRITE & flow->flow_flags)) {
                    pending_write_transfer = 1;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:%s:\tWrite flow under transfer for copy %p; deferring task readiness",
                                         gpu_device->super.device_index, gpu_device->super.name, task_name,
                                         gpu_elem);
                }

            }
            parsec_atomic_unlock(&master->lock);
            continue;
        }

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Allocate GPU copy %p sz %zu [ref_count %d] for data %p",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             gpu_elem, gpu_task->flow_info[i].flow_span, gpu_elem->super.super.obj_reference_count, master);
        gpu_elem->flags = PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED;
    malloc_data:
        copy_readers_update = 0;
        assert(0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        gpu_elem->device_private = zone_malloc(gpu_device->memory, gpu_task->flow_info[i].flow_span);
        gpu_elem->arena_chunk = (parsec_arena_chunk_t *)gpu_device->memory;
        if( NULL == gpu_elem->device_private ) {
#endif

        find_another_data:
            temp_loc[i] = NULL;
            /* Look for a data_copy to free */
            lru_gpu_elem = (parsec_gpu_data_copy_t*)parsec_list_pop_front(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the beginning of
                 * the LRU (in order to be reused asap) and return with error.
                 */
            release_temp_and_return:
#if defined(PARSEC_DEBUG_NOISIER)
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\tRequest space on GPU failed for flow %s index %d/%d for task %s",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     flow->name, i, gpu_task->nb_flows, task_name );
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
                for( j = 0; j <= i; j++ ) {
                    /* This flow could be a control flow */
                    if( NULL == temp_loc[j] ) continue;
                    this_task->data[j].data_out = NULL;  /* reset the data out */
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
                PARSEC_DATA_COPY_RELEASE(gpu_elem);
#endif
                parsec_atomic_unlock(&master->lock);
                if( data_avail_epoch ) {  /* update the memory epoch */
                    gpu_device->data_avail_epoch++;
                }
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
                int release_protected_cpu_mirror = parsec_gpu_has_protected_cpu_mirror(oldmaster);
                int do_unlock = oldmaster->super.obj_reference_count != 1;
                parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
                parsec_atomic_wmb();
                /* detach could have released the oldmaster if it only had a single refcount */
                if( do_unlock )
                    parsec_atomic_unlock( &oldmaster->lock );
                if( release_protected_cpu_mirror ) {
                    parsec_data_release_self_contained_data(oldmaster);
                }

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
            PARSEC_DATA_COPY_RELEASE(lru_gpu_elem);
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
                                         &gpu_task->flow_info[i].flow_span, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
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
    if( pending_write_transfer ) {
        /* Existing GPU copies can remain attached to the task while it waits
         * for a writable input transfer to complete. Copies created
         * speculatively during this pass are tracked in temp_loc[] and must be
         * returned to the GPU LRU exactly like an ordinary reservation retry.
         */
        for( j = 0; j < gpu_task->nb_flows; j++ ) {
            /* This flow could be a control flow or an existing GPU copy. */
            if( NULL == temp_loc[j] ) continue;
            this_task->data[j].data_out = NULL;  /* reset the data out */
            /* This flow could be non-parsec-owned, in which case we can't reclaim it */
            if( 0 == (temp_loc[j]->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s:\tAdd copy %p [ref_count %d] back to the LRU list",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 temp_loc[j], temp_loc[j]->super.super.obj_reference_count);
            /* push them at the head to reach them again at the next iteration */
            parsec_list_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
        }
        /* The task's GPU dependencies are located, but at least one writable
         * copy is still being filled on the input stream. Keep pre-existing
         * data_in/data_out bindings in place and retry after the transfer
         * completion bumps data_avail_epoch. If this pass also freed GPU
         * memory while searching for reservations, bump the epoch before
         * returning so other delayed tasks are not stuck behind the old view of
         * available data/memory.
         */
        if( data_avail_epoch ) {
            gpu_device->data_avail_epoch++;
        }
        return PARSEC_HOOK_RETURN_AGAIN;
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
    parsec_data_copy_t * src_copy;
    parsec_data_copy_t * dst_copy;
    parsec_device_gpu_module_t *src_dev;
    parsec_device_gpu_module_t *dst_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;

    for(uint32_t i = 0; i < gtask->nb_flows  /* not task->task_class->nb_flows */; i++) {
        if( !(flow_mask & (1U << i)) ) continue;
        src_copy = gtask->flow_info[i].source;
        assert(src_copy->device_private != NULL);
        dst_copy = task->data[i].data_out;
        src_dev = (parsec_device_gpu_module_t *)parsec_mca_device_get(src_copy->device_index);
        dst_dev = (parsec_device_gpu_module_t *)parsec_mca_device_get(dst_copy->device_index);

        if((src_dev->super.type & PARSEC_DEV_ANY_TYPE) == (dst_dev->super.type & PARSEC_DEV_ANY_TYPE)) {
            assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
            dir = parsec_device_gpu_transfer_direction_d2d;
        } else {
            dir = parsec_device_gpu_transfer_direction_h2d;
        }

        count = (src_copy->original->span <= dst_copy->original->span) ? src_copy->original->span : dst_copy->original->span;
        ret = dst_dev->memcpy_async(dst_dev, gpu_stream,
                                    dst_copy->device_private,
                                    src_copy->device_private,
                                    count,
                                    dir);
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
    parsec_data_copy_t * src_copy;
    parsec_data_copy_t * dst_copy;
    parsec_device_gpu_module_t *dst_dev, *src_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;

    for(uint32_t i = 0; i < gtask->nb_flows  /* not task->task_class->nb_flows */; i++){
        if(flow_mask & (1U << i)){
            src_copy = task->data[i].data_out;
            dst_copy = src_copy->original->device_copies[0];
            if( NULL == dst_copy ) {
                /* Kernel-pop normally materializes a host mirror before
                 * calling the default stage-out path. Keep this guard here for
                 * direct/custom callers so a missing CPU target fails cleanly
                 * instead of dereferencing a NULL device_copies[0].
                 */
                return PARSEC_HOOK_RETURN_ERROR;
            }
            dst_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(dst_copy->device_index);
            src_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(src_copy->device_index);
            count = (src_copy->original->span <= dst_copy->original->span) ? src_copy->original->span : dst_copy->original->span;
            if( (src_dev->super.type & PARSEC_DEV_ANY_TYPE) == (dst_dev->super.type & PARSEC_DEV_ANY_TYPE) ) {
                assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
                dir = parsec_device_gpu_transfer_direction_d2d;
            } else {
                dir = parsec_device_gpu_transfer_direction_d2h;
                if (dst_copy->device_private == NULL && dst_copy->alloc_cb != NULL) {
                    dst_copy->alloc_cb(dst_copy, 0); // allocate on host
                }
                if (dst_copy->device_private == NULL) {
                    return PARSEC_HOOK_RETURN_ERROR;
                }
            }
            ret = src_dev->memcpy_async( src_dev, gpu_stream,
                                         dst_copy->device_private,
                                         src_copy->device_private,
                                         count,
                                         dir );
            if(PARSEC_SUCCESS != ret) {
                return PARSEC_HOOK_RETURN_ERROR;
            }
        }
    }
    return PARSEC_HOOK_RETURN_DONE;
}

static inline int
parsec_gpu_data_copy_acquire_reader(parsec_data_copy_t *copy,
                                    parsec_data_t *expected_original,
                                    uint32_t expected_version)
{
    parsec_device_module_t *device = parsec_mca_device_get(copy->device_index);
    if( (NULL == device) || !PARSEC_DEV_IS_GPU(device->type) ) {
        return 1;
    }

    int readers = parsec_atomic_fetch_inc_int32(&copy->readers);
    if( readers >= 0 ) {
        parsec_atomic_rmb();
        if( (copy->original == expected_original) &&
            (copy->version == expected_version) ) {
            return 1;
        }
    }

    parsec_atomic_fetch_add_int32(&copy->readers, -1);
    return 0;
}

static inline int
parsec_gpu_data_copy_release_reader(parsec_device_gpu_module_t *gpu_device,
                                    parsec_data_copy_t *copy,
                                    int make_available)
{
    int readers = parsec_atomic_fetch_sub_int32(&copy->readers, 1) - 1;
    if( (0 == readers) && make_available ) {
        parsec_device_module_t *copy_device;
        parsec_device_gpu_module_t *copy_gpu_device;
        /* Only PaRSEC-owned copies can be reclaimed through the device LRU.
         * D2D readers do not change ownership, so dirty GPU-only data must
         * also stay off the clean LRU until the W2R backup path sees it.
         */
        if( 0 != (copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) {
            copy_device = parsec_mca_device_get(copy->device_index);
            assert(NULL != copy_device);
            assert(PARSEC_DEV_IS_GPU(copy_device->type));
            copy_gpu_device = (parsec_device_gpu_module_t*)copy_device;
            if( copy_gpu_device != gpu_device ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: released reader for copy %p owned by GPU[%d:%s]; requeue on owner device",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     copy, copy_gpu_device->super.device_index, copy_gpu_device->super.name);
            }
            parsec_list_item_ring_chop((parsec_list_item_t*)copy);
            PARSEC_LIST_ITEM_SINGLETON(copy);
            /* D2D source copies can still own the latest version after the
             * transfer completes. Keep dirty copies out of the clean LRU so
             * they are not reclaimed as reusable read-cache memory.
             */
            if( PARSEC_DATA_COHERENCY_OWNED == copy->coherency_state ) {
                parsec_list_push_back(&copy_gpu_device->gpu_mem_owned_lru,
                                      (parsec_list_item_t*)copy);
            } else {
                parsec_list_push_back(&copy_gpu_device->gpu_mem_lru,
                                      (parsec_list_item_t*)copy);
            }
        }
    }
    return readers;
}

#if defined(PARSEC_DEBUG_NOISIER)
/**
 * State of all the copies of a data, captured before the runtime starts
 * mutating coherency states for a staging decision.
 */
typedef struct parsec_device_gpu_copy_snapshot_s {
    char     table[1024];  /**< one entry per existing copy */
    uint32_t newest_version;
    int      newest_device;  /**< -1 when no copy is readable */
    int      owner_device;
} parsec_device_gpu_copy_snapshot_t;

/** Caller must hold original->lock, or accept a best-effort read. */
static void
parsec_device_gpu_snapshot_copies(parsec_data_t *original,
                                  parsec_device_gpu_copy_snapshot_t *snapshot)
{
    int offset = 0;

    snapshot->newest_version = 0;
    snapshot->newest_device = -1;
    snapshot->owner_device = original->owner_device;

    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_data_copy_t *copy = original->device_copies[i];
        if( NULL == copy ) continue;
        if( (PARSEC_DATA_COHERENCY_INVALID != copy->coherency_state) &&
            (PARSEC_DATA_STATUS_UNDER_TRANSFER != copy->data_transfer_status) &&
            ((-1 == snapshot->newest_device) || (copy->version > snapshot->newest_version)) ) {
            snapshot->newest_version = copy->version;
            snapshot->newest_device = (int)i;
        }
        if( offset < (int)sizeof(snapshot->table) ) {
            offset += snprintf(snapshot->table + offset, sizeof(snapshot->table) - offset,
                               " %u:(v%u coh%d xfer%d rd%d)",
                               i, copy->version, copy->coherency_state,
                               copy->data_transfer_status, copy->readers);
        }
    }
}

/**
 * Report which copy ends up feeding an input flow, and in which state every
 * copy of the data was when the decision was taken. The source is reported as
 * stale when another copy that was neither invalid nor under transfer carried
 * a more recent version, which is the situation where the task is about to
 * read data that has already been superseded on another device.
 */
static void
parsec_device_gpu_audit_source(parsec_device_gpu_module_t *gpu_device,
                               const parsec_flow_t *flow,
                               parsec_gpu_task_t *gpu_task,
                               parsec_data_t *original,
                               const parsec_device_gpu_copy_snapshot_t *snapshot,
                               parsec_data_copy_t *data_in,
                               parsec_data_copy_t *effective_source,
                               const char *decision)
{
    char task_name[MAX_TASK_STRLEN];
    int stale = (-1 != snapshot->newest_device) &&
                (effective_source->version < snapshot->newest_version);

    if( !stale && (parsec_device_audit_stage_in > 1) ) return;

    if( NULL != gpu_task->ec ) {
        parsec_task_snprintf(task_name, MAX_TASK_STRLEN, gpu_task->ec);
    } else {
        snprintf(task_name, MAX_TASK_STRLEN, "<gpu task type %d>", gpu_task->task_type);
    }

    parsec_inform("STAGEIN%s GPU[%d:%s] %s flow %s access %s key %x: %s from dev %d v%u,"
                  " data_in dev %d v%u, newest dev %d v%u, owner %d, copies:%s",
                  stale ? "-STALE" : "",
                  gpu_device->super.device_index, gpu_device->super.name,
                  task_name, flow->name,
                  (PARSEC_FLOW_ACCESS_WRITE & flow->flow_flags)
                      ? ((PARSEC_FLOW_ACCESS_READ & flow->flow_flags) ? "RW" : "W") : "R",
                  original->key, decision,
                  effective_source->device_index, effective_source->version,
                  data_in->device_index, data_in->version,
                  snapshot->newest_device, snapshot->newest_version,
                  snapshot->owner_device, snapshot->table);
}
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

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
    size_t span = gpu_task->flow_info[flow->flow_index].flow_span;
    int transfer_from = -1;
    /* True once a GPU source copy has a readers reference held for this transfer. */
    int source_acquired = 0;

    if( gpu_task->task_type == PARSEC_GPU_TASK_TYPE_PREFETCH ) {
        PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream,
                             "GPU[%d:%s]: Prefetch task %p is staging in",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_task);
    }
    if( gpu_elem == candidate ) {  /* data already located in the right place */
        if( candidate->device_index == gpu_device->super.device_index ) {
            int wait_on_input_stream = (PARSEC_DATA_STATUS_UNDER_TRANSFER == candidate->data_transfer_status);
            /* The candidate is already located on the GPU. If it is still being
             * filled by earlier work on the input stream, there is no extra copy
             * to schedule here, but this stage must still record an input event.
             * The event preserves stream ordering and prevents execution before
             * the queued transfer has completed.
             */
            if( wait_on_input_stream ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\t\tInput data copy %p for flow %s is still under transfer; recording input stream event",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     candidate, flow->name);
            }
            if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) ) {
                candidate->version++;
                parsec_list_item_ring_chop((parsec_list_item_t *)candidate);
                PARSEC_LIST_ITEM_SINGLETON(candidate);
            }
            if( PARSEC_FLOW_ACCESS_READ & type ) {
                parsec_atomic_fetch_add_int32(&candidate->readers, 1);
            }
#if defined(PARSEC_DEBUG_NOISIER)
            if( parsec_device_audit_stage_in ) {
                parsec_device_gpu_copy_snapshot_t snapshot;
                parsec_atomic_lock( &original->lock );
                parsec_device_gpu_snapshot_copies(original, &snapshot);
                parsec_atomic_unlock( &original->lock );
                parsec_device_gpu_audit_source(gpu_device, flow, gpu_task, original, &snapshot,
                                               task_data->data_in, candidate, "INPLACE");
            }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
            return wait_on_input_stream ? 1 : PARSEC_HOOK_RETURN_DONE;
        }
        parsec_warning("GPU[%d:%s]:\t device_data_stage_in without a proper data_out on the device "
                       "and with a data_in (%p) located on another device %d",
                       gpu_device->super.device_index, gpu_device->super.name,
                       candidate, candidate->device_index);
    }
    parsec_atomic_lock( &original->lock );

    gpu_task->flow_info[flow->flow_index].source = candidate; /* default source for the transfer */
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

    /* If it is already under transfer, don't schedule the transfer again.
     * This happens if the task refers twice (or more) to the same input flow.
     * This is the only path that intentionally calls start_transfer_ownership
     * before selecting/acquiring a source: no new transfer will be issued here,
     * but start_transfer_ownership still reserves the destination reader that
     * will be released when the task pops.
     */
    if( (PARSEC_FLOW_ACCESS_READ & type) &&
        (gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER) ) {
        transfer_from = parsec_data_start_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);
        gpu_device->super.required_data_in += original->span;
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tMove data copy %p [ref_count %d, key %x] of %zu bytes: data copy is already under transfer, ignoring double request",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, span);
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
        if( (gpu_device->super.type & PARSEC_DEV_ANY_TYPE) == (candidate_dev->super.type & PARSEC_DEV_ANY_TYPE) ) {
            if( gpu_device->peer_access_mask & (1 << candidate_dev->super.device_index) ) {
                /* The fast path bypasses the full source scan below, so it must
                 * enforce the same readiness checks before acquiring a reader. */
                if( (PARSEC_DATA_COHERENCY_INVALID != candidate->coherency_state) &&
                    (PARSEC_DATA_STATUS_UNDER_TRANSFER != candidate->data_transfer_status) ) {
                    if( parsec_gpu_data_copy_acquire_reader(candidate, original, task_data->data_in->version) ) {
                        source_acquired = 1;
                        /* We can directly do D2D, so let's skip the selection */
                        PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                             "GPU[%d:%s]:\tskipping candidate lookup: data_in copy %p on %s has PEER ACCESS",
                                             gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, candidate_dev->super.name);
                        goto src_selected;
                    }
                    PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tdata_in copy %p on %s has PEER ACCESS but is being repurposed; looking for another source",
                                         gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, candidate_dev->super.name);
                } else {
                    PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tdata_in copy %p on %s has PEER ACCESS but is not ready; looking for another source",
                                         gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, candidate_dev->super.name);
                }
            }
        }

        for(int t = 1; t < (int)parsec_nb_devices; t++) {
            parsec_device_gpu_module_t *target = (parsec_device_gpu_module_t*)parsec_mca_device_get(t);
            /* Ownership has not been transferred yet, so the destination copy
             * may still look valid. Do not pick ourselves as a D2D source.
             */
            if( target == gpu_device ) {
                continue;
            }
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
            if( parsec_gpu_data_copy_acquire_reader(candidate, original, task_data->data_in->version) ) {
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tData copy %p [ref_count %d] on PaRSEC device %s is the best candidate to do Device to Device copy, increasing its readers to %d",
                                     gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name, candidate->readers);
                candidate_dev = target;
                source_acquired = 1;
                goto src_selected;
            }
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tCandidate %p [ref_count %d] on PaRSEC device %s is being repurposed by owner device. Looking for another candidate",
                                 gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name);
        }
        /* We fall back on the CPU copy, which is only a source at all if it
         * holds the version this task expects and is not itself in flight. */
        parsec_data_copy_t *cpu_copy = original->device_copies[0];
        if( (NULL == cpu_copy) ||
            (NULL == cpu_copy->device_private) ||
            (cpu_copy->version != task_data->data_in->version) ||
            (PARSEC_DATA_COHERENCY_INVALID == cpu_copy->coherency_state) ||
            (PARSEC_DATA_STATUS_UNDER_TRANSFER == cpu_copy->data_transfer_status) ) {
            if( potential_alt_src ) {
                /* An accelerator does hold this version and is only momentarily
                 * unable to serve it, so waiting is enough. No ownership or
                 * coherency state has been changed yet, so deferring is safe. */
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tThere is a potential alternative source for data_in %p [ref_count %d] in original %p to go in copy %p [ref_count %d], but neither the GPU nor the CPU copy is ready; retry later",
                                     gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original, gpu_elem, gpu_elem->super.super.obj_reference_count);
                parsec_atomic_unlock( &original->lock );
                return PARSEC_HOOK_RETURN_AGAIN;
            }
            /* Nothing holds this version where it can be reached: no peer was
             * even a candidate, and the host copy is not it either. Waiting
             * would wait forever and reading it would compute on the wrong
             * bytes, so say so instead of doing either silently. */
            char task_name[MAX_TASK_STRLEN];
            parsec_fatal("GPU[%d:%s]: no source for v%d of data %p [key %x] on flow %s of %s:"
                         " the host copy is v%d, coherency %d, transfer status %d, %s"
                         " -- the value is only on an accelerator this device cannot read",
                         gpu_device->super.device_index, gpu_device->super.name,
                         task_data->data_in->version, original, original->key,
                         flow->name, parsec_task_snprintf(task_name, MAX_TASK_STRLEN, gpu_task->ec),
                         (NULL == cpu_copy) ? -1 : (int)cpu_copy->version,
                         (NULL == cpu_copy) ? -1 : (int)cpu_copy->coherency_state,
                         (NULL == cpu_copy) ? -1 : (int)cpu_copy->data_transfer_status,
                         (NULL == cpu_copy) ? "absent" :
                             ((NULL == cpu_copy->device_private) ? "unallocated" : "allocated"));
        }

        /** TODO: when considering RW accesses, don't forget to chop gpu_elem
         *        from its queue... */
        candidate = cpu_copy;
        candidate_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get( candidate->device_index );
    }

 src_selected:
    /* Acquire a GPU source before changing ownership/coherency on the
     * destination. parsec_data_start_transfer_ownership_to_copy increments the
     * destination readers for read accesses and may update owner/coherency
     * state, so any retry/deferral must happen before that call.
     */
    if( !source_acquired &&
        (PARSEC_FLOW_ACCESS_READ & type) &&
        PARSEC_DEV_IS_GPU(candidate_dev->super.type) ) {
        if( !parsec_gpu_data_copy_acquire_reader(candidate, original, task_data->data_in->version) ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tCould not acquire GPU source copy %p [ref_count %d, key %x] on device %d; retry later",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 candidate, candidate->super.super.obj_reference_count,
                                 original->key, candidate_dev->super.device_index);
            parsec_atomic_unlock( &original->lock );
            return PARSEC_HOOK_RETURN_NEXT;
        }
        source_acquired = 1;
    }

#if defined(PARSEC_DEBUG_NOISIER)
    parsec_device_gpu_copy_snapshot_t snapshot;
    int new_data_shortcut = 0;
    if( parsec_device_audit_stage_in ) {
        parsec_device_gpu_snapshot_copies(original, &snapshot);
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    transfer_from = parsec_data_start_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

    /* If data is from NEW (it doesn't have a source_repo_entry and is not a direct data collection reference),
     * and nobody has touched it yet, then we don't need to pull it in, we have created it already, that's enough.
     * Having no data collection and a version of 0 does not on its own mean the
     * source is empty: data received from a remote peer also satisfies both, and
     * its content lives only in the host copy until it is staged in. Skip the
     * transfer only when the source holds no value at all. */
    if( (NULL == task_data->source_repo_entry) &&
        (NULL == task_data->data_in->original->dc) &&
        (0 == task_data->data_in->version) &&
        (PARSEC_DATA_COHERENCY_INVALID == task_data->data_in->coherency_state) ) {
        transfer_from = -1;
#if defined(PARSEC_DEBUG_NOISIER)
        new_data_shortcut = 1;
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
    }

    /* Update the transferred required_data_in size */
    gpu_device->super.required_data_in += original->span;

    if( -1 == transfer_from ) {  /* Do not need to be transferred */
        /* No transfer completion callback will run on the source in this path,
         * so release the temporary source reader immediately.
         */
        if( source_acquired ) {
            int readers = parsec_gpu_data_copy_release_reader(candidate_dev, candidate, 1);
            assert(readers >= 0);
        }
        gpu_elem->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;

        parsec_data_end_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

        if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) ) {
            gpu_elem->version = candidate->version + 1;
        }

        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tNO Move for data copy %p v%d [ref_count %d, key %x]",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->version, gpu_elem->super.super.obj_reference_count, original->key);
#if defined(PARSEC_DEBUG_NOISIER)
        if( parsec_device_audit_stage_in ) {
            /* Nothing moves, so what the task reads is the copy already sitting
             * on this device, not the source that was selected above. */
            parsec_device_gpu_audit_source(gpu_device, flow, gpu_task, original, &snapshot,
                                           task_data->data_in, gpu_elem,
                                           new_data_shortcut ? "NOMOVE-NEW" : "NOMOVE");
        }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
        parsec_atomic_unlock( &original->lock );
        /* TODO: data keeps the same coherence flags as before */
        return PARSEC_HOOK_RETURN_DONE;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]:\t\tMove %s data copy %p [ref_count %d, key %x] of %zu bytes\t(src dev: %d, v:%d, ptr:%p, copy:%p [ref_count %d, under_transfer: %d, coherency_state: %d] / dst dev: %d, v:%d, ptr:%p)",
                         gpu_device->super.device_index, gpu_device->super.name,
                         PARSEC_DEV_IS_GPU(candidate_dev->super.type) ? "D2D": "H2D",
                         gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, span,
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
        gpu_task->prof_stage_key_end = -1;

        if( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type && (gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_PREFETCH) ) {
            gpu_task->prof_stage_key_end = parsec_gpu_prefetch_key_end;
            gpu_task->prof_event_id = (int64_t)gpu_elem->device_private;
            gpu_task->prof_stage_object_id = gpu_device->super.device_index;
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   parsec_gpu_prefetch_key_start,
                                   gpu_task->prof_event_id,
                                   gpu_task->prof_stage_object_id,
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
            _info.size = (uint64_t)span;
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
    gpu_task->flow_info[flow->flow_index].source = candidate;  /* save the candidate for release on transfer completion */
    /* Push data into the GPU from the source device */
    int rc = gpu_task->stage_in ? gpu_task->stage_in(gpu_task, (1U << flow->flow_index), gpu_stream): PARSEC_SUCCESS;
    if(PARSEC_SUCCESS != rc) {
        parsec_warning( "GPU[%d:%s]: gpu_task->stage_in to device rc=%d @%s:%d\n"
                        "\t<<%p on device %d:%s>> -> <<%p on device %d:%s>> [%zu, %s]",
                        gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                        candidate->device_private, candidate_dev->super.device_index, candidate_dev->super.name,
                        gpu_elem->device_private, gpu_device->super.device_index, gpu_device->super.name,
                        span,
                        (candidate_dev->super.type & gpu_device->super.type & PARSEC_DEV_ANY_TYPE)? "D2D": "H2D");
        if( source_acquired ) {
            int readers = parsec_gpu_data_copy_release_reader(candidate_dev, candidate, 1);
            assert(readers >= 0);
        }
        parsec_atomic_unlock( &original->lock );
        assert(0);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    assert(candidate_dev->super.device_index < gpu_device->super.data_in_array_size);
    gpu_device->super.data_in_from_device[candidate_dev->super.device_index] += span;
    if( PARSEC_GPU_TASK_TYPE_KERNEL == gpu_task->task_type )
        gpu_device->super.nb_data_faults += span;

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
#if defined(PARSEC_DEBUG_NOISIER)
    if( parsec_device_audit_stage_in ) {
        parsec_device_gpu_audit_source(gpu_device, flow, gpu_task, original, &snapshot,
                                       task_data->data_in, candidate,
                                       PARSEC_DEV_IS_GPU(candidate_dev->super.type) ? "D2D" : "H2D");
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: GPU copy %p [ref_count %d] gets the version %d from copy %p version %d [ref_count %d]",
                         gpu_device->super.device_index, gpu_device->super.name,
                         gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->version, candidate, candidate->version, candidate->super.super.obj_reference_count);

    parsec_atomic_unlock( &original->lock );
    return 1;  /* positive returns have special meaning and are used for optimizations */
}

static inline int
parsec_gpu_task_is_singleton(parsec_gpu_task_t *task)
{
    parsec_list_item_t *item = &task->list_item;

#if defined(PARSEC_DEBUG_PARANOID)
    if( ((parsec_list_item_t *)(void *)0xdeadbeefL == item->list_next) ||
        ((parsec_list_item_t *)(void *)0xdeadbeefL == item->list_prev) ) {
        return 0;
    }
#endif
    return (item->list_next == item) && (item->list_prev == item);
}

/* Merge a singleton or task ring into a stream using the stream's ordering
 * policy. The generic list sorter needs an integer embedded in every list
 * item, so refresh the wrapper priority snapshots before the merge instead of
 * incorrectly applying parsec_task_t's priority offset to parsec_gpu_task_t.
 */
static inline void
parsec_gpu_stream_chain_pending(parsec_gpu_exec_stream_t *stream,
                                parsec_list_item_t *ring)
{
#if PARSEC_GPU_USE_PRIORITIES
    parsec_gpu_task_t *task = (parsec_gpu_task_t *)ring;
    parsec_gpu_task_t *current = task;

    do {
        assert(NULL != current->ec);
        current->priority = current->ec->priority;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != task );

    parsec_list_nolock_chain_sorted(stream->fifo_pending, ring,
                                    offsetof(parsec_gpu_task_t, priority));
#else
    parsec_list_nolock_chain_back(stream->fifo_pending, ring);
#endif
}

static inline void
parsec_gpu_stream_push_pending(parsec_gpu_exec_stream_t *stream,
                               parsec_gpu_task_t *task)
{
    /* Singleton and batched tasks must obey the same stream policy; otherwise
     * appending a successful batch can invalidate a priority-sorted FIFO.
     */
    parsec_gpu_stream_chain_pending(stream, &task->list_item);
}

/* NEXT applies to the selected head rather than its tentatively collected
 * followers. Merge those followers back under one lock using the stream's
 * insertion policy, then leave the head as a singleton for normal NEXT handling.
 */
static inline void
parsec_gpu_stream_rollback_batch(parsec_gpu_exec_stream_t *stream,
                                 parsec_gpu_task_t *batch_head)
{
    parsec_list_item_t *ring;

    ring = parsec_list_item_ring_chop(&batch_head->list_item);
    PARSEC_LIST_ITEM_SINGLETON(&batch_head->list_item);
    if( NULL != ring ) {
        parsec_gpu_stream_chain_pending(stream, ring);
    }
}

#if defined(PARSEC_PROF_TRACE)
/* Return whether any member owns an open logical GPU execution interval. The
 * state lives on each wrapper so it follows followers that are manually
 * detached and returned to runtime scheduling.
 */
static inline int
parsec_gpu_profile_exec_ring_is_open(parsec_gpu_task_t *ring)
{
    _LIST_ITEM_ITERATOR(&ring->list_item, &ring->list_item, item, {
        parsec_gpu_task_t *task = (parsec_gpu_task_t *)item;

        if( PARSEC_GPU_TASK_PROF_EXEC_OPEN == task->prof_exec_state ) {
            return 1;
        }
    });
    return 0;
}

/* Start each logical task at its first committed GPU submission. The user hook
 * has returned at this point, so the runtime can see the complete finalized
 * ring. Members already opened by an earlier AGAIN continuation are skipped.
 */
static inline void
parsec_gpu_profile_exec_ring_start(parsec_gpu_exec_stream_t *stream,
                                   parsec_gpu_task_t *ring)
{
    parsec_gpu_task_t *task;
    parsec_task_t *ec;
    const parsec_task_class_t *tc;
    int key_start, key_end;

    if( !stream->prof_event_track_enable || !parsec_profile_enabled ) {
        return;
    }
    _LIST_ITEM_ITERATOR(&ring->list_item, &ring->list_item, item, {
        task = (parsec_gpu_task_t *)item;
        ec = task->ec;
        tc = ec->task_class;

        if( (PARSEC_GPU_TASK_PROF_EXEC_PENDING == task->prof_exec_state) &&
            (NULL != ec->taskpool->profiling_array) ) {
            key_start = PARSEC_PROF_FUNC_KEY_START(ec->taskpool, tc->task_class_id);
            key_end = PARSEC_PROF_FUNC_KEY_END(ec->taskpool, tc->task_class_id);
            if( (key_start >= 2) && (key_end >= 2) ) {
                task->prof_event_id = tc->key_functions->key_hash(
                    tc->make_key(ec->taskpool, ec->locals), NULL);
                PARSEC_PROFILING_TRACE_INFO_FN(stream->profiling, key_start,
                                               task->prof_event_id,
                                               ec->taskpool->taskpool_id,
                                               tc->profile_info, (void *)ec);
                task->prof_exec_state = PARSEC_GPU_TASK_PROF_EXEC_OPEN;
            }
        }
    });
}

/* Close every open logical execution interval in a ring exactly once. This is
 * called only for final completion or a terminal ownership transfer, never for
 * an intermediate AGAIN event.
 */
static inline int
parsec_gpu_profile_exec_ring_end(parsec_gpu_exec_stream_t *stream,
                                 parsec_gpu_task_t *ring)
{
    parsec_gpu_task_t *task = ring;
    parsec_task_t *ec;
    const parsec_task_class_t *tc;
    int key_end;
    int found = 0;

    do {
        if( PARSEC_GPU_TASK_PROF_EXEC_OPEN == task->prof_exec_state ) {
            found = 1;
            if( stream->prof_event_track_enable ) {
                ec = task->ec;
                tc = ec->task_class;
                key_end = PARSEC_PROF_FUNC_KEY_END(ec->taskpool,
                                                   tc->task_class_id);
                PARSEC_PROFILING_TRACE(stream->profiling, key_end,
                                       task->prof_event_id,
                                       ec->taskpool->taskpool_id, NULL);
            }
            task->prof_exec_state = PARSEC_GPU_TASK_PROF_EXEC_PENDING;
        }
        task = (parsec_gpu_task_t *)task->list_item.list_next;
    } while( task != ring );
    return found;
}

/* Complete the profiling interval associated with one recorded stream event.
 * Transfer stages use the legacy head-only key. Execution batches instead
 * close every logical member, except when AGAIN retains the interval.
 */
static inline void
parsec_gpu_profile_event_complete(parsec_gpu_exec_stream_t *stream,
                                  parsec_gpu_task_t *ring,
                                  int keep_exec_open)
{
    if( keep_exec_open && parsec_gpu_profile_exec_ring_is_open(ring) ) {
        return;
    }
    if( !keep_exec_open &&
        parsec_gpu_profile_exec_ring_end(stream, ring) ) {
        return;
    }
    if( stream->prof_event_track_enable &&
        (ring->prof_stage_key_end != -1) ) {
        PARSEC_PROFILING_TRACE(stream->profiling, ring->prof_stage_key_end,
                               ring->prof_event_id,
                               ring->prof_stage_object_id, NULL);
    }
}
#endif /* defined(PARSEC_PROF_TRACE) */

/* Release every wrapper in a task ring and return the number released. ASYNC
 * transfers ownership of each underlying execution context to the submit hook,
 * but the GPU engine remains responsible for returning all device wrappers to
 * their allocator and removing each one from the manager's outstanding count.
 */
static int
parsec_gpu_task_ring_release(parsec_gpu_task_t *ring)
{
    int count = 0;

    while( NULL != ring ) {
        parsec_gpu_task_t *task = ring;

        ring = (parsec_gpu_task_t *)parsec_list_item_ring_chop(&task->list_item);
        PARSEC_LIST_ITEM_SINGLETON(&task->list_item);
        task->release_device_task(task);
        count++;
    }
    return count;
}

/* A terminal submit error can be reported after the hook queued partial work.
 * Record and wait for a stream event before cleaning the affected batch. The
 * runtime currently treats this as fatal; draining the rest of the device is a
 * separate recovery problem.
 */
static int
parsec_gpu_stream_quiesce_after_failure(parsec_device_gpu_module_t *gpu_device,
                                        parsec_gpu_exec_stream_t *stream)
{
    struct timespec delay = { .tv_sec = 0, .tv_nsec = 100 };
    int rc;

    rc = gpu_device->event_record(gpu_device, stream, stream->start);
    if( PARSEC_SUCCESS != rc ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }

    do {
        rc = gpu_device->event_query(gpu_device, stream, stream->start);
        if( 0 == rc ) {
            nanosleep(&delay, NULL);
        }
    } while( 0 == rc );

    return (1 == rc) ? PARSEC_SUCCESS : PARSEC_HOOK_RETURN_ERROR;
}

static inline int
parsec_gpu_task_selected_chore_allows_batch(parsec_task_t *task,
                                            parsec_device_module_t *device)
{
    const __parsec_chore_t *chore;

    if( (NULL == task) ||
        (NULL == task->task_class) ||
        (NULL == device) ||
        (task->selected_device != device) ||
        (task->selected_chore < 0) ) {
        return 0;
    }

    chore = &task->task_class->incarnations[task->selected_chore];
    return (PARSEC_DEV_NONE != (chore->type & PARSEC_DEV_ANY_TYPE)) &&
           (0 != (chore->type & device->type)) &&
           (0 != (chore->type & PARSEC_DEV_CHORE_ALLOW_BATCH));
}

int
parsec_gpu_task_collect_batch(parsec_gpu_exec_stream_t *gpu_stream,
                              parsec_gpu_task_t *batch_head,
                              parsec_gpu_task_batch_cb_t callback,
                              void *callback_data)
{
    parsec_list_t *fifo_pending;
    parsec_list_item_t *item, *next;
    parsec_task_t *head_task;
    parsec_device_module_t *device;
    int nb_tasks = 0;
    int rc;

    assert(NULL != gpu_stream);
    assert(NULL != batch_head);
    assert(NULL != callback);

    /* A batched AGAIN continuation already owns a committed ring. Let hooks
     * call the collector on every coroutine resume without detaching followers
     * or collecting unrelated pending tasks into the in-flight batch.
     */
    if( !parsec_gpu_task_is_singleton(batch_head) ) {
        parsec_gpu_task_t *current =
            (parsec_gpu_task_t *)batch_head->list_item.list_next;

        while( current != batch_head ) {
            nb_tasks++;
            current = (parsec_gpu_task_t *)current->list_item.list_next;
        }
        return nb_tasks;
    }

    parsec_list_item_singleton(&batch_head->list_item);

    head_task = batch_head->ec;
    if( (NULL == head_task) || (NULL == head_task->selected_device) ) {
        return nb_tasks;
    }
    device = head_task->selected_device;
    if( !parsec_mca_device_type_supports_batch(device->type) ||
        !parsec_gpu_task_selected_chore_allows_batch(head_task, device) ) {
        return nb_tasks;
    }

    fifo_pending = gpu_stream->fifo_pending;
    assert(NULL != fifo_pending);

    /* The collector deliberately has no scan bound. Submit-hook policy decides
     * when enough work has been considered and returns STOP at that candidate.
     */
    for(item = (parsec_list_item_t *)fifo_pending->ghost_element.list_next;
        item != &fifo_pending->ghost_element;
        item = next) {
        parsec_gpu_task_t *candidate = (parsec_gpu_task_t *)item;
        parsec_task_t *candidate_task = candidate->ec;

        next = (parsec_list_item_t *)item->list_next;
        if( !parsec_gpu_task_selected_chore_allows_batch(candidate_task, device) ) {
            continue;
        }
        rc = callback(candidate, batch_head, callback_data);
        if( PARSEC_GPU_TASK_BATCH_STOP == rc ) {
            break;
        }
        if( PARSEC_GPU_TASK_BATCH_ACCEPT == rc ) {
            (void)parsec_list_nolock_remove(fifo_pending, item);
            (void)parsec_list_item_ring_push(&batch_head->list_item, item);
            nb_tasks++;
            continue;
        }
        if( PARSEC_GPU_TASK_BATCH_REJECT != rc ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
    }

    return nb_tasks;
}

int
parsec_gpu_task_split_batch(parsec_gpu_exec_stream_t *gpu_stream,
                            parsec_gpu_task_t *batch_head,
                            parsec_gpu_task_split_cb_t callback,
                            void *callback_data)
{
    parsec_gpu_task_t *member, *next;
    parsec_list_item_t *returned = NULL;
    int nb_tasks = 0;
    int rc = PARSEC_GPU_TASK_SPLIT_KEEP;

    assert(NULL != gpu_stream);
    assert(NULL != batch_head);
    assert(NULL != callback);

    if( parsec_gpu_task_is_singleton(batch_head) ) {
        return nb_tasks;
    }

    member = (parsec_gpu_task_t *)batch_head->list_item.list_next;
    while( member != batch_head ) {
        /* Detaching rewrites the member's own links, so read the successor
         * while the ring is still intact.
         */
        next = (parsec_gpu_task_t *)member->list_item.list_next;

        rc = callback(member, batch_head, callback_data);
        if( PARSEC_GPU_TASK_SPLIT_STOP == rc ) {
            break;
        }
        if( PARSEC_GPU_TASK_SPLIT_RETURN == rc ) {
            (void)parsec_list_item_ring_chop(&member->list_item);
            PARSEC_LIST_ITEM_SINGLETON(&member->list_item);
            if( NULL == returned ) {
                returned = &member->list_item;
            } else {
                (void)parsec_list_item_ring_push(returned, &member->list_item);
            }
            nb_tasks++;
        } else if( PARSEC_GPU_TASK_SPLIT_KEEP != rc ) {
            break;
        }
        member = next;
    }

    /* Merge once, after the walk: a priority-sorted FIFO costs a scan per
     * insertion, and the members are not reachable from anywhere until this
     * runs. That also covers the malformed-result exit below, so a buggy
     * callback cannot strand the members it already declined.
     */
    if( NULL != returned ) {
        parsec_gpu_stream_chain_pending(gpu_stream, returned);
    }

    if( (PARSEC_GPU_TASK_SPLIT_KEEP != rc) &&
        (PARSEC_GPU_TASK_SPLIT_RETURN != rc) &&
        (PARSEC_GPU_TASK_SPLIT_STOP != rc) ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }
    return nb_tasks;
}

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
    parsec_gpu_task_t *gpu_task = (parsec_gpu_task_t *)PARSEC_OBJ_NEW(parsec_gpu_dsl_task_t);
    gpu_task->task_type = PARSEC_GPU_TASK_TYPE_D2D_COMPLETE;
    gpu_task->ec = calloc(1, sizeof(parsec_task_t));
    PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
    gpu_task->ec->task_class = &parsec_device_d2d_complete_tc;
    gpu_task->ec->priority = INT32_MAX; /* This task should be executed as soon as possible */
    /* The pending heap compares gpu_task->priority, not gpu_task->ec->priority: priority
     * inheritance from ec normally happens in parsec_device_kernel_scheduler(), which this
     * direct-enqueue path bypasses. Set it explicitly so this command is not starved behind
     * normal work. */
    gpu_task->priority = INT32_MAX;
    gpu_task->nb_flows = 1;
    gpu_task->flow_info[0].flow = &parsec_device_d2d_complete_flow;
    gpu_task->flow_info[0].flow_span = copy->original->span;
    gpu_task->stage_in  = parsec_default_gpu_stage_in;
    gpu_task->stage_out = parsec_default_gpu_stage_out;
    gpu_task->ec->data[0].data_in = copy;  /* We need to set not-null in data_in, so that the fake flow is
                                            * not ignored when popping the data from the fake task */
    gpu_task->ec->data[0].data_out = copy; /* We "free" data[i].data_out if its readers reaches 0 */
    gpu_task->ec->data[0].source_repo_entry = NULL;
    gpu_task->ec->data[0].source_repo = NULL;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_stage_key_end = -1; /* D2D complete tasks are pure internal management, we do not trace them */
#endif
    (void)current_dev;
    PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                         "GPU[%d:%s]: data copy %p [ref_count %d] D2D transfer is complete, sending order to count it "
                         "to GPU Device %d:%s",
                         current_dev->device_index, current_dev->name, gpu_task->ec->data[0].data_out,
                         gpu_task->ec->data[0].data_out->super.super.obj_reference_count,
                         dst_dev->device_index, dst_dev->name);
    parsec_lifo_push( &(((parsec_device_gpu_module_t*)dst_dev)->pending), (parsec_list_item_t*)gpu_task );
}

static int
parsec_device_callback_complete_push(parsec_device_gpu_module_t   *gpu_device,
                                     parsec_gpu_task_t           **gpu_task,
                                     parsec_gpu_exec_stream_t     *gpu_stream)
{
    (void)gpu_stream;

    parsec_gpu_task_t *gtask = *gpu_task;
    parsec_task_t *task;
    uint32_t i;
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

    for (i = 0; i < gtask->nb_flows /* not task->task_class->nb_flows */; i++){
        /* Make sure data_in is not NULL */
        if( NULL == task->data[i].data_in ) continue;
        /* We also don't push back non-parsec-owned copies */
        if(NULL != task->data[i].data_out &&
           0 == (task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)) continue;
        if( gpu_device->super.device_index == task->data[i].data_in->device_index ) continue;

        flow = gtask->flow_info[i].flow;
        assert( flow );
        assert( flow->flow_index == i );
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if(PARSEC_DATA_STATUS_UNDER_TRANSFER == task->data[i].data_out->data_transfer_status ) {
            /* only the task who did the PUSH can modify the status */
            parsec_atomic_lock(&task->data[i].data_out->original->lock);
            task->data[i].data_out->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;
            /* A completed input transfer makes this device copy usable even if
             * no GPU memory was freed. Wake tasks that previously backed off on
             * UNDER_TRANSFER and cached the old data availability epoch; they
             * must re-enter reserve_space() to revalidate the preserved
             * data_out binding.
             */
            gpu_device->data_avail_epoch++;
            parsec_data_end_transfer_ownership_to_copy(task->data[i].data_out->original,
                                                       gpu_device->super.device_index,
                                                       flow->flow_flags);

            parsec_data_copy_t* source = gtask->flow_info[i].source;
            parsec_device_gpu_module_t *src_device =
                    (parsec_device_gpu_module_t*)parsec_mca_device_get( source->device_index );
            if (task->data[i].data_in->flags & PARSEC_DATA_FLAG_EVICTED) {
                /**
                 * The device copy had been evicted to the host and brought back in.
                 * If this is the only device on which that data is used we can release
                 * the host memory back to the application. If there are other devices
                 * we cannot release the host memory because the data may actually be used
                 * by a host task (e.g., after being sent there from a different device)
                 * or be used as input to the other device.
                 */
                parsec_data_copy_t *cpu_copy = task->data[i].data_out->original->device_copies[0];
                parsec_data_copy_t *gpu_copy = task->data[i].data_out;
                parsec_data_t *original = task->data[i].data_out->original;
                /* release host memory if requested */
                if (cpu_copy->device_private != NULL &&
                    cpu_copy->release_cb != NULL) {
                    bool may_release = true;
                    /* check if there are any other device copies */
                    for (uint32_t i = 1; i < parsec_nb_devices; ++i) {
                        parsec_data_copy_t *copy = original->device_copies[i];
                        if (NULL != copy && copy != gpu_copy) {
                            may_release = false;
                            break;
                        }
                    }
                    if (may_release) {
                        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                            "GPU[%d:%s]: CPU copy %p [ref_count %d] memory %p eager release",
                                            gpu_device->super.device_index, gpu_device->super.name,
                                            cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->device_private);
                        cpu_copy->release_cb(cpu_copy, 0);
                        cpu_copy->device_private = NULL;
                    }
                }
                task->data[i].data_in->flags ^= PARSEC_DATA_FLAG_EVICTED;
            }

            parsec_atomic_unlock(&task->data[i].data_out->original->lock);

#if defined(PARSEC_PROF_TRACE)
            if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_IN) {
                PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                       parsec_gpu_movein_key_end,
                                       (int64_t)(int64_t)task->data[i].data_out->device_private,
                                       gpu_device->super.device_index,
                                       NULL);
            }
#endif
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
                    int readers = parsec_gpu_data_copy_release_reader(src_device, source, 1);
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
        int readers = parsec_gpu_data_copy_release_reader(gpu_device, gpu_copy, 1);
        assert(readers >= 0);
        if( 0 == readers ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tMake copy %p [ref_count %d] available after prefetch from gpu_task %p, ec %p",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, gtask, gtask->ec);
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
 * function progresses a task. If a batched submit function returns a task
 * ring, the ring is returned as the out_task parameter and chained into the
 * next stream by the caller's next invocation of this helper.
 *
 * Beware: this function does not generate errors by itself, instead
 * it propagates upward the return code of the progress function.
 * However, by convention the error code follows the parsec_hook_return_e
 * enum.
 */
static inline int
parsec_device_progress_stream( parsec_device_gpu_module_t* gpu_device,
                               parsec_execution_stream_t* es,
                               parsec_gpu_exec_stream_t* stream,
                               parsec_gpu_step_function_t progress_fct,
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
        parsec_gpu_stream_push_pending(stream, task);
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
            parsec_gpu_profile_event_complete(
                stream, task, PARSEC_HOOK_RETURN_AGAIN == task->last_status);
#endif /* (PARSEC_PROF_TRACE) */
            if( PARSEC_HOOK_RETURN_AGAIN == task->last_status ) {
                /* AGAIN means the submit hook made progress on this exact task
                 * or batch. Keep the complete ring as continuation state and
                 * re-enter the same stage only after its event has completed.
                 */
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: GPU task ring %p[%p] is ready to continue on the same GPU device and same stream",
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
    assert(NULL == task);
    if( NULL == stream->tasks[stream->start] ) {  /* there is room on the stream */
        task = (parsec_gpu_task_t*)parsec_list_nolock_pop_front(stream->fifo_pending);  /* get the next task */
    }
    if( NULL == task ) {  /* No tasks, we're done */
        return PARSEC_HOOK_RETURN_DONE;
    }
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)task);

    assert( NULL == stream->tasks[stream->start] );

  schedule_task:
    /* New queue entries are singletons. An AGAIN continuation deliberately
     * keeps the complete submitted ring together across event completions.
     */
    assert(parsec_gpu_task_is_singleton(task) ||
           (PARSEC_HOOK_RETURN_AGAIN == task->last_status));
    rc = progress_fct( gpu_device, es, task, stream );
    if( 0 == rc && parsec_device_skip_empty_events ) {
#if defined(PARSEC_PROF_TRACE)
        parsec_gpu_profile_event_complete(stream, task, 0);
#endif
        /* If progress_fct added nothing on that stream, skip the GPU event.
         * Input stages with copies already queued on the input stream return a
         * positive value, so they still record an event and preserve ordering.
         */
        if( task->complete_stage )
            rc = task->complete_stage(gpu_device, &task, stream);
        *out_task = task;
        return rc;
    }
    if( 0 > rc ) {
        if( PARSEC_HOOK_RETURN_AGAIN != rc ) {
            if( PARSEC_HOOK_RETURN_NEXT == rc ) {
                /* NEXT applies only to the selected head. Restore tentative
                 * followers, then use the unchanged singleton NEXT path.
                 */
                parsec_gpu_stream_rollback_batch(stream, task);
                /* Don't reorder the push_back, we are running into physical constraints and need to delay
                 * the resubmission of this task as much as possible, but without losing track of it
                 * (aka. returning it to the upper level).
                 */
                parsec_gpu_stream_push_pending(stream, task);
            } else if( PARSEC_HOOK_RETURN_ASYNC == rc ) {
                /* A batch-aware hook transfers every execution context in the
                 * ring. The manager will release all device wrappers.
                 */
#if defined(PARSEC_PROF_TRACE)
                parsec_gpu_profile_exec_ring_end(stream, task);
#endif
                *out_task = task;
            } else {
                /* ERROR and DISABLE are terminal. A hook may have queued work
                 * before failing, so quiesce the stream before returning the
                 * complete ring for cleanup. Unknown negative values are
                 * normalized to ERROR at this boundary.
                 */
                rc = (PARSEC_HOOK_RETURN_DISABLE == rc) ?
                     PARSEC_HOOK_RETURN_DISABLE : PARSEC_HOOK_RETURN_ERROR;
                task->last_status = rc;
                if( PARSEC_SUCCESS !=
                    parsec_gpu_stream_quiesce_after_failure(gpu_device, stream) ) {
                    rc = task->last_status = PARSEC_HOOK_RETURN_ERROR;
                }
#if defined(PARSEC_PROF_TRACE)
                parsec_gpu_profile_exec_ring_end(stream, task);
#endif
                *out_task = task;
            }
            return rc;
        }

        *out_task = NULL;
        /**
         * The task requested to be rescheduled but it might have added kernels on the
         * stream and we need to wait for their completion. Thus, treat the task as usual,
         * create and event and upon completion of this event add the task back into the
         * execution stream pending list (to be executed again).
         */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]: GPU task ring %p returned AGAIN; continue it after the event completes",
                             gpu_device->super.device_index, gpu_device->super.name, (void*)task);
    }
    task->last_status = rc;
    /* Keep the progress result with the task while it is parked behind the
     * event. In particular, AGAIN is not an error: it means the event
     * completion path must retry this same stage instead of advancing the task.
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
                           parsec_execution_stream_t       *es,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t *flow;
    int ret = 0, input_stream_work = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    (void)es;

    /* If no changes were made to the available memory/data state, avoid
     * immediately rewalking the same reservation path. This shortcut is safe
     * only if every state change that can make a previous reservation failure
     * succeed bumps data_avail_epoch. In particular, transfer completion must
     * bump the epoch, otherwise a task that deferred readiness after seeing
     * UNDER_TRANSFER can livelock here without re-entering reserve_space().
     */
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
            parsec_data_copy_t *gpu_copy =
                gpu_task->ec->data[0].data_in->original->device_copies[gpu_device->super.device_index];
            /* There is already a copy of this data in the GPU */
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s data_copy at index %d is %p, destroying prefetch request",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task),
                                 gpu_device->super.device_index,
                                 gpu_copy);
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
    gpu_task->last_status = 0;  /* mark the task as clean */
    for( uint32_t i = 0; i < gpu_task->nb_flows  /* not this_task->task_class->nb_flows */; i++ ) {

        flow = gpu_task->flow_info[i].flow;
        /* Skip CTL flows */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* If there is already a GPU data copy (set by reserve_device_space), and this copy
         * is not parsec-owned, don't stage in. We still record read uses so the pop
         * path can balance readers without handing the copy to PaRSEC's LRU.
         */
        if( NULL != this_task->data[i].data_out &&
            (0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) ) {
            if( PARSEC_DATA_STATUS_UNDER_TRANSFER == this_task->data[i].data_out->data_transfer_status ) {
                /* Non-owned copies are not staged by PaRSEC, but an existing
                 * under-transfer marker still means the input stream has work
                 * ordered before this task. Record the event and let stream
                 * ordering protect the access.
                 */
                input_stream_work++;
            }
            if( flow->flow_flags & PARSEC_FLOW_ACCESS_READ ) {
                parsec_atomic_fetch_add_int32(&this_task->data[i].data_out->readers, 1);
            }
            continue;
        }

        assert( NULL != parsec_data_copy_get_ptr(this_task->data[i].data_in)
             || NULL != this_task->data[i].data_in->alloc_cb );

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
        /* A positive return means either a transfer was submitted, or an
         * already-submitted transfer for this input is still ordered before us
         * on the input stream. In both cases the push stage must record an
         * event before the task can advance to execution.
         */
        input_stream_work += ret;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Push task %s DONE",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task));
    gpu_task->complete_stage = parsec_device_callback_complete_push;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_stage_key_end = -1; /* We do not log that event as the completion of this task */
#endif
    return input_stream_work;
}

/**
 * @brief Prepare a task for execution on the GPU. Invoke the task submission
 * body, then start execution profiling for every member of the finalized ring.
 * Upon return from the body, handle the task state machine, including AGAIN
 * continuations and ASYNC ownership transfer.
 * @returns An error if anything unexpected came out of the task submission body, otherwise
 */
static int
parsec_device_kernel_exec( parsec_device_gpu_module_t      *gpu_device,
                           parsec_execution_stream_t       *es,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_advance_task_function_t progress_fct = gpu_task->submit;
    parsec_task_t* this_task = gpu_task->ec;
    int rc;
#if defined(PARSEC_PROF_TRACE)
    int continuing_batch;
#endif

#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    if( 0 != parsec_device_inject_disable ) {
        static int32_t nb_submitted = 0;
        if( parsec_atomic_fetch_inc_int32(&nb_submitted) + 1 == parsec_device_inject_disable ) {
            /* Decline the device exactly once, as an unsupported kernel would, so
             * the fallback onto another incarnation is exercised on hardware that
             * otherwise runs every kernel successfully. */
            return PARSEC_HOOK_RETURN_DISABLE;
        }
    }
#endif  /* defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER) */

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]:\tEnqueue on device %s stream %s"     ,
                         gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(tmp, MAX_TASK_STRLEN,
                         (parsec_task_t *) this_task), gpu_stream->name);
#endif /* defined(PARSEC_DEBUG_NOISIER) */
    (void)es;
#if defined(PARSEC_DEBUG_PARANOID)
    const parsec_flow_t *flow;
    for( uint i = 0; i < gpu_task->nb_flows  /* this_task->task_class->nb_flows */; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow_info[i].flow;
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if( 0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
        assert(this_task->data[i].data_out->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER);
    }
#endif /* defined(PARSEC_DEBUG_PARANOID) */

    /* New submissions start from a clean singleton so stale release-mode list
     * links cannot be mistaken for a batch. AGAIN is different: its ring is
     * the submit hook's continuation state and must be passed back intact.
     */
#if defined(PARSEC_PROF_TRACE)
    /* A committed non-singleton AGAIN ring was fully accounted when it first
     * yielded. Avoid rescanning every member on each coroutine progress step.
     */
    continuing_batch = (PARSEC_HOOK_RETURN_AGAIN == gpu_task->last_status) &&
                       !parsec_gpu_task_is_singleton(gpu_task);
#endif
    if( PARSEC_HOOK_RETURN_AGAIN != gpu_task->last_status ) {
        PARSEC_LIST_ITEM_SINGLETON(&gpu_task->list_item);
    }

    (void)this_task;
    rc = progress_fct( gpu_device, gpu_task, gpu_stream );
#if defined(PARSEC_PROF_TRACE)
    if( !continuing_batch &&
        ((rc >= 0) || (PARSEC_HOOK_RETURN_AGAIN == rc)) ) {
        /* The hook has finalized the submitted ring. Start every new logical
         * member here so generated and user-defined hooks share one policy.
         */
        parsec_gpu_profile_exec_ring_start(gpu_stream, gpu_task);
    }
#endif
    gpu_task->last_status = rc;
    /* Empty-stage event skipping is only valid for input/output streams.
     * A non-negative kernel submit result means the execution stream needs an
     * event, even when the submit hook itself returned HOOK_DONE.
     */
    if( rc < 0 )
        return rc;
    return 1;
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
                          parsec_execution_stream_t    *es,
                          parsec_gpu_task_t            *gpu_task,
                          parsec_gpu_exec_stream_t     *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy;
    parsec_data_t              *original;
    size_t                      span;
    const parsec_flow_t        *flow;
    int return_code = 0, rc, how_many = 0, update_data_epoch = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    if (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER) {
        for( int i = 0; i < this_task->locals[0].value; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            /* If the gpu copy is not owned by parsec, we don't manage it at all */
            if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
            original = gpu_copy->original;
            parsec_atomic_lock(&original->lock);
            /* Background D2H cleanup has no successor flow to query for an
             * arena/datatype, so it can only use an already-existing host
             * mirror. Task pushout below can materialize a missing mirror from
             * the receiving flow metadata.
             */
            parsec_data_copy_t *cpu_copy = original->device_copies[0];
            parsec_atomic_unlock(&original->lock);
            if( NULL == cpu_copy ) {
                parsec_warning("GPU[%d:%s]: unable to find host mirror for D2H transfer of copy %p\n",
                               gpu_device->super.device_index, gpu_device->super.name, gpu_copy);
                return_code = PARSEC_HOOK_RETURN_DISABLE;
                goto release_and_return_error;
            }
            assert(cpu_copy->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER);
            rc = gpu_task->stage_out ? gpu_task->stage_out(gpu_task, (1U << i), gpu_stream): PARSEC_SUCCESS;
            if(PARSEC_SUCCESS != rc) {
                parsec_warning( "GPU[%d:%s]: gpu_task->stage_out from device rc=%d @%s:%d\n"
                                "\tdata %s <<%p>> -> <<%p>>\n",
                                gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                                this_task->task_class->out[i]->name,
                                gpu_copy->device_private, cpu_copy->device_private);
                return_code = PARSEC_HOOK_RETURN_DISABLE;
                goto release_and_return_error;
            }
            /* stage_out only enqueues the device-to-host transfer. The runtime
             * owns the copy state transition so custom stage_out callbacks do
             * not need to know about the GPU copy-transfer bookkeeping.
             */
            cpu_copy->data_transfer_status = PARSEC_DATA_STATUS_UNDER_TRANSFER;
            how_many++;
        }
        return how_many;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                        "GPU[%d:%s]: Try to Pop %s",
                        gpu_device->super.device_index, gpu_device->super.name,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    for( uint32_t i = 0; i < gpu_task->nb_flows  /* not this_task->task_class->nb_flows */; i++ ) {
        /* We need to manage all data that has been used as input, even if they were read only */

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow_info[i].flow;
        if( PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags) )  continue;  /* control flow */

        gpu_copy = this_task->data[i].data_out;

        original = gpu_copy->original;
        span = gpu_task->flow_info[i].flow_span;

        assert( this_task->data[i].data_in == NULL || original == this_task->data[i].data_in->original );

#if 0
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
#endif
        parsec_atomic_lock(&original->lock);
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_READ ) {
            int current_readers = parsec_gpu_data_copy_release_reader(gpu_device, gpu_copy,
                                                                      !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE));
            if( current_readers < 0 ) {
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: While trying to Pop %s, gpu_copy %p [ref_count %d] on flow %d with original %p had a negative number of readers (%d)",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task),
                                     gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     i, original, current_readers);
            }
            assert(current_readers >= 0);
            /* Non-owned copies may be used as GPU inputs, but their lifetime is
             * not managed through PaRSEC's device LRUs. After balancing readers,
             * leave ownership, availability, and reclamation to the external owner.
             */
            if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) {
                parsec_atomic_unlock(&original->lock);
                continue;
            }
            if( (0 == current_readers) && !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available on flow %s",
                                     gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);
                update_data_epoch = 1;
                parsec_atomic_unlock(&original->lock);
                continue;  /* done with this element, go for the next one */
            }
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tread copy %p [ref_count %d] on flow %s has readers (%i)",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name, current_readers);
        }
        /* Non-owned copies may be used as GPU inputs, but their lifetime is not
         * managed through PaRSEC's device LRUs. Write-only users did not acquire
         * a reader above, so leave ownership, write-back, and reclamation to the
         * external owner.
         */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) {
            parsec_atomic_unlock(&original->lock);
            continue;
        }
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == parsec_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                "GPU[%d:%s]:\tOUT Data copy %p [ref_count %d] for flow %s",
                                gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += span;
            assert( ((parsec_list_item_t*)gpu_copy)->list_next == (parsec_list_item_t*)gpu_copy );
            assert( ((parsec_list_item_t*)gpu_copy)->list_prev == (parsec_list_item_t*)gpu_copy );

            assert( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
            if( gpu_task->pushout & (1 << i) ) {
                parsec_data_copy_t *cpu_copy;
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                /* Pushout means the runtime needs a host-visible copy after
                 * this GPU task. For self-contained temporaries, the original
                 * CPU copy may have been released earlier when it no longer
                 * had local successors. Materialize the mirror lazily instead
                 * of assuming device_copies[0] survived the GPU propagation.
                 */
                cpu_copy = parsec_device_gpu_ensure_host_mirror(es, this_task, flow, gpu_copy);
                if( NULL == cpu_copy ) {
                    parsec_warning("GPU[%d:%s]: unable to create host mirror for data <%s:%x> copy %p\n",
                                   gpu_device->super.device_index, gpu_device->super.name,
                                   flow->name, original->key, gpu_copy);
                    return_code = PARSEC_HOOK_RETURN_DISABLE;
                    parsec_atomic_unlock(&original->lock);
                    goto release_and_return_error;
                }
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                    "GPU[%d:%s]:\tMove D2H data <%s:%x> copy %p [ref_count %d] -- D:%p -> H:%p requested",
                                    gpu_device->super.device_index, gpu_device->super.name, flow->name, original->key, gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     (void*)gpu_copy->device_private, cpu_copy->device_private);
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
                        gpu_task->prof_stage_key_end = parsec_gpu_moveout_key_end;
                        gpu_task->prof_stage_object_id = this_task->taskpool->taskpool_id;
                        gpu_task->prof_event_id = this_task->task_class->key_functions->key_hash(this_task->task_class->make_key(this_task->taskpool, this_task->locals), NULL);
                        PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                               parsec_gpu_moveout_key_start,
                                               gpu_task->prof_event_id,
                                               gpu_task->prof_stage_object_id,
                                               &info);
                    } else {
                        gpu_task->prof_stage_key_end = -1;
                    }
                }
#endif
                /* Move the data back into main memory */
                assert(cpu_copy->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER);
                rc = gpu_task->stage_out? gpu_task->stage_out(gpu_task, (1U << flow->flow_index), gpu_stream): PARSEC_SUCCESS;
                if( PARSEC_SUCCESS != rc ) {
                    parsec_warning( "GPU[%d:%s]: gpu_task->stage_out from device rc=%d @%s:%d\n"
                                    "\tdata %s <<%p>> -> <<%p>>\n",
                                    gpu_device->super.device_index, gpu_device->super.name, rc, __func__, __LINE__,
                                    this_task->task_class->out[i]->name,
                                    gpu_copy->device_private, cpu_copy->device_private);
                    return_code = PARSEC_HOOK_RETURN_DISABLE;
                    parsec_atomic_unlock(&original->lock);
                    goto release_and_return_error;
                }
                /* stage_out only enqueues the device-to-host transfer. The runtime
                 * owns the copy state transition so custom stage_out callbacks do
                 * not need to know about the GPU copy-transfer bookkeeping.
                 */
                cpu_copy->data_transfer_status = PARSEC_DATA_STATUS_UNDER_TRANSFER;
                gpu_device->super.data_out_to_host += span; /* TODO: not hardcoded, use datatype size */
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
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), return_code < 0 ? return_code: how_many, gpu_device->data_avail_epoch );

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
    parsec_gpu_data_copy_t     *gpu_copy;

#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
    (void)parsec_task_snprintf(task_str, MAX_TASK_STRLEN, this_task);
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Epilog of %s",
                         gpu_device->super.device_index, gpu_device->super.name, task_str);
#endif

    for( uint32_t i = 0; i < gpu_task->nb_flows  /* not this_task->task_class->nb_flows */; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;


        if( !(gpu_task->flow_info[i].flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;

        /* If it is a copy managed by the user, don't bother either */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
#if 0
        parsec_data_t *original = gpu_copy->original;
        parsec_gpu_data_copy_t *cpu_copy = original->device_copies[0];
        if( this_task->data[i].data_in == this_task->data[i].data_out ) {
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
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s: CPU copy %p [ref_count %d] gets the same version %d as GPU copy %p [ref_count %d]",
                                 gpu_device->super.device_index, gpu_device->super.name, task_str,
                                 cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->version, gpu_copy, gpu_copy->super.super.obj_reference_count);

            /**
             * Let's lie to the engine by reporting that working version of this
             * data is now on the CPU.
             */
            this_task->data[i].data_out = cpu_copy;
        }
#endif
        assert(0 <= gpu_copy->readers);

        if( gpu_task->pushout & (1 << i) ) {
            parsec_data_t *original = gpu_copy->original;
            parsec_gpu_data_copy_t *cpu_copy = original->device_copies[0];
            /* Update the CPU copy to reflect the current status */
            assert(cpu_copy->version < gpu_copy->version);
            cpu_copy->version = gpu_copy->version;
            cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            assert(PARSEC_DATA_STATUS_UNDER_TRANSFER == cpu_copy->data_transfer_status);
            cpu_copy->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;
            /* If the communication engine cannot send directly from GPU memory,
             * report the CPU copy as the task output. Otherwise, keep the GPU copy
             * as the output so remote successors can use the GPU-aware send path.
             */
            if( 0 == (parsec_mpi_allow_gpu_memory_communications & PARSEC_RUNTIME_SEND_GPU_MEMORY) ) {
                /* Report the CPU copy as the output of the task. */
                this_task->data[i].data_out = cpu_copy;
                PARSEC_DEBUG_VERBOSE(100, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s: GPU copy %p [ref_count %d] replaced by the CPU copy %p [ref_count %d] in %s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_str,
                                 gpu_copy, gpu_copy->super.super.obj_reference_count,
                                 cpu_copy, cpu_copy->super.super.obj_reference_count, __func__);
            } else {
                /* Self-contained temporaries still need their CPU mirror while
                 * GPU copies are propagated through GPU-aware remote deps.
                 */
                parsec_data_protect_cpu_mirror(original);
            }
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s: GPU copy %p [ref_count %d] moved to the read LRU in %s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_str,
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            parsec_list_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        } else {
            /* No need to detach the GPU copy it does not belong to any lists because it was owned by the task */
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s: GPU copy %p [ref_count %d] moved to the owned LRU in %s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_str,
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
    int data_avail_epoch = 0;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Cleanup of %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( uint32_t i = 0; i < gpu_task->nb_flows  /* not this_task->task_class->nb_flows */; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        if( !(gpu_task->flow_info[i].flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
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

/* parsec_lifo_detach_chain() returns tasks in LIFO order (most recently
 * pushed first). parsec_heap_push_chain() stamps the FIFO tie-break
 * sequence number in traversal order, so pushing a LIFO-ordered chain
 * directly would give newer tasks smaller (i.e. "older") sequence numbers,
 * inverting arrival-order fairness among equal-priority tasks within a
 * single detached batch. Reverse the chain in place so it is walked
 * oldest-first, matching true arrival order. */
static parsec_list_item_t *
parsec_gpu_pending_chain_reverse(parsec_list_item_t *chain)
{
    parsec_list_item_t *prev = NULL;
    while (NULL != chain) {
        parsec_list_item_t *next = (parsec_list_item_t *)chain->list_next;
        chain->list_next = prev;
        prev = chain;
        chain = next;
    }
    return prev;
}

/* Clean every kernel task in a terminally failed submit batch after its
 * execution stream has been quiesced. Device-wide queue recovery is
 * intentionally left to the failure path rather than hidden here.
 */
static void
parsec_device_kernel_cleanout_ring(parsec_device_gpu_module_t *gpu_device,
                                   parsec_gpu_task_t *ring)
{
    parsec_gpu_task_t *task = ring;

    do {
        if( PARSEC_GPU_TASK_TYPE_KERNEL == task->task_type ) {
            parsec_device_kernel_cleanout(gpu_device, task);
        }
        task = (parsec_gpu_task_t *)task->list_item.list_next;
    } while( task != ring );
}

/* Report whether every member of a declined ring still has an incarnation to
 * fall back on. Retiring a chore mutates the task, so the ring has to be probed
 * before any member is modified: retiring part of a batch and then discovering
 * that another member is stuck would leave the batch half-retired with no way
 * back.
 */
static int
parsec_device_kernel_ring_can_retry_elsewhere(parsec_gpu_task_t *ring)
{
    parsec_gpu_task_t *task = ring;

    do {
        parsec_task_t *ec = task->ec;

        if( (PARSEC_GPU_TASK_TYPE_KERNEL != task->task_type) || (NULL == ec) )
            return 0;
        if( 0 == (ec->chore_mask & ~(1 << ec->selected_chore)) )
            return 0;
        task = (parsec_gpu_task_t *)task->list_item.list_next;
    } while( task != ring );
    return 1;
}

/* Retire the declined incarnation for every member of the ring and hand each
 * execution context back to the scheduler. Only call this once
 * parsec_device_kernel_ring_can_retry_elsewhere() has agreed that no member
 * would be left without an incarnation.
 */
static void
parsec_device_kernel_retire_incarnation_ring(parsec_device_gpu_module_t *gpu_device,
                                             parsec_execution_stream_t *es,
                                             parsec_gpu_task_t *ring)
{
    parsec_gpu_task_t *task = ring;

    do {
        parsec_task_t *ec = task->ec;
        char decl[MAX_TASK_STRLEN];

        ec->chore_mask &= ~(1 << ec->selected_chore);
        assert(0 != ec->chore_mask);
        parsec_warning("GPU[%d:%s]: %s cannot run on this device, falling back to another incarnation",
                       gpu_device->super.device_index, gpu_device->super.name,
                       parsec_task_snprintf(decl, MAX_TASK_STRLEN, ec));
        parsec_device_kernel_cleanout(gpu_device, task);
        /* Advance before rescheduling: ec re-enters the scheduler immediately
         * and must no longer be reachable through this walk.
         */
        task = (parsec_gpu_task_t *)task->list_item.list_next;
        __parsec_reschedule(es, ec);
    } while( task != ring );
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
    int rc, exec_stream = 0, released_tasks;
    parsec_gpu_task_t *progress_task = NULL;
    parsec_gpu_task_t *gpu_task = (parsec_gpu_task_t*)_gpu_task;
    parsec_gpu_task_t *failed_batch = NULL;
    int failed_batch_staged = 1;
    parsec_hook_return_t failure_status = PARSEC_HOOK_RETURN_DISABLE;
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

    if (gpu_task != NULL && gpu_task->priority < 0) {
        gpu_task->priority = (gpu_task->ec != NULL) ? gpu_task->ec->priority : 0;
    }

    /* Check the GPU status -- three kinds of values for rc:
     *   - rc < 0: somebody owns an exclusive no-manager transition, so wait.
     *   - rc == 0: there is no manager, and at the exit of the while this
     *             worker changed the mutex from 0 to 1 and became the manager.
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
        parsec_lifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    /* Only the worker that changed mutex from 0 to 1 becomes the long-lived
     * manager. Expose its private ready task before entering GPU progress;
     * workers that merely enqueue behind this manager retain next_task. Other
     * submitters may enqueue during the flush, but this manager has not started
     * consuming the device queues yet.
     */
    rc = __parsec_schedule_flush_private(es);
    if( PARSEC_SUCCESS != rc ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }

    PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream, "GPU[%d:%s]: Entering GPU management",
                         gpu_device->super.device_index, gpu_device->super.name);

#if defined(PARSEC_PROF_TRACE)
    if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_start,
                                (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc) {
        /* This thread is already the manager and has already opened its own-GPU
         * interval, so it cannot just return: take the common exit so the
         * wrapper, the outstanding count and the profiling interval are all
         * settled. Nothing has been staged for this task yet, which is why the
         * cleanout is suppressed below.
         */
        failed_batch = gpu_task;
        failed_batch_staged = 0;
        gpu_task = NULL;
        goto disable_gpu;
    }

 check_in_deps:
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tUpload data (if any) for %s",
                             gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
    }
    rc = parsec_device_progress_stream( gpu_device,
                                        es,
                                        gpu_device->exec_stream[0],
                                        parsec_device_kernel_push,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {  /* In case of error progress_task is the task that raised it */
        if( PARSEC_HOOK_RETURN_ERROR == rc ) {
            failure_status = PARSEC_HOOK_RETURN_ERROR;
            failed_batch = progress_task;
            progress_task = NULL;
            goto disable_gpu;
        }
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
                                        es,
                                        gpu_device->exec_stream[2+exec_stream],
                                        parsec_device_kernel_exec,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {
        if( PARSEC_HOOK_RETURN_ERROR == rc ) {
            failure_status = rc;
            failed_batch = progress_task;
            progress_task = NULL;
            goto disable_gpu;
        }
        if( PARSEC_HOOK_RETURN_DISABLE == rc ) {
            /* The body declined this device. That is a statement about this
             * incarnation, not a reason to abort the execution, so retire the
             * incarnation and let another one take over. progress_stream hands
             * back the task, or the complete batch, that raised the error.
             *
             * A declined batch is retired as a unit, so that no follower is
             * stranded by rescheduling the head alone. The ring is probed for a
             * surviving incarnation before any member is modified.
             */
            if( (NULL != progress_task) &&
                parsec_device_kernel_ring_can_retry_elsewhere(progress_task) ) {
                parsec_device_kernel_retire_incarnation_ring(gpu_device, es, progress_task);
                gpu_task = progress_task;
                progress_task = NULL;
                goto remove_gpu_task;
            }
            failure_status = rc;
            failed_batch = progress_task;
            progress_task = NULL;
            goto disable_gpu;  /* no incarnation left to fall back on */
        }
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

  get_data_out_of_device:
    if( (NULL != gpu_task) && (PARSEC_GPU_TASK_TYPE_KERNEL == gpu_task->task_type) ) {
        parsec_gpu_task_update_pushout(es, gpu_task);
    }
    if( NULL != gpu_task ) {  /* This task has completed its execution */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tRetrieve data (if any) for %s", gpu_device->super.device_index, gpu_device->super.name,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    }
    /* Task is ready to move the data back to main memory */
    rc = parsec_device_progress_stream( gpu_device,
                                        es,
                                        gpu_device->exec_stream[1],
                                        parsec_device_kernel_pop,
                                        gpu_task, &progress_task );
    if( rc < 0 ) {
        if( (PARSEC_HOOK_RETURN_ERROR == rc) || (PARSEC_HOOK_RETURN_DISABLE == rc) ) {
            failure_status = rc;
            failed_batch = progress_task;
            progress_task = NULL;
            goto disable_gpu;
        }
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

 fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    {
        parsec_list_item_t *chain = parsec_lifo_detach_chain(&gpu_device->pending);
        if (NULL != chain) {
            chain = parsec_gpu_pending_chain_reverse(chain);
            parsec_heap_push_chain(&gpu_device->pending_heap, chain);
        }
    }
    gpu_task = (parsec_gpu_task_t*)parsec_heap_pop(&gpu_device->pending_heap);
    if( NULL != gpu_task ) {
        pop_null = 0;
        /* parsec_heap_pop() singletonizes the returned item before returning it. */
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch - 1;  /* force at least one tour */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tGet from shared queue %s", gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
        if( PARSEC_GPU_TASK_TYPE_D2D_COMPLETE == gpu_task->task_type ) {
            goto get_data_out_of_device;
        }
    } else {
        pop_null++;
        if( pop_null % 1024 == 1023 ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,  "GPU[%d:%s]:\tStill waiting for %d tasks to execute, but popped NULL the last %d times I tried to pop something...",
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
    /* Completing a GPU task can reserve one newly enabled successor in this
     * manager's private next_task slot. The manager does not return to normal
     * task selection while GPU work remains, so make that successor stealable.
     */
    rc = __parsec_schedule_flush_private(es);
    assert(PARSEC_SUCCESS == rc);
    gpu_device->super.executed_tasks++;
  remove_gpu_task:
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]: gpu_task %p freed",
                         gpu_device->super.device_index, gpu_device->super.name,
                         gpu_task);
    /* Release the GPU task */
    released_tasks = parsec_gpu_task_ring_release(gpu_task);
    assert(released_tasks > 0);

    rc = parsec_atomic_fetch_sub_int32(&(gpu_device->mutex), released_tasks);
    assert(rc >= released_tasks);
    if( released_tasks == rc ) {  /* I released the last outstanding task(s) */
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
    /* The scheduler currently treats device failure as fatal. Clean the batch
     * that observed the failure before propagating its terminal status.
     */
    if( NULL != failed_batch ) {
        /* Only a batch that reached the device pipeline owns device copies.
         * parsec_device_kernel_cleanout() reads data[i].data_out expecting the
         * device copy installed by stage-in; on a task that never got that far
         * it is still the host copy, and detaching that from this device index
         * would corrupt the original.
         */
        if( failed_batch_staged ) {
            parsec_device_kernel_cleanout_ring(gpu_device, failed_batch);
        }
        released_tasks = parsec_gpu_task_ring_release(failed_batch);
        /* The wrappers are gone, so they must also leave the manager's
         * outstanding count. Skipping this pins the count above zero forever:
         * every later caller would then observe a live manager, hand its task
         * to gpu_device->pending and return ASYNC, and that task would never be
         * picked up again because no thread can win the election.
         */
        rc = parsec_atomic_fetch_sub_int32(&(gpu_device->mutex), released_tasks);
        assert(rc >= released_tasks); (void)rc;
    }
    /* TODO: Recover the tasks in every pending FIFO, recorded event slot, and
     * gpu_device->pending before making DISABLE recoverable. The upper scheduler
     * currently treats this return as fatal, so only the failed batch is
     * quiesced and cleaned here.
     */
#if defined(PARSEC_PROF_TRACE)
    /* This thread is the manager and is leaving for good, so the interval it
     * opened on election has to be closed here as well as on the normal exit.
     */
    if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_end,
                                (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */
    parsec_warning("GPU[%d:%s]: Critical issue related to the GPU discovered. Giving up",
                   gpu_device->super.device_index, gpu_device->super.name);
    return failure_status;
}
