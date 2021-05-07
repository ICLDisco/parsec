/*
 *
 * Copyright (c) 2021      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/device_gpu.h"
#include "parsec/utils/mca_param.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/argv.h"
#include "parsec/parsec_internal.h"

void* parsec_gpu_pop_workspace(parsec_device_gpu_module_t* gpu_device,
                               parsec_gpu_exec_stream_t* gpu_stream, size_t size)
{
    parsec_device_level_zero_module_t *level_zero_device = (parsec_device_level_zero_module_t*)gpu_device;
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (parsec_gpu_workspace_t *)malloc(sizeof(parsec_gpu_workspace_t));
        gpu_stream->workspace->total_workspace = PARSEC_GPU_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = PARSEC_GPU_MAX_WORKSPACE - 1;

        for( int i = 0; i < PARSEC_GPU_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
            PARSEC_DEBUG_VERBOSE(2, parsec_level_zero_output_stream,
                                 "GPU[%s] Succeeded Allocating workspace %d (device_ptr %p)",
                                 gpu_device->super.name,
                                 i, gpu_stream->workspace->workspace[i]);
#if defined(PARSEC_PROF_TRACE)
            if((parsec_level_zero_trackable_events & PARSEC_PROFILE_LEVEL_ZERO_TRACK_MEM_USE) &&
               (level_zero_device->exec_stream[0].super.prof_event_track_enable ||
                level_zero_device->exec_stream[1].super.prof_event_track_enable)) {
                parsec_profiling_trace_flags(gpu_stream->profiling,
                                             parsec_level_zero_allocate_memory_key, (int64_t)gpu_stream->workspace->workspace[i], level_zero_device->level_zero_index,
                                             &size, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
            }
#endif
        }
    }
    assert (gpu_stream->workspace->stack_head >= 0);
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE) */
    return work;
}

int parsec_gpu_push_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_GPU_MAX_WORKSPACE);
#endif /* !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE) */
    return 0;
}

int parsec_gpu_free_workspace(parsec_device_gpu_module_t * gpu_device)
{
    parsec_device_level_zero_module_t *level_zero_device = (parsec_device_level_zero_module_t*)gpu_device;
    (void)gpu_device;
#if !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *gpu_stream = &(level_zero_device->exec_stream[i].super);
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
#if defined(PARSEC_PROF_TRACE)
                if((parsec_level_zero_trackable_events & PARSEC_PROFILE_LEVEL_ZERO_TRACK_MEM_USE) &&
                   (level_zero_device->exec_stream[0].super.prof_event_track_enable ||
                    level_zero_device->exec_stream[1].super.prof_event_track_enable)) {
                    parsec_profiling_trace_flags(gpu_stream->profiling,
                                                 parsec_level_zero_allocate_memory_key, (int64_t)gpu_stream->workspace->workspace[i], level_zero_device->level_zero_index,
                                                 NULL, PARSEC_PROFILING_EVENT_COUNTER);
                }
#endif
                PARSEC_DEBUG_VERBOSE(2, parsec_level_zero_output_stream,
                                     "GPU[%s] Release workspace %d (device_ptr %p)",
                                     gpu_device->super.name,
                                     j, gpu_stream->workspace->workspace[j]);
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(PARSEC_GPU_LEVEL_ZERO_ALLOC_PER_TILE) */
    return 0;
}
