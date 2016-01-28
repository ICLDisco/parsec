/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _pingpong_gpu_h_
#define _pingpong_gpu_h_

#include "dague.h"
#include "dague/data_distribution.h"
#include "dague/data.h"
#include "dague/devices/cuda/dev_cuda.h"
#include "dague/utils/output.h"
#include "dague/class/fifo.h"
#include "dague/scheduling.h"

#define flow_T 0

#define KERNEL_NAME bandwidth
#include <dague/devices/cuda/cuda_scheduling.h>

static int
gpu_kernel_submit_bandwidth( gpu_device_t            *gpu_device,
                             dague_gpu_context_t     *gpu_task,
                             dague_gpu_exec_stream_t *gpu_stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);
    (void)gpu_device; (void)gpu_stream; (void)this_task;
    return 0;
}

static inline
int bandwidth_cuda(dague_execution_unit_t* eu_context,
                   dague_execution_context_t* this_task)
{
    dague_gpu_context_t* gpu_task;

    gpu_task = (dague_gpu_context_t*)malloc(sizeof(dague_gpu_context_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->ec = this_task;
    gpu_task->pushout[flow_T] = 1;

    return gpu_kernel_scheduler_bandwidth( eu_context, gpu_task, 1 );
}

#endif /* _bandwidth_gpu_h_ */
