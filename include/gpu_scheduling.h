/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague_internal.h"
#include "gpu_data.h"
#include "gpu_malloc.h"
#include "profiling.h"
#include "execution_unit.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "lifo.h"

extern volatile uint32_t dague_cpu_counter;
extern gpu_device_t** gpu_enabled_devices;

/**
 * Define functions names
 */
#ifndef KERNEL_NAME
#error "KERNEL_NAME must be defined before to include this file"
#endif

#define GENERATE_NAME_v2( _func_, _kernel_ ) _func_##_##_kernel_
#define GENERATE_NAME( _func_, _kernel_ ) GENERATE_NAME_v2( _func_, _kernel_ )

#define gpu_kernel_push      GENERATE_NAME( gpu_kernel_push     , KERNEL_NAME )
#define gpu_kernel_submit    GENERATE_NAME( gpu_kernel_submit   , KERNEL_NAME )
#define gpu_kernel_pop       GENERATE_NAME( gpu_kernel_pop      , KERNEL_NAME )
#define gpu_kernel_epilog    GENERATE_NAME( gpu_kernel_epilog   , KERNEL_NAME )
#define gpu_kernel_profile   GENERATE_NAME( gpu_kernel_profile  , KERNEL_NAME )
#define gpu_kernel_scheduler GENERATE_NAME( gpu_kernel_scheduler, KERNEL_NAME )

/**
 * Try to execute a kernel on a GPU.
 *
 * Returns:
 *  0 - if the kernel should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the kernel is scheduled to be executed on a GPU.
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
static inline
int gpu_kernel_scheduler( dague_execution_unit_t *eu_context,
                          dague_gpu_context_t    *this_task,
                          int which_gpu )
{
    gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_gpu_context_t *next_task;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    gpu_device = gpu_enabled_devices[which_gpu];

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(gpu_device->pending), (dague_list_item_t*)this_task );
        return -1;
    }

    /**
     * There might be a small race condition here, between the moment when the previous
     * owner of the GPU context release it, and the moment where I can get it.
     */
    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                               (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                            {return -2;} );

 check_in_deps:
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPush data for %s priority %d\n", gpu_device->device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[0]),
                          gpu_kernel_push,
                          this_task, &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    this_task = next_task;

    /* Stage-in completed for this Task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tExecute %s priority %d\n", gpu_device->device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[2+exec_stream]),
                          gpu_kernel_submit,
                          this_task, &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    } 
    this_task = next_task;

    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPop data for %s priority %d\n", gpu_device->device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[1]),
                          gpu_kernel_pop,
                          this_task,
                          &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    if( NULL != next_task ) {
        /* We have a succesfully completed task. However, it is not this_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the next_task.
         */
        this_task = next_task;
        next_task = NULL;
        goto complete_task;
    }
    this_task = next_task;

 fetch_task_from_shared_queue:
    assert( NULL == this_task );
    this_task = (dague_gpu_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tGet from shared queue %s priority %d\n", gpu_device->device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != this_task );
    DEBUG2(( "GPU[%1d]:\tComplete %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
             this_task->ec->priority ));
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    gpu_kernel_epilog( gpu_device, this_task );
    __dague_complete_execution( eu_context, this_task->ec );
    device_load[gpu_device->device_index+1] -= device_weight[gpu_device->device_index+1];
    gpu_device->executed_tasks++;
    free( this_task );
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_end,
                                   (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
        dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );

        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {return -1;} );
        return -1;
    }
    this_task = next_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    exit(-20);
    return -2;
}

#endif /* HAVE_CUDA */

