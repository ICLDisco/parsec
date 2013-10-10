/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague_internal.h"
#include <dague/devices/cuda/dev_cuda.h>
#include <dague/devices/device_malloc.h>
#include "profiling.h"
#include "execution_unit.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "lifo.h"

#if defined(DAGUE_PROF_TRACE)
#include "dbp.h"
#endif /* defined(DAGUE_PROF_TRACE) */

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
 * Returns: one of the dague_hook_return_t values
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
static inline dague_hook_return_t
gpu_kernel_scheduler( dague_execution_unit_t *eu_context,
                      dague_gpu_context_t    *this_task,
                      int which_gpu )
{
    gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_gpu_context_t *progress_task;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    gpu_device = (gpu_device_t*)dague_devices_get(which_gpu);

#if defined(DAGUE_PROF_TRACE)
    DAGUE_PROFILING_TRACE_FLAGS( eu_context->eu_profile,
				 DAGUE_PROF_FUNC_KEY_END(this_task->ec->dague_handle,
							 this_task->ec->function->function_id),
				 this_task->ec->function->key( this_task->ec->dague_handle, this_task->ec->locals),
				 this_task->ec->dague_handle->handle_id, NULL,
				 DAGUE_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(DAGUE_PROF_TRACE) */

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(gpu_device->pending), (dague_list_item_t*)this_task );
        return DAGUE_HOOK_RETURN_ASYNC;
    }

    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        DAGUE_PROFILING_TRACE( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                               (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                            {return DAGUE_HOOK_RETURN_DISABLE;} );

 check_in_deps:
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPush data for %s priority %d\n", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[0]),
                          gpu_kernel_push,
                          this_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    this_task = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tExecute %s priority %d\n", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[2+exec_stream]),
                          gpu_kernel_submit,
                          this_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    this_task = progress_task;

    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPop data for %s priority %d\n", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[1]),
                          gpu_kernel_pop,
                          this_task,
                          &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a succesfully completed task. However, it is not this_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        this_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    this_task = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == this_task );
    this_task = (dague_gpu_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tGet from shared queue %s priority %d\n", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority ));
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != this_task );
    DEBUG2(( "GPU[%1d]:\tComplete %s priority %d\n", gpu_device->cuda_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
             this_task->ec->priority ));
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    gpu_kernel_epilog( gpu_device, this_task );
    __dague_complete_execution( eu_context, this_task->ec );
    dague_device_load[gpu_device->super.device_index] -= dague_device_sweight[gpu_device->super.device_index];
    gpu_device->super.executed_tasks++;
    free( this_task );
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            DAGUE_PROFILING_TRACE( eu_context->eu_profile, dague_cuda_own_GPU_key_end,
                                   (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
        dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );

        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {return DAGUE_HOOK_RETURN_ASYNC;} );
        return DAGUE_HOOK_RETURN_ASYNC;
    }
    this_task = progress_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    return DAGUE_HOOK_RETURN_DISABLE;
}

#endif /* HAVE_CUDA */

