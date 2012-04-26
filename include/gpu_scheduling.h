/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague.h"
#include "gpu_data.h"
#include "gpu_malloc.h"
#include "profiling.h"
#include "execution_unit.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "lifo.h"

extern dague_gpu_data_map_t dague_gpu_map;
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

#if !defined(DAGUE_GPU_STREAM_PER_TASK)

#if DAGUE_GPU_USE_PRIORITIES
static inline dague_list_item_t* dague_fifo_push_ordered( dague_list_t* fifo,
                                                          dague_list_item_t* elem )
{
    dague_ulist_push_sorted(fifo, elem, dague_execution_context_priority_comparator);
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_ulist_fifo_push
#endif

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
static inline
int gpu_kernel_scheduler( dague_execution_unit_t    *eu_context,
                          dague_execution_context_t *this_task,
                          int which_gpu )
{
    gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int rc, exec_stream = 0;
    
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
        dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_start, (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                            {return -2;} );

    DEBUG2(( "GPU:\tSchedule task %s with priority %d\n",
            this_task->function->name, this_task->priority ));
 check_in_deps:
    if( NULL != this_task ) {
        if( NULL != gpu_device->in_array[gpu_device->in_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in, (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_ulist_is_empty(gpu_device->fifo_pending_in) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in, (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_in);
            }
        }
    } else {
        if( NULL == gpu_device->in_array[gpu_device->in_submit] ) {
            this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_in);
        }
    }
    if( NULL != this_task ) {
        DEBUG3(( "GPU:\tWork on %s with priority %d\n",
                 this_task->function->name, this_task->priority ));
        assert( NULL == gpu_device->in_array[gpu_device->in_submit] );
        rc = gpu_kernel_push( gpu_device, this_task, gpu_device->streams[0] );
        if( 0 > rc ) {
            if( -1 == rc )
                goto disable_gpu;  /* Critical issue */
            /* No more room on the GPU. Push the task back on the queue and check the completion queue. */
            /* TODO maybe push the task into another queue to buy us some time */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in, (dague_list_item_t*)this_task);
            DEBUG2(( "GPU:\tReschedule %s with priority %d: no room available on the GPU for data\n",
                     this_task->function->name, this_task->priority ));
        } else {
            /**
             * Do not skip the cuda event generation. The problem is that some of the inputs
             * might be in the pipe of being transferred to the GPU. If we activate this task
             * too early, it might get executed before the data is available on the GPU.
             * Obviously, this lead to incorrect results.
             */
            /*if( 0 == rc ) goto exec_task;*/  /* No data to be moved for this task */
            rc = cuEventRecord( gpu_device->in_array_events[gpu_device->in_submit], gpu_device->streams[0] );
            gpu_device->in_array[gpu_device->in_submit] = this_task;
            gpu_device->in_submit = (gpu_device->in_submit + 1) % gpu_device->max_in_tasks;
        }
        this_task = NULL;
    }
    assert( NULL == this_task );
    if( NULL != gpu_device->in_array[gpu_device->in_waiting] ) {
        rc = cuEventQuery(gpu_device->in_array_events[gpu_device->in_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_exec_completion;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG3(("GPU Completion of GPU Request number %d\n", gpu_device->in_array_events[gpu_device->in_waiting]));
            this_task = gpu_device->in_array[gpu_device->in_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
                dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_end,
                                       (unsigned long)this_task, this_task->dague_object->object_id,
                                       NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->in_array[gpu_device->in_waiting] = NULL;
            gpu_device->in_waiting = (gpu_device->in_waiting + 1) % gpu_device->max_in_tasks;
            goto exec_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }
 exec_task:
    if( NULL != this_task ) {
        if( NULL != gpu_device->exec_array[gpu_device->exec_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec, (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_ulist_is_empty(gpu_device->fifo_pending_exec) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec, (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_exec);
            }
        }
    } else {
        if( NULL == gpu_device->exec_array[gpu_device->exec_submit] ) {
            this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_exec);
        }
    }
    if( NULL != this_task ) {
        assert( NULL == gpu_device->exec_array[gpu_device->exec_submit] );
        /* Choose an exec_stream */
        exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams);
        DEBUG(( "GPU:\tExecute %s with priority %d\n",
                this_task->function->name, this_task->priority ));
        rc = gpu_kernel_submit( gpu_device, this_task, gpu_device->streams[2 + exec_stream] );
        DEBUG3(("GPU:\tRequest number %d/%d\n", gpu_device->exec_array_events[gpu_device->exec_submit], gpu_device->streams[2 + exec_stream]));
        gpu_device->exec_array[gpu_device->exec_submit] = this_task;
        this_task = NULL;
        if( 0 != rc )  goto disable_gpu;
        rc = cuEventRecord( gpu_device->exec_array_events[gpu_device->exec_submit], gpu_device->streams[2 + exec_stream] );
        gpu_device->exec_submit = (gpu_device->exec_submit + 1) % gpu_device->max_exec_tasks;
    }
 check_exec_completion:
    assert( NULL == this_task );
    if( NULL != gpu_device->exec_array[gpu_device->exec_waiting] ) {
        rc = cuEventQuery(gpu_device->exec_array_events[gpu_device->exec_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_out_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG3(("GPU:\tCompletion of GPU Request number %d\n", gpu_device->exec_array_events[gpu_device->exec_waiting]));
            this_task = gpu_device->exec_array[gpu_device->exec_waiting];
#if defined(DAGUE_PROF_TRACE)
            gpu_kernel_profile( gpu_device, this_task, dague_gpu_map.desc);
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->exec_array[gpu_device->exec_waiting] = NULL;
            gpu_device->exec_waiting = (gpu_device->exec_waiting + 1) % gpu_device->max_exec_tasks;
            goto out_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }
 out_task:
    if( NULL != this_task ) {
        if( NULL != gpu_device->out_array[gpu_device->out_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out, (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_ulist_is_empty(gpu_device->fifo_pending_out) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out, (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_out);
            }
        }
    } else {
        if( NULL == gpu_device->out_array[gpu_device->out_submit] ) {
            this_task = (dague_execution_context_t*)dague_ulist_fifo_pop(gpu_device->fifo_pending_out);
        }
    }
    if( NULL != this_task ) {
        assert( NULL == gpu_device->out_array[gpu_device->out_submit] );
        rc = gpu_kernel_pop( gpu_device, this_task, gpu_device->streams[1] );
        DEBUG3(("GPU:\tRequest number %d/%d\n", gpu_device->out_array_events[gpu_device->out_submit], gpu_device->streams[1]));
        if( 0 == rc ) goto complete_task;  /* no data to be moved */
        gpu_device->out_array[gpu_device->out_submit] = this_task;
        this_task = NULL;
        if( 0 > rc ) goto disable_gpu;
        rc = cuEventRecord( gpu_device->out_array_events[gpu_device->out_submit], gpu_device->streams[1] );
        gpu_device->out_submit = (gpu_device->out_submit + 1) % gpu_device->max_out_tasks;
    }
 check_out_deps:
    assert( NULL == this_task );
    if( NULL != gpu_device->out_array[gpu_device->out_waiting] ) {
        rc = cuEventQuery(gpu_device->out_array_events[gpu_device->out_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_in_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG3(("GPU:\tCompletion of GPU Request number %d\n", gpu_device->out_array_events[gpu_device->out_waiting]));
            this_task = gpu_device->out_array[gpu_device->out_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
                dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_end,
                                       (unsigned long)this_task, this_task->dague_object->object_id,
                                       NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->out_array[gpu_device->out_waiting] = NULL;
            gpu_device->out_waiting = (gpu_device->out_waiting + 1) % gpu_device->max_out_tasks;
            goto complete_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }

 fetch_task_from_shared_queue:
    assert( NULL == this_task );
    this_task = (dague_execution_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DEBUG2(( "GPU:\tAdd %s with priority %d\n",
                this_task->function->name, this_task->priority ));
    }
    goto check_in_deps;

 complete_task:
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    gpu_kernel_epilog( gpu_device, this_task );
    dague_complete_execution( eu_context, this_task );
    gpu_device->executed_tasks++;
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        assert( (NULL == gpu_device->in_array[gpu_device->in_waiting]) &&
                (NULL == gpu_device->exec_array[gpu_device->exec_waiting]) &&
                (NULL == gpu_device->out_array[gpu_device->out_waiting]) );
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
    this_task = NULL;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    exit(-20);
    return -2;
}
#else /* !defined(DAGUE_GPU_STREAM_PER_TASK)*/
#error "Not implemented"
#endif

#endif /* HAVE_CUDA */

