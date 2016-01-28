/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague/dague_internal.h"
#include "dague/devices/cuda/dev_cuda.h"
#include "dague/profiling.h"
#include "dague/execution_unit.h"
#include "dague/scheduling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "dague/class/lifo.h"

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

static inline
int gpu_kernel_submit( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream);

/**
 * Try to execute a kernel on a GPU.
 *
 * Returns: one of the dague_hook_return_t values
 */

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns:
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
static inline int
gpu_kernel_push( gpu_device_t            *gpu_device,
                 dague_gpu_context_t     *gpu_task,
                 dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret = 0;
    int space_needed = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_data_t              *original;
    dague_data_copy_t         *data, *local;
    const dague_flow_t        *flow;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

        this_task->data[i].data_out = NULL;
        data = this_task->data[i].data_in;
        original = data->original;
        flow = this_task->function->in[i];
        if(NULL == flow) {
            flow = this_task->function->out[i];
        }
        if( NULL != (local = dague_data_get_copy(original, gpu_device->super.device_index)) ) {
            if ( (flow->flow_flags & FLOW_ACCESS_WRITE) && local->readers > 0 ) {
                return -86;
            }
            this_task->data[i].data_out = local;

            /* Check the most up2date version of the data */
            if( data->device_index != gpu_device->super.device_index ) {
                if(data->version <= local->version) {
                    if(data->version == local->version) continue;
                    /* Trouble: there are two versions of this data coexisting in same
                     * time, one using a read-only path and one that has been updated.
                     * We don't handle this case yet!
                     * TODO:
                     */
                    assert(0);
                }
            }
            continue;  /* space available on the device */
        }

        /* If the data is needed as an input load it up */
        if(this_task->function->in[i]->flow_flags & FLOW_ACCESS_READ)
            space_needed++;
    }

    if( 0 != space_needed ) { /* Try to reserve enough room for all data */
        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   space_needed );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_stream->profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if(NULL == this_task->function->in[i]) continue;
        assert( NULL != dague_data_copy_get_ptr(this_task->data[i].data_in) );

        DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                              "GPU[%1d]:\tIN  Data of %s <%x> on GPU\n",
                              gpu_device->cuda_index, this_task->function->in[i]->name,
                              this_task->data[i].data_out->original->key));
        ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[i]->flow_flags,
                                       &(this_task->data[i]), gpu_task, gpu_stream );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

  release_and_return_error:
    return ret;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
static inline int
gpu_kernel_pop( gpu_device_t            *gpu_device,
                dague_gpu_context_t     *gpu_task,
                dague_gpu_exec_stream_t *gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    const dague_flow_t        *flow;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < 1; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            original = gpu_copy->original;
            status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                      gpu_copy->device_private,
                                      original->nb_elts,
                                      cudaMemcpyDeviceToHost,
                                      gpu_stream->cuda_stream );
            DAGUE_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                    { dague_warning("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                               gpu_copy->device_private, original->device_copies[0]->device_private);
                                        return_code = -2;
                                        goto release_and_return_error;} );
        }
        return return_code;
    }

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        flow = this_task->function->in[i];
        if(NULL == flow)
            flow = this_task->function->out[i];

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(original == this_task->data[i].data_in->original);
        if( flow->flow_flags & FLOW_ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
                continue;  /* done with this element, go for the next one */
            }
        }
        if( flow->flow_flags & FLOW_ACCESS_WRITE ) {
            gpu_copy->version++;  /* on to the next version */
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );
            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                  "GPU[%1d]:\tOUT Data of %s\n", gpu_device->cuda_index, flow->name));
            if( gpu_task->pushout[i] ) {
                original = gpu_copy->original;
                DAGUE_OUTPUT_VERBOSE((2, dague_cuda_output_stream,
                                      "GPU:\tMove D2H data <%s:%x> from GPU %d %p -> %p requested\n",
                                      this_task->function->in[i]->name, original->key, gpu_device->cuda_index,
                                      (void*)gpu_copy->device_private, original->device_copies[0]->device_private));
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_stream->profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* Move the data back into main memory */
                status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                          gpu_copy->device_private,
                                          original->nb_elts,
                                          cudaMemcpyDeviceToHost,
                                          gpu_stream->cuda_stream );
                DAGUE_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                        { dague_warning("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                                   gpu_copy->device_private, original->device_copies[0]->device_private);
                                            return_code = -2;
                                            goto release_and_return_error;} );
                gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            }
        }
    }

  release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
static inline int
gpu_kernel_epilog( gpu_device_t        *gpu_device,
                   dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    dague_data_t              *original;
    const dague_flow_t        *flow;
    int i;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        flow = this_task->function->out[i];
        if(NULL == flow) continue;

        gpu_copy = this_task->data[flow->flow_index].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        if( !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[flow->flow_index].data_out = cpu_copy;
            continue;
        }

        /* There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         */
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        /* TODO: make sure no readers are working on the CPU version */
        cpu_copy->version = gpu_copy->version;

        /* Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[flow->flow_index].data_out = cpu_copy;

        if( gpu_task->pushout[i] ) {
            dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                  "CUDA copy %p [ref_count %d] moved to the read LRU in %s\n",
                                  gpu_copy, gpu_copy->super.super.obj_reference_count, __func__));
        } else {
            dague_ulist_fifo_push(&gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

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
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_gpu_context_t *progress_task, *out_task_push, *out_task_submit, *out_task_pop;
#if defined(DAGUE_DEBUG_NOISIER)
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

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        DAGUE_PROFILING_TRACE( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                               (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(gpu_kernel_scheduler) cudaSetDevice ", status,
                            {return DAGUE_HOOK_RETURN_DISABLE;} );

 check_in_deps:
    if( NULL != this_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tUpload data (if any) for %s priority %d", gpu_device->super.device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority );
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
    out_task_push = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != this_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tExecute %s priority %d", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority );
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
    out_task_submit = progress_task;

    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != this_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tRetrieve data (if any) for %s priority %d", gpu_device->super.device_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority );
    }
    if (out_task_submit == NULL && out_task_push == NULL) {
        this_task = dague_gpu_create_W2R_task(gpu_device, eu_context);
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
    out_task_pop = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == this_task );
    if (out_task_submit == NULL && out_task_pop == NULL) {
        dague_gpu_sort_pending_list(gpu_device);
    }
    this_task = (dague_gpu_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tGet from shared queue %s priority %d", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                 this_task->ec->priority );
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != this_task );
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tComplete %s priority %d", gpu_device->cuda_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
             this_task->ec->priority );
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    if (this_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        dague_gpu_W2R_task_fini(gpu_device, this_task, eu_context);
        this_task = progress_task;
        goto fetch_task_from_shared_queue;
    }
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
