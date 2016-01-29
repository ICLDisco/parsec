/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include <dague_config.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <core_blas.h>
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif
#include "dague.h"
#include "dague/execution_unit.h"
#include "dague/class/fifo.h"
#include "data_dist/matrix/matrix.h"
#include "dague/data_internal.h"
#include "dague/utils/output.h"
#include "cuda_zgemm.h"
#include <cublas.h>

#define flow_A  1
#define flow_B  2
#define flow_C  0

#define KERNEL_NAME zgemm
#if CUDA_VERSION < 4000 || 1
typedef void (*cublas_zgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                 dague_complex64_t alpha, dague_complex64_t *d_A, int lda,
                                 dague_complex64_t *d_B, int ldb,
                                 dague_complex64_t beta,  dague_complex64_t *d_C, int ldc
                                 );
#else
typedef cublas_status_t (*cublas_zgemm_t) ( cublas_handle_t h,
                                            char TRANSA, char TRANSB, int m, int n, int k,
                                            dague_complex64_t alpha, dague_complex64_t *d_A, int lda,
                                            dague_complex64_t *d_B, int ldb,
                                            dague_complex64_t beta,  dague_complex64_t *d_C, int ldc
                                            );
#endif

extern int dague_cuda_output_stream;

#define FORCE_LINK_STATIC_SYMBOL(x) void* __ ## x ## _fp =(void*)&x;
FORCE_LINK_STATIC_SYMBOL(cublasZgemm)

static inline
int gpu_kernel_push_zgemm( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           dague_gpu_exec_stream_t* gpu_stream);

static inline
int gpu_kernel_submit_zgemm( gpu_device_t* gpu_device,
                             dague_gpu_context_t* this_task,
                             dague_gpu_exec_stream_t* gpu_stream);

static inline
int gpu_kernel_pop_zgemm( gpu_device_t* gpu_device,
                          dague_gpu_context_t* this_task,
                          dague_gpu_exec_stream_t* gpu_stream);

static inline
int  gpu_kernel_epilog_zgemm( gpu_device_t* gpu_device,
                              dague_gpu_context_t* this_task );

typedef struct dague_zgemm_args_s {
    dague_gpu_context_t super;
    int pushout;
    dague_complex64_t alpha, beta;
    PLASMA_enum transA, transB;
    int M, N, K;
    int lda, ldb, ldc;
} dague_zgemm_args_t;

#include <dague/devices/cuda/cuda_scheduling.h>

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
gpu_kernel_push_zgemm( gpu_device_t            *gpu_device,
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


static inline int
gpu_kernel_submit_zgemm( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task,
                         dague_gpu_exec_stream_t* gpu_stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    void *d_A, *d_B, *d_C;
    cublasStatus_t status;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif

    cublas_zgemm_t cublas_fnzgemm = (cublas_zgemm_t) this_task->function->incarnations[gpu_device->cuda_index].dyld_fn;
    assert( NULL != cublas_fnzgemm );

    assert(this_task->data[flow_A].data_out->device_index == gpu_device->super.device_index);
    d_A = this_task->data[flow_A].data_out->device_private;
    assert(this_task->data[flow_B].data_out->device_index == gpu_device->super.device_index);
    d_B = this_task->data[flow_B].data_out->device_private;
    assert(this_task->data[flow_C].data_out->device_index == gpu_device->super.device_index);
    /*assert( DATA_COHERENCY_OWNED == this_task->data[flow_C].data_out->coherency_state );*/
    d_C = this_task->data[flow_C].data_out->device_private;

    DEBUG2( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->cuda_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority );

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_stream->profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    status = CUBLAS_STATUS_SUCCESS;
#if (CUDA_VERSION < 4000) || 1 /* todo: always use legacy cublas until we understand how to get the cublas_handle in API v5 */
    cublasSetKernelStream( gpu_stream->cuda_stream );
    cublas_fnzgemm( lapack_const(args->transA), lapack_const(args->transB),
                    args->M, args->N, args->K,
                    args->alpha, (dague_complex64_t*)d_A, args->lda,
                    (dague_complex64_t*)d_B, args->ldb,
                    args->beta,  (dague_complex64_t*)d_C, args->ldc );
    status = cublasGetError();
#else
    {
        cudaStream_t current_stream;
        cublasHandle_t handle = cublasGetCurrentCtx(); /* todo: available in cuda API 4 only */
        cublasGetStream_v2 ( handle, &current_stream );
        cublasSetStream_v2 ( handle, &gpu_stream->cuda_srtream );
        status =
            cublas_fnzgemm( handle,
                            lapack_const(args->transA), lapack_const(args->transB),
                            args->M, args->N, args->K,
                            args->alpha, (dague_complex64_t*)d_A, args->lda,
                            (dague_complex64_t*)d_B, args->ldb,
                            args->beta,  (dague_complex64_t*)d_C, args->ldc );
        cublasSetStream_v2 ( handle, &current_stream );
    }
#endif /* CUDA_VERSION < 4000 */
    DAGUE_CUDA_CHECK_ERROR( "cublasZgemm ", status,
                            {return -1;} );
    return 0;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
static inline int
gpu_kernel_pop_zgemm( gpu_device_t        *gpu_device,
                      dague_gpu_context_t *gpu_task,
                      dague_gpu_exec_stream_t* gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
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
                                    { WARNING("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
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
            if( args->pushout ) {  /* n == (k + 1) */
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
                                        { WARNING("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
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
gpu_kernel_epilog_zgemm( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if(NULL == this_task->function->out[i]) continue;

        gpu_copy = this_task->data[this_task->function->out[i]->flow_index].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        if( !(this_task->function->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[this_task->function->out[i]->flow_index].data_out = cpu_copy;
            continue;
        }

        /* There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         */
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state =  DATA_COHERENCY_SHARED;
        /* TODO: make sure no readers are working on the CPU version */
        cpu_copy->version = gpu_copy->version;

        /* Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[this_task->function->out[i]->flow_index].data_out = cpu_copy;

        if( args->pushout ) {  /* n == (k  + 1) */
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
 * Try to execute a GEMM on a GPU.
 *
 * Returns:
 *  0 - if the GEMM should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the GEMM is scheduled to be executed on a GPU.
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int gpu_zgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int pushout, int nb,
               PLASMA_enum transA, PLASMA_enum transB,
               int M, int N, int K,
               dague_complex64_t alpha, int lda,
               int ldb,
               dague_complex64_t beta,  int ldc )
{
    int i, dev_index, data_index = 0;
    dague_zgemm_args_t *gpu_task;
    dague_handle_t* handle = this_task->dague_handle;

    /* Step one: Find the first data in WRITE mode */
    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if( (NULL == this_task->function->out[i]) ||
            (this_task->function->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            data_index = this_task->function->out[i]->flow_index;
            break;
        }
    }

    /* Which device is the owner of the data */
    dev_index = this_task->data[data_index].data_in->original->owner_device;

    /* 0 is CPU, and 1 is recursive device */
    if( dev_index <= 1 ) {  /* this is the first time we see this tile.
                             * Let's decide which GPU will work on it. */
        int best_index = 0;  /* default value: first CPU device */
        float weight, best_weight = dague_device_load[0] + nb * dague_device_sweight[0];
        for( dev_index = 2; dev_index < dague_devices_enabled(); dev_index++ ) {
            /* Skip the device if it is not configured */
            if(!(handle->devices_mask & (1 << dev_index))) continue;
            weight = dague_device_load[dev_index] + nb * dague_device_sweight[dev_index];
            if( best_weight > weight ) {
                best_index = dev_index;
                best_weight = weight;
            }
        }
        dague_device_load[best_index] += nb * dague_device_sweight[best_index];
        if( best_index == 0 ) {
            return DAGUE_HOOK_RETURN_NEXT;  /* Fall back */
        }
        assert( best_index != 1 );
        dev_index = best_index;
    }

    gpu_task = (dague_zgemm_args_t*)malloc(sizeof(dague_zgemm_args_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->super.ec = this_task;
    gpu_task->super.task_type = 0;
    gpu_task->pushout  = pushout;
    gpu_task->alpha    = alpha;
    gpu_task->beta     = beta;
    gpu_task->transA   = transA;
    gpu_task->transB   = transB;
    gpu_task->M        = M;
    gpu_task->N        = N;
    gpu_task->K        = K;
    gpu_task->lda      = lda;
    gpu_task->ldb      = ldb;
    gpu_task->ldc      = ldc;

    return gpu_kernel_scheduler_zgemm( eu_context, (dague_gpu_context_t*)gpu_task, dev_index );
}
