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

typedef struct dague_zgemm_args_s {
    dague_gpu_context_t super;
    dague_complex64_t alpha, beta;
    PLASMA_enum transA, transB;
    int M, N, K;
    int lda, ldb, ldc;
} dague_zgemm_args_t;

#include <dague/devices/cuda/cuda_scheduling.h>

static inline int
gpu_kernel_submit_zgemm( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task,
                         dague_gpu_exec_stream_t* gpu_stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    void *d_A, *d_B, *d_C;
    cublasStatus_t status;
#if defined(DAGUE_DEBUG_NOISIER)
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

    DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->cuda_index,
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
    gpu_task->super.pushout[flow_A] = 0;
    gpu_task->super.pushout[flow_B] = 0;
    gpu_task->super.pushout[flow_C] = pushout;
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
