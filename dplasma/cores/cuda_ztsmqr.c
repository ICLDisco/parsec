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
#include <cublas.h>
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif
#include "dague.h"
#include "dague/execution_unit.h"
#include "dague/class/fifo.h"
#include "data_dist/matrix/matrix.h"
#include "dague/data_internal.h"
#include "dague/utils/output.h"
#include "cuda_ztsmqr.h"
#include "dplasma_zcores.h"

#define flow_A1  0
#define flow_A2  1
#define flow_V   2
#define flow_T   3

#define KERNEL_NAME ztsmqr
#if CUDA_VERSION < 4000 || 1
typedef void (*cublas_ztsmqr_t) ( PLASMA_enum side, PLASMA_enum trans,
                                  int M1, int N1, int M2, int N2, int K, int IB,
                                  dague_complex64_t *A1, int LDA1,
                                  dague_complex64_t *A2, int LDA2,
                                  dague_complex64_t *V, int LDV,
                                  dague_complex64_t *T, int LDT,
                                  dague_complex64_t *WORK,  int LDWORK,
                                  dague_complex64_t *WORKC, int LDWORKC,
                                  cudaStream_t stream );
#else
typedef cublas_status_t (*cublas_ztsmqr_t) ( PLASMA_enum side, PLASMA_enum trans,
                                             int M1, int N1, int M2, int N2, int K, int IB,
                                             dague_complex64_t *A1, int LDA1,
                                             dague_complex64_t *A2, int LDA2,
                                             dague_complex64_t *V, int LDV,
                                             dague_complex64_t *T, int LDT,
                                             dague_complex64_t *WORK,  int LDWORK,
                                             dague_complex64_t *WORKC, int LDWORKC,
                                             cublas_handle_t h );
#endif

extern int dague_cuda_output_stream;

#define FORCE_LINK_STATIC_SYMBOL(x) void* __ ## x ## _fp =(void*)&x;
FORCE_LINK_STATIC_SYMBOL(dplasma_cuda_ztsmqr)

typedef struct dague_ztsmqr_args_s {
    dague_gpu_context_t super;
    int pushout_A1, pushout_A2;
    PLASMA_enum side, trans;
    int M1, N1, M2, N2, K, IB;
    int lda1, lda2, ldv, ldt;
} dague_ztsmqr_args_t;

#include <dague/devices/cuda/cuda_scheduling.h>

static inline int
gpu_kernel_submit_ztsmqr( gpu_device_t            *gpu_device,
                          dague_gpu_context_t     *gpu_task,
                          dague_gpu_exec_stream_t *gpu_stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_ztsmqr_args_t        *args = (dague_ztsmqr_args_t*)gpu_task;
    void *d_A1, *d_A2, *d_V, *d_T, *WORK, *WORKC;
    cublasStatus_t status;
#if defined(DAGUE_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    int Wn, Wld;

    cublas_ztsmqr_t cuda_fnztsmqr = (cublas_ztsmqr_t) this_task->function->incarnations[gpu_device->cuda_index].dyld_fn;
    assert( NULL != cuda_fnztsmqr );

    assert(this_task->data[flow_A1].data_out->device_index == gpu_device->super.device_index);
    d_A1 = this_task->data[flow_A1].data_out->device_private;
    assert(this_task->data[flow_A2].data_out->device_index == gpu_device->super.device_index);
    d_A2 = this_task->data[flow_A2].data_out->device_private;
    assert(this_task->data[flow_V].data_out->device_index == gpu_device->super.device_index);
    d_V  = this_task->data[flow_V].data_out->device_private;
    assert(this_task->data[flow_T].data_out->device_index == gpu_device->super.device_index);
    d_T  = this_task->data[flow_T].data_out->device_private;

    if ( args->side == PlasmaLeft ) {
        Wn = args->N1;
        Wld = args->IB;
    }
    else {
        Wn = args->IB;
        Wld = args->M1;
    }

    WORK  = dague_gpu_pop_workspace(gpu_device, gpu_stream, Wn * Wld * sizeof(dague_complex64_t));
    WORKC = dague_gpu_pop_workspace(gpu_device, gpu_stream, args->M2 * args->IB * sizeof(dague_complex64_t));

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
    cuda_fnztsmqr( args->side, args->trans,
                   args->M1, args->N1, args->M2, args->N2, args->K, args->IB,
                   (dague_complex64_t*)d_A1, args->lda1,
                   (dague_complex64_t*)d_A2, args->lda2,
                   (dague_complex64_t*)d_V,  args->ldv,
                   (dague_complex64_t*)d_T,  args->ldt,
                   (dague_complex64_t*)WORK,  Wld,
                   (dague_complex64_t*)WORKC, args->M2,
                   gpu_stream->cuda_stream );
    status = cublasGetError();
#else
    {
        cudaStream_t current_stream;
        cublasHandle_t handle = cublasGetCurrentCtx(); /* todo: available in cuda API 4 only */
        cublasGetStream_v2 ( handle, &current_stream );
        cublasSetStream_v2 ( handle, &gpu_stream->cuda_stream );
        status =
            cuda_fnztsmqr( args->side, args->trans,
                           args->M1, args->N1, args->M2, args->N2, args->K, args->IB,
                           (dague_complex64_t*)d_A1, args->lda1,
                           (dague_complex64_t*)d_A2, args->lda2,
                           (dague_complex64_t*)d_V,  args->ldv,
                           (dague_complex64_t*)d_T,  args->ldt,
                           (dague_complex64_t*)WORK,  Wld,
                           (dague_complex64_t*)WORKC, args->M2,
                           handle );
        cublasSetStream_v2 ( handle, &current_stream );
    }
#endif /* CUDA_VERSION < 4000 */
    DAGUE_CUDA_CHECK_ERROR( "dplasma_cuda_ztsmqr", status,
                            {return -1;} );

    dague_gpu_push_workspace(gpu_device, gpu_stream);
    dague_gpu_push_workspace(gpu_device, gpu_stream);

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
gpu_kernel_pop_ztsmqr( gpu_device_t        *gpu_device,
                       dague_gpu_context_t *gpu_task,
                       dague_gpu_exec_stream_t* gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_ztsmqr_args_t       *args = (dague_ztsmqr_args_t*)gpu_task;
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
            if( (args->pushout_A1 && (i == flow_A1)) || (args->pushout_A2 && (i == flow_A2)) ) {
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
gpu_kernel_epilog_ztsmqr( gpu_device_t        *gpu_device,
                          dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_ztsmqr_args_t       *args = (dague_ztsmqr_args_t*)gpu_task;
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

        if( (args->pushout_A1 && (i == flow_A1)) || (args->pushout_A2 && (i == flow_A2)) ) {
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
int gpu_ztsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* this_task,
                int pushout_A1, int pushout_A2, int m, int n, int k,
                PLASMA_enum side, PLASMA_enum trans,
                int M1, int N1, int M2, int N2, int K, int IB,
                int LDA1, int LDA2, int LDV, int LDT )
{
    int dev_index;
    dague_ztsmqr_args_t *gpu_task;
    (void)m; (void)n; (void)k;

    dev_index = this_task->data[flow_A1].data_in->original->owner_device;

#if defined(HAVE_MPI) || 1 /* defined(GPU_STATIC) */
    /**
     * Columns are distributed in a round-robin fashion
     */
    if ( dev_index <= 0 ) {
        dev_index = n % (dague_devices_enabled()-2)+2;
    }
#else
    /**
     * Only the tsmqr task located in (k, k+1, n) can choose the device;
     * If the task goes to GPU, then the whole column will stay in this GPU for ever.
     * If the task goes to CPU, then the whole column will stay in CPU in k iteration,
     *    then it has the opportunity to choose again a device at iteration k+1.
     * This way we choose to place the computation where A1 is.
     */
    if (m == (k+1)) {
        int A1_dev_index = this_task->data[flow_A1].data_in->original->owner_device;
        int A2_dev_index = this_task->data[flow_A2].data_in->original->owner_device;

        if ( A1_dev_index > 0) {
            dev_index = A1_dev_index;
            assert(dev_index != 0);
        }
        else if ( A2_dev_index > 0 ) {
            dev_index = A2_dev_index;
            assert(dev_index != 0);
        }
        else {
            int best_index = 0;  /* default value: first CPU device */
            float weight, best_weight = dague_device_load[0] + dague_device_sweight[0];
            dague_handle_t* handle = this_task->dague_handle;

            assert( (A1_dev_index <= 0) && (A2_dev_index <= 0) );

            for( dev_index = 2; dev_index < dague_devices_enabled(); dev_index++ ) {
                /* Skip the device if it is not configured */
                if(!(handle->devices_mask & (1 << dev_index))) continue;
                weight = dague_device_load[dev_index] + dague_device_sweight[dev_index];
                if( best_weight > weight ) {
                    best_index = dev_index;
                    best_weight = weight;
                }
            }
            dague_device_load[best_index] += dague_device_sweight[best_index];
            dev_index = best_index;
        }
    }
#endif

    if( dev_index == 0 ) {
        return DAGUE_HOOK_RETURN_NEXT;  /* Fall back */
    }
    assert( dev_index > 1 );

    gpu_task = (dague_ztsmqr_args_t*)malloc(sizeof(dague_ztsmqr_args_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->super.ec = this_task;
    gpu_task->super.task_type = 0;
    gpu_task->pushout_A1  = pushout_A1;
    gpu_task->pushout_A2  = pushout_A2;
    gpu_task->side  = side;
    gpu_task->trans = trans;
    gpu_task->M1    = M1;
    gpu_task->N1    = N1;
    gpu_task->M2    = M2;
    gpu_task->N2    = N2;
    gpu_task->K     = K;
    gpu_task->IB    = IB;
    gpu_task->lda1  = LDA1;
    gpu_task->lda2  = LDA2;
    gpu_task->ldv   = LDV;
    gpu_task->ldt   = LDT;

    return gpu_kernel_scheduler_ztsmqr( eu_context, (dague_gpu_context_t*)gpu_task, dev_index );
}
