/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @generated s Thu Feb 27 15:42:30 2014
 *
 */
#include <dague_config.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <core_blas.h>
#include <core_blas.h>
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "datarepo.h"
#include "data_dist/matrix/matrix.h"
#include "dague/utils/output.h"
#include "cuda_stsmqr.h"

#define flow_A1  0
#define flow_A2  1
#define flow_V   2
#define flow_T   3

#define KERNEL_NAME stsmqr

/*typedef void (*cuda_sgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                               float alpha, float *d_A, int lda,
                                                        float *d_B, int ldb,
                               float beta,  float *d_C, int ldc,
                               CUstream stream );*/
/* TO DISSAPEAR */
extern void** cuda_gemm_functions;
extern int dague_cuda_output_stream;

/*
#define FORCE_UNDEFINED_SYMBOL(x) void* __ ## x ## _fp =(void*)&x;
extern cuda_sgemm_t magmablas_SGEMM_SM11;
FORCE_UNDEFINED_SYMBOL(magmablas_SGEMM_SM11)
extern cuda_sgemm_t magmablas_SGEMM_SM13;
FORCE_UNDEFINED_SYMBOL(magmablas_SGEMM_SM13)
extern cuda_sgemm_t magmablas_SGEMM_SM20;
FORCE_UNDEFINED_SYMBOL(magmablas_SGEMM_SM20)*/

typedef void (*cuda_stsmqr_t) (PLASMA_enum side, PLASMA_enum trans,
                               int M1, int N1, int M2, int N2, int K, int IB,
                               float *A1, int LDA1,
                               float *A2, int LDA2,
                         const float *V, int LDV,
                         const float *T, int LDT,
                               float *WORK, int LDWORK,
                               CUstream stream);

static inline
int gpu_kernel_push_stsmqr( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           dague_gpu_exec_stream_t* gpu_stream);

static inline
int gpu_kernel_submit_stsmqr( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           dague_gpu_exec_stream_t* gpu_stream);

static inline
int gpu_kernel_pop_stsmqr( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           dague_gpu_exec_stream_t* gpu_stream);

static inline
int  gpu_kernel_epilog_stsmqr( gpu_device_t* gpu_device,
                              dague_gpu_context_t* this_task );

typedef struct dague_stsmqr_args_s {
    dague_gpu_context_t super;
    int pushout_A1, pushout_A2;
    PLASMA_enum side, trans;
    int M1, N1, M2, N2, K, IB;
    int A1m, A1n, lda1, A2m, A2n, lda2, Vm, Vn, ldv, Tm, Tn, ldt;
    dague_ddesc_t *ddescA1, *ddescA2, *ddescV, *ddescT;
} dague_stsmqr_args_t;

#include <dague/devices/cuda/cuda_scheduling.h>

//#define WEI_DEBUG
inline static void wei_debug_printf(const char *fmt, ...)
{
#if defined (WEI_DEBUG)	
	va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif /* WEI_DEBUG */
}

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
gpu_kernel_push_stsmqr( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret = 0;
    int space_needed = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_data_t              *original;
    dague_data_copy_t         *data, *local;

    dague_stsmqr_args_t       *args = (dague_stsmqr_args_t*)gpu_task;
    int k = args->A1m;
    int m = args->A2m;
    int n = args->A1n;
    wei_debug_printf("------------------I am in push m %d, n %d, k %d, device %d\n", m, n, k, gpu_device->super.device_index);

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

        this_task->data[i].data_out = NULL;
        data = this_task->data[i].data_in;
        original = data->original;
        if( NULL != (local = dague_data_get_copy(original, gpu_device->super.device_index)) ) {
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
            wei_debug_printf("i %d\n", i);
            space_needed++;
    }

    wei_debug_printf("space needed %d\n", space_needed);

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
       
        wei_debug_printf("PUSH %d, nb %d\n", i, this_task->function->nb_flows);

        DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                              "GPU[%1d]:\tIN  Data of %s <%x> on GPU\n",
                              gpu_device->cuda_index, this_task->function->in[i]->name,
                              this_task->data[i].data_out->original->key));
        ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[i]->flow_flags,
                                       &(this_task->data[i]), gpu_task, gpu_stream->cuda_stream );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

  release_and_return_error:
    return ret;
}


static inline int
gpu_kernel_submit_stsmqr( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task,
                         dague_gpu_exec_stream_t* gpu_stream )
{
    wei_debug_printf("I am in submit\n");
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_stsmqr_args_t        *args = (dague_stsmqr_args_t*)gpu_task;
    CUdeviceptr d_A1, d_A2, d_V, d_T, WORK;
    cudaError_t status;

    cuda_stsmqr_t cuda_stsmqr = (cuda_stsmqr_t)cuda_gemm_functions[gpu_device->cuda_index];

    assert(this_task->data[flow_A1].data_out->device_index == gpu_device->super.device_index);
    d_A1 = (CUdeviceptr)this_task->data[flow_A1].data_out->device_private;
    assert(this_task->data[flow_A2].data_out->device_index == gpu_device->super.device_index);
    d_A2 = (CUdeviceptr)this_task->data[flow_A2].data_out->device_private;
    assert(this_task->data[flow_V].data_out->device_index == gpu_device->super.device_index);
    d_V  = (CUdeviceptr)this_task->data[flow_V].data_out->device_private;
    assert(this_task->data[flow_T].data_out->device_index == gpu_device->super.device_index);
    d_T  = (CUdeviceptr)this_task->data[flow_T].data_out->device_private;

    tiled_matrix_desc_t *descT = (tiled_matrix_desc_t *)args->ddescT;
    WORK = (CUdeviceptr)gpu_malloc( gpu_device->memory, 1 );
    int LDWORK = args->IB;

    wei_debug_printf("nb %d, ib %d, WORK %p\n", descT->nb, args->IB, (void*)WORK);

    cuda_stsmqr(args->side, args->trans,
                args->M1, args->N1, args->M2, args->N2, args->K, args->IB,
                (float*)d_A1, args->lda1,
                (float*)d_A2, args->lda2,
                (float*)d_V,  args->ldv,
                (float*)d_T,  args->ldt,
                (float*)WORK, LDWORK,
                gpu_stream->cuda_stream);

    gpu_free( gpu_device->memory, (void*)WORK );

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
gpu_kernel_pop_stsmqr( gpu_device_t        *gpu_device,
                      dague_gpu_context_t *gpu_task,
                      dague_gpu_exec_stream_t* gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_stsmqr_args_t        *args = (dague_stsmqr_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    const dague_flow_t        *flow;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    int k = args->A1m;
    int m = args->A2m;
    int n = args->A1n;
    wei_debug_printf("++++++++++++++++++++I am in pop m %d, n %d, k %d, device %d\n", m, n, k, gpu_device->super.device_index);


    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        flow = this_task->function->in[i];
        if(NULL == flow)
            flow = this_task->function->out[i];

        original = this_task->data[i].data_out->original;
        gpu_copy = this_task->data[i].data_out;
        assert(original == this_task->data[i].data_in->original);
        if( flow->flow_flags & FLOW_ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
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
            if( (args->pushout_A1 && i == flow_A1) || (args->pushout_A2 && i == flow_A2) ) { 
                original = gpu_copy->original;
                DAGUE_OUTPUT_VERBOSE((2, dague_cuda_output_stream,
                                      "GPU:\tMove D2H data <%x> from GPU %d %p -> %p requested\n",
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
                wei_debug_printf("POP %d, nb %d\n", i, this_task->function->nb_flows);
                wei_debug_printf("POP from %p to %p, size %d\n", gpu_copy->device_private, original->device_copies[0]->device_private, original->nb_elts);
                status = (cudaError_t)cuMemcpyDtoHAsync( original->device_copies[0]->device_private,
                                                         (CUdeviceptr)gpu_copy->device_private,
                                                         original->nb_elts, gpu_stream->cuda_stream );
                DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                        { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                                   gpu_copy->device_private, original->device_copies[0]->device_private));
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
gpu_kernel_epilog_stsmqr( gpu_device_t        *gpu_device,
                          dague_gpu_context_t *gpu_task )
{
    wei_debug_printf("I am in epilog\n");

    dague_execution_context_t *this_task = gpu_task->ec;
    dague_stsmqr_args_t        *args = (dague_stsmqr_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if(NULL == this_task->function->out[i]) continue;
        if( !(this_task->function->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) continue;

        gpu_copy = this_task->data[this_task->function->out[i]->flow_index].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];
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

        if( (args->pushout_A1 && i == flow_A1) || (args->pushout_A2 && i == flow_A2) ) {  
            dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
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
int gpu_stsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* this_task,
                int pushout_A1, int pushout_A2,
                PLASMA_enum side, PLASMA_enum trans,
                int M1, int N1, int M2, int N2, int K, int IB,
                int A1m, int A1n, const tiled_matrix_desc_t *descA1, int LDA1,
                int A2m, int A2n, const tiled_matrix_desc_t *descA2, int LDA2,
                int Vm,  int Vn,  const tiled_matrix_desc_t *descV,  int LDV,
                int Tm,  int Tn,  const tiled_matrix_desc_t *descT,  int LDT)
{
    int i, dev_index, data_index = 0, m, n, k;
    dague_stsmqr_args_t *gpu_task;
    dague_handle_t* handle = this_task->dague_handle;
    int A1_dev_index, A2_dev_index;

    dev_index = 1;

    k = A1m;
    m = A2m;
    n = A1n;

    A1_dev_index = this_task->data[flow_A1].data_in->original->owner_device;
    A2_dev_index = this_task->data[flow_A2].data_in->original->owner_device;
    wei_debug_printf("m %d, n %d, k %d, A1 owner %d, A2 owner %d\n", m, n, k, A1_dev_index, A2_dev_index);

        
    /* only the tsmqr task located in (k, k+1, n) can choose the device;
     * if the task goes to GPU, then the whole column will stay in this GPU for ever. 
     * if the task goes to CPU, then the whole column will stay in CPU in k iteration, 
     *    then he has the oppotunity to choose devices in k+1 iteration.  
     */
    if (m == (k+1)) {
        assert(A1_dev_index == 0);
        if (A2_dev_index == 0) {  
            int best_index = 0;  /* default value: first CPU device */
            float weight, best_weight = dague_device_load[0] + dague_device_sweight[0];
            for( dev_index = 1; dev_index < dague_devices_enabled(); dev_index++ ) {
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
        } else {
            /* task allocation has been decided in previous iterations, so task will goes to where A2 located */
            dev_index = A2_dev_index;
            assert(dev_index != 0);
        }
    } else {
       dev_index = A1_dev_index;
    }
    wei_debug_printf("m %d, n %d, k %d, A1 owner %d, A2 owner %d, dev_index %d\n", m, n, k, A1_dev_index, A2_dev_index, dev_index);

    if( dev_index == 0 ) {
        wei_debug_printf("!!!!!!!!!!!!!!!!!!!!!!!! m %d, n %d, k %d go back to CPU\n", m, n, k);
        return DAGUE_HOOK_RETURN_NEXT;  /* Fall back */
    }
  //  dev_index = 1;

    gpu_task = (dague_stsmqr_args_t*)malloc(sizeof(dague_stsmqr_args_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->super.ec = this_task;
    gpu_task->pushout_A1  = pushout_A1;
    gpu_task->pushout_A2  = pushout_A2;
    gpu_task->side     = side;
    gpu_task->trans    = trans;
    gpu_task->M1       = M1;
    gpu_task->N1       = N1;
    gpu_task->M2       = M2;
    gpu_task->N2       = N2;
    gpu_task->K        = K;
    gpu_task->IB       = IB;
    gpu_task->A1m      = A1m;
    gpu_task->A1n      = A1n;
    gpu_task->lda1     = LDA1;
    gpu_task->A2m      = A2m;
    gpu_task->A2n      = A2n;
    gpu_task->lda2     = LDA2;
    gpu_task->Vm       = Vm;
    gpu_task->Vn       = Vn;
    gpu_task->ldv      = LDV;
    gpu_task->Tm       = Tm;
    gpu_task->Tn       = Tn;
    gpu_task->ldt      = LDT;
    gpu_task->ddescA1   = (dague_ddesc_t*)descA1;
    gpu_task->ddescA2   = (dague_ddesc_t*)descA2;
    gpu_task->ddescV   = (dague_ddesc_t*)descV;
    gpu_task->ddescT   = (dague_ddesc_t*)descT;

    return gpu_kernel_scheduler_stsmqr( eu_context, (dague_gpu_context_t*)gpu_task, dev_index );
}
