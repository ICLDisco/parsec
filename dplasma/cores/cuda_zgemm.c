/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include <dague_config.h>
#include <stdlib.h>
#include <plasma.h>
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

#include "cuda_zgemm.h"

#define KERNEL_NAME zgemm

typedef void (*cuda_zgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                               dague_complex64_t alpha, dague_complex64_t *d_A, int lda,
                                                        dague_complex64_t *d_B, int ldb,
                               dague_complex64_t beta,  dague_complex64_t *d_C, int ldc,
                               CUstream stream );

extern void** cuda_gemm_functions;

#define FORCE_UNDEFINED_SYMBOL(x) void* __ ## x ## _fp =(void*)&x;
extern cuda_zgemm_t magmablas_ZGEMM_SM11;
FORCE_UNDEFINED_SYMBOL(magmablas_ZGEMM_SM11)
extern cuda_zgemm_t magmablas_ZGEMM_SM13;
FORCE_UNDEFINED_SYMBOL(magmablas_ZGEMM_SM13)
extern cuda_zgemm_t magmablas_ZGEMM_SM20;
FORCE_UNDEFINED_SYMBOL(magmablas_ZGEMM_SM20)

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
    int Am, An, lda, Bm, Bn, ldb, Cm, Cn, ldc;
    dague_ddesc_t *ddescA, *ddescB, *ddescC;
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
    int i, ret, move_data_count = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_data_t              *original;
    dague_data_copy_t         *data, *local;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

        data = this_task->data[i].data;
        original = data->original;
        if( NULL != (local = dague_data_get_copy(original, gpu_device->super.device_index)) ) {
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
        }
        /* If the data is needed as an input load it up */
        if(this_task->function->in[i]->access_type & ACCESS_READ)
            move_data_count++;
    }

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;
        assert( NULL != dague_data_copy_get_ptr(this_task->data[i].data) );

        DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d) on GPU\n",
                gpu_device->cuda_index, this_task->function->in[i]->name,
                (int)this_task->data[i].data->original.key));
        ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[i]->access_type,
                                       &(this_task->data[i]), gpu_stream->cuda_stream );
        if( ret < 0 ) {
            return ret;
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
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    cuda_zgemm_t cuda_zgemm = (cuda_zgemm_t)cuda_gemm_functions[gpu_device->cuda_index];

    d_A = (CUdeviceptr)this_task->data[0].data->original->device_copies[gpu_device->super.device_index]->device_private;
    d_B = (CUdeviceptr)this_task->data[1].data->original->device_copies[gpu_device->super.device_index]->device_private;
    d_C = (CUdeviceptr)this_task->data[2].data->original->device_copies[gpu_device->super.device_index]->device_private;

    DEBUG2(( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->cuda_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    status = cudaSuccess;
    cuda_zgemm( lapack_const(args->transA), lapack_const(args->transB), args->M, args->N, args->K,
                args->alpha, (dague_complex64_t*)d_A, args->lda,
                             (dague_complex64_t*)d_B, args->ldb,
                args->beta,  (dague_complex64_t*)d_C, args->ldc,
                gpu_stream->cuda_stream );

    DAGUE_CUDA_CHECK_ERROR( "cuda_zgemm ", status,
                              {return -1;} );

/*     fprintf(stderr, "cuda_zgemm( %d, %d, %d )\n\t( %c, %c, %d, %d, %d, %e, A(%d,%d)[%p], %d, A(%d,%d)[%p], %d, %e, A(%d,%d)[%p], %d)\n", */
/*             this_task->locals[0].value, this_task->locals[1].value, this_task->locals[2].value, */
/*             lapack_const( args->transA ),  lapack_const( args->transB ), */
/*             args->M, args->N, args->K, */
/*             args->alpha, args->Am, args->An, (dague_complex64_t*)d_A, args->lda, */
/*                          args->Bm, args->Bn, (dague_complex64_t*)d_B, args->ldb, */
/*             args->beta,  args->Cm, args->Cn, (dague_complex64_t*)d_C, args->ldc); */
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

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        /* Don't bother if there is no real data (aka. CTL) */
        if(NULL == this_task->data[i].data) continue;
        flow = this_task->function->in[i];
        if(NULL == flow)
            flow = this_task->function->out[i];

        original = this_task->data[i].data->original;
        gpu_copy = dague_data_get_copy(original, gpu_device->super.device_index);
        if( flow->access_type & ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            }
        }
        if( flow->access_type & ACCESS_WRITE ) {
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );
            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            if( args->pushout ) {  /* n == (k + 1) */
                DEBUG3(("GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->cuda_index,
                        this_task->function->out[i]->name, this_task->data[i].data->original->key));
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_device->super.profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* TODO: Move the data back into main memory, but not always on the first device (!) */
                original = gpu_copy->original;
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
gpu_kernel_epilog_zgemm( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->out[i]) continue;
        if(!(this_task->function->out[i]->access_type & ACCESS_WRITE)) continue;

        original = this_task->data[i].data->original;
        gpu_copy = dague_data_get_copy(original, gpu_device->super.device_index);
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        original = gpu_copy->original;
        original->version = gpu_copy->version;
        original->owner_device = -1;

        if( args->pushout ) {  /* n == (k  + 1) */
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
int gpu_zgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int pushout,
               PLASMA_enum transA, PLASMA_enum transB,
               int M, int N, int K,
               dague_complex64_t alpha, int Am, int An, const tiled_matrix_desc_t *descA, int lda,
                                        int Bm, int Bn, const tiled_matrix_desc_t *descB, int ldb,
               dague_complex64_t beta,  int Cm, int Cn, const tiled_matrix_desc_t *descC, int ldc )
{
    int i, dev_index, data_index = 0;
    dague_zgemm_args_t *gpu_task;
    dague_handle_t* handle = this_task->dague_handle;

    /* Step one: which write enabled data we will look at */
    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( (NULL == this_task->function->out[i]) ||
            (this_task->function->out[i]->access_type & ACCESS_WRITE) ) {
            data_index = i;
            break;
        }
    }
    /* Which device is the owner of the data */
    dev_index = this_task->data[data_index].data->original->owner_device;
    if( dev_index <= 0 ) {  /* this is the first time we see this tile.
                             * Let's decide which GPU will work on it. */
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
        if( best_index == 0 ) {
            return DAGUE_HOOK_RETURN_NEXT;  /* Fall back */
        }
        dev_index = best_index;
    }

    gpu_task = (dague_zgemm_args_t*)malloc(sizeof(dague_zgemm_args_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->super.ec = this_task;
    gpu_task->pushout  = pushout;
    gpu_task->alpha    = alpha;
    gpu_task->beta     = beta;
    gpu_task->transA   = transA;
    gpu_task->transB   = transB;
    gpu_task->M        = M;
    gpu_task->N        = N;
    gpu_task->K        = K;
    gpu_task->Am       = Am;
    gpu_task->An       = An;
    gpu_task->lda      = lda;
    gpu_task->Bm       = Bm;
    gpu_task->Bn       = Bn;
    gpu_task->ldb      = ldb;
    gpu_task->Cm       = Cm;
    gpu_task->Cn       = Cn;
    gpu_task->ldc      = ldc;
    gpu_task->ddescA   = (dague_ddesc_t*)descA;
    gpu_task->ddescB   = (dague_ddesc_t*)descB;
    gpu_task->ddescC   = (dague_ddesc_t*)descC;

    return gpu_kernel_scheduler_zgemm( eu_context, (dague_gpu_context_t*)gpu_task, dev_index );
}
