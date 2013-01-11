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
#include <dlfcn.h>
#include <plasma.h>
#include <core_blas.h>
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif
#include "dague.h"
#include "gpu_data.h"
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

cuda_zgemm_t* zgemm_functions;

#define FORCE_UNDEFINED_SYMBOL(x) void* __ ## x ## _fp =(void*)&x;
extern cuda_zgemm_t magmablas_zgemm_SM11;
FORCE_UNDEFINED_SYMBOL(magmablas_zgemm_SM11)
extern cuda_zgemm_t magmablas_zgemm_SM13;
FORCE_UNDEFINED_SYMBOL(magmablas_zgemm_SM13)
extern cuda_zgemm_t magmablas_zgemm_SM20;
FORCE_UNDEFINED_SYMBOL(magmablas_zgemm_SM20)

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
    size_t sizeA, sizeB, sizeC;
    dague_ddesc_t *ddescA, *ddescB, *ddescC;
} dague_zgemm_args_t;

#include "gpu_scheduling.h"

static int ndevices = 0;

int gpu_kernel_init_zgemm( dague_context_t* dague_context )
{
    char *env;
    int i, dindex, nbgpus;
    (void)dague_context;

    nbgpus = dague_active_gpu();
    zgemm_functions = calloc(nbgpus, sizeof(cuda_zgemm_t));

    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        void* fn;
        void* dlh;
        char library_name[FILENAME_MAX];
        char function_name[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];
        fn = NULL;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, {continue;} );
        int major = gpu_device->major, minor = gpu_device->minor;

    retry_lesser_sm_version:
        snprintf(function_name, FILENAME_MAX, "magmablas_zgemm_SM%d%d", major, minor);
        env = getenv("DAGUE_CUCORES_LIB");
        if(NULL == env) {
            snprintf(library_name,  FILENAME_MAX, "libdplasma_cucores_sm%d%d.so",  major, minor);
        }
        else {
            snprintf(library_name,  FILENAME_MAX, "%s", env);
        }

        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            if(env) ERROR(("Could not find %s library: %s\n"
                           "  It is derived from environment DAGUE_CUCORES_LIB=%s\n"
                           "  To resolve this issue, set this variable to the correct path\n"
                           "    ex: /path/libdplasma_cucores_sm20.so\n"
                           "  Or unset it to use the default GPU kernels\n"
                           , library_name, dlerror(), env));
            DEBUG3(("Could not find %s dynamic library (%s)\n", library_name, dlerror()));
        }
        else {
            fn = dlsym(dlh, function_name);
            dlclose(dlh);
        }

        /* Couldn't load from dynamic libs, try static */
        if(NULL == fn) {
            DEBUG3(("No dynamic function %s found, loading from statically linked\n", function_name));
            dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            fn = dlsym(dlh, function_name);
            if(env && fn) WARNING(("Internal static function %s used (because library %s didn't loaded correctly)\n", function_name, library_name));
            dlclose(dlh);
        }

        /* Still not found?? skip this GPU */
        if(NULL == fn) {
            STATUS(("No function %s found for GPU %d\n", function_name, i));
            if(minor > 0) {
                minor--;
                goto retry_lesser_sm_version;
            } else
            {
                major--; minor = 9;
                if(major > 0) goto retry_lesser_sm_version;
            }
            status = cuCtxPopCurrent(NULL);
            continue;
        }

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );

        gpu_device->index = (uint8_t)dindex;
        zgemm_functions[dindex] = (cuda_zgemm_t)fn;
        gpu_enabled_devices[dindex++] = gpu_device;
    }

    /* Update the number of GPUs available */
    dague_data_enable_gpu( dindex );
    ndevices = dindex;
    assert( nbgpus == ndevices ); /* the code for when some devices can load some functions but not others is not yet correct, blanket protection against this */

    return 0;
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
gpu_kernel_push_zgemm( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret, move_data_count = 0;
    int sizeloc[MAX_PARAM_COUNT];
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    dague_data_t* data;
    dague_data_copy_t* local;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( !(this_task->function->in[0]->access_type & ACCESS_READ) )
            continue;

        data = this_task->data[i].data->original;
        if( NULL == (local = dague_data_get_copy(data, gpu_device->index)) ) {
            move_data_count++;
        } else {
            /**
             * In case the data copy I got is not on my local device, swap the
             * reference with the most recent version on the local device. Otherwise,
             * use the original copy. This allow copy-on-write to work seamlesly.
             */
            if( this_task->data[i].data->device_index != gpu_device->index ) {
                /* Attach the GPU copy to the task */
                this_task->data[i].data = local;
            }
        }
    }

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        sizeloc[0] = args->sizeA;
        sizeloc[1] = args->sizeB;
        sizeloc[2] = args->sizeC;

        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   sizeloc,
                                                   move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    assert( NULL != dague_data_copy_get_ptr(this_task->data[0].data) );
    assert( NULL != dague_data_copy_get_ptr(this_task->data[1].data) );
    assert( NULL != dague_data_copy_get_ptr(this_task->data[2].data) );

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[0]->name, args->Am, args->An));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), args->sizeA, gpu_stream->cuda_stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[1]->name, args->Bm, args->Bn));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[1]->access_type,
                                   &(this_task->data[1]), args->sizeB, gpu_stream->cuda_stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[2]->name, args->Cm, args->Cn));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[2]->access_type,
                                   &(this_task->data[2]), args->sizeC, gpu_stream->cuda_stream );
    if( ret < 0 ) {
        goto release_and_return_error;
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

    cuda_zgemm_t cuda_zgemm = zgemm_functions[gpu_device->index];

    d_A = (CUdeviceptr)this_task->data[0].data->device_private;
    d_B = (CUdeviceptr)this_task->data[1].data->device_private;
    d_C = (CUdeviceptr)this_task->data[2].data->device_private;

    DEBUG2(( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->profiling,
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

    DAGUE_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
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
    dague_gpu_data_copy_t     *gpu_copy = NULL;
    dague_data_t              *original;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_copy = this_task->data[i].data;
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->index) );
            /* Stage the transfer of the data back to main memory */
            gpu_device->required_data_out += args->sizeC;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            if( args->pushout ) {  /* n == (k + 1) */
                DEBUG3(("GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->device_index,
                        this_task->function->in[i]->name, this_task->data[i].data->original->key));
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_device->profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* TODO: Move the data back into main memory, but not always on the first device (!) */
                original = gpu_copy->original;
                status = (cudaError_t)cuMemcpyDtoHAsync( original->device_copies[0]->device_private,
                                                         (CUdeviceptr)gpu_copy->device_private,
                                                         args->sizeC, gpu_stream->cuda_stream );
                DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                        { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[i]->name,
                                                   gpu_copy->device_private, original->device_copies[0]->device_private));
                                            return_code = -2;
                                            goto release_and_return_error;} );
                gpu_device->transferred_data_out += args->sizeC; /* TODO: not hardcoded, use datatype size */
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
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_copy = this_task->data[i].data;
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        original = gpu_copy->original;
        original->version = gpu_copy->version;
        original->owner_device = -1;

        if( args->pushout ) {  /* n == (k  + 1) */
            dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
        } else {
            dague_ulist_fifo_push(gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_copy);
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
    int which_gpu;
    dague_zgemm_args_t *gpu_task = (dague_zgemm_args_t*)malloc(sizeof(dague_zgemm_args_t));

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
    gpu_task->sizeA    = sizeof(dague_complex64_t) * (size_t)lda * (( transA == PlasmaNoTrans ) ? K : M );
    gpu_task->sizeB    = sizeof(dague_complex64_t) * (size_t)ldb * (( transB == PlasmaNoTrans ) ? N : K );
    gpu_task->sizeC    = sizeof(dague_complex64_t) * (size_t)ldc * N;
    gpu_task->ddescA   = (dague_ddesc_t*)descA;
    gpu_task->ddescB   = (dague_ddesc_t*)descB;
    gpu_task->ddescC   = (dague_ddesc_t*)descC;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = this_task->data[2].data->original->owner_device;
    if( which_gpu <= 0 ) {  /* this is the first time we see this tile.
                            * Let's decide which GPU will work on it. */
        int best_index = -1;  /* cores */
        /* There are 3 types of GEMMs kernels: the ones waiting on the
         * execution contextes queues to be investigated, the current one
         * which is investigated for execution on the context of the current
         * execution context, and the ones already queued on the GPUs. The
         * decision regarding the status of the current GEMM should be therefore
         * based only on the number of pending tasks on the GPUs.
         */
        float weight, best_weight = device_load[0] + device_weight[0];
        for( which_gpu = 0; which_gpu < ndevices; which_gpu++ ) {
            weight = device_load[which_gpu+1] + device_weight[which_gpu+1];
            if( best_weight > weight ) {
                best_index = which_gpu;
                best_weight = weight;
            }
        }
        if( best_index == -1 ) {
            dague_atomic_inc_32b( &dague_cpu_counter );
            return -99;
        }
        which_gpu = best_index;
    }
    /* Update the load of the selected GPU */
    device_load[which_gpu+1] += device_weight[which_gpu+1];

    return gpu_kernel_scheduler_zgemm( eu_context, (dague_gpu_context_t*)gpu_task, which_gpu );
}
