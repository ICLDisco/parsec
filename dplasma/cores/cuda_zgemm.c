/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include "dague_config.h"
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
                               Dague_Complex64_t alpha, Dague_Complex64_t *d_A, int lda,
                                                        Dague_Complex64_t *d_B, int ldb,
                               Dague_Complex64_t beta,  Dague_Complex64_t *d_C, int ldc,
                               CUstream stream );

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
                           CUstream stream );

static inline
int gpu_kernel_submit_zgemm( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           CUstream stream );

static inline
int gpu_kernel_pop_zgemm( gpu_device_t* gpu_device,
                           dague_gpu_context_t* this_task,
                           CUstream stream );

static inline
int  gpu_kernel_epilog_zgemm( gpu_device_t* gpu_device,
                              dague_gpu_context_t* this_task );

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgemm( gpu_device_t        *gpu_device,
                               dague_gpu_context_t *this_task );
#endif

typedef struct dague_zgemm_args_s {
    dague_gpu_context_t super;
    int pushout;
    Dague_Complex64_t alpha, beta;
    PLASMA_enum transA, transB;
    int M, N, K;
    int Am, An, lda, Bm, Bn, ldb, Cm, Cn, ldc;
    size_t sizeA, sizeB, sizeC;
    dague_ddesc_t *ddescA, *ddescB, *ddescC;
} dague_zgemm_args_t;

#include "gpu_scheduling.h"

static int ndevices = 0;

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgemm( gpu_device_t        *gpu_device,
                               dague_gpu_context_t *this_task )
{
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        dague_execution_context_t *ec   = this_task->ec;
        dague_zgemm_args_t        *args = (dague_zgemm_args_t*)this_task;
        dague_ddesc_t *ddesc = (dague_ddesc_t*)(args->ddescC);
        int data_id =
            ddesc->data_key(ddesc, 
                            args->Cm, 
                            args->Cn );

        uint64_t task_id =
            ec->function->key( ec->dague_object, ec->locals );
        
        dague_profile_ddesc_info_t info;
        info.desc = ddesc;
        info.id = data_id;
        dague_profiling_trace( gpu_device->profiling,
                               DAGUE_PROF_FUNC_KEY_START(ec->dague_object,
                                                         ec->function->function_id),
                               task_id, ec->dague_object->object_id,
                               (void*)&info);
    }
}
#endif  /* defined(DAGUE_PROF_TRACE) */

int gpu_kernel_init_zgemm( dague_context_t* dague_context )
{
    char *env;
    int i, dindex, nbgpus;
    (void)dague_context;

    nbgpus = dague_active_gpu();
    //gpu_active_devices = (gpu_device_t** )calloc(nbgpus, sizeof(gpu_device_t*));
    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        void* dlh;
        char library_name[FILENAME_MAX];
        char function_name[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];
        gpu_device->function = NULL;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, {continue;} );

/* If not found statically, try shared lib */
/*         if(NULL == gpu_device->hcuFunction) { */
/*             env = getenv("DAGUE_CUBIN_PATH"); */
/*             snprintf(module_path, FILENAME_MAX, "%s/zgemm_sm%1d%1d.cubin", */
/*                      env?env:"../cores", gpu_device->major, gpu_device->minor); */
/*             status = cuModuleLoad(&(gpu_device->hcuModule), module_path); */
/*             DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status, */
/*                                     { */
/*                                         WARNING(("GPU:\tUnable to load `%s'\n", module_path)); */
/*                                         continue; */
/*                                     } ); */
/*             snprintf(module_path, FILENAME_MAX, "zgemmNT_SM%d%d", gpu_device->major, gpu_device->minor); */
/*             DEBUG3(("CUDA MODULE %s\n", module_path)); */
/*             status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, module_path ); */
/*             DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status, */
/*                                     { */
/*                                         WARNING(("GPU:\tUnable to find the function `%s'\n", module_path)); */
/*                                         continue; */
/*                                     } ); */
/*         } */

        snprintf(function_name, FILENAME_MAX, "magmablas_zgemm_SM%d%d", gpu_device->major, gpu_device->minor);
        env = getenv("DAGUE_CUBIN_LIBNAME");
        if(NULL == env) {
            snprintf(library_name,  FILENAME_MAX, "libdplasma_cucores_sm%d%d.so",  gpu_device->major, gpu_device->minor);
        }
        else {
            snprintf(library_name,  FILENAME_MAX, "%s_sm%d%d.so", env, gpu_device->major, gpu_device->minor);
        }
            
        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            if(env) ERROR(("Could not find %s library: %s\n"
                           "  It is derived from environment DAGUE_CUBIN_LIBNAME=%s\n"
                           "  To resolve this issue, set this variable to the correct path\n"
                           "    ex: if /path/libdplasma_cucores_sm20.so exists, \n"
                           "    set it to /path/libdplasma_cucores\n"
                           "  Or unset it to use the default GPU kernels\n"
                           , library_name, dlerror(), env));
            DEBUG3(("Could not find %s library (%s)\n", library_name, dlerror()));
        }
        else {
            gpu_device->function = dlsym(dlh, function_name);
            dlclose(dlh);
        }

        /* Couldn't load from dynamic libs, try static */
        if(NULL == gpu_device->function) {
            DEBUG3(("No dynamic function %s found, loading from statically linked\n", function_name));
            dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            gpu_device->function = dlsym(dlh, function_name);
            if(env && gpu_device->function) WARNING(("Internal static function %s used (because library %s didn't loaded correctly)\n", function_name, library_name));
            dlclose(dlh);
        }

        /* Still not found?? skip this GPU */
        if(NULL == gpu_device->function) {
            STATUS(("No function %s found for GPU %d\n", function_name, i));
            status = cuCtxPopCurrent(NULL);
            continue;
        }

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );

        gpu_device->index = (uint8_t)dindex;
        gpu_enabled_devices[dindex++] = gpu_device;
    }

    /* Update the number of GPUs available */
    dague_data_enable_gpu( dindex );
    ndevices = dindex;

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
gpu_kernel_push_zgemm( gpu_device_t        *gpu_device,
                       dague_gpu_context_t *gpu_task,
                       CUstream stream )
{
    int ret, move_data_count = 0;
    int sizeloc[MAX_PARAM_COUNT];
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;

    /* WARNING: A has to be the first data,
     *          B the second one and
     *          C the third one.
     * if the kernel swapp A and B it won't work 
     */
    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(args->ddescA, args->Am, args->An ),
                           &(this_task->data[0].mem2dev_data));
    if( NULL == (this_task->data[0].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(args->ddescB, args->Bm, args->Bn ),
                           &(this_task->data[1].mem2dev_data));
    if( NULL == (this_task->data[1].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(args->ddescC, args->Cm, args->Cn ),
                           &(this_task->data[2].mem2dev_data));
    if( NULL == (this_task->data[2].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    this_task->data[3].mem2dev_data =  NULL;  /* last element */

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        sizeloc[0] = args->sizeA;
        sizeloc[1] = args->sizeB;
        sizeloc[2] = args->sizeC;

        ret = dague_gpu_find_space_for_elts( gpu_device,
                                             this_task,
                                             sizeloc,
                                             move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    assert( NULL != this_task->data[0].mem2dev_data->device_elem[gpu_device->index] );
    assert( NULL != this_task->data[1].mem2dev_data->device_elem[gpu_device->index] );
    assert( NULL != this_task->data[2].mem2dev_data->device_elem[gpu_device->index] );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start,
                               (unsigned long)this_task, this_task->dague_object->object_id,
                               NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[0]->name, args->Am, args->An));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), args->sizeA, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[1]->name, args->Bm, args->Bn));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[1]->access_type,
                                   &(this_task->data[1]), args->sizeB, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[2]->name, args->Cm, args->Cn));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[2]->access_type,
                                   &(this_task->data[2]), args->sizeC, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }
  release_and_return_error:
    return ret;
}


static inline int
gpu_kernel_submit_zgemm( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task,
                         CUstream stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    cuda_zgemm_t cuda_zgemm = (cuda_zgemm_t) gpu_device->function;
    
    gpu_elem_A = (gpu_elem_t *)this_task->data[0].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_B = (gpu_elem_t *)this_task->data[1].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_C = (gpu_elem_t *)this_task->data[2].mem2dev_data->device_elem[gpu_device->index];
    d_A = gpu_elem_A->gpu_mem;
    d_B = gpu_elem_B->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;

    DEBUG2(( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, gpu_task );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = cudaSuccess;
    cuda_zgemm( lapack_const(args->transA), lapack_const(args->transB), args->M, args->N, args->K, 
                args->alpha, (Dague_Complex64_t*)d_A, args->lda,
                             (Dague_Complex64_t*)d_B, args->ldb,
                args->beta,  (Dague_Complex64_t*)d_C, args->ldc,
                stream );

    DAGUE_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
                              {return -1;} );

/*     fprintf(stderr, "cuda_zgemm( %d, %d, %d )\n\t( %c, %c, %d, %d, %d, %e, A(%d,%d)[%p], %d, A(%d,%d)[%p], %d, %e, A(%d,%d)[%p], %d)\n", */
/*             this_task->locals[0].value, this_task->locals[1].value, this_task->locals[2].value, */
/*             lapack_const( args->transA ),  lapack_const( args->transB ), */
/*             args->M, args->N, args->K, */
/*             args->alpha, args->Am, args->An, (Dague_Complex64_t*)d_A, args->lda, */
/*                          args->Bm, args->Bn, (Dague_Complex64_t*)d_B, args->ldb, */
/*             args->beta,  args->Cm, args->Cn, (Dague_Complex64_t*)d_C, args->ldc); */
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
                      CUstream stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( gpu_elem->generic.memory_elem == this_task->data[i].mem2dev_data );
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->generic.readers--; assert(gpu_elem->generic.readers >= 0);
            if( (0 == gpu_elem->generic.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_item_ring_chop((dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_CONSTRUCT(gpu_elem);
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            /* If we're not using this anymore on the GPU it should be moved back to the CPU */
        }
    }

    gpu_elem = (gpu_elem_t*)this_task->data[2].mem2dev_data->device_elem[gpu_device->index];

    /* Stage the transfer of the data back to main memory */
    gpu_device->required_data_out += args->sizeC;
    assert( ((dague_list_item_t*)gpu_elem)->list_next == (dague_list_item_t*)gpu_elem );
    assert( ((dague_list_item_t*)gpu_elem)->list_prev == (dague_list_item_t*)gpu_elem );

    if( args->pushout ) {  /* n == (k + 1) */
        DEBUG3(("GPU Request out of GPU for %s key %d\n", this_task->function->in[2]->name, this_task->data[2].mem2dev_data->key));
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start,
                                   (unsigned long)this_task, this_task->dague_object->object_id,
                                   NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        /* Move the data back into main memory */
        status = (cudaError_t)cuMemcpyDtoHAsync( ADATA(this_task->data[2].data), gpu_elem->gpu_mem, args->sizeC, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[2]->name,
                                           (void*)(long)gpu_elem->gpu_mem, (void*)ADATA(this_task->data[2].data)));
                                  return_code = -2;
                                  goto release_and_return_error;} );
        gpu_device->transferred_data_out += args->sizeC;
        how_many++;
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
    gpu_elem_t* gpu_elem;
    int i;

    for( i = 0; NULL != this_task->data[i].mem2dev_data; i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( DAGUE_DATA_OWNED == gpu_elem->generic.coherency_state );
        gpu_elem->generic.coherency_state = DAGUE_DATA_SHARED;
        gpu_elem->generic.memory_elem->version = gpu_elem->generic.version;
        this_task->data[2].mem2dev_data->device_owner = -1;

#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_end,
                                   (unsigned long)this_task, this_task->dague_object->object_id,
                                   NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        if( args->pushout ) {  /* n == (k  + 1) */
            dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        } else {
            dague_ulist_fifo_push(gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_elem);
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
               Dague_Complex64_t alpha, int Am, int An, const tiled_matrix_desc_t *descA, int lda,
                                        int Bm, int Bn, const tiled_matrix_desc_t *descB, int ldb,
               Dague_Complex64_t beta,  int Cm, int Cn, const tiled_matrix_desc_t *descC, int ldc )
{
    int which_gpu;
    dague_zgemm_args_t *gpu_task = (dague_zgemm_args_t*)malloc(sizeof(dague_zgemm_args_t));

    DAGUE_LIST_ITEM_CONSTRUCT(gpu_task);
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
    gpu_task->sizeA    = sizeof(Dague_Complex64_t) * (size_t)lda * (( transA == PlasmaNoTrans ) ? K : M );
    gpu_task->sizeB    = sizeof(Dague_Complex64_t) * (size_t)ldb * (( transB == PlasmaNoTrans ) ? N : K );
    gpu_task->sizeC    = sizeof(Dague_Complex64_t) * (size_t)ldc * N;
    gpu_task->ddescA   = (dague_ddesc_t*)descA;
    gpu_task->ddescB   = (dague_ddesc_t*)descB;
    gpu_task->ddescC   = (dague_ddesc_t*)descC;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dague_gpu_data_elt_write_owner( &dague_gpu_map, GEMM_KEY( descC, Cm, Cn) );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile.
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
