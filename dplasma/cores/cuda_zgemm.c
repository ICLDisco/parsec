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
    dague_complex64_t alpha, beta;
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
    zgemm_functions = calloc(nbgpus, sizeof(cuda_zgemm_t));

    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        void* fn;
        void* dlh;
        char library_name[FILENAME_MAX];
        char function_name[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, {continue;} );

        snprintf(function_name, FILENAME_MAX, "magmablas_zgemm_SM%d%d", gpu_device->major, gpu_device->minor);
        env = getenv("DAGUE_CUCORES_LIB");
        if(NULL == env) {
            snprintf(library_name,  FILENAME_MAX, "libdplasma_cucores_sm%d%d.so",  gpu_device->major, gpu_device->minor);
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
     * if the kernel swap A and B it won't work
     */
    moesi_get_master(args->ddescA->moesi_map, GEMM_KEY(args->ddescA, args->Am, args->An ),
                           &(this_task->data[0].moesi_master));
    if( NULL == (this_task->data[0].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    moesi_get_master(args->ddescB->moesi_map, GEMM_KEY(args->ddescB, args->Bm, args->Bn ),
                           &(this_task->data[1].moesi_master));
    if( NULL == (this_task->data[1].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    moesi_get_master(args->ddescC->moesi_map, GEMM_KEY(args->ddescC, args->Cm, args->Cn ),
                           &(this_task->data[2].moesi_master));
    if( NULL == (this_task->data[2].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    this_task->data[3].moesi_master =  NULL;  /* last element */

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

    assert( NULL != gpu_elem_obtain_from_master(this_task->data[0].moesi_master, gpu_device->index) );
    assert( NULL != gpu_elem_obtain_from_master(this_task->data[1].moesi_master, gpu_device->index) );
    assert( NULL != gpu_elem_obtain_from_master(this_task->data[2].moesi_master, gpu_device->index) );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start,
                               (unsigned long)this_task, this_task->dague_object->object_id,
                               NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n", gpu_device->device_index, this_task->function->in[0]->name, args->Am, args->An));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), args->sizeA, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n", gpu_device->device_index, this_task->function->in[1]->name, args->Bm, args->Bn));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[1]->access_type,
                                   &(this_task->data[1]), args->sizeB, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n", gpu_device->device_index, this_task->function->in[2]->name, args->Cm, args->Cn));
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

    cuda_zgemm_t cuda_zgemm = zgemm_functions[gpu_device->index];

    gpu_elem_A = gpu_elem_obtain_from_master(this_task->data[0].moesi_master, gpu_device->index);
    gpu_elem_B = gpu_elem_obtain_from_master(this_task->data[1].moesi_master, gpu_device->index);
    gpu_elem_C = gpu_elem_obtain_from_master(this_task->data[2].moesi_master, gpu_device->index);
    d_A = gpu_elem_A->gpu_mem_ptr;
    d_B = gpu_elem_B->gpu_mem_ptr;
    d_C = gpu_elem_C->gpu_mem_ptr;

    DEBUG2(( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, gpu_task );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = cudaSuccess;
    cuda_zgemm( lapack_const(args->transA), lapack_const(args->transB), args->M, args->N, args->K,
                args->alpha, (dague_complex64_t*)d_A, args->lda,
                             (dague_complex64_t*)d_B, args->ldb,
                args->beta,  (dague_complex64_t*)d_C, args->ldc,
                stream );

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
                      CUstream stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_zgemm_args_t        *args = (dague_zgemm_args_t*)gpu_task;
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_elem = gpu_elem_obtain_from_master(this_task->data[i].moesi_master, gpu_device->index);
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->moesi.readers--; assert(gpu_elem->moesi.readers >= 0);
            if( (0 == gpu_elem->moesi.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_CONSTRUCT(gpu_elem); /* TODO: singleton instead? */
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            gpu_elem = gpu_elem_obtain_from_master(this_task->data[i].moesi_master, gpu_device->index);

            /* Stage the transfer of the data back to main memory */
            gpu_device->required_data_out += args->sizeC;
            assert( ((dague_list_item_t*)gpu_elem)->list_next == (dague_list_item_t*)gpu_elem );
            assert( ((dague_list_item_t*)gpu_elem)->list_prev == (dague_list_item_t*)gpu_elem );

            if( args->pushout ) {  /* n == (k + 1) */
                DEBUG3(("GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->device_index, this_task->function->in[i]->name, this_task->data[i].moesi_master->key));
#if defined(DAGUE_PROF_TRACE)
                if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
                    dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start,
                                           (unsigned long)this_task, this_task->dague_object->object_id,
                                           NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
                /* Move the data back into main memory */
                status = (cudaError_t)cuMemcpyDtoHAsync( ADATA(this_task->data[2].data), gpu_elem->gpu_mem_ptr, args->sizeC, stream );
                DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                        { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[2]->name,
                                                  (void*)gpu_elem->gpu_mem_ptr, (void*)ADATA(this_task->data[2].data)));
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
    gpu_elem_t* gpu_elem;
    moesi_master_t* master;
    int i;

    for( i = 0; NULL != (master = this_task->data[i].moesi_master); i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_elem = gpu_elem_obtain_from_master(master, gpu_device->index);
        assert( MOESI_OWNED == gpu_elem->moesi.coherency_state );
        gpu_elem->moesi.coherency_state = MOESI_SHARED;
        master->version = gpu_elem->moesi.version;
        master->owner_device = -1;

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
               dague_complex64_t alpha, int Am, int An, const tiled_matrix_desc_t *descA, int lda,
                                        int Bm, int Bn, const tiled_matrix_desc_t *descB, int ldb,
               dague_complex64_t beta,  int Cm, int Cn, const tiled_matrix_desc_t *descC, int ldc )
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
    gpu_task->sizeA    = sizeof(dague_complex64_t) * (size_t)lda * (( transA == PlasmaNoTrans ) ? K : M );
    gpu_task->sizeB    = sizeof(dague_complex64_t) * (size_t)ldb * (( transB == PlasmaNoTrans ) ? N : K );
    gpu_task->sizeC    = sizeof(dague_complex64_t) * (size_t)ldc * N;
    gpu_task->ddescA   = (dague_ddesc_t*)descA;
    gpu_task->ddescB   = (dague_ddesc_t*)descB;
    gpu_task->ddescC   = (dague_ddesc_t*)descC;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = moesi_locate_device_with_valid_copy( descC->super.moesi_map, GEMM_KEY( descC, Cm, Cn) );
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
