/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#define PRECISION_z

#include "dague_config.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <plasma.h>
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


int gpu_kernel_init_zgemm( dague_context_t* dague_context,
                           tiled_matrix_desc_t *tileA );

static inline
int gpu_kernel_push_zgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );

static inline
int gpu_kernel_submit_zgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );

static inline
int gpu_kernel_pop_zgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );

static inline
int  gpu_kernel_epilog_zgemm( gpu_device_t* gpu_device,
                              dague_execution_context_t* this_task );

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgemm( gpu_device_t              *gpu_device,
                               dague_execution_context_t *this_task,
                               dague_ddesc_t             *ddesca );
#endif

#include "gpu_scheduling.h"

static tiled_matrix_desc_t* UGLY_A;
static int ndevices = 0;

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgemm( gpu_device_t              *gpu_device,
                               dague_execution_context_t *this_task,
                               dague_ddesc_t             *ddesca )
{
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        int data_id =
            ddesca->data_key(ddesca, this_task->locals[1].value, this_task->locals[2].value);
        uint64_t task_id =
            this_task->function->key( this_task->dague_object, this_task->locals );

        dague_profile_ddesc_info_t info;
        info.desc = ddesca;
        info.id = data_id;
        dague_profiling_trace( gpu_device->profiling,
                               DAGUE_PROF_FUNC_KEY_START(this_task->dague_object,
                                                         this_task->function->function_id),
                               task_id, this_task->dague_object->object_id,
                               (void*)&info);
    }
}
#endif  /* defined(DAGUE_PROF_TRACE) */

int gpu_kernel_init_zgemm( dague_context_t* dague_context,
                           tiled_matrix_desc_t *tileA )
{
    char *env;
    int i, dindex, nbgpus;
    (void)dague_context;

    UGLY_A = tileA;

    /**
     * Right now the zgemm function available with DPLASMA can only handle
     * square tiles with a size multiple of 64.
     */
    if( (tileA->mb != tileA->nb) || ((tileA->nb % 64) != 0) ) {
        ERROR(("The CUDA GEMM version provided by DPLASMA is limited to 64 multiple square tiles\n"));
        return -1;
    }

    nbgpus = dague_active_gpu();
    //gpu_active_devices = (gpu_device_t** )calloc(nbgpus, sizeof(gpu_device_t*));
    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        char library_name[FILENAME_MAX];
        char function_name[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];
        gpu_device->function = NULL;

        snprintf(library_name,  FILENAME_MAX, "libdplasma-sm_%d%d.so", gpu_device->major, gpu_device->minor);
        snprintf(function_name, FILENAME_MAX, "magmablas_zgemm_SM%d%d", gpu_device->major, gpu_device->minor);

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, {continue;} );

        /* If not disallowed by env, load from static linked kernels */
        env = getenv("DAGUE_CUBIN_NOSTATIC");
        if( !env || (('1' != env[0]) && ('y' != env[0])) ) {
            void* dlh;
            dlh = dlopen(NULL, RTLD_NOW);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            gpu_device->function = dlsym(dlh, function_name);
            dlclose(dlh);
        }

        /* If not found statically, try shared lib */
/*         if(NULL == gpu_device->hcuFunction) { */
/*             env = getenv("DAGUE_CUBIN_PATH"); */
/*             snprintf(module_path, FILENAME_MAX, "%s/zgemm-sm_%1d%1d.cubin", */
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

        if(NULL == gpu_device->function) {
            void* dlh;
            dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
            if(NULL == dlh) ERROR(("Could not find %s library (%s)\n", library_name, dlerror()));
            gpu_device->function = dlsym(dlh, function_name);
            dlclose(dlh);
        }

        if(NULL == gpu_device->function) return -1;

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

#define ddescA(ec) (UGLY_A)
#define ddescB(ec) ddescA(ec)
#define ddescC(ec) ddescA(ec)

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
gpu_kernel_push_zgemm( gpu_device_t* gpu_device,
                       dague_execution_context_t* this_task,
                       CUstream stream )
{
    int tile_size, ret, k, n, m, move_data_count = 0;
    int sizeloc[MAX_PARAM_COUNT];

    k = this_task->locals[0].value;
    m = this_task->locals[1].value;
    n = this_task->locals[2].value;

    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(n, k),
                           &(this_task->data[0].mem2dev_data));
    if( NULL == (this_task->data[0].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(m, k),
                           &(this_task->data[1].mem2dev_data));
    if( NULL == (this_task->data[1].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    dague_gpu_data_get_elt(&dague_gpu_map, GEMM_KEY(m, n),
                           &(this_task->data[2].mem2dev_data));
    if( NULL == (this_task->data[2].mem2dev_data)->device_elem[gpu_device->index])
        move_data_count++;

    this_task->data[3].mem2dev_data =  NULL;  /* last element */

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        tile_size = UGLY_A->mb*UGLY_A->nb*sizeof(Dague_Complex64_t);
        sizeloc[0] = tile_size;
        sizeloc[1] = tile_size;
        sizeloc[2] = tile_size;

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

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[0]->name, n, k));
    tile_size = ddescA(this_task)->mb * ddescA(this_task)->nb * dague_datadist_getsizeoftype(ddescA(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[1]->name, m, k));
    tile_size = ddescB(this_task)->mb * ddescB(this_task)->nb * dague_datadist_getsizeoftype(ddescB(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[1]->access_type,
                                   &(this_task->data[1]), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[2]->name, m, n));
    tile_size = ddescC(this_task)->mb * ddescC(this_task)->nb * dague_datadist_getsizeoftype(ddescC(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[2]->access_type,
                                   &(this_task->data[2]), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }
  release_and_return_error:
    return ret;
}

static inline int
gpu_kernel_submit_zgemm( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
    Dague_Complex64_t alpha = -1.0;
    Dague_Complex64_t beta  = 1.0;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    void (*cuda_zgemm) ( char TRANSA, char TRANSB, int m, int n, int k,
                         Dague_Complex64_t alpha, Dague_Complex64_t *d_A, int lda,
                                                  Dague_Complex64_t *d_B, int ldb,
                         Dague_Complex64_t beta,  Dague_Complex64_t *d_C, int ldc,
                         CUstream stream ) = gpu_device->function;
    
    gpu_elem_A = (gpu_elem_t *)this_task->data[0].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_B = (gpu_elem_t *)this_task->data[1].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_C = (gpu_elem_t *)this_task->data[2].mem2dev_data->device_elem[gpu_device->index];
    d_A = gpu_elem_A->gpu_mem;
    d_B = gpu_elem_B->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;

    DEBUG2(( "GPU[%1d]:\Enqueue on device %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, this_task, dague_gpu_map.desc);
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = cudaSuccess;
    cuda_zgemm( 'N', 'T', ddescA(this_task)->nb, ddescA(this_task)->nb, ddescA(this_task)->nb,
                alpha, (Dague_Complex64_t*)d_A, ddescA(this_task)->nb,
                       (Dague_Complex64_t*)d_B, ddescA(this_task)->nb,
                beta,  (Dague_Complex64_t*)d_C, ddescA(this_task)->nb,
                stream );

    DAGUE_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
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
gpu_kernel_pop_zgemm( gpu_device_t* gpu_device,
                      dague_execution_context_t* this_task,
                      CUstream stream )
{
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, tile_size, how_many = 0, i;
    cudaError_t status;

    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( gpu_elem->generic.memory_elem == this_task->data[i].mem2dev_data );
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->generic.readers--; assert(gpu_elem->generic.readers >= 0);
            if( (0 == gpu_elem->generic.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_ulist_remove( gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_CONSTRUCT(gpu_elem);
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            /* If we're not using this anymore on the GPU it should be moved back to the CPU */
        }
    }

    gpu_elem = (gpu_elem_t*)this_task->data[2].mem2dev_data->device_elem[gpu_device->index];
    tile_size = ddescC(this_task)->mb * ddescC(this_task)->nb * dague_datadist_getsizeoftype(ddescC(this_task)->mtype);

    /* Stage the transfer of the data back to main memory */
    gpu_device->required_data_out += tile_size;
    assert( ((dague_list_item_t*)gpu_elem)->list_next == (dague_list_item_t*)gpu_elem );
    assert( ((dague_list_item_t*)gpu_elem)->list_prev == (dague_list_item_t*)gpu_elem );
    if( this_task->locals[2].value == (this_task->locals[0].value+1) ) {  /* n == (k + 1) */
        DEBUG3(("GPU Request out of GPU for %s key %d\n", this_task->function->in[2]->name, this_task->data[2].mem2dev_data->key));
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start,
                                   (unsigned long)this_task, this_task->dague_object->object_id,
                                   NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        /* Move the data back into main memory */
        status = (cudaError_t)cuMemcpyDtoHAsync( ADATA(this_task->data[2].data), gpu_elem->gpu_mem, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[2]->name,
                                           (void*)(long)gpu_elem->gpu_mem, (void*)ADATA(this_task->data[2].data)));
                                  return_code = -2;
                                  goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
        how_many++;
    }
 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
static inline int
gpu_kernel_epilog_zgemm( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task )
{
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
        if( this_task->locals[2].value == (this_task->locals[0].value+1) ) {  /* n == (k  + 1) */
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
               int uplo )
{
    int which_gpu, n, m;

    m = this_task->locals[1].value;
    n = this_task->locals[2].value;
    (void)uplo;
    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dague_gpu_data_elt_write_owner( &dague_gpu_map, GEMM_KEY(m, n) );
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
        device_load[best_index+1] = best_weight;  /* update the expected load: 0 is for the cores */
        which_gpu = best_index;
    }
    return gpu_kernel_scheduler_zgemm( eu_context, this_task, which_gpu );
}
