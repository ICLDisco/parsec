/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <plasma.h>
#include <cublas.h>
#include "dague.h"
#include "gpu_data.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "datarepo.h"
#include "data_dist/matrix/matrix.h"

#include "cuda_sgemm.h"

#define KERNEL_NAME sgemm


int gpu_kernel_init_sgemm( dague_context_t* dague_context, 
                           tiled_matrix_desc_t *tileA );

static inline
int gpu_kernel_push_sgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );
    
static inline
int gpu_kernel_submit_sgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );

static inline
int gpu_kernel_pop_sgemm( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream );

static inline
int  gpu_kernel_epilog_sgemm( gpu_device_t* gpu_device,
                              dague_execution_context_t* this_task );

static inline 
void gpu_kernel_profile_sgemm( gpu_device_t              *gpu_device,
                               dague_execution_context_t *this_task,
                               dague_ddesc_t             *ddesca );

#include "gpu_scheduling.h"

#if DPLASMA_SCHEDULING
uint32_t *gpu_set;
#endif

static tiled_matrix_desc_t* UGLY_A;
static int ndevices = 0;

static int OHM_N = 5;
static int OHM_M = 3;

/* FIXME */
#define dague_gpu_1gpu_fini( ... )

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_sgemm( gpu_device_t              *gpu_device,
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

int gpu_kernel_init_sgemm( dague_context_t* dague_context, 
                           tiled_matrix_desc_t *tileA )
{
    char *env;
    int i, dindex;
    int nbgpus;
    (void)dague_context;

    UGLY_A = tileA;

    /**
     * Right now the sgemm function available with DPLASMA can only handle
     * square tiles with a size multiple of 64.
     */
    if( (tileA->mb != tileA->nb) || ((tileA->nb % 64) != 0) ) {
        ERROR(("The CUDA GEMM version provided by DPLASMA is limited to 64 multiple square tiles\n"));
        return -1;
    }

    env = getenv("OHM_N");
    if( NULL != env )
        OHM_N = atoi(env);

    env = getenv("OHM_M");
    if( NULL != env )
        OHM_M = atoi(env);

    nbgpus = dague_active_gpu();
#if DPLASMA_SCHEDULING
    gpu_set = (uint32_t*)calloc(UGLY_A->nt, sizeof(uint32_t));
#endif
    //gpu_active_devices = (gpu_device_t** )calloc(nbgpus, sizeof(gpu_device_t*));
    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        char module_path[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, { dague_gpu_1gpu_fini(gpu_device); continue;} );

        /* If not disallowed by env, load from static linked kernels */
        /* This is non functional, as the ptr is not a CuFunction. */
        gpu_device->hcuFunction = NULL;
        env = getenv("DAGUE_CUBIN_NOSTATIC");
        if( !env || (('1' != env[0]) && ('y' != env[0])) ) {
            void* dlh;
            snprintf(module_path, FILENAME_MAX, "sgemmNT_SM%d%d", gpu_device->major, gpu_device->minor);
            dlh = dlopen(NULL, RTLD_NOW);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            gpu_device->hcuFunction = dlsym(dlh, module_path);
            dlclose(dlh);
        }

        /* If not found statically, cuload it */
        if(NULL == gpu_device->hcuFunction) {
            env = getenv("DAGUE_CUBIN_PATH");
            snprintf(module_path, FILENAME_MAX, "%s/sgemm-sm_%1d%1d.cubin",
                     env?env:"../cores", gpu_device->major, gpu_device->minor);
            status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                    {
                                        WARNING(("GPU:\tUnable to load `%s'\n", module_path));
                                        dague_gpu_1gpu_fini(gpu_device); 
                                        continue;
                                    } );
            snprintf(module_path, FILENAME_MAX, "sgemmNT_SM%d%d", gpu_device->major, gpu_device->minor);
            DEBUG3(("CUDA MODULE %s\n", module_path));
            status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, module_path );
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                    {
                                        WARNING(("GPU:\tUnable to find the function `%s'\n", module_path));
                                        dague_gpu_1gpu_fini(gpu_device); 
                                        continue;
                                    } );
        }
        if(NULL == gpu_device->hcuFunction) return -1;
        if( 1 == gpu_device->major ) {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
        } else {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 64, 4, 1 );
        }


        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {dague_gpu_1gpu_fini(gpu_device); continue;} );
        gpu_device->index = (uint8_t)dindex;
        gpu_enabled_devices[dindex++] = gpu_device;
    }

    /* Update the number of GPUs available */
    dague_data_enable_gpu( dindex );
    ndevices = dindex;

    return 0;
}

#define ALIGN_UP(OFFSET, ALIGN) \
    (OFFSET) = ((OFFSET) + (ALIGN) - 1) & ~((ALIGN) - 1)
#define CU_PUSH_POINTER( FUNCTION, OFFSET, PTR )                        \
    do {                                                                \
        void* __ptr = (void*)(size_t)(PTR);                             \
        ALIGN_UP((OFFSET), __alignof(void*));                           \
        cuParamSetv( (FUNCTION), (OFFSET), &__ptr, sizeof(void*));      \
        (OFFSET) += sizeof(void*);                                      \
    } while (0)
#define CU_PUSH_INT( FUNCTION, OFFSET, VALUE )                          \
    do {                                                                \
        ALIGN_UP((OFFSET), __alignof(int));                             \
        cuParamSeti( (FUNCTION), (OFFSET), (VALUE) );                   \
        (OFFSET) += sizeof(int);                                        \
    } while (0)
#define CU_PUSH_FLOAT( FUNCTION, OFFSET, VALUE )                        \
    do {                                                                \
        ALIGN_UP((OFFSET), __alignof(float));                           \
        cuParamSetf( (FUNCTION), (OFFSET), (VALUE) );                   \
        (OFFSET) += sizeof(float);                                      \
    } while (0)

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
gpu_kernel_push_sgemm( gpu_device_t* gpu_device,
                       dague_execution_context_t* this_task,
                       CUstream stream )
{
    int sizeloc[MAX_PARAM_COUNT];
    int tile_size, ret;
    int k, n, m, move_data_count = 0;
    int eltsize = 0;
    gpu_elem_t* gpu_elem;
    (void)eltsize;

    k = this_task->locals[0].value;
    m = this_task->locals[1].value;
    n = this_task->locals[2].value;

    gpu_elem = dague_gpu_get_data_on_gpu(gpu_device, &dague_gpu_map, GEMM_KEY(n, k),
                                         &(this_task->data[0].mem2dev_data) );
    if( NULL == gpu_elem ) move_data_count++;
    gpu_elem = dague_gpu_get_data_on_gpu(gpu_device, &dague_gpu_map, GEMM_KEY(m, k),
                                         &(this_task->data[1].mem2dev_data));
    if( NULL == gpu_elem ) move_data_count++;
    gpu_elem = dague_gpu_get_data_on_gpu(gpu_device, &dague_gpu_map, GEMM_KEY(m, n),
                                         &(this_task->data[2].mem2dev_data));
    if( NULL == gpu_elem ) move_data_count++;

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        tile_size = UGLY_A->mb*UGLY_A->nb*sizeof(float);
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

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start,
                               (unsigned long)this_task, this_task->dague_object->object_id,
                               NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[0]->name, n, k));
    tile_size = ddescA(this_task)->mb * ddescA(this_task)->nb * dague_datadist_getsizeoftype(ddescA(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, GEMM_KEY(n, k), this_task->function->in[0]->access_type,
                                   this_task->data[0].mem2dev_data,
                                   ADATA(this_task->data[0].data), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[1]->name, m, k));
    tile_size = ddescB(this_task)->mb * ddescB(this_task)->nb * dague_datadist_getsizeoftype(ddescB(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, GEMM_KEY(m, k), this_task->function->in[1]->access_type,
                                   this_task->data[1].mem2dev_data,
                                   ADATA(this_task->data[1].data), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }
    
    DEBUG3(("GPU:\tRequest Data of %s(%d, %d) on GPU\n", this_task->function->in[2]->name, m, n));
    tile_size = ddescC(this_task)->mb * ddescC(this_task)->nb * dague_datadist_getsizeoftype(ddescC(this_task)->mtype);
    ret = dague_gpu_data_stage_in( gpu_device, GEMM_KEY(m, n), this_task->function->in[2]->access_type,
                                   this_task->data[2].mem2dev_data,
                                   ADATA(this_task->data[2].data), tile_size, stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }
    assert( NULL != this_task->data[0].mem2dev_data->device_elem[gpu_device->index] );
    assert( NULL != this_task->data[1].mem2dev_data->device_elem[gpu_device->index] );
    assert( NULL != this_task->data[2].mem2dev_data->device_elem[gpu_device->index] );
  release_and_return_error:
    return ret;
}

static inline int
gpu_kernel_submit_sgemm( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
    int grid_width, grid_height;
    float alpha = -1.0, beta = 1.0;
    int offset;

    gpu_elem_A = (gpu_elem_t *)this_task->data[0].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_B = (gpu_elem_t *)this_task->data[1].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_C = (gpu_elem_t *)this_task->data[2].mem2dev_data->device_elem[gpu_device->index];
    d_A = gpu_elem_A->gpu_mem;
    d_B = gpu_elem_B->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;

    DEBUG2(("GPU:\tRequest GPU runs GEMM(%d, %d, %d)\n", this_task->locals[0], this_task->locals[1], this_task->locals[2]));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, this_task, dague_gpu_map.desc);
#endif  /* defined(DAGUE_PROF_TRACE) */

    offset = 0;
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(this_task)->nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_A );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(this_task)->nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(this_task)->nb );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(this_task)->nb );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
    cuParamSetSize( gpu_device->hcuFunction, offset );

    /* cuLaunch: we kick off the CUDA */
    if( 1 == gpu_device->major ) {
        grid_width  = ddescA(this_task)->nb / 64 + (ddescA(this_task)->nb % 64 != 0);
        grid_height = ddescA(this_task)->nb / 16 + (ddescA(this_task)->nb % 16 != 0);
    } else {
        /* Change bx and by to match the values in the fermi gemm code */
#define bx 4
#define by 4
        grid_width  = ddescA(this_task)->nb / (16*bx) + (ddescA(this_task)->nb % (16*bx) != 0);
        grid_height = ddescA(this_task)->nb / (16*by) + (ddescA(this_task)->nb % (16*by) != 0);
    }
    status = (cudaError_t)cuLaunchGridAsync( gpu_device->hcuFunction,
                                             grid_width, grid_height, stream);

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
gpu_kernel_pop_sgemm( gpu_device_t* gpu_device,
                      dague_execution_context_t* this_task,
                      CUstream stream )
{
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, tile_size, how_many = 0, i;
    cudaError_t status;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( gpu_elem->generic.memory_elem == this_task->data[i].mem2dev_data );
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->generic.readers--;
            if( (0 == gpu_elem->generic.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_ulist_remove( gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
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

static inline int
gpu_kernel_epilog_sgemm( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task )
{
    gpu_elem_t* gpu_elem;
    int i;
    
    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( DAGUE_DATA_OWNED == gpu_elem->generic.coherency_state );
        gpu_elem->generic.coherency_state = DAGUE_DATA_SHARED;
        gpu_elem->generic.memory_elem->version = gpu_elem->generic.version;
        this_task->data[2].mem2dev_data->device_owner = -1;

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

#if !defined(DAGUE_GPU_STREAM_PER_TASK)

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int uplo )
{
    int which_gpu;
    int n, m;

    m = this_task->locals[1].value;
    n = this_task->locals[2].value;
    (void)uplo;
    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dague_gpu_data_elt_write_owner( &dague_gpu_map, GEMM_KEY(m, n) );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
        assert( n < UGLY_A->nt );
        if( ndevices > 1){
            /* reverse odd-even */
            /* homogeneous GPU */
            {
                if(n % 2 == 0){
                    which_gpu = gpu_set[n] % ndevices;
                }
                else{
                    which_gpu = ndevices - (gpu_set[n] % ndevices + 1);
                }
            }

            /* heterogenous GPU */
            /* weight by percentage of getting n of (n) with performance factor */
            {


            }

            dague_atomic_inc_32b( &(gpu_set[n]) );
        }
#if DPLASMA_ONLY_GPU
#else
        /*
        **Rectangular Mesh **
        1. Fact, a number of tile ahd GEMMs comes from Matrix size and tile size
        - we may have to change m,n in every tile size/ matrix size
        2. m and n is assign the size of squares which're going to mark over the
        * triangular bunch of GEMMs
        * 3. m % (?) == (?) and n % (?) == (?) marks which tile is gonna be executed on CPU
        * 4. all (?) values affect "square size" and "position"-- which affects how many GEMMs will be executed on CPU
        * 5. Once we superpose/pile up "many square(m,n) -- like a mesh" on to triangular GEMMs, we will be able to caluculate how many GEMMs will be on CPU, also know which tiles 
        * 6. The number GEMMs on GPU and CPU would meet "how many times GPU faster than CPU "
        * I usually use m % 3 == 0 && n % 2 == 0 on C1060 (3x2 square)
        * I usaully use m % 4 == 0 && n % 2 == 0 on C2050 (4x2 square)
        * chance is lower that 1:6 or 1:8 becasue we pile up this square on to triangular
        * Why this method ?
        *  - try to finish "each bunch of GEMMs" as soon as poosible with GPU+CPU
        *  - plus "balancing" between CPU/GPU
        **/

        if( ((m % OHM_M) == 0) && ( (n % OHM_N) == 0) ){
            dague_atomic_inc_32b( &(dague_cpu_counter) );
            return -99;
        }
#endif  /* DPLASMA_ONLY_GPU */
#endif  /* DPLASMA_SCHEDULING */
    }

    return gpu_kernel_scheduler_sgemm( eu_context, this_task, which_gpu );
}

#else
#error "This case is not correct right now"

static int
gpu_sgemm_internal( gpu_device_t* gpu_device,
                    dague_execution_unit_t* eu_context,
                    dague_execution_context_t* this_task,
                    CUstream stream, int uplo )
{
    int return_code = 0;  /* by default suppose an error */

    (void)eu_context;
    (void)uplo;

   // DEBUG(("GPU:\tExecute GEMM( k = %d, m = %d, n = %d ) [%d] on device %d stream %p\n",
     //      k, m, n, this_task->priority, gpu_device->device_index, (void*)stream));

    return_code = gpu_sgemm_internal_push( gpu_device,
                                           this_task,
                                           stream );
    if( 0 > return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_submit( gpu_device,
                                             this_task,
                                             stream );
    if( 0 != return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_pop( gpu_device,
                                          this_task,
                                          stream );

 release_and_return_error:
    return (return_code < 0 ? return_code : 0);
}

/**
 * This version is based on 4 streams, each of them potentially containing
 * all transfers from memory to the GPU, the kernel execution on the GPU and
 * the transfers from the GPU to the main memory. The synchronizations are
 * based on the fact that each stream contains only tasks related to a single
 * kernel, so waiting for the stream to be empty means everything related to
 * a task has been completed. There might be overlap between the operations on
 * different streams, however it is difficult to schedule in advance transfers
 * related to kernel that will be executed later.
 */
int gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int uplo )
{
    int which_gpu, rc, stream_rc, waiting = 0, submit = 0;
    gpu_device_t* gpu_device;
    cudaError_t status;
    dague_execution_context_t* progress_array[DAGUE_MAX_STREAMS];
    int n, m;

    m = this_task->locals[1].value;
    n = this_task->locals[2].value;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dague_gpu_data_elt_write_owner( &dague_gpu_map, GEMM_KEY(m, n) );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
        assert( n < UGLY_A->nt );
        if(ndevices > 1) {
        /* reverse odd-even */
        /* homogeneous GPU */
        if(n % 2 == 0) {
            which_gpu = gpu_set[n] % ndevices;
        }
        else {
            which_gpu = ndevices - (gpu_set[n] % ndevices + 1);
        }

        /* heterogenous GPU */
        /* weight by percentage of getting n of (n) with performance factor */
        {

        }
        dague_atomic_inc_32b( &(gpu_set[n]) );
    }
    /*c1060 4 - 2  384-448  3-0-2-0 960 */
    /*c2050 5 - 2 448       4-2 960 */

#if DPLASMA_ONLY_GPU

#else

     /*
      **Rectangular Mesh **

       1. Fact, number of tile,GEMMs is come from Matrix size and tile size
       	- we may have to change m,n in every tile size/ matrix size
       2. m and n is assign the size of squares which're going to mark over the
     * triangular bunch of GEMMs
     * 3. m % (?) == (?) and n % (?) == (?) marks which tile is gonna be executed on CPU
     * 4. all (?) values affect "square size" and "position"-- which affects how many GEMMs will be executed on CPU
     * 5. Once we superpose/pile up "many square(m,n) -- like a mesh" on to triangular GEMMs, we will be able to caluculate how many GEMMs will be on CPU, also know which tiles 
     * 6. The number GEMMs on GPU and CPU would meet "how many times GPU faster than CPU "
     * I usually use m % 3 == 0 && n % 2 == 0 on C1060 (3x2 square)
     * I usaully use m % 4 == 0 && n % 2 == 0 on C2050 (4x2 square)
     * chance is lower that 1:6 or 1:8 becasue we pile up this square on to triangular
     *
     * Why this method ?
     * 	 - try to finish "each bunch of GEMMs" as soon as poosible with GPU+CPU
     * 	 - plus "balancing" between CPU/GPU
     */
    if( ((m % OHM_M) == 0) && ( (n % OHM_N) == 0) ){
        dague_atomic_inc_32b( &(dague_cpu_counter) );
        return -99;
    }
#endif

#endif
    }
    gpu_device = gpu_enabled_devices[which_gpu];

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(gpu_device->pending), (dague_list_item_t*)this_task );
        return -1;
    }

    status = (cudaError_t)cuCtxPushCurrent(gpu_device->ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {return -2;} );
    for( rc = 0; rc < DAGUE_MAX_STREAMS; rc++ )
        progress_array[rc] = NULL;

 more_work_to_do:
    if( (NULL != this_task) && (NULL == progress_array[submit]) ) {
        progress_array[submit] = this_task;

        /* Push this task into the GPU */
        rc = gpu_sgemm_internal( gpu_device, eu_context, this_task, gpu_device->streams[submit], uplo );
        if( 0 != rc ) {  /* something fishy happened. Reschedule the pending tasks on the cores */
            goto disable_gpu;
        }
        DEBUG3(( "GPU:\tsubmit %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[submit], k, m, n, submit ));
        submit = (submit + 1) % gpu_device->max_streams;
        this_task = NULL;
    }

    if( NULL != progress_array[waiting] ) {
    wait_for_completion:
        stream_rc = cuStreamQuery(gpu_device->streams[waiting]);
        if( CUDA_ERROR_NOT_READY == stream_rc ) {
            goto fetch_more_work;
            /* Task not yet completed */
        } else if( CUDA_SUCCESS == stream_rc ) {  /* Done with this task */
            goto complete_previous_work;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuStreamQuery ", stream_rc,
                                      {return -2;} );
        }
    }

    if( NULL == this_task ) {
        goto fetch_more_work;
    }
    goto more_work_to_do;

 complete_previous_work:
    /* Everything went fine so far, the result is correct and back in the main memory */
    DEBUG3(( "GPU:\tcomplete %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[waiting], k, m, n, waiting ));
    dague_complete_execution( eu_context, progress_array[waiting] );
    progress_array[waiting] = NULL;
    waiting = (waiting + 1) % gpu_device->max_streams;

    gpu_device->executed_tasks++;
/*	dague_atomic_dec_32b( &(gpu_device->workload) );*/
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {return -1;} );
        return -1;
    }

 fetch_more_work:
    /* Do we still have room in the progress_array? */
    if( NULL != progress_array[submit] )
        goto wait_for_completion;

    this_task = (dague_execution_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL == this_task ) {  /* Collisions, save time and come back here later */
        goto more_work_to_do;
    }

    m = this_task->locals[1].value;
    n = this_task->locals[2].value;

    goto more_work_to_do;

    /* a device ... */
 disable_gpu:
    __dague_schedule( eu_context, this_task);
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    while( rc != 0 ) {
        this_task = (dague_execution_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
        if( NULL != this_task ) {
            __dague_schedule( eu_context, this_task);
            rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
        }
    }
    status = (cudaError_t)cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {} );
    return -2;
}
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */

