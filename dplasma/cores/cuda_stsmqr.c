/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "cuda_stsmqr.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"

#include <plasma.h>

#include <stdio.h>
#include <cublas.h>

#include "data_distribution.h"

#define DPLASMA_SCHEDULING 1
#define DPLASMA_ONLY_GPU 0
static volatile uint32_t cpu_counter = 0;
static int ndevices = 0;
#if DPLASMA_SCHEDULING
uint32_t *gpu_set;
int *gpu_load;
int MAX_QUEUE = 80;
#endif
#include "data_dist/matrix/matrix.h"
#include "dplasma/lib/memory_pool.h"

static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit );

static tiled_matrix_desc_t* UGLY_A;
static tiled_matrix_desc_t* UGLY_T;

int stsmqr_cuda_init( dague_context_t* dague_context,
                      tiled_matrix_desc_t *tileA,
                      tiled_matrix_desc_t *tileT )
{
    CUdevice hcuDevice;
    int i, nbdevfound;

    UGLY_A = tileA;
    UGLY_T = tileT;

    ndevices = dague_using_gpu();
#if DPLASMA_SCHEDULING
	gpu_set = (uint32_t*)calloc(400, sizeof(uint32_t));
	for( i = 0; i < 400 ; i++){
		gpu_set[i] = 0;
	}
	gpu_load = (int*)calloc(ndevices, sizeof(int));
	for( i = 0; i < ndevices;i++){
		gpu_load[i] = 0;
	}
#endif
    nbdevfound = 0;

    for( i = 0; i < ndevices; i++ ) {
        size_t total_mem, tile_size, thread_gpu_mem, free_mem;
        unsigned int nb_allocations = 0;
        gpu_device_t* gpu_device;
        CUresult status;
        int major, minor;
        char module_path[20];

        status = cuDeviceGet( &hcuDevice, i );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {ndevices = 0; return -1;} );

        status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {ndevices = 0; return -1;} );

        gpu_device = gpu_devices[i];
        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status,
                                {free(gpu_device); gpu_devices[i] = NULL; continue; } );
        
        assert(gpu_device->major < 10 && gpu_device->minor < 10);
        snprintf(module_path, 20, "stsmqr-sm_%1d%1d.cubin", gpu_device->major, gpu_device->minor);
        status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                {
                                    fprintf(stderr, "*** unable to load `%s'\n", module_path);
                                    cuCtxDestroy( gpu_device->ctx );
                                    free(gpu_device);
                                    gpu_devices[i] = NULL;
                                    continue;
                                 } );
                    
        status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, "stsmqrNT" );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                {
                                    cuCtxDestroy( gpu_device->ctx );
                                    free(gpu_device);
                                    gpu_devices[i] = NULL;
                                    continue;
                                } );
        if( 1 == gpu_device->major ) {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
        } else {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 64, 4, 1 );
        }
        /**
         * Prepare the reusable memory on the GPU.
         */
        gpu_qr_data_map_init( 0, gpu_device, tileA );
        gpu_qr_data_map_init( 1, gpu_device, tileT );

        /**
         * It appears that CUDA allocate the memory in chunks of 1MB,
         * so we need to adapt to this.
         */
        tile_size = tileA->bsiz * sizeof(float);
        cuMemGetInfo( &free_mem, &total_mem );
        /* We allocate 9/10 of the total memory */
        thread_gpu_mem = (total_mem - total_mem / 10);

        while( free_mem > (total_mem - thread_gpu_mem) ) {
            gpu_elem_t* gpu_elem;
            cudaError_t cuda_status;

            if( nb_allocations > ((tileA->mt * tileA->nt) >> 1) )
                break;
            gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
            dague_linked_list_item_construct( (dague_list_item_t*)gpu_elem );
            
            cuda_status = (cudaError_t)cuMemAlloc( &(gpu_elem->gpu_mem), tile_size);
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                    ({
                                        size_t _free_mem, _total_mem;
                                        cuMemGetInfo( &_free_mem, &_total_mem );
                                        printf("Per context: free mem %zu total mem %zu\n", _free_mem, _total_mem);
                                        free( gpu_elem );
                                        break;
                                    }) );
            nb_allocations++;
            gpu_elem->memory_elem = NULL;
            dague_linked_list_add_tail( gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem );
            cuMemGetInfo( &free_mem, &total_mem );
        }
        if( 0 == nb_allocations ) {
            printf("Rank %d Cannot allocate memory on GPU %d. Skip it!\n", dague_context->my_rank, i);
            cuCtxDestroy( gpu_device->ctx );
            free(gpu_device);
            gpu_devices[i] = NULL;
            continue;
        }
        printf( "Allocate %u tiles on the GPU memory\n", nb_allocations );
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); return -1;} );
        nbdevfound++;
    }

    if( nbdevfound == 0 ) {
        fprintf(stderr, "STRSM-GPU: No suitable device found\n");
        return -1;
    }
    return 0;
}

int stsmqr_cuda_fini(dague_context_t* dague_context)
{
    cudaError_t status;
    gpu_elem_t* gpu_elem;
    gpu_device_t* gpu_device;
    int total = 0, *gpu_counter, i, j, active_devices = 0;
    uint64_t *transferred_in, *transferred_out, total_data_in = 0, total_data_out = 0;
    uint64_t *required_in, *required_out;
    float gtotal = 0.0, best_data_in, best_data_out;
    char *data_in_unit, *data_out_unit;

    if (ndevices <= 0)
        return 0;

    /* GPU counter for STSMQR / each */
    gpu_counter = (int*)calloc(ndevices, sizeof(int));
    transferred_in  = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    transferred_out = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));

    for(i = 0; i < ndevices; i++) {
        gpu_device = gpu_devices[i];

        if( NULL == gpu_device )
            continue;

        status = (cudaError_t)cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                {continue;} );
        status = (cudaError_t)cuCtxSynchronize();
        DAGUE_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
                                {continue;} );
        /* Save the statistics */
        gpu_counter[gpu_device->id]     += gpu_device->executed_tasks;
        transferred_in[gpu_device->id]  += gpu_device->transferred_data_in;
        transferred_out[gpu_device->id] += gpu_device->transferred_data_out;
        required_in[gpu_device->id]     += gpu_device->required_data_in;
        required_out[gpu_device->id]    += gpu_device->required_data_out;
        
        /**
         * Release the GPU memory.
         */
        while( NULL != (gpu_elem = (gpu_elem_t*)dague_linked_list_remove_head( gpu_device->gpu_mem_lru )) ) {
            cuMemFree( gpu_elem->gpu_mem );
            free( gpu_elem );
        }
        /**
         * Release all streams
         */
        for( j = 0; j < gpu_device->max_streams; j++ ) {
            cuStreamDestroy( gpu_device->streams[j] );
        }
        
        status = (cudaError_t)cuCtxDestroy( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(FINI) cuCtxDestroy ", status,
                                {continue;} );
        free(gpu_device->gpu_mem_lru);
        free(gpu_device);
        active_devices++;
    }

    /* No active devices */
    if( 0 == active_devices )
        return 0;

    /* Print statisitics */
    for( i = 0; i < ndevices; i++ ) {
        total += gpu_counter[i];
        total_data_in  += transferred_in[i];
        total_data_out += transferred_out[i];
    }
    if( 0 == total_data_in ) total_data_in = 1;
    if( 0 == total_data_out ) total_data_out = 1;
    
    gtotal = (float)total + (float)cpu_counter;
    printf("------------------------------------------------------------------------------\n");
    printf("|PU %4d  |  # STSMQR   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n", dague_context->my_rank);
    printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
    for( i = 0; i < ndevices; i++ ) {
        compute_best_unit( transferred_in[i],  &best_data_in, &data_in_unit );
        compute_best_unit( transferred_out[i], &best_data_out, &data_out_unit );
        printf("|GPU:  %2d |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
               i, gpu_counter[i], (gpu_counter[i]/gtotal)*100.00,
               best_data_in, data_in_unit, (((float)transferred_in[i]) / required_in[i]) * 100.0,
               best_data_out, data_out_unit, (((float)transferred_out[i]) / required_out[i]) * 100.0 );
    }
    printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
    compute_best_unit( total_data_in,  &best_data_in, &data_in_unit );
    compute_best_unit( total_data_out, &best_data_out, &data_out_unit );
    printf("|All GPUs |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
           total, (total/gtotal)*100.00,
           best_data_in, data_in_unit, 100.0,
           best_data_out, data_out_unit, 100.0);
    printf("|All CPUs |%10u | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
           cpu_counter, (cpu_counter / gtotal)*100.00,
           0.0, " ", 0.0, 0.0, " ", 0.0);
    printf("------------------------------------------------------------------------------\n");
    free(gpu_counter);
    free(transferred_in);
    free(transferred_out);
    free(required_in);
    free(required_out);

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

//#include "generated/sgeqrf.h"
//#define ddescA(ec) ((tiled_matrix_desc_t *)(((dague_sgeqrf_object_t*)(ec)->dague_object)->A))
//#define ddescT(ec) ((tiled_matrix_desc_t *)(((dague_sgeqrf_object_t*)(ec)->dague_object)->T))
#define ddescA(ec) (UGLY_A)
#define ddescT(ec) (UGLY_T)

static inline int
gpu_stsmqr_internal_push( gpu_device_t* gpu_device,
                         dague_execution_context_t* exec_context,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    dague_arena_chunk_t *aA, *aB, *aC;
    int tile_size, return_code = 0, on_gpu;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
    void *A, *B, *C;
    int k, n, m;

    k = exec_context->locals[0].value;
    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;
    aA = exec_context->data[0].data;
    aB = exec_context->data[1].data;
    aC = exec_context->data[2].data;
    A = ADATA(aA);
    B = ADATA(aB);
    C = ADATA(aC);

    tile_size = ddescA(exec_context)->mb*ddescA(exec_context)->nb*sizeof(float);
#if defined(DAGUE_PROF_TRACE)
    dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start, 0, NULL );
#endif  /* defined(PROFILING) */

    on_gpu = gpu_qr_data_is_on_gpu(0, gpu_device, ddescA(exec_context), DAGUE_READ, n, k, &gpu_elem_A);
    gpu_elem_A->memory_elem->memory = A;
    d_A = gpu_elem_A->gpu_mem;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push A into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_A, A, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_A) ", status, 
                                  {printf("<<%p>> -> <<%p>> [%d]\n", (void*)A, (void*)(long)d_A, tile_size); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }
    exec_context->data[0].gpu_data = (struct gpu_elem_t *)gpu_elem_A;

    on_gpu = gpu_qr_data_is_on_gpu(0, gpu_device, ddescA(exec_context), DAGUE_READ, m, k, &gpu_elem_B);
    d_B = gpu_elem_B->gpu_mem;
    gpu_elem_B->memory_elem->memory = B;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push B into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_B, B, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_B) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)B, (void*)(long)d_B); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }
    exec_context->data[1].gpu_data = (struct gpu_elem_t *)gpu_elem_B;

    on_gpu = gpu_qr_data_is_on_gpu(0, gpu_device, ddescA(exec_context), DAGUE_READ | DAGUE_WRITE, m, n, &gpu_elem_C);
    d_C = gpu_elem_C->gpu_mem;
    gpu_elem_C->memory_elem->memory = C;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push C into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_C, C, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)C, (void*)(long)d_C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }
    exec_context->data[2].gpu_data = (struct gpu_elem_t *)gpu_elem_C;

#if defined(DAGUE_PROF_TRACE)
    dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_end, 0, NULL );
#endif  /* defined(PROFILING) */

 release_and_return_error:
    return return_code;
}

static inline int
gpu_stsmqr_internal_submit( gpu_device_t* gpu_device,
                           dague_execution_context_t* exec_context,
                           CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
    int grid_width, grid_height;
    float alpha = -1.0, beta = 1.0;
    int offset;

    gpu_elem_A = (gpu_elem_t *)exec_context->data[0].gpu_data;
    gpu_elem_B = (gpu_elem_t *)exec_context->data[1].gpu_data;
    gpu_elem_C = (gpu_elem_t *)exec_context->data[2].gpu_data;
    d_A = gpu_elem_A->gpu_mem;
    d_B = gpu_elem_B->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;

#if defined(DAGUE_PROF_TRACE)
    dague_profiling_trace( gpu_device->profiling, exec_context->dague_object->profiling_array[0 + 2 * exec_context->function->function_id], 1, NULL );
#endif  /* defined(PROFILING) */
    offset = 0;
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(exec_context)->nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_A );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(exec_context)->nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(exec_context)->nb );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA(exec_context)->nb );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
    cuParamSetSize( gpu_device->hcuFunction, offset );

    /* cuLaunch: we kick off the CUDA */
    if( 1 == gpu_device->major ) {
        grid_width  = ddescA(exec_context)->nb / 64 + (ddescA(exec_context)->nb % 64 != 0);
        grid_height = ddescA(exec_context)->nb / 16 + (ddescA(exec_context)->nb % 16 != 0);
    } else {
        grid_width  = ddescA(exec_context)->nb / 64 + (ddescA(exec_context)->nb % 64 != 0);
        grid_height = ddescA(exec_context)->nb / 64 + (ddescA(exec_context)->nb % 64 != 0);
    }
    status = (cudaError_t)cuLaunchGridAsync( gpu_device->hcuFunction,
                                             grid_width, grid_height, stream);

    DAGUE_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
                              {return -1;} );

#if defined(DAGUE_PROF_TRACE)
    dague_profiling_trace( gpu_device->profiling, exec_context->dague_object->profiling_array[1 + 2 * exec_context->function->function_id], 1, NULL );
#endif  /* defined(PROFILING) */
    return 0;
}

static inline int
gpu_stsmqr_internal_pop( gpu_device_t* gpu_device,
                        dague_execution_context_t* exec_context,
                        CUstream stream )
{
    dague_arena_chunk_t *aC;
    gpu_elem_t *gpu_elem_C = NULL;
    int return_code = 0, tile_size;
    cudaError_t status;
    CUdeviceptr d_C;
    void* C;
    int n, k;

    k = exec_context->locals[0].value;
    n = exec_context->locals[2].value;

    gpu_elem_C = (gpu_elem_t *)exec_context->data[2].gpu_data;
    aC = exec_context->data[2].data;
    d_C = gpu_elem_C->gpu_mem;
    C = ADATA(aC);

    tile_size = ddescA(exec_context)->mb*ddescA(exec_context)->nb*sizeof(float);

    /* Pop C from the GPU */
    gpu_device->required_data_out += tile_size;
    if( (n == k+1) ) {
#if defined(DAGUE_PROF_TRACE)
        dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start, 2, NULL );
#endif  /* defined(PROFILING) */
        /* Pop C from the GPU */
        status = (cudaError_t)cuMemcpyDtoHAsync( C, d_C, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)(long)d_C, (void*)C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
#if defined(DAGUE_PROF_TRACE)
        dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_end, 2, NULL );
#endif  /* defined(PROFILING) */
    }
 release_and_return_error:
    return return_code;
}

static int
gpu_stsmqr_internal( gpu_device_t* gpu_device,
                    dague_execution_unit_t* eu_context,
                    dague_execution_context_t* exec_context,
                    CUstream stream )
{
    int return_code = 0;  /* by default suppose an error */

    (void)eu_context;

    DEBUG(("Execute STSMQR( k = %d, m = %d, n = %d ) [%d] on device %d stream %p\n",
           exec_context->locals[0], exec_context->locals[1], exec_context->locals[2], exec_context->priority, gpu_device->id, (void*)stream));

    return_code = gpu_stsmqr_internal_push( gpu_device,
                                           exec_context,
                                           stream );
    if( 0 != return_code ) goto release_and_return_error;

    return_code = gpu_stsmqr_internal_submit( gpu_device,
                                             exec_context,
                                             stream );
    if( 0 != return_code ) goto release_and_return_error;

    return_code = gpu_stsmqr_internal_pop( gpu_device,
                                          exec_context,
                                          stream );

 release_and_return_error:
    return return_code;
}

/* Try to execute a STSMQR on a GPU.
 *
 * Returns:
 *  0 - if the STSMQR should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the STSMQR is scheduled to be executed on a GPU.
 */
int gpu_stsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* exec_context )
{
    int which_gpu, rc, stream_rc, waiting = 0, submit = 0;
    gpu_device_t* gpu_device;
    cudaError_t status;
    dague_execution_context_t* progress_array[DAGUE_MAX_STREAMS];
    int n, m;

    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;

    DEBUG(("STSMQR( k = %d, m = %d, n = %d )\n", exec_context->locals[0], exec_context->locals[1], exec_context->locals[2]));
    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = gpu_qr_data_tile_write_owner( 0, ddescA(exec_context), m, n );
/*    printf("k=%d, m=%d, n=%d\n",k,m,n);*/
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
    if(ndevices > 1){
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
#endif
    }
    gpu_device = gpu_devices[which_gpu];\

#if DPLASMA_SCHEDULING	

#if DPLASMA_ONLY_GPU

#else
/* return task to CPU when GPU got too much in queue */
/* MAX QUEUE would derive from CPU(CORE) performance VS GPU performance individually*/
    if(gpu_device->mutex > MAX_QUEUE ){
        dague_atomic_inc_32b( &(cpu_counter) );
        return -99;
    }
#endif
    /* keep n -- not being used yet*/
    gpu_load[gpu_device->id]+=n;
#endif

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)exec_context );
        dague_dequeue_push_back( &(gpu_device->pending), (dague_list_item_t*)exec_context );
        return -1;
    }

    status = (cudaError_t)cuCtxPushCurrent(gpu_device->ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {return -2;} );
    for( rc = 0; rc < DAGUE_MAX_STREAMS; rc++ )
        progress_array[rc] = NULL;

 more_work_to_do:
    if( (NULL != exec_context) && (NULL == progress_array[submit]) ) {
        progress_array[submit] = exec_context;

        /* Push this task into the GPU */
        rc = gpu_stsmqr_internal( gpu_device, eu_context, exec_context, gpu_device->streams[submit] );
        if( 0 != rc ) {  /* something fishy happened. Reschedule the pending tasks on the cores */
            goto disable_gpu;
        }
        /*printf( "GPU submit %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[submit], k, m, n, submit );*/
        submit = (submit + 1) % gpu_device->max_streams;
        exec_context = NULL;
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
            DAGUE_CUDA_CHECK_ERROR( "cuStreamQuery ", status,
                                      {return -2;} );
        }
    }

    if( NULL == exec_context ) {
        goto fetch_more_work;
    }
    goto more_work_to_do;

 complete_previous_work:
    /* Everything went fine so far, the result is correct and back in the main memory */
    /*printf( "GPU complete %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[waiting], k, m, n, waiting );*/
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

    exec_context = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
    if( NULL == exec_context ) {  /* Collisions, save time and come back here later */
        goto more_work_to_do;
    }

    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;

    goto more_work_to_do;

    /* a device ... */
 disable_gpu:
    __dague_schedule( eu_context, exec_context, 0 );
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    while( rc != 0 ) {
        exec_context = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
        if( NULL != exec_context ) {
            __dague_schedule( eu_context, exec_context, 0 );
            rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
        }
    }
    status = (cudaError_t)cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {} );
    return -2;
}


/****************************************************
 ** GPU-DATA that is Qr Specific Starts Here **
 ****************************************************/

#include "gpu_data.h"
#include "data_distribution.h"
#include "linked_list.h"

static memory_elem_t** data_mapT = NULL;
static memory_elem_t** data_mapA = NULL;
extern int ndevices;

int gpu_qr_mark_data_usage( int matrixIsT, tiled_matrix_desc_t* data, int type, int col, int row )
{
    memory_elem_t* this_data;
    memory_elem_t** data_map;

    if( matrixIsT )
        data_map = data_mapT;
    else
        data_map = data_mapA;

    if( (NULL == data_map) || (NULL == (this_data = data_map[col * data->lnt + row])) ) {
        /* Data not on the GPU. Nothing to do */
        return 0;
    }
    if( type & DAGUE_WRITE ) {
        this_data->memory_version++;
        this_data->writer++;
    }
    if( type & DAGUE_READ ) {
        this_data->readers++;
    }
    return 0;
}

int gpu_qr_data_map_init( int matrixIsT,
                          gpu_device_t* gpu_device,
                          tiled_matrix_desc_t* data )
{
    memory_elem_t** data_map;

    if( matrixIsT )
        data_map = data_mapT;
    else
        data_map = data_mapA;

    if( NULL == data_map ) {
        data_map = (memory_elem_t**)calloc(data->lmt * data->lnt, sizeof(memory_elem_t*));
    }
    gpu_device->gpu_mem_lru = (dague_linked_list_t*)malloc(sizeof(dague_linked_list_t));
    dague_linked_list_construct(gpu_device->gpu_mem_lru);
    return 0;
}

int gpu_qr_data_tile_write_owner( int matrixIsT, 
                                  tiled_matrix_desc_t* data,
                                  int col, int row )
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;
    int i;
    memory_elem_t** data_map;

    if( matrixIsT )
        data_map = data_mapT;
    else
        data_map = data_mapA;

    if( NULL == (memory_elem = data_map[col * data->lnt + row]) ) {
        return -1;
    }
    for( i = 0; i < ndevices; i++ ) {
        gpu_elem = memory_elem->gpu_elems[i];
        if( NULL == gpu_elem )
            continue;
        if( gpu_elem->type & DAGUE_WRITE )
            return i;
    }
    return -2;
}

int gpu_qr_data_get_tile( int matrixIsT, 
                          tiled_matrix_desc_t* data,
                          int col, int row,
                          memory_elem_t **pmem_elem )
{
    memory_elem_t* memory_elem;
    int rc = 0;  /* the tile already existed */
    memory_elem_t** data_map;

    if( matrixIsT )
        data_map = data_mapT;
    else
        data_map = data_mapA;

    if( NULL == (memory_elem = data_map[col * data->lnt + row]) ) {
        memory_elem = (memory_elem_t*)calloc(1, sizeof(memory_elem_t) + (ndevices-1) * sizeof(gpu_elem_t*));
        memory_elem->col = col;
        memory_elem->row = row;
        memory_elem->memory_version = 0;
        memory_elem->readers = 0;
        memory_elem->writer = 0;
        memory_elem->memory = NULL;
        rc = 1;  /* the tile has just been created */
        if( 0 == dague_atomic_cas( &(data_map[col * data->lnt + row]), NULL, memory_elem ) ) {
            free(memory_elem);
            rc = 0;  /* the tile already existed */
            memory_elem = data_map[col * data->lnt + row];
        }
    }
    *pmem_elem = memory_elem;
    return rc;
}

/**
 * This function check if the target tile is already on the GPU memory. If it is the case,
 * it check if the version on the GPU match with the one in memory. In all cases, it
 * propose a section in the GPU memory where the data should be transferred.
 *
 * It return 1 if no transfer should be initiated, a 0 if a transfer is
 * necessary, and a negative value if no memory is currently available on the GPU.
 */
int gpu_qr_data_is_on_gpu( int matrixIsT, 
                           gpu_device_t* gpu_device,
                           tiled_matrix_desc_t* data,
                           int type, int col, int row,
                           gpu_elem_t **pgpu_elem)
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;

    gpu_qr_data_get_tile( matrixIsT, data, col, row, &memory_elem );

    if( NULL == (gpu_elem = memory_elem->gpu_elems[gpu_device->id]) ) {
        /* Get the LRU element on the GPU and transfer it to this new data */
        gpu_elem = (gpu_elem_t*)dague_linked_list_remove_head(gpu_device->gpu_mem_lru);
        if( memory_elem != gpu_elem->memory_elem ) {
            if( NULL != gpu_elem->memory_elem ) {
                memory_elem_t* old_mem = gpu_elem->memory_elem;
                old_mem->gpu_elems[gpu_device->id] = NULL;
            }
            gpu_elem->type = 0;
        }
        gpu_elem->type |= type;
        gpu_elem->memory_elem = memory_elem;
        memory_elem->gpu_elems[gpu_device->id] = gpu_elem;
        *pgpu_elem = gpu_elem;
        dague_linked_list_add_tail(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    } else {
        dague_linked_list_remove_item(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        dague_linked_list_add_tail(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        gpu_elem->type |= type;
        *pgpu_elem = gpu_elem;
        if( memory_elem->memory_version == gpu_elem->gpu_version ) {
            /* The GPU version of the data matches the one in memory. We're done */
            return 1;
        }
        /* The version on the GPU doesn't match the one in memory. Let the
         * upper level know a transfer is required.
         */
    }
    gpu_elem->gpu_version = memory_elem->memory_version;
    /* Transfer is required */
    return 0;
}


static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
{
    float measure = (float)length;

    *best_unit = "B";
    if( measure > 1024.0f ) { /* 1KB */
        *best_unit = "KB";
        measure = measure / 1024.0f;
        if( measure > 1024.0f ) { /* 1MB */
            *best_unit = "MB";
            measure = measure / 1024.0f;
            if( measure > 1024.0f ) {
                *best_unit = "GB";
                measure = measure / 1024.0f;
            }
        }
    }
    *updated_value = measure;
    return;
}
