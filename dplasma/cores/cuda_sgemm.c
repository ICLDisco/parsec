/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "cuda_sgemm.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"

#include <plasma.h>

#include <stdio.h>
#include <cublas.h>
#include <dlfcn.h>

#include "data_distribution.h"

#define DPLASMA_SCHEDULING 1
#define DPLASMA_ONLY_GPU 0
#define DAGUE_GPU_USE_PRIORITIES 1

static volatile uint32_t cpu_counter = 0;
static int ndevices = 0;
#if DPLASMA_SCHEDULING
uint32_t *gpu_set;
int *gpu_load;
int MAX_QUEUE = 55;
#endif
#include "data_dist/matrix/matrix.h"

static int OHM_N = 2;
static int OHM_M = 3;

#define TRACE_WITH_REF(prof, key, eid, refdesc, refdescid) do {         \
        dague_profile_ddesc_info_t info;                                \
        info.desc = refdesc;                                            \
        info.id = refdescid;                                            \
        dague_profiling_trace(prof, key, eid, (void*)&info);            \
    } while(0)

static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit );

static tiled_matrix_desc_t* UGLY_A;

int sgemm_cuda_init( dague_context_t* dague_context, tiled_matrix_desc_t *tileA )
{
    CUdevice hcuDevice;
    int i, j;

    char *env;

    UGLY_A = tileA;

    /**
     * Right now the sgemm function available with DPLASMA can only handle
     * square tiles with a size multiple of 64.
     */
    if( (tileA->mb != tileA->nb) || ((tileA->nb % 64) != 0) ) {
        printf("#\n# The CUDA GEMM version provided by DPLASMA is limitted to square tiles\n"
               "# with a size multiple of 64.\n");
        return -1;
    }

    env = getenv("OHM_N");
    if( NULL != env )
        OHM_N = atoi(env);

    env = getenv("OHM_M");
    if( NULL != env )
        OHM_M = atoi(env);

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
    for( i = 0; i < ndevices; i++ ) {
        size_t tile_size, thread_gpu_mem;
#if CUDA_VERSION < 3020
        unsigned int total_mem, free_mem;
#else
        size_t total_mem, free_mem;
#endif  /* CUDA_VERSION < 3020 */
        unsigned int nb_allocations = 0;
        gpu_device_t* gpu_device;
        CUresult status;
        int major, minor;
        char module_path[FILENAME_MAX];

        status = cuDeviceGet( &hcuDevice, i );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {ndevices = 0; return -1;} );

        status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {ndevices = 0; return -1;} );

        gpu_device = gpu_devices[i];
        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status,
                                {free(gpu_device); gpu_devices[i] = NULL; continue; } );
       
        /* If not disallowed by env, load from static linked kernels */
        /* This is non functional, as the ptr is not a CuFunction. */
        gpu_device->hcuFunction = NULL;
        env = getenv("DAGUE_CUBIN_NOSTATIC");
        if(!env || (('1' != env[0]) && ('y' != env[0])))
        {
            void* dlh;
            snprintf(module_path, FILENAME_MAX, "sgemmNT_SM%d%d", gpu_device->major, gpu_device->minor);
            dlh = dlopen(NULL, RTLD_NOW);
            if(NULL == dlh) printf("Error parsing static libs: %s\n", dlerror());
            gpu_device->hcuFunction = dlsym(dlh, module_path);
            dlclose(dlh);
        }
        
        /* If not found statically, cuload it */
        if(NULL == gpu_device->hcuFunction)
        {
            env = getenv("DAGUE_CUBIN_PATH");
            snprintf(module_path, FILENAME_MAX, "%s/sgemm-sm_%1d%1d.cubin", 
                     env?env:"../cores", gpu_device->major, gpu_device->minor);
            status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                    {
                                        fprintf(stderr, "*** unable to load `%s'\n", module_path);
                                        cuCtxDestroy( gpu_device->ctx );
                                        free(gpu_device);
                                        gpu_devices[i] = NULL;
                                        continue;
                                    } );
            snprintf(module_path, FILENAME_MAX, "sgemmNT_SM%d%d", gpu_device->major, gpu_device->minor);         
            printf("CUDA MODULE %s\n", module_path);
            status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, module_path );
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                    {
                                        cuCtxDestroy( gpu_device->ctx );
                                        free(gpu_device);
                                        gpu_devices[i] = NULL;
                                        continue;
                                    } );
        }
        if(NULL == gpu_device->hcuFunction) return -1;
        if( 1 == gpu_device->major ) {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
        } else {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 64, 4, 1 );
        }
        /**
         * Prepare the reusable memory on the GPU.
         */
        gpu_data_map_init( gpu_device, tileA );
        /**
         * It appears that CUDA allocate the memory in chunks of 1MB,
         * so we need to adapt to this.
         */
        tile_size = tileA->bsiz * tileA->mtype;
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
#if CUDA_VERSION < 3020
                                        unsigned int _free_mem, _total_mem;
#else
                                        size_t _free_mem, _total_mem;
#endif  /* CUDA_VERSION < 3020 */
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
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
        /* Prepare the management arrays */
        gpu_device->max_in_tasks   = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->max_exec_tasks = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->max_out_tasks  = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->in_submit   = gpu_device->in_waiting   = 0;
        gpu_device->exec_submit = gpu_device->exec_waiting = 0;
        gpu_device->out_submit  = gpu_device->out_waiting  = 0;

        gpu_device->max_exec_streams = gpu_device->max_streams - 2;

        gpu_device->fifo_pending_in = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_in );
        gpu_device->fifo_pending_exec = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_exec );
        gpu_device->fifo_pending_out = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_out );

        gpu_device->in_array = (struct dague_execution_context_t**)malloc(gpu_device->max_in_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->in_array_events = (CUevent*)malloc(gpu_device->max_in_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_in_tasks; j++ ) {
            gpu_device->in_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->in_array_events[j]), CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->in_array_events[j]), CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ", (cudaError_t)status,
                                    {continue;} );
        }
        gpu_device->exec_array = (struct dague_execution_context_t**)malloc(gpu_device->max_exec_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->exec_array_events = (CUevent*)malloc(gpu_device->max_exec_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_exec_tasks; j++ ) {
            gpu_device->exec_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->exec_array_events[j]), CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->exec_array_events[j]), CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ", (cudaError_t)status,
                                    {continue;} );
        }
        gpu_device->out_array = (struct dague_execution_context_t**)malloc(gpu_device->max_out_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->out_array_events = (CUevent*)malloc(gpu_device->max_out_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_out_tasks; j++ ) {
            gpu_device->out_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->out_array_events[j]), CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->out_array_events[j]), CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ", (cudaError_t)status,
                                    {continue;} );
        }
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); return -1;} );
    }

    return 0;
}

int sgemm_cuda_fini(dague_context_t* dague_context)
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

    /* GPU counter for GEMM / each */
    gpu_counter     = (int*)calloc(ndevices, sizeof(int));
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
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
        /* Release all registered events */
        for( j= 0; j < gpu_device->max_in_tasks; j++ ) {
            assert( NULL == gpu_device->in_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->in_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->in_array); gpu_device->in_array = NULL;
        free(gpu_device->in_array_events); gpu_device->in_array_events = NULL;
        for( j= 0; j < gpu_device->max_exec_tasks; j++ ) {
            assert( NULL == gpu_device->exec_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->exec_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->exec_array); gpu_device->exec_array = NULL;
        free(gpu_device->exec_array_events); gpu_device->exec_array_events = NULL;
        for( j= 0; j < gpu_device->max_out_tasks; j++ ) {
            assert( NULL == gpu_device->out_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->out_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->out_array); gpu_device->out_array = NULL;
        free(gpu_device->out_array_events); gpu_device->out_array_events = NULL;
        free( gpu_device->fifo_pending_in ); gpu_device->fifo_pending_in = NULL;
        free( gpu_device->fifo_pending_exec ); gpu_device->fifo_pending_exec = NULL;
        free( gpu_device->fifo_pending_out ); gpu_device->fifo_pending_out = NULL;
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */
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
    printf("|PU % 5d |  # GEMM   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n", dague_context->my_rank);
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


#if defined(DAGUE_PROF_TRACE)
#include "../lib/generated/spotrf_rl.h"
/**
 * This function has benn copied by hand from the generated code. It should be
 * kept in sync with the hash function from there.
 */
static inline int GEMM_hash(const dague_spotrf_rl_object_t* __dague_object, int k, int m, int n)
{
    int __h = 0;
    int k_min = 0;
    int k_range = (__dague_object->SIZE - 1) - k_min + 1;
    int m_min = (k + 2);
    int m_range = (__dague_object->SIZE - 1) - m_min + 1;
    int n_min = (k + 1);
    __h += (k - k_min);
    __h += (m - m_min) * k_range;
    __h += (n - n_min) * k_range * m_range;
    /* Ensure we avoid collisions with the GEMM ID on the CPU */
    return __h + (__dague_object->SIZE * __dague_object->SIZE * __dague_object->SIZE);
}
#endif  /* defined(DAGUE_PROF_TRACE) */

#define ddescA(ec) (UGLY_A)
#define ddescB(ec) ddescA(ec)
#define ddescC(ec) ddescA(ec)

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
static inline int
gpu_sgemm_internal_push( gpu_device_t* gpu_device,
                         dague_execution_context_t* exec_context,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    dague_arena_chunk_t *aA, *aB, *aC;
    int tile_size, return_code = 0, on_gpu, how_many = 0;
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

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start, (unsigned long)exec_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG(("Request Data of A(%d, %d) on GPU\n", n, k));
    tile_size = ddescA(exec_context)->mb * ddescA(exec_context)->nb * ddescA(exec_context)->mtype;
    on_gpu = gpu_data_is_on_gpu(gpu_device, ddescA(exec_context), DAGUE_READ, n, k, &gpu_elem_A);
    gpu_elem_A->memory_elem->memory = A;
    d_A = gpu_elem_A->gpu_mem;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push A into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_A, A, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_A) ", status, 
                                  {printf("<<%p>> -> <<%p>> [%d]\n", (void*)A, (void*)(long)d_A, tile_size); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
        how_many++;
    }
    exec_context->data[0].gpu_data = (struct gpu_elem_t *)gpu_elem_A;

    DEBUG(("Request Data of B(%d, %d) on GPU\n", m, k));
    tile_size = ddescB(exec_context)->mb * ddescB(exec_context)->nb * ddescB(exec_context)->mtype;
    on_gpu = gpu_data_is_on_gpu(gpu_device, ddescB(exec_context), DAGUE_READ, m, k, &gpu_elem_B);
    d_B = gpu_elem_B->gpu_mem;
    gpu_elem_B->memory_elem->memory = B;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push B into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_B, B, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_B) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)B, (void*)(long)d_B); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
        how_many++;
    }
    exec_context->data[1].gpu_data = (struct gpu_elem_t *)gpu_elem_B;

    DEBUG(("Request Data of C(%d, %d) on GPU\n", m, n));
    tile_size = ddescC(exec_context)->mb * ddescC(exec_context)->nb * ddescC(exec_context)->mtype;
    on_gpu = gpu_data_is_on_gpu(gpu_device, ddescC(exec_context), DAGUE_READ | DAGUE_WRITE, m, n, &gpu_elem_C);
    d_C = gpu_elem_C->gpu_mem;
    gpu_elem_C->memory_elem->memory = C;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push C into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_C, C, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)C, (void*)(long)d_C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
        how_many++;
    }
    exec_context->data[2].gpu_data = (struct gpu_elem_t *)gpu_elem_C;

 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

static inline int
gpu_sgemm_internal_submit( gpu_device_t* gpu_device,
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

    DEBUG(("Request GPU runs GEMM(%d, %d, %d)\n", exec_context->locals[0], exec_context->locals[1], exec_context->locals[2]));

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        dague_spotrf_rl_object_t* __dague_object = (dague_spotrf_rl_object_t*)exec_context->dague_object;
        TRACE_WITH_REF(gpu_device->profiling, 
                       DAGUE_PROF_FUNC_KEY_START(exec_context->dague_object,exec_context->function->function_id),
                       GEMM_hash( __dague_object, exec_context->locals[0].value, exec_context->locals[1].value, exec_context->locals[2].value),
                       ((dague_ddesc_t*)__dague_object->A),
                       ((dague_ddesc_t*)__dague_object->A)->data_key((dague_ddesc_t*)__dague_object->A, exec_context->locals[1].value, exec_context->locals[2].value));
    }
#endif  /* defined(DAGUE_PROF_TRACE) */
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
        /* Change bx and by to match the values in the fermi gemm code */
#define bx 4
#define by 4
        grid_width  = ddescA(exec_context)->nb / (16*bx) + (ddescA(exec_context)->nb % (16*bx) != 0);
        grid_height = ddescA(exec_context)->nb / (16*by) + (ddescA(exec_context)->nb % (16*by) != 0);
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
gpu_sgemm_internal_pop( gpu_device_t* gpu_device,
                        dague_execution_context_t* exec_context,
                        CUstream stream )
{
    dague_arena_chunk_t *aC;
    gpu_elem_t *gpu_elem_C = NULL;
    int return_code = 0, tile_size, how_many = 0;
    cudaError_t status;
    CUdeviceptr d_C;
    void* C;
    int n, k, m;

    k = exec_context->locals[0].value;
    m = exec_context->locals[1].value; (void)m;
    n = exec_context->locals[2].value;

    gpu_elem_C = (gpu_elem_t *)exec_context->data[2].gpu_data;
    aC = exec_context->data[2].data;
    d_C = gpu_elem_C->gpu_mem;
    C = ADATA(aC);

    tile_size = ddescC(exec_context)->mb * ddescC(exec_context)->nb * ddescC(exec_context)->mtype;

    /* Pop C from the GPU */
    gpu_device->required_data_out += tile_size;
    if( (n == k+1) ) {
        DEBUG(("Request out of GPU for C(%d, %d)\n", m, n));
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start, (unsigned long)exec_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        /* Pop C from the GPU */
        status = (cudaError_t)cuMemcpyDtoHAsync( C, d_C, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)(long)d_C, (void*)C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
        how_many++;
    }
 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/* Try to execute a GEMM on a GPU.
 *
 * Returns:
 *  0 - if the GEMM should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the GEMM is scheduled to be executed on a GPU.
 */

#if !defined(DAGUE_GPU_STREAM_PER_TASK)

#if DAGUE_GPU_USE_PRIORITIES
static inline dague_list_item_t* dague_fifo_push_ordered( dague_fifo_t* fifo,
                                                          dague_list_item_t* elem )
{
    dague_execution_context_t* ec;
    dague_execution_context_t* input = (dague_execution_context_t*)elem;
    dague_list_item_t* current = (dague_list_item_t*)fifo->fifo_ghost.list_next;

    if( 0 == input->priority ) {
        while( current != &(fifo->fifo_ghost) ) {
            ec = (dague_execution_context_t*)current;
            if( ec->priority < input->priority )
                break;
            current = (dague_list_item_t *)current->list_next;
        }
    } else {
        current = &(fifo->fifo_ghost);
    }
    /* Add the input element before the current one */
    elem->list_prev = current->list_prev;
    elem->list_next = current;
    elem->list_prev->list_next = elem;
    elem->list_next->list_prev = elem;
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_fifo_push
#endif

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* exec_context,
               int uplo )
{
    int which_gpu, rc, exec_stream = 0;
    gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int n, m;

    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;
    (void)uplo;
    //DEBUG(("GEMM( k = %d, m = %d, n = %d )\n", k, m, n));
    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = gpu_data_tile_write_owner( ddescA(exec_context), m, n );
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
            dague_atomic_inc_32b( &(cpu_counter) );
            return -99;
        }
#endif

#endif
    }
    gpu_device = gpu_devices[which_gpu];

#if DPLASMA_SCHEDULING	
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

    /**
     * There might be a small race condition here, between the moment when the previous
     * owner of the GPU context release it, and the moment where I can get it.
     */
    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );
        
#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_start, (unsigned long)eu_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                            {return -2;} );

    DEBUG(( "Add gemm(k = %d, m = %d, n = %d) priority %d\n",
            exec_context->locals[0].value, exec_context->locals[1].value, exec_context->locals[2].value,
            exec_context->priority ));
 check_in_deps:
    if( NULL != exec_context ) {
        if( NULL != gpu_device->in_array[gpu_device->in_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in, (dague_list_item_t*)exec_context);
            exec_context = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_in) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in, (dague_list_item_t*)exec_context);
                exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_in);
            }
        }
    } else {
        if( NULL == gpu_device->in_array[gpu_device->in_submit] ) {
            exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_in);
        }
    }
    if( NULL != exec_context ) {
        assert( NULL == gpu_device->in_array[gpu_device->in_submit] );
        rc = gpu_sgemm_internal_push( gpu_device, exec_context, gpu_device->streams[0] );
        /**
         * Do not skip the cuda event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to bad results.
         */
        /*if( 0 == rc ) goto exec_task;*/  /* No data to be moved for this task */
        gpu_device->in_array[gpu_device->in_submit] = exec_context;
        DEBUG(("GPU Request number %d/%d\n", gpu_device->in_array_events[gpu_device->in_submit], gpu_device->streams[0]));
        exec_context = NULL;
        if( 0 > rc ) goto disable_gpu;
        rc = cuEventRecord( gpu_device->in_array_events[gpu_device->in_submit], gpu_device->streams[0] );
        gpu_device->in_submit = (gpu_device->in_submit + 1) % gpu_device->max_in_tasks;
    }
    assert( NULL == exec_context );
    if( NULL != gpu_device->in_array[gpu_device->in_waiting] ) {
        rc = cuEventQuery(gpu_device->in_array_events[gpu_device->in_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_exec_completion;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n", gpu_device->in_array_events[gpu_device->in_waiting]));
            exec_context = gpu_device->in_array[gpu_device->in_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
                dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_end, (unsigned long)exec_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->in_array[gpu_device->in_waiting] = NULL;
            gpu_device->in_waiting = (gpu_device->in_waiting + 1) % gpu_device->max_in_tasks;
            goto exec_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }
 exec_task:
    if( NULL != exec_context ) {
        if( NULL != gpu_device->exec_array[gpu_device->exec_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec, (dague_list_item_t*)exec_context);
            exec_context = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_exec) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec, (dague_list_item_t*)exec_context);
                exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_exec);
            }
        }
    } else {
        if( NULL == gpu_device->exec_array[gpu_device->exec_submit] ) {
            exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_exec);
        }
    }
    if( NULL != exec_context ) {
        assert( NULL == gpu_device->exec_array[gpu_device->exec_submit] );
        /* Choose an exec_stream */
        exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams);
        DEBUG(( "Execute gemm(k = %d, m = %d, n = %d) priority %d\n",
                exec_context->locals[0].value, exec_context->locals[1].value, exec_context->locals[2].value,
                exec_context->priority ));
        rc = gpu_sgemm_internal_submit( gpu_device, exec_context, gpu_device->streams[2 + exec_stream] );
        DEBUG(("GPU Request number %d/%d\n", gpu_device->exec_array_events[gpu_device->exec_submit], gpu_device->streams[2 + exec_stream]));
        gpu_device->exec_array[gpu_device->exec_submit] = exec_context;
        exec_context = NULL;
        if( 0 != rc )  goto disable_gpu;
        rc = cuEventRecord( gpu_device->exec_array_events[gpu_device->exec_submit], gpu_device->streams[2 + exec_stream] );
        gpu_device->exec_submit = (gpu_device->exec_submit + 1) % gpu_device->max_exec_tasks;
    }
 check_exec_completion:
    assert( NULL == exec_context );
    if( NULL != gpu_device->exec_array[gpu_device->exec_waiting] ) {
        rc = cuEventQuery(gpu_device->exec_array_events[gpu_device->exec_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_out_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n", gpu_device->exec_array_events[gpu_device->exec_waiting]));
            exec_context = gpu_device->exec_array[gpu_device->exec_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
                dague_spotrf_rl_object_t* __dague_object = (dague_spotrf_rl_object_t*)exec_context->dague_object;
                TRACE_WITH_REF(gpu_device->profiling, 
                               DAGUE_PROF_FUNC_KEY_END(exec_context->dague_object, exec_context->function->function_id),
                               GEMM_hash( __dague_object, exec_context->locals[0].value, exec_context->locals[1].value, exec_context->locals[2].value),
                               (dague_ddesc_t*)__dague_object->A,
                               ((dague_ddesc_t*)__dague_object->A)->data_key((dague_ddesc_t*)__dague_object->A, exec_context->locals[1].value, exec_context->locals[2].value));
            }
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->exec_array[gpu_device->exec_waiting] = NULL;
            gpu_device->exec_waiting = (gpu_device->exec_waiting + 1) % gpu_device->max_exec_tasks;
            goto out_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }
 out_task:
    if( NULL != exec_context ) {
        if( NULL != gpu_device->out_array[gpu_device->out_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out, (dague_list_item_t*)exec_context);
            exec_context = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_out) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out, (dague_list_item_t*)exec_context);
                exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_out);
            }
        }
    } else {
        if( NULL == gpu_device->out_array[gpu_device->out_submit] ) {
            exec_context = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_out);
        }
    }
    if( NULL != exec_context ) {
        assert( NULL == gpu_device->out_array[gpu_device->out_submit] );
        rc = gpu_sgemm_internal_pop( gpu_device, exec_context, gpu_device->streams[1] );
        DEBUG(("GPU Request number %d/%d\n", gpu_device->out_array_events[gpu_device->out_submit], gpu_device->streams[1]));
        if( 0 == rc ) goto complete_task;  /* no data to be moved */
        gpu_device->out_array[gpu_device->out_submit] = exec_context;
        exec_context = NULL;
        if( 0 > rc ) goto disable_gpu;
        rc = cuEventRecord( gpu_device->out_array_events[gpu_device->out_submit], gpu_device->streams[1] );
        gpu_device->out_submit = (gpu_device->out_submit + 1) % gpu_device->max_out_tasks;
    }
 check_out_deps:
    assert( NULL == exec_context );
    if( NULL != gpu_device->out_array[gpu_device->out_waiting] ) {
        rc = cuEventQuery(gpu_device->out_array_events[gpu_device->out_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_in_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n", gpu_device->out_array_events[gpu_device->out_waiting]));
            exec_context = gpu_device->out_array[gpu_device->out_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
                dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_end, (unsigned long)exec_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->out_array[gpu_device->out_waiting] = NULL;
            gpu_device->out_waiting = (gpu_device->out_waiting + 1) % gpu_device->max_out_tasks;
            goto complete_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {goto disable_gpu;} );
        }
    }

 fetch_task_from_shared_dequeue:
    assert( NULL == exec_context );
    exec_context = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
    if( NULL != exec_context ) {
        DEBUG(( "Add gemm(k = %d, m = %d, n = %d) priority %d\n",
                exec_context->locals[0].value, exec_context->locals[1].value, exec_context->locals[2].value,
                exec_context->priority ));
    }
    goto check_in_deps;

 complete_task:
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(exec_context);
    dague_complete_execution( eu_context, exec_context );
    gpu_device->executed_tasks++;
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        assert( (NULL == gpu_device->in_array[gpu_device->in_waiting]) &&
                (NULL == gpu_device->exec_array[gpu_device->exec_waiting]) &&
                (NULL == gpu_device->out_array[gpu_device->out_waiting]) );
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_end, (unsigned long)eu_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
        dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );

        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {return -1;} );
        return -1;
    }
    exec_context = NULL;
    goto fetch_task_from_shared_dequeue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    exit(-20);
    return -2;
}
#else
static int
gpu_sgemm_internal( gpu_device_t* gpu_device,
                    dague_execution_unit_t* eu_context,
                    dague_execution_context_t* exec_context,
                    CUstream stream, int uplo )
{
    int return_code = 0;  /* by default suppose an error */

    (void)eu_context;
    (void)uplo;

   // DEBUG(("Execute GEMM( k = %d, m = %d, n = %d ) [%d] on device %d stream %p\n",
     //      k, m, n, exec_context->priority, gpu_device->id, (void*)stream));

    return_code = gpu_sgemm_internal_push( gpu_device,
                                           exec_context,
                                           stream );
    if( 0 > return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_submit( gpu_device,
                                             exec_context,
                                             stream );
    if( 0 != return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_pop( gpu_device,
                                          exec_context,
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
               dague_execution_context_t* exec_context,
               int uplo )
{
    int which_gpu, rc, stream_rc, waiting = 0, submit = 0;
    gpu_device_t* gpu_device;
    cudaError_t status;
    dague_execution_context_t* progress_array[DAGUE_MAX_STREAMS];
    int n, m;

    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;

    //DEBUG(("GEMM( k = %d, m = %d, n = %d )\n", k, m, n));
    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = gpu_data_tile_write_owner( ddescA(exec_context), m, n );
/*    printf("k=%d, m=%d, n=%d\n",k,m,n);*/
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
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
        dague_atomic_inc_32b( &(cpu_counter) );
        return -99;
    }
#endif
    
#endif
    }
    gpu_device = gpu_devices[which_gpu];

#if DPLASMA_SCHEDULING

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
        rc = gpu_sgemm_internal( gpu_device, eu_context, exec_context, gpu_device->streams[submit], uplo );
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
            DAGUE_CUDA_CHECK_ERROR( "cuStreamQuery ", stream_rc,
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
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

#include "gpu_data.h"
#include "data_distribution.h"
#include "linked_list.h"

static memory_elem_t** data_map = NULL;
extern int ndevices;

int gpu_mark_data_usage( tiled_matrix_desc_t* data, int type, int col, int row )
{
    memory_elem_t* this_data;

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

int gpu_data_map_init( gpu_device_t* gpu_device,
                       tiled_matrix_desc_t* data )
{
    if( NULL == data_map ) {
        data_map = (memory_elem_t**)calloc(data->lmt * data->lnt, sizeof(memory_elem_t*));
    }
    gpu_device->gpu_mem_lru = (dague_linked_list_t*)malloc(sizeof(dague_linked_list_t));
    dague_linked_list_construct(gpu_device->gpu_mem_lru);
    return 0;
}

int gpu_data_tile_write_owner( tiled_matrix_desc_t* data,
                               int col, int row )
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;
    int i;

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

int gpu_data_get_tile( tiled_matrix_desc_t* data,
                       int col, int row,
                       memory_elem_t **pmem_elem )
{
    memory_elem_t* memory_elem;
    int rc = 0;  /* the tile already existed */

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
int gpu_data_is_on_gpu( gpu_device_t* gpu_device,
                        tiled_matrix_desc_t* data,
                        int type, int col, int row,
                        gpu_elem_t **pgpu_elem)
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;

    gpu_data_get_tile( data, col, row, &memory_elem );

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
