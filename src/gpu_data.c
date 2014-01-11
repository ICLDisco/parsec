/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"

#if defined(HAVE_CUDA)
#include "dague.h"
#include "gpu_data.h"
#include "gpu_malloc.h"
#include "profiling.h"
#include "execution_unit.h"
#include "arena.h"
#include "moesi.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>

/*
 *  dague_gpu_init()           : Initialize the ndevices GPU asked
 *  dague_gpu_kernel_init()    : Check which GPUs can execute the kernel and initialize function ptr
 *  dague_gpu_data_register()  : Register the dague_ddesc on which the gpu kernels will work
 *  dague_gpu_data_unregister(): Unregister the dague_ddesc on which the gpu kernels will work
 *  dague_gpu_kernel_fini()    : Show kernel statistics and clean all kernels
 *  dague_gpu_fini()           : Show global data movment statistics and clean all GPUs
 *
 */

static int __dague_active_gpu = 0;
static uint32_t __dague_gpu_mask = 0xffffffff;
static CUcontext dague_allocate_on_gpu_context;
static int dague_gpu_allocation_initialized = 0;

volatile uint32_t dague_cpu_counter = 0;
float *device_load = NULL, *device_weight = NULL;

#define DAGUE_LIST_DESTRUCT(__dague_list)       \
    {                                           \
        dague_list_destruct( __dague_list );    \
        free( __dague_list );                   \
        __dague_list = NULL;                    \
    }

static void* dague_gpu_data_allocate(size_t matrix_size)
{
    void* mat = NULL;

    if( 0 == matrix_size ) return NULL;

    if( __dague_active_gpu ) {
        CUresult status;
        if( dague_gpu_allocation_initialized ) {
            status = cuCtxPushCurrent( dague_allocate_on_gpu_context );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                    {
                                        ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                    } );
        }
        else {
            int rc;
            gpu_device_t* gpu_device = gpu_enabled_devices[0];
            assert( gpu_device != NULL );
            /* Check the GPU status */
            rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
            if( rc != 1 ) {
                ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
            }
            status = cuCtxPushCurrent( gpu_device->ctx );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                    {
                                        ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                    } );
        }
        status = cuMemHostAlloc( (void**)&mat, matrix_size, CU_MEMHOSTALLOC_PORTABLE);
        DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuMemHostAlloc failed ", status,
                                {
                                    ERROR(("Unable to allocate %ld bytes of GPU-compatible data as requested.\n", matrix_size));
                                } );
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {} );
        if( !dague_gpu_allocation_initialized ) {
            gpu_device_t* gpu_device = gpu_enabled_devices[0];
            dague_atomic_dec_32b( &(gpu_device->mutex) );
        }
    } else {
        mat = malloc( matrix_size );
    }

    if( NULL == mat ) {
        WARNING(("memory allocation of %lu failed (%s)\n", (unsigned long) matrix_size, strerror(errno)));
        return NULL;
    }
    return mat;
}

/**
 * free a buffer allocated by dague_allocate_data
 */
static void dague_gpu_data_free(void *dta)
{
    unsigned int flags, call_free = 1;

    if( NULL == dta ) return;

    if( __dague_active_gpu ) {
        CUresult status;
        if( dague_gpu_allocation_initialized ) {
            status = cuCtxPushCurrent( dague_allocate_on_gpu_context );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                    {
                                        ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                    } );
        }
        else {
            int rc;
            gpu_device_t* gpu_device = gpu_enabled_devices[0];
            assert( gpu_device != NULL );
            /* Check the GPU status */
            rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
            if( rc != 1 ) {
                ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
            }
            status = cuCtxPushCurrent( gpu_device->ctx );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                    {
                                        ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                    } );
        }

        status = cuMemHostGetFlags( &flags, dta );
        DAGUE_CUDA_CHECK_ERROR( "cuMemHostGetFlags ", status,
                                {goto release_cuda_context;} );

        status = cuMemFreeHost( dta );
        DAGUE_CUDA_CHECK_ERROR( "cuMemFreeHost ", status,
                                {goto release_cuda_context;} );
        call_free = 0;
    release_cuda_context:
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPopCurrent ", status,
                                {} );
        if( !dague_gpu_allocation_initialized ) {
            gpu_device_t* gpu_device = gpu_enabled_devices[0];
            dague_atomic_dec_32b( &(gpu_device->mutex) );
        }
    }

    if( call_free ) free( dta );
}

/**
 * Enable GPU-compatible memory if possible
 */
void dague_data_enable_gpu( int nbgpu )
{
    __dague_active_gpu = nbgpu;

    dague_data_allocate = dague_gpu_data_allocate;
    dague_data_free     = dague_gpu_data_free;
}

void dague_data_disable_gpu(void) {
    __dague_active_gpu = 0;
    dague_data_allocate = malloc;
    dague_data_free     = free;
}

int dague_active_gpu(void)
{
    return __dague_active_gpu;
}

#if defined(DAGUE_PROF_TRACE)
/* Accepted values are: DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      DAGUE_PROFILE_CUDA_TRACK_OWN | DAGUE_PROFILE_CUDA_TRACK_EXEC
 */
int dague_cuda_trackable_events = DAGUE_PROFILE_CUDA_TRACK_EXEC | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT
  | DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_OWN;
int dague_cuda_movein_key_start;
int dague_cuda_movein_key_end;
int dague_cuda_moveout_key_start;
int dague_cuda_moveout_key_end;
int dague_cuda_own_GPU_key_start;
int dague_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

/* We don't use gpu_devices, instead we use a subset of gpu-array
 * gpu_array - list of GPU by order of their performance
 */
gpu_device_t** gpu_enabled_devices = NULL;

/* Dirty selection for now */
float gpu_speeds[2][2] ={
    /* C1060, C2050 */
    { 622.08, 1030.4 },
    {  77.76,  515.2 }
};

int dague_gpu_data_verbose = 0;

int dague_gpu_init(dague_context_t *dague_context,
                   int* puse_gpu,
                   int dague_show_detailed_capabilities )
{
    int ndevices, i, j, k, dindex, nb_cores;
    float total_perf;
    CUresult status;
    int isdouble = 0;

    if( (*puse_gpu) == -1 ) {
        return -1;  /* Nothing to do around here */
    }
    status = cuInit(0);
    DAGUE_CUDA_CHECK_ERROR( "cuInit ", status, {*puse_gpu = 0; return -1;} );
    dague_gpu_data_verbose = dague_show_detailed_capabilities;

    cuDeviceGetCount( &ndevices );

    if( ndevices > (*puse_gpu) )
        ndevices = (*puse_gpu);
    /* Update the number of GPU for the upper layer */
    *puse_gpu = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

#if defined(DAGUE_PROF_TRACE)
    dague_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                            0, NULL,
                                            &dague_cuda_movein_key_start, &dague_cuda_movein_key_end);
    dague_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                            0, NULL,
                                            &dague_cuda_moveout_key_start, &dague_cuda_moveout_key_end);
    dague_profiling_add_dictionary_keyword( "cuda", "fill:#66ff66",
                                            0, NULL,
                                            &dague_cuda_own_GPU_key_start, &dague_cuda_own_GPU_key_end);
#endif  /* defined(PROFILING) */

    gpu_enabled_devices = (gpu_device_t**)calloc(ndevices, sizeof(gpu_device_t*));

    device_load   = (float*)calloc(ndevices+1, sizeof(float));  /* 0 for the cores */
    device_weight = (float*)calloc(ndevices+1, sizeof(float));

    memset( device_load, 0, (ndevices+1 )* sizeof(float) );

    for( i = nb_cores = 0; i < dague_context->nb_vp; i++ )
        nb_cores += dague_context->virtual_processes[i]->nb_cores;

    /* TODO: Change this to a more generic approach */
    /* Theoritical perf in double 2.27 is the frequency of dancer */
    total_perf = (float)nb_cores * 2.27f * 4.f;
    if ( ! isdouble )
        total_perf *= 2;
    device_weight[0] = total_perf;

    for( i = dindex = 0; i < ndevices; i++ ) {
#if CUDA_VERSION >= 3020
        size_t total_mem;
#else
        unsigned int total_mem;
#endif  /* CUDA_VERSION >= 3020 */
        gpu_device_t* gpu_device;
        CUdevprop devProps;
        char szName[256];
        int major, minor, concurrency, computemode;
        CUdevice hcuDevice;

        status = cuDeviceGet( &hcuDevice, i );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {continue;} );
        status = cuDeviceGetName( szName, 256, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGetName ", status, {continue;} );

        status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {continue;} );

        status = cuDeviceGetProperties( &devProps, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGetProperties ", status, {continue;} );

        status = cuDeviceGetAttribute( &concurrency, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGetAttribute ", status, {continue;} );

        status = cuDeviceGetAttribute( &computemode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGetAttribute ", status, {continue;} );

        if ( isdouble )
            device_weight[i+1] = ( major == 1 ) ? gpu_speeds[1][0] : gpu_speeds[1][1];
        else
            device_weight[i+1] = ( major == 1 ) ? gpu_speeds[0][0] : gpu_speeds[0][1];

        //device_weight[i+1] = ((float)devProps.maxThreadsPerBlock * (float)devProps.clockRate) * 2;
        total_perf += device_weight[i+1];
        //device_weight[i+1] *= (concurrency == 1 ? 2 : 1);

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & __dague_gpu_mask) ) continue;

        if( dague_show_detailed_capabilities ) {
            STATUS(("GPU Device %d (capability %d.%d): %s\n", i, major, minor, szName ));
            STATUS(("\tmaxThreadsPerBlock : %d\n", devProps.maxThreadsPerBlock ));
            STATUS(("\tmaxThreadsDim      : [%d %d %d]\n", devProps.maxThreadsDim[0],
                    devProps.maxThreadsDim[1], devProps.maxThreadsDim[2] ));
            STATUS(("\tmaxGridSize        : [%d %d %d]\n", devProps.maxGridSize[0],
                    devProps.maxGridSize[1], devProps.maxGridSize[2] ));
            STATUS(("\tsharedMemPerBlock  : %d\n", devProps.sharedMemPerBlock ));
            STATUS(("\tconstantMemory     : %d\n", devProps.totalConstantMemory ));
            STATUS(("\tSIMDWidth          : %d\n", devProps.SIMDWidth ));
            STATUS(("\tmemPitch           : %d\n", devProps.memPitch ));
            STATUS(("\tregsPerBlock       : %d\n", devProps.regsPerBlock ));
            STATUS(("\tclockRate          : %d\n", devProps.clockRate ));
            STATUS(("\tconcurrency        : %s\n", (concurrency == 1 ? "yes" : "no") ));
            STATUS(("\tcomputeMode        : %d\n", computemode ));
        }
        status = cuDeviceTotalMem( &total_mem, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceTotalMem ", status, {continue;} );

        gpu_device = (gpu_device_t*)calloc(1, sizeof(gpu_device_t));
        gpu_enabled_devices[dindex] = gpu_device;
        dague_list_construct(&gpu_device->pending);
        gpu_device->major = (uint8_t)major;
        gpu_device->minor = (uint8_t)minor;

        if( dague_gpu_allocation_initialized == 0 ) {
            if( computemode == CU_COMPUTEMODE_DEFAULT || computemode == CU_COMPUTEMODE_EXCLUSIVE_PROCESS ) {
                status = cuCtxCreate( &dague_allocate_on_gpu_context, 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
                DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                        {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );
                status = cuCtxPopCurrent(NULL);
                DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                        {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );
                dague_gpu_allocation_initialized = 1;
            }
        }

        gpu_device->index                = (uint8_t)dindex;
        gpu_device->device_index         = (uint8_t)i;
        gpu_device->executed_tasks       = 0;
        gpu_device->transferred_data_in  = 0;
        gpu_device->transferred_data_out = 0;

        /* Initialize LRU */
        gpu_device->gpu_mem_lru       = (dague_list_t*)malloc(sizeof(dague_list_t));
        gpu_device->gpu_mem_owned_lru = (dague_list_t*)malloc(sizeof(dague_list_t));
        dague_list_construct(gpu_device->gpu_mem_lru);
        dague_list_construct(gpu_device->gpu_mem_owned_lru);

        /* cuCtxCreate: Function works on floating contexts and current context */
        status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue; } );

        gpu_device->max_exec_streams = DAGUE_MAX_STREAMS;
        gpu_device->exec_stream =
            (dague_gpu_exec_stream_t*)malloc(gpu_device->max_exec_streams
                                             * sizeof(dague_gpu_exec_stream_t));
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            cudaError_t cudastatus;
            dague_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

            /* Allocate the stream */
            cudastatus = cudaStreamCreate( &(exec_stream->cuda_stream) );
            DAGUE_CUDA_CHECK_ERROR( "cudaStreamCreate ", cudastatus,
                                    {break;} );
            exec_stream->max_events   = DAGUE_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;
            exec_stream->fifo_pending = (dague_list_t*)malloc( sizeof(dague_list_t) );
            dague_list_construct( exec_stream->fifo_pending );
            exec_stream->tasks  = (dague_gpu_context_t**)malloc(exec_stream->max_events
                                                                * sizeof(dague_gpu_context_t*));
            exec_stream->events = (CUevent*)malloc(exec_stream->max_events * sizeof(CUevent));
            /* and the corresponding events */
            for( k = 0; k < exec_stream->max_events; k++ ) {
                exec_stream->events[k] = NULL;
                exec_stream->tasks[k]  = NULL;
#if CUDA_VERSION >= 3020
                status = cuEventCreate(&(exec_stream->events[k]), CU_EVENT_DISABLE_TIMING);
#else
                status = cuEventCreate(&(exec_stream->events[k]), CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
                DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ", (cudaError_t)status,
                                        {break;} );
            }
#if defined(DAGUE_PROF_TRACE)
            exec_stream->profiling = dague_profiling_thread_init( 2*1024*1024, DAGUE_PROFILE_STREAM_STR, i, j );
#endif  /* defined(PROFILING) */
#if defined(DAGUE_PROF_TRACE)
            exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC;
            exec_stream->prof_event_key_start    = -1;
            exec_stream->prof_event_key_end      = -1;
#endif  /* defined(DAGUE_PROF_TRACE) */
        }

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );

        dindex++;
    }

    /* Compute the weight of each device including the cores */
    DEBUG(("Global Theoritical performance: %2.4f\n", total_perf ));
    for( i = 0; i < (ndevices+1); i++ ) {
        if( 0 == i ) {
            DEBUG(("CPU             ->ratio %2.4e (%2.4e)\n",
                   device_weight[i],
                   device_weight[i] / nb_cores ));
        } else
            DEBUG(("Device index %2d ->ratio %2.4e\n",
                   i-1, device_weight[i]));
        device_weight[i] = (total_perf / device_weight[i]);
        if( dague_show_detailed_capabilities ) {
            if( 0 == i )
                STATUS(("CPU             ->ratio %2.4f\n", device_weight[i]));
            else
                STATUS(("Device index %2d ->ratio %2.4f\n", i-1, device_weight[i]));
        }
    }
#if defined(DAGUE_PROF_TRACE)
    /**
     * Reconfigure the stream 0 and 1 for input and outputs.
     */
    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t *gpu_device = gpu_enabled_devices[i];
        if( NULL == gpu_device ) continue;
        gpu_device->exec_stream[0].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN;
        gpu_device->exec_stream[0].prof_event_key_start    = dague_cuda_movein_key_start;
        gpu_device->exec_stream[0].prof_event_key_end      = dague_cuda_movein_key_end;

        gpu_device->exec_stream[1].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT;
        gpu_device->exec_stream[1].prof_event_key_start    = dague_cuda_moveout_key_start;
        gpu_device->exec_stream[1].prof_event_key_end      = dague_cuda_moveout_key_end;
    }
#endif  /* defined(DAGUE_PROF_TRACE) */

#if defined(DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t *source_gpu, *target_gpu;
        CUdevice source, target;
        int canAccessPeer;

        if( NULL == (source_gpu = gpu_enabled_devices[i]) ) continue;

        source_gpu->peer_access_mask = 0;
        status = cuDeviceGet( &source, source_gpu->device_index );
        DAGUE_CUDA_CHECK_ERROR( "No peer memory access: cuDeviceGet ", status, {continue;} );
        status = cuCtxPushCurrent( source_gpu->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cuCtxPushCurrent ", status,
                                {continue;} );

        for( j = 0; j < ndevices; j++ ) {
            if( (NULL == (target_gpu = gpu_enabled_devices[j])) || (i == j) ) continue;

            status = cuDeviceGet( &target, target_gpu->device_index );
            DAGUE_CUDA_CHECK_ERROR( "No peer memory access: cuDeviceGet ", status, {continue;} );

            /* Communication mask */
            status = cuDeviceCanAccessPeer( &canAccessPeer, source, target );
            DAGUE_CUDA_CHECK_ERROR( "cuDeviceCanAccessPeer ", status,
                                    {continue;} );
            if( 1 == canAccessPeer ) {
                status = cuCtxEnablePeerAccess( target_gpu->ctx, 0 );
                DAGUE_CUDA_CHECK_ERROR( "cuCtxEnablePeerAccess ", status,
                                        {continue;} );
                source_gpu->peer_access_mask |= (int16_t)(1 << target_gpu->device_index);
            }
        }
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cuCtxPopCurrent ", status,
                                {continue;} );
    }
#endif

    /* Set the initial load of the cores to twice their weight, so that GPU will
     * offload on CPU only if the work goes over that. */
    device_load[0] = device_weight[0] * 2;
    /* Now we set the weight to only one core */
    device_weight[0] = device_weight[0] / nb_cores;
    dague_data_enable_gpu( ndevices );
    return 0;
}

int dague_gpu_fini( void )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i, j, k;

    for(i = 0; i < __dague_active_gpu; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;
        gpu_enabled_devices[i] = NULL;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );
        /*
         * Release pending queue
         */
        dague_list_destruct(&gpu_device->pending);

        /**
         * Release all streams
         */
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            dague_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

            exec_stream->max_events   = DAGUE_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;

            for( k = 0; k < exec_stream->max_events; k++ ) {
                assert( NULL == exec_stream->tasks[k] );
                status = cuEventDestroy(exec_stream->events[k]);
                DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                        {continue;} );
            }
            free(exec_stream->events); exec_stream->events = NULL;
            free(exec_stream->tasks); exec_stream->tasks = NULL;
            free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;

            /* Release the stream */
            cudaStreamDestroy( exec_stream->cuda_stream );
        }

        free(gpu_device->exec_stream);

        status = cuCtxDestroy( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxDestroy ", status,
                                {continue;} );
        gpu_device->ctx = NULL;

        /*
         * Release the GPU memory.
         */
        DAGUE_LIST_DESTRUCT(gpu_device->gpu_mem_lru);
        DAGUE_LIST_DESTRUCT(gpu_device->gpu_mem_owned_lru);

        free(gpu_device);

    }
    free(gpu_enabled_devices);
    gpu_enabled_devices = NULL;

    if( dague_gpu_allocation_initialized == 1 ) {
        cuCtxDestroy( dague_allocate_on_gpu_context );
        dague_gpu_allocation_initialized = 0;
    }

    free(device_load); device_load = NULL;
    free(device_weight); device_weight = NULL;

    __dague_active_gpu = 0;
    return 0;
}


int dague_gpu_data_register( dague_context_t *dague_context,
                             dague_ddesc_t   *data,
                             int              nbelem, /* Could be a function of the dague_desc_t */
                             size_t           eltsize )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i;
    (void)eltsize;

    moesi_map_create(&data->moesi_map, nbelem, __dague_active_gpu);
    DEBUG2(("GPU:\tregister ddesc %p, with %d tiles of size %zu (moesi %p at %p)\n",
            data, nbelem, eltsize, &data->moesi_map, data->moesi_map));

    for(i = 0; i < __dague_active_gpu; i++) {
        size_t how_much_we_allocate;
#if CUDA_VERSION < 3020
        unsigned int total_mem, free_mem, initial_free_mem;
#else
        size_t total_mem, free_mem, initial_free_mem;
#endif  /* CUDA_VERSION < 3020 */
        uint32_t mem_elem_per_gpu = 0;

        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );

        /**
         * It appears that CUDA allocate the memory in chunks of 1MB,
         * so we need to adapt to this.
         */
        cuMemGetInfo( &initial_free_mem, &total_mem );
        free_mem = initial_free_mem;
        /* We allocate 9/10 of the available memory */
        how_much_we_allocate = (9 * initial_free_mem) / 10;

#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        /*
         * We allocate a bunch of tiles that will be used
         * during the computations
         */
        while( (free_mem > eltsize )
               && ((total_mem - free_mem) < how_much_we_allocate)
               && !(mem_elem_per_gpu > (uint32_t)(nbelem/2*3)) ) {
            gpu_elem_t* gpu_elem;
            cudaError_t cuda_status;
#if 0
            /* Enable to stress the GPU memory subsystem and the coherence protocol */
            if( mem_elem_per_gpu > 10 )
                break;
#endif
            gpu_elem = (gpu_elem_t*)calloc(1, sizeof(gpu_elem_t));
            gpu_elem_construct(gpu_elem, NULL);

            cuda_status = (cudaError_t)cuMemAlloc( &(gpu_elem->gpu_mem_ptr), eltsize);
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                    ({
#if CUDA_VERSION < 3020
                                        unsigned int _free_mem, _total_mem;
#else
                                        size_t _free_mem, _total_mem;
#endif  /* CUDA_VERSION < 3020 */
                                        cuMemGetInfo( &_free_mem, &_total_mem );
                                        WARNING(("Per context: free mem %zu total mem %zu (allocated tiles %u)\n",
                                                 _free_mem, _total_mem, mem_elem_per_gpu));
                                        free( gpu_elem );
                                        break;
                                     }) );
            mem_elem_per_gpu++;
            dague_ulist_fifo_push( gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem );
            cuMemGetInfo( &free_mem, &total_mem );
        }
        if( 0 == mem_elem_per_gpu && dague_ulist_is_empty( gpu_device->gpu_mem_lru ) ) {
            WARNING(("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!\n",
                     dague_context->my_rank, i));
        }
        else {
            DEBUG3(( "GPU:\tAllocate %u tiles on the GPU memory\n", mem_elem_per_gpu ));
        }
#else
        if( NULL == gpu_device->memory ) {
            /*
             * We allocate all the memory on the GPU and we use our memory management
             */
            mem_elem_per_gpu = (how_much_we_allocate + GPU_MALLOC_UNIT_SIZE - 1 ) / GPU_MALLOC_UNIT_SIZE ;
            gpu_device->memory = gpu_malloc_init( mem_elem_per_gpu, GPU_MALLOC_UNIT_SIZE );

            if( gpu_device->memory == NULL ) {
                WARNING(("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!\n",
                         dague_context->my_rank, i));
            }
            else {
                DEBUG3(( "GPU:\tAllocate %u segment of size %d on the GPU memory\n",
                         mem_elem_per_gpu, GPU_MALLOC_UNIT_SIZE ));

            }
        }
#endif

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    return 0;
}

int dague_gpu_data_unregister( dague_ddesc_t* ddesc )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i;

    for(i = 0; i < __dague_active_gpu; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );
        /* Free memory on GPU */
        DAGUE_ULIST_ITERATOR(gpu_device->gpu_mem_lru, item, {
                gpu_elem_t* gpu_elem = (gpu_elem_t*)item;
                moesi_master_t* master = gpu_elem->moesi.master;
                DEBUG3(("Suppressing %p, attached to master %p, in map %p\n",
                        gpu_elem, master, ddesc->moesi_map));
                if( !master || master->map == ddesc->moesi_map ) {
                    if( master ) master->device_copies[i] = NULL;
                }
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                cuMemFree( gpu_elem->gpu_mem_ptr );
#else
                gpu_free( gpu_device->memory, (void*)gpu_elem->gpu_mem_ptr );
#endif
                item = dague_ulist_remove(gpu_device->gpu_mem_lru, item);
                free(gpu_elem);
            });
        DAGUE_ULIST_ITERATOR(gpu_device->gpu_mem_owned_lru, item, {
                gpu_elem_t* gpu_elem = (gpu_elem_t*)item;
                moesi_master_t* master = gpu_elem->moesi.master;
                DEBUG3(("Suppresing %p, attached to master %p, in map %p\n",
                        gpu_elem, master, ddesc->moesi_map));
                if( !master || master->map == ddesc->moesi_map ) {
                    if( master ) master->device_copies[i] = NULL;
                }
                if( MOESI_OWNED == gpu_elem->moesi.coherency_state ) {
                    WARNING(("GPU[%d] still OWNS the master memory copy for data %d and it is discarding it!\n", i, gpu_elem->moesi.master->key));
                    assert( gpu_elem->moesi.master->owner_device == i );
                }
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                cuMemFree( gpu_elem->gpu_mem_ptr );
#else
                gpu_free( gpu_device->memory, (void*)gpu_elem->gpu_mem_ptr );
#endif
                item = dague_ulist_remove(gpu_device->gpu_mem_owned_lru, item);
                free(gpu_elem);
            });

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        if( gpu_device->memory ) {
            gpu_malloc_fini( gpu_device->memory );
            free( gpu_device->memory );
            gpu_device->memory = NULL;
        }
#endif

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    moesi_map_destroy(&ddesc->moesi_map);
    return 0;
}

/**
 * Try to find memory space to move all data on the GPU. We attach a device_elem to
 * a memory_elem as soon as a device_elem is available. If we fail to find enough
 * available elements, we push all the elements handled during this allocation
 * back into the pool of available device_elem, to be picked up by another call
 * (this call will remove them from the current task).
 * Returns:
 *    0: All gpu_mem/mem_elem have been initialized
 *   -2: The task needs to rescheduled
 */
int dague_gpu_data_reserve_device_space( gpu_device_t* gpu_device,
                                         dague_execution_context_t *this_task,
                                         int *array_of_eltsize,
                                         int  move_data_count )
{
    gpu_elem_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    moesi_master_t* master;
    int eltsize = 0, i, j;
    (void)array_of_eltsize;
    (void)eltsize;

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0;  NULL != (master = this_task->data[i].moesi_master); i++ ) {
        temp_loc[i] = NULL;

        gpu_elem = gpu_elem_obtain_from_master(master, gpu_device->index);
        if( NULL != gpu_elem ) continue;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        gpu_elem = (gpu_elem_t*)calloc(1, sizeof(gpu_elem_t));
        gpu_elem_construct(gpu_elem, master);

        eltsize = array_of_eltsize[i];
        eltsize = (eltsize + GPU_MALLOC_UNIT_SIZE - 1) / GPU_MALLOC_UNIT_SIZE;

    malloc_data:
        gpu_elem->gpu_mem_ptr = (CUdeviceptr)gpu_malloc( gpu_device->memory, eltsize );
        if( NULL == (void*)(gpu_elem->gpu_mem_ptr) ) {
#endif

        find_another_data:
            lru_gpu_elem = (gpu_elem_t*)dague_ulist_fifo_pop(gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* Make sure all remaining temporary locations are set to NULL */
                for( ;  NULL != (master = this_task->data[i].moesi_master); temp_loc[i++] = NULL );
                break;  /* Go and cleanup */
            }
            DAGUE_LIST_ITEM_SINGLETON(lru_gpu_elem);

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the dague_gpu_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->moesi.readers ) {
                goto find_another_data;
            }
            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->moesi.master );
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
            assert(NULL != lru_gpu_elem->moesi.master);
#endif
            if( master != lru_gpu_elem->moesi.master ) {
                if( NULL != lru_gpu_elem->moesi.master ) {
                    moesi_master_t* oldmaster = lru_gpu_elem->moesi.master;
                    /* Let's check we're not trying to steal one of our own data */
                    for( j = 0; NULL != this_task->data[j].moesi_master; j++ ) {
                        if( this_task->data[j].moesi_master == oldmaster ) {
                            temp_loc[j] = lru_gpu_elem;
                            goto find_another_data;
                        }
                    }

                    oldmaster->device_copies[gpu_device->index] = NULL;
                    DEBUG3(("GPU[%d]:\tRepurpose moesi copy %p to mirror block %p (in task %s:i) instead of %p\n", gpu_device->index, lru_gpu_elem, master, this_task->function->name, i, oldmaster));

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                    gpu_free( gpu_device->memory, (void*)(lru_gpu_elem->gpu_mem_ptr) );
                    free(lru_gpu_elem);
                    goto malloc_data;
#endif
                }
            }
            gpu_elem = lru_gpu_elem;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        }
#endif
        assert( 0 == gpu_elem->moesi.readers );
        master->device_copies[gpu_device->index] = &gpu_elem->moesi;
        gpu_elem->moesi.master = master;
        gpu_elem->moesi.coherency_state = MOESI_INVALID;
        gpu_elem->moesi.version = 0;
        move_data_count--;
        temp_loc[i] = gpu_elem;
        dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    }
    if( 0 != move_data_count ) {
        WARNING(("GPU:\tRequest space on GPU failed for %d out of %d data\n",
                 move_data_count, this_task->function->nb_parameters));
        /* We can't find enough room on the GPU. Insert the tiles in the begining of
         * the LRU (in order to be reused asap) and return without scheduling the task.
         */
        for( i = 0; NULL != this_task->data[i].moesi_master; i++ ) {
            if( NULL == temp_loc[i] ) continue;
            dague_ulist_lifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)temp_loc[i]);
        }
        return -2;
    }
    return 0;
}


/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns:
 *    0: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   -1: A copy cannot be issued due to CUDA.
 */
int dague_gpu_data_stage_in( gpu_device_t* gpu_device,
                             int32_t type,
                             dague_data_pair_t* task_data,
                             size_t length,
                             CUstream stream )
{
    moesi_master_t* master = task_data->moesi_master;
    uint32_t key = master->key;
    gpu_elem_t* gpu_elem = gpu_elem_obtain_from_master(master, gpu_device->index);
    void* memptr = ADATA(task_data->data);
    int transfer_required = 0;

    /* If the data will be accessed in write mode, remove it from any lists
     * until the task is completed.
     */
    if( FLOW_ACCESS_WRITE & type ) {
        dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
        DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
    }

    transfer_required = moesi_prepare_transfer_to_device(master->map, key, gpu_device->index, (uint8_t)type);
    gpu_device->required_data_in += length;
    if( transfer_required ) {
        cudaError_t status;

        DEBUG3(("GPU:\tMove H2D data %x (H %p:D %p) %d bytes to GPU %d\n",
                key, memptr, (void*)gpu_elem->gpu_mem_ptr, length, gpu_device->device_index));
        /* Push data into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( gpu_elem->gpu_mem_ptr, memptr, length, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device ", status,
                                { WARNING(("<<%p>> -> <<%p>> [%d]\n", memptr, (void*)(long)gpu_elem->gpu_mem_ptr, length));
                                    return -1; } );
        gpu_device->transferred_data_in += length;
        master->mem_ptr = memptr;
        /* TODO: take ownership of the data */
        return 1;
    }
    /* TODO: data keeps the same coherence flags as before */
    return 0;
}


#if DAGUE_GPU_USE_PRIORITIES
static inline dague_list_item_t* dague_fifo_push_ordered( dague_list_t* fifo,
                                                          dague_list_item_t* elem )
{
    dague_ulist_push_sorted(fifo, elem, dague_execution_context_priority_comparator);
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_ulist_fifo_push
#endif

int progress_stream( gpu_device_t* gpu_device,
                     dague_gpu_exec_stream_t* exec_stream,
                     advance_task_function_t progress_fct,
                     dague_gpu_context_t* task,
                     dague_gpu_context_t** out_task )
{
    int saved_rc = 0, rc;
    *out_task = NULL;

    if( NULL != task ) {
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        task = NULL;
    }
 grab_a_task:
    if( NULL == exec_stream->tasks[exec_stream->start] ) {
        /* get the best task */
        task = (dague_gpu_context_t*)dague_ulist_fifo_pop(exec_stream->fifo_pending);
    }
    if( NULL == task ) {
        /* No more room on the event list or no tasks. Keep moving */
        goto check_completion;
    }
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)task);

    assert( NULL == exec_stream->tasks[exec_stream->start] );
    /**
     * In case the task is succesfully progressed, the corresponding profiling
     * event is triggered.
     */
    rc = progress_fct( gpu_device, task, exec_stream );
    if( 0 > rc ) {
        if( -1 == rc ) return -1;  /* Critical issue */
        assert(0); // want to debug this. It happens too often
        /* No more room on the GPU. Push the task back on the queue and check the completion queue. */
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        DEBUG2(( "GPU: Reschedule %s(task %p) priority %d: no room available on the GPU for data\n",
                 task->ec->function->name, (void*)task->ec, task->ec->priority ));
        saved_rc = rc;  /* keep the info for the upper layer */
    } else {
        /**
         * Do not skip the cuda event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to incorrect results.
         */
        rc = cuEventRecord( exec_stream->events[exec_stream->start], exec_stream->cuda_stream );
        exec_stream->tasks[exec_stream->start] = task;
        exec_stream->start = (exec_stream->start + 1) % exec_stream->max_events;
        DEBUG3(( "GPU: Submitted %s(task %p) priority %d on stream %p\n",
                 task->ec->function->name, (void*)task->ec, task->ec->priority,
                 (void*)exec_stream->cuda_stream ));
    }
    task = NULL;

 check_completion:
    if( (NULL == *out_task) && (NULL != exec_stream->tasks[exec_stream->end]) ) {
        rc = cuEventQuery(exec_stream->events[exec_stream->end]);
        if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            task = *out_task = exec_stream->tasks[exec_stream->end];
            DEBUG3(("GPU: Complete %s(task %p) on stream %p\n", task->ec->function->name, (void*)task,
                    (void*)exec_stream->cuda_stream));
            exec_stream->tasks[exec_stream->end] = NULL;
            exec_stream->end = (exec_stream->end + 1) % exec_stream->max_events;
            DAGUE_TASK_PROF_TRACE_IF(exec_stream->prof_event_track_enable,
                                     exec_stream->profiling,
                                     (-1 == exec_stream->prof_event_key_end ?
                                      DAGUE_PROF_FUNC_KEY_END(task->ec->dague_object,
                                                              task->ec->function->function_id) :
                                      exec_stream->prof_event_key_end),
                                     task->ec);
            task = NULL;  /* Try to schedule another task */
            goto grab_a_task;
        }
        if( CUDA_ERROR_NOT_READY != rc ) {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {return -1;} );
        }
#if 0
        else {
            static cudaEvent_t ev = NULL;
            static double first = 0.0;
            static double last = 0.0;
            double new = MPI_Wtime();
            if(exec_stream->events[exec_stream->end] != ev) {
                first = new;
                ev = exec_stream->events[exec_stream->end];
                printf("%p : %f\tNEW\tsince last poll (on the prev. event)\n", ev, first - last);
            } else {
                printf("%p : %f\tsame\tsince last poll (on the same event)\tTOTAL: %f\n", ev, new - last, new - first);
            }
            last = new;
        }
#endif
    }
    return saved_rc;
}

void dague_compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
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

int dague_gpu_kernel_fini(dague_context_t* dague_context,
                          char *kernelname)
{
    gpu_device_t *gpu_device;
    CUresult status;
    int *gpu_counter;
    int i, total = 0, active_devices = 0;
    uint64_t *transferred_in, *transferred_out;
    uint64_t *required_in,    *required_out;
    float gtotal = 0.0;
    float best_data_in, best_data_out;
    float best_required_in, best_required_out;
    char *data_in_unit, *data_out_unit;
    char *required_in_unit, *required_out_unit;
    uint64_t total_data_in = 0,     total_data_out = 0;
    uint64_t total_required_in = 0, total_required_out = 0;

    if (__dague_active_gpu <= 0)
        return 0;

    /* GPU counter for GEMM / each */
    gpu_counter     = (int*)     calloc(__dague_active_gpu, sizeof(int)     );
    transferred_in  = (uint64_t*)calloc(__dague_active_gpu, sizeof(uint64_t));
    transferred_out = (uint64_t*)calloc(__dague_active_gpu, sizeof(uint64_t));
    required_in     = (uint64_t*)calloc(__dague_active_gpu, sizeof(uint64_t));
    required_out    = (uint64_t*)calloc(__dague_active_gpu, sizeof(uint64_t));

    for(i = 0; i < __dague_active_gpu; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                {continue;} );
        status = cuCtxSynchronize();
        DAGUE_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
                                {continue;} );
        /* Save the statistics */
        gpu_counter[gpu_device->index]     += gpu_device->executed_tasks;
        transferred_in[gpu_device->index]  += gpu_device->transferred_data_in;
        transferred_out[gpu_device->index] += gpu_device->transferred_data_out;
        required_in[gpu_device->index]     += gpu_device->required_data_in;
        required_out[gpu_device->index]    += gpu_device->required_data_out;

        active_devices++;

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    if( 0 == active_devices )  /* No active devices */
        return 0;

    /* Print statistics */
    for( i = 0; i < __dague_active_gpu; i++ ) {
        total              += gpu_counter[i];
        total_data_in      += transferred_in[i];
        total_data_out     += transferred_out[i];
        total_required_in  += required_in[i];
        total_required_out += required_out[i];
    }

    if( 0 == total_data_in )  total_data_in  = 1;
    if( 0 == total_data_out ) total_data_out = 1;
    gtotal = (float)total + (float)dague_cpu_counter;

    if(dague_gpu_data_verbose) {
        printf("-------------------------------------------------------------------------------------------------\n");
        printf("|         |                   |         Data In                |         Data Out               |\n");
        printf("|PU % 5d |  # %5s  |   %%   |  Required  |   Transfered(%%)   |  Required  |   Transfered(%%)   |\n",
                 dague_context->my_rank, kernelname);
        printf("|---------|-----------|-------|------------|-------------------|------------|-------------------|\n");
        for( i = 0; i < __dague_active_gpu; i++ ) {
            CUdevice hcuDevice;
            char szName[256];

            gpu_device = gpu_enabled_devices[i];

            dague_compute_best_unit( required_in[i],     &best_required_in,  &required_in_unit  );
            dague_compute_best_unit( required_out[i],    &best_required_out, &required_out_unit );
            dague_compute_best_unit( transferred_in[i],  &best_data_in,      &data_in_unit      );
            dague_compute_best_unit( transferred_out[i], &best_data_out,     &data_out_unit     );

            status = cuDeviceGet( &hcuDevice, gpu_device->device_index );
            DAGUE_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {continue;} );
            status = cuDeviceGetName( szName, 256, hcuDevice );
            DAGUE_CUDA_CHECK_ERROR( "cuDeviceGetName ", status, {continue;} );

            printf("|  GPU %2d |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) | %s\n",
                   gpu_device->device_index, gpu_counter[i], (gpu_counter[i]/gtotal)*100.00,
                   best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
                   (((double)transferred_in[i])  / (double)required_in[i] ) * 100.0,
                   best_required_out, required_out_unit, best_data_out, data_out_unit,
                   (((double)transferred_out[i]) / (double)required_out[i]) * 100.0, szName );
        }
        printf("|---------|-----------|-------|------------|-------------------|------------|-------------------|\n");

        dague_compute_best_unit( total_required_in,  &best_required_in,  &required_in_unit  );
        dague_compute_best_unit( total_required_out, &best_required_out, &required_out_unit );
        dague_compute_best_unit( total_data_in,      &best_data_in,      &data_in_unit      );
        dague_compute_best_unit( total_data_out,     &best_data_out,     &data_out_unit     );

        printf("|All GPUs |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) |\n",
                total, (total/gtotal)*100.00,
                best_required_in,  required_in_unit,  best_data_in,  data_in_unit,
                ((double)total_data_in  / (double)total_required_in ) * 100.0,
                best_required_out, required_out_unit, best_data_out, data_out_unit,
                ((double)total_data_out / (double)total_required_out) * 100.0);
        printf("|All CPUs |%10d | %5.2f | %8.2f%2s | %8.2f%2s(%5.2f) | %8.2f%2s | %8.2f%2s(%5.2f) |\n",
                (int)dague_cpu_counter, (dague_cpu_counter / gtotal)*100.00,
                0.0, " ", 0.0, " ", 0.0, 0.0, " ", 0.0, " ", 0.0);
        printf("-------------------------------------------------------------------------------------------------\n");
    }
    free(gpu_counter);
    free(transferred_in);
    free(transferred_out);
    free(required_in);
    free(required_out);
    return 0;
}

void dump_exec_stream(dague_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    printf( "Dump GPU exec stream %p [events = %d, start = %d, end = %d, executed = %d]\n",
            exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
            exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        printf( "    %d: %s\n", i, dague_snprintf_execution_context(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void dump_GPU_state(gpu_device_t* gpu_device)
{
    int i;

    printf("\n\n");
    printf("Device %d:%d (%p)\n", gpu_device->device_index, gpu_device->index, gpu_device);
    printf("\tpeer mask %x executed tasks %d max streams %d\n",
           gpu_device->peer_access_mask, gpu_device->executed_tasks, gpu_device->max_exec_streams);
    printf("\tstats transferred [in %lu out %lu] required [in %lu out %lu]\n",
           gpu_device->transferred_data_in, gpu_device->transferred_data_out,
           gpu_device->required_data_in, gpu_device->required_data_out);
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dump_exec_stream(&gpu_device->exec_stream[i]);
    }
    if( !dague_ulist_is_empty(gpu_device->gpu_mem_lru) ) {
        printf("#\n# LRU list\n#\n");
        i = 0;
        DAGUE_LIST_ITERATOR(gpu_device->gpu_mem_lru, item,
                            {
                                gpu_elem_t* gpu_elem = (gpu_elem_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_elem, (void*)(uintptr_t)gpu_elem->gpu_mem_ptr);
                                moesi_dump_moesi_copy(&gpu_elem->moesi);
                                i++;
                            });
    };
    if( !dague_ulist_is_empty(gpu_device->gpu_mem_owned_lru) ) {
        printf("#\n# Owned LRU list\n#\n");
        i = 0;
        DAGUE_LIST_ITERATOR(gpu_device->gpu_mem_owned_lru, item,
                            {
                                gpu_elem_t* gpu_elem = (gpu_elem_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_elem, (void*)(uintptr_t)gpu_elem->gpu_mem_ptr);
                                moesi_dump_moesi_copy(&gpu_elem->moesi);
                                i++;
                            });
    };
    printf("\n\n");
}

#endif /* HAVE_CUDA */
