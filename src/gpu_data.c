/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague.h"
#include "gpu_data.h"
#include "profiling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "lifo.h"

static int __dague_active_gpu = 0;
static uint32_t __dague_gpu_mask = 0xffffffff;
static CUcontext dague_allocate_on_gpu_context;
static int dague_gpu_allocation_initialized = 0;

static void* dague_allocate_data_gpu(size_t matrix_size)
{
    void* mat = NULL;

    if( __dague_active_gpu ) {
        CUresult status;

        status = cuCtxPushCurrent( dague_allocate_on_gpu_context );
        DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                {
                                    ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                } );

        status = cuMemHostAlloc( (void**)&mat, matrix_size, CU_MEMHOSTALLOC_PORTABLE);
        DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuMemHostAlloc failed ", status,
                                {
                                    ERROR(("Unable to allocate GPU-compatible data as requested.\n"));
                                } );
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {} );
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
static void dague_free_data_gpu(void *dta)
{
    unsigned int flags, call_free = 1;

    if( dague_gpu_allocation_initialized ) {
        CUresult status;

        status = cuCtxPushCurrent( dague_allocate_on_gpu_context );
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                { goto clib_free; } );

        status = cuMemHostGetFlags( &flags, dta );
        DAGUE_CUDA_CHECK_ERROR( "cuMemHostGetFlags ", status,
                                {goto clib_free;} );

        status = cuMemFreeHost( dta );
        DAGUE_CUDA_CHECK_ERROR( "cuMemFreeHost ", status,
                                {goto clib_free;} );
        call_free = 0;
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPopCurrent ", status,
                                {} );
    }

  clib_free:
    if( call_free ) free( dta );
}

/**
 * Enable GPU-compatible memory if possible
 */
void dague_data_enable_gpu( int nbgpu )
{
    __dague_active_gpu = nbgpu;

    dague_data_allocate = dague_allocate_data_gpu;
    dague_data_free     = dague_free_data_gpu;
}

int dague_active_gpu(void)
{
    return __dague_active_gpu;
}

#if defined(DAGUE_PROF_TRACE)
/* Accepted values are: DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      DAGUE_PROFILE_CUDA_TRACK_OWN | DAGUE_PROFILE_CUDA_TRACK_EXEC
 */
int dague_cuda_trackable_events = DAGUE_PROFILE_CUDA_TRACK_EXEC | DAGUE_PROFILE_CUDA_TRACK_OWN;
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

int dague_gpu_init(int* puse_gpu, int dague_show_detailed_capabilities)
{
    int ndevices, i, dindex;
    CUresult status;

    if( (*puse_gpu) == -1 ) {
        return -1;  /* Nothing to do around here */
    }
    status = cuInit(0);
    DAGUE_CUDA_CHECK_ERROR( "cuInit ", status, {*puse_gpu = 0; return -1;} );

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
    for( i = dindex = 0; i < ndevices; i++ ) {
#if CUDA_VERSION >= 3020
        size_t total_mem;
#else
        unsigned int total_mem;
#endif  /* CUDA_VERSION >= 3020 */
        gpu_device_t* gpu_device;
        CUdevprop devProps;
        char szName[256];
        int major, minor, concurrency;
        CUdevice hcuDevice;

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & __dague_gpu_mask) ) continue;

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
        }
        status = cuDeviceTotalMem( &total_mem, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceTotalMem ", status, {continue;} );

        gpu_device = (gpu_device_t*)calloc(1, sizeof(gpu_device_t));
        gpu_enabled_devices[dindex] = gpu_device;
        dague_list_construct(&gpu_device->pending);
        gpu_device->major = (uint8_t)major;
        gpu_device->minor = (uint8_t)minor;

        if( dague_gpu_allocation_initialized == 0 ) {
            status = cuCtxCreate( &dague_allocate_on_gpu_context, 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                    {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );
            status = cuCtxPopCurrent(NULL);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                    {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );
            dague_gpu_allocation_initialized = 1;
        }

        /* cuCtxCreate: Function works on floating contexts and current context */
        status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue; } );

        /**
         * Allocate the streams
         */
        {
            int stream_id;
            cudaError_t cuda_status;

            gpu_device->max_streams = DAGUE_MAX_STREAMS;
            for( stream_id = 0; stream_id < gpu_device->max_streams; stream_id++ ) {
                cuda_status = (cudaError_t)cuStreamCreate( &(gpu_device->streams[stream_id]), 0 );
                DAGUE_CUDA_CHECK_ERROR( "cuStreamCreate ", cuda_status,
                                        ({
                                            gpu_device->max_streams = stream_id - 1;
                                            break;
                                        }) );
            }
        }
        gpu_device->index                = (uint8_t)dindex;
        gpu_device->device_index         = (uint8_t)i;
        gpu_device->executed_tasks       = 0;
        gpu_device->transferred_data_in  = 0;
        gpu_device->transferred_data_out = 0;

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); gpu_enabled_devices[dindex] = NULL; continue;} );

#if defined(DAGUE_PROF_TRACE)
        gpu_device->profiling = dague_profiling_thread_init( 2*1024*1024, "GPU %d.0", i );
#endif  /* defined(PROFILING) */
        dindex++;
    }

    dague_data_enable_gpu( ndevices );
    return 0;
}

int dague_gpu_fini( void )
{
    gpu_device_t* gpu_device;
    cudaError_t status;
    int i, j;

    for(i = 0; i < __dague_active_gpu; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;
        gpu_enabled_devices[i] = NULL;

        status = (cudaError_t)cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );
        /**
         * Release all streams
         */
        for( j = 0; j < gpu_device->max_streams; j++ ) {
            cuStreamDestroy( gpu_device->streams[j] );
        }
        status = (cudaError_t)cuCtxDestroy( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxDestroy ", status,
                                {continue;} );
        gpu_device->ctx = NULL;
    }
    free(gpu_enabled_devices);
    gpu_enabled_devices = NULL;
    __dague_active_gpu = 0;
    return 0;
}

#endif /* HAVE_CUDA */

