/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include <dague/utils/mca_param.h>
#include <dague/constants.h>

#if defined(HAVE_CUDA)
#include "dague.h"
#include "data.h"
#include <dague/devices/cuda/dev_cuda.h>
#include <dague/devices/device_malloc.h>
#include "profiling.h"
#include "execution_unit.h"
#include "arena.h"
#include "dague/utils/output.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <dlfcn.h>

#if defined(DAGUE_PROF_TRACE)
/* Accepted values are: DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      DAGUE_PROFILE_CUDA_TRACK_OWN | DAGUE_PROFILE_CUDA_TRACK_EXEC
 */
int dague_cuda_trackable_events = DAGUE_PROFILE_CUDA_TRACK_EXEC | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT | DAGUE_PROFILE_CUDA_TRACK_DATA_IN;
int dague_cuda_movein_key_start;
int dague_cuda_movein_key_end;
int dague_cuda_moveout_key_start;
int dague_cuda_moveout_key_end;
int dague_cuda_own_GPU_key_start;
int dague_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

int dague_cuda_output_stream = -1;

/* Dirty selection for now */
float gpu_speeds[2][2] ={
    /* C1060, C2050 */
    { 622.08, 1030.4 },
    {  77.76,  515.2 }
};

static int dague_cuda_device_fini(dague_device_t* device)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    CUresult status;
    int j, k;

    status = cuCtxPushCurrent( gpu_device->ctx );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_device_fini) cuCtxPushCurrent ", status,
                            {continue;} );
    /* Release pending queue */
    OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
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

    status = cuCtxDestroy( gpu_device->ctx );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_device_fini) cuCtxDestroy ", status,
                                {continue;} );
    gpu_device->ctx = NULL;

    /* Cleanup the GPU memory. */
    OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    free(gpu_device);

    return DAGUE_SUCCESS;
}

static int dague_cuda_memory_register(dague_device_t* device, void* ptr, size_t length)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    CUresult status;
    CUcontext ctx;
    int rc = DAGUE_ERROR;

    /* Atomically get the GPU context */
    do {
        ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), ctx, NULL );
    } while( NULL == ctx );

    status = cuCtxPushCurrent( ctx );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cuCtxPushCurrent ", status,
                            {goto restore_and_return;} );

    status = cuMemHostRegister(ptr, length, CU_MEMHOSTREGISTER_PORTABLE);
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cuMemHostRegister ", status,
                            {goto restore_and_return;} );

    status = cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cuCtxPopCurrent ", status,
                            {goto restore_and_return;} );
    rc = DAGUE_SUCCESS;

  restore_and_return:
    /* Restore the context so the others can steal it */
    dague_atomic_cas( &(gpu_device->ctx), NULL, ctx );

    return rc;
}

static int dague_cuda_memory_unregister(dague_device_t* device, void* ptr)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    CUresult status;
    CUcontext ctx;
    int rc = DAGUE_ERROR;

    /* Atomically get the GPU context */
    do {
        ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), ctx, NULL );
    } while( NULL == ctx );

    status = cuCtxPushCurrent( ctx );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cuCtxPushCurrent ", status,
                            {goto restore_and_return;} );

    status = cuMemHostUnregister(ptr);
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_unregister) cuMemHostUnregister ", status,
                            {goto restore_and_return;} );

    status = cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cuCtxPopCurrent ", status,
                            {goto restore_and_return;} );
    rc = DAGUE_SUCCESS;

  restore_and_return:
    /* Restore the context so the others can steal it */
    dague_atomic_cas( &(gpu_device->ctx), NULL, ctx );

    return rc;
}

static int cuda_legal_compute_capabilitites[] = {10, 11, 12, 13, 20, 21, 30, 35};

void* cuda_solve_handle_dependencies(gpu_device_t* gpu_device,
                                     const char* fname)
{
    char library_name[FILENAME_MAX], function_name[FILENAME_MAX], *env;
    int i, index, capability = gpu_device->major * 10 + gpu_device->minor;
    CUresult status;
    void *fn = NULL, *dlh = NULL;

    status = cuCtxPushCurrent( gpu_device->ctx );
    DAGUE_CUDA_CHECK_ERROR( "(cuda_solve_handle_dependencies) cuCtxPushCurrent ", status, {continue;} );

    for( i = 0, index = -1; i < sizeof(cuda_legal_compute_capabilitites); i++ ) {
        if(cuda_legal_compute_capabilitites[i] == capability) {
            index = i;
            break;
        }
    }
    if( -1 == index ) {  /* This shouldn't have happened */
        return NULL;
    }
  retry_lesser_sm_version:
    capability = cuda_legal_compute_capabilitites[index];
    snprintf(function_name, FILENAME_MAX, "magmablas_S%s_SM%2d", fname, capability);
    env = getenv("DAGUE_CUCORES_LIB");
    if(NULL == env) {
        snprintf(library_name,  FILENAME_MAX, "libdplasma_cucores_sm%d.so",  capability);
    } else {
        snprintf(library_name,  FILENAME_MAX, "%s", env);
    }

    dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
    if(NULL == dlh) {
        dague_output_verbose(5, dague_cuda_output_stream,
                             "Could not find %s dynamic library (%s)\n", library_name, dlerror());
        if(env) ERROR(("Could not find %s library: %s\n"
                       "  It is derived from environment DAGUE_CUCORES_LIB=%s\n"
                       "  To resolve this issue, set this variable to the correct path\n"
                       "    ex: /path/libdplasma_cucores_sm20.so\n"
                       "  Or unset it to use the default GPU kernels\n"
                       , library_name, dlerror(), env));
    } else {
        fn = dlsym(dlh, function_name);
        dlclose(dlh);
    }
    /* Couldn't load from dynamic libs, try static */
    if(NULL == fn) {
        dague_output_verbose(5, dague_cuda_output_stream,
                             "No dynamic function %s found, trying from  statically linked\n",
                             function_name);
        dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
        if(NULL == dlh) {
            ERROR(("Error parsing static libs: %s\n", dlerror()));
        }
        fn = dlsym(dlh, function_name);
        if(env && fn) {
            dague_output_verbose(10, dague_cuda_output_stream,
                                 "Function %s found in the application object\n",
                                 function_name);
        }
        dlclose(dlh);
    } else {
        dague_output_verbose(10, dague_cuda_output_stream,
                             "Function %s found in shared object %s\n",
                             function_name, library_name);
    }

    /* Still not found?? skip this GPU */
    if(NULL == fn) {
        dague_output_verbose(1, dague_cuda_output_stream,
                             "No function %s found for CUDA device %s\n",
                             function_name, gpu_device->super.name);
        index--;
        if(0 < index)
            goto retry_lesser_sm_version;
    }

    status = cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                            {continue;} );

    return fn;
}

/* TODO: Ugly code to be removed ASAP */
void** cuda_gemm_functions = NULL;
/* TODO: Ugly code to be removed ASAP */

static int
dague_cuda_handle_register(dague_device_t* device, dague_handle_t* handle)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    uint32_t i, dev_mask = 0x0;

    /**
     * Let's suppose it is not our job to detect if a particular body can
     * run or not. We will need to add some properties that will allow the
     * user to write the code to assess this.
     */
    assert(DAGUE_DEV_CUDA == device->type);
    for( i = 0; i < handle->nb_functions; i++ ) {
        const dague_function_t* function = handle->functions_array[i];
        __dague_chore_t* chores = (__dague_chore_t*)function->incarnations;
        for( uint32_t j = 0; NULL != chores[j].hook; j++ ) {
            dev_mask |= (1 << chores[j].type);
        }
        if(dev_mask & (1 << device->type)) {  /* find the function */
            void* devf = cuda_solve_handle_dependencies(gpu_device, function->name);
            /* TODO: Ugly code to be removed ASAP */
            if( NULL == cuda_gemm_functions ) {
                cuda_gemm_functions = (void**)calloc(100, sizeof(void*));
            }
            cuda_gemm_functions[gpu_device->cuda_index] = devf;
            /* TODO: Ugly code to be removed ASAP */
        }
    }
    /* Not a single chore supports this device, there is no reason to check anything further */
    if(!(dev_mask & (1 << device->type))) {
        handle->devices_mask &= ~(device->device_index);
    }

    return DAGUE_SUCCESS;
}

static int
dague_cuda_handle_unregister(dague_device_t* device, dague_handle_t* handle)
{
    return DAGUE_SUCCESS;
}

int dague_gpu_init(dague_context_t *dague_context)
{
    int show_caps_index, show_caps = 0;
    int use_cuda_index, use_cuda;
    int cuda_mask_index, cuda_mask;
    int cuda_output_index, cuda_verbosity;
    int ndevices, i, j, k;
    CUresult status;

    use_cuda_index = dague_mca_param_reg_int_name("device_cuda", "enabled",
                                                  "The number of CUDA device to enable for the next PaRSEC context",
                                                  false, false, 0, &use_cuda);
    cuda_mask_index = dague_mca_param_reg_int_name("device_cuda", "mask",
                                                   "The bitwise mask of CUDA devices to be enabled (default all)",
                                                   false, false, 0xffffffff, &cuda_mask);
    cuda_output_index = dague_mca_param_reg_int_name("device_cuda", "verbose",
                                                     "Set the verbosity level of the CUDA device (negative value turns all output off, higher is less verbose)\n",
                                                     false, false, -1, &cuda_verbosity);
    if( 0 == use_cuda ) {
        return -1;  /* Nothing to do around here */
    }

    if( cuda_verbosity >= 0 ) {
        dague_cuda_output_stream = dague_output_open(NULL);
        dague_output_set_verbosity(dague_cuda_output_stream, cuda_verbosity);
    }

    status = cuInit(0);
    DAGUE_CUDA_CHECK_ERROR( "cuInit ", status,
                            {
                                if( 0 < use_cuda_index )
                                    dague_mca_param_set_int(use_cuda_index, 0);
                                return -1;
                            } );

    cuDeviceGetCount( &ndevices );

    if( ndevices < use_cuda ) {
        if( 0 < use_cuda_index )
            dague_mca_param_set_int(use_cuda_index, ndevices);
    }
    /* Update the number of GPU for the upper layer */
    use_cuda = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }
    show_caps_index = dague_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        dague_mca_param_lookup_int(show_caps_index, &show_caps);
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

    for( i = 0; i < ndevices; i++ ) {
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

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & cuda_mask) ) continue;

        if( show_caps ) {
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
        OBJ_CONSTRUCT(gpu_device, dague_list_item_t);
        gpu_device->major = (uint8_t)major;
        gpu_device->minor = (uint8_t)minor;
        gpu_device->super.name = strdup(szName);

        /* cuCtxCreate: Function works on floating contexts and current context */
        status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                {free(gpu_device); continue; } );

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
            exec_stream->fifo_pending = (dague_list_t*)OBJ_NEW(dague_list_t);
            OBJ_CONSTRUCT(exec_stream->fifo_pending, dague_list_t);
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
            exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC;
            exec_stream->prof_event_key_start    = -1;
            exec_stream->prof_event_key_end      = -1;
#endif  /* defined(DAGUE_PROF_TRACE) */
        }

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); continue;} );

        gpu_device->cuda_index                 = (uint8_t)i;
        gpu_device->super.type                 = DAGUE_DEV_CUDA;
        gpu_device->super.executed_tasks       = 0;
        gpu_device->super.transferred_data_in  = 0;
        gpu_device->super.transferred_data_out = 0;
        gpu_device->super.required_data_in     = 0;
        gpu_device->super.required_data_out    = 0;

        gpu_device->super.device_fini              = dague_cuda_device_fini;
        gpu_device->super.device_memory_register   = dague_cuda_memory_register;
        gpu_device->super.device_memory_unregister = dague_cuda_memory_unregister;
        gpu_device->super.device_handle_register   = dague_cuda_handle_register;
        gpu_device->super.device_handle_unregister = dague_cuda_handle_unregister;

        /**
         * TODO: Find a better ay to evaluate the performance of the current GPU.
         * device_weight[i+1] = ((float)devProps.maxThreadsPerBlock * (float)devProps.clockRate) * 2;
         * device_weight[i+1] *= (concurrency == 1 ? 2 : 1);
         */
        gpu_device->super.device_dweight = ( major == 1 ) ? gpu_speeds[1][0] : gpu_speeds[1][1];
        gpu_device->super.device_sweight = ( major == 1 ) ? gpu_speeds[0][0] : gpu_speeds[0][1];

        /* Initialize internal lists */
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       dague_list_t);
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, dague_list_t);
        OBJ_CONSTRUCT(&gpu_device->pending,           dague_list_t);

#if defined(DAGUE_PROF_TRACE)
        gpu_device->super.profiling = dague_profiling_thread_init( 2*1024*1024, "GPU %d.0", i );
        /**
         * Reconfigure the stream 0 and 1 for input and outputs.
         */
        gpu_device->exec_stream[0].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN;
        gpu_device->exec_stream[0].prof_event_key_start    = dague_cuda_movein_key_start;
        gpu_device->exec_stream[0].prof_event_key_end      = dague_cuda_movein_key_end;

        gpu_device->exec_stream[1].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT;
        gpu_device->exec_stream[1].prof_event_key_start    = dague_cuda_moveout_key_start;
        gpu_device->exec_stream[1].prof_event_key_end      = dague_cuda_moveout_key_end;
#endif  /* defined(PROFILING) */
        dague_devices_add(dague_context, &(gpu_device->super));
    }

#if defined(DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t *source_gpu, *target_gpu;
        CUdevice source, target;
        int canAccessPeer;

        if( NULL == (source_gpu = (gpu_device_t*)dague_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( DAGUE_DEV_CUDA != source_gpu->super.type ) continue;

        source_gpu->peer_access_mask = 0;
        status = cuDeviceGet( &source, source_gpu->cuda_index );
        DAGUE_CUDA_CHECK_ERROR( "No peer memory access: cuDeviceGet ", status, {continue;} );
        status = cuCtxPushCurrent( source_gpu->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cuCtxPushCurrent ", status,
                                {continue;} );

        for( j = 0; j < ndevices; j++ ) {
            if( (NULL == (target_gpu = (gpu_device_t*)dague_devices_get(j))) || (i == j) ) continue;
            /* Skip all non CUDA devices */
            if( DAGUE_DEV_CUDA != target_gpu->super.type ) continue;

            status = cuDeviceGet( &target, target_gpu->cuda_index );
            DAGUE_CUDA_CHECK_ERROR( "No peer memory access: cuDeviceGet ", status, {continue;} );

            /* Communication mask */
            status = cuDeviceCanAccessPeer( &canAccessPeer, source, target );
            DAGUE_CUDA_CHECK_ERROR( "cuDeviceCanAccessPeer ", status,
                                    {continue;} );
            if( 1 == canAccessPeer ) {
                status = cuCtxEnablePeerAccess( target_gpu->ctx, 0 );
                DAGUE_CUDA_CHECK_ERROR( "cuCtxEnablePeerAccess ", status,
                                        {continue;} );
                source_gpu->peer_access_mask |= (int16_t)(1 << target_gpu->cuda_index);
            }
        }
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cuCtxPopCurrent ", status,
                                {continue;} );
    }
#endif

    return 0;
}

int dague_gpu_fini(void)
{
    dague_output_close(dague_cuda_output_stream);
    dague_cuda_output_stream = -1;

    return DAGUE_SUCCESS;
}

/*
 * TODO: this function should not be necessary anymore.
 */
int dague_gpu_data_register( dague_context_t *dague_context,
                             dague_ddesc_t   *data,
                             int              nbelem, /* Could be a function of the dague_desc_t */
                             size_t           eltsize )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i;
    (void)eltsize;

    for(i = 0; i < dague_nb_devices; i++) {
        size_t how_much_we_allocate;
#if CUDA_VERSION < 3020
        unsigned int total_mem, free_mem, initial_free_mem;
#else
        size_t total_mem, free_mem, initial_free_mem;
#endif  /* CUDA_VERSION < 3020 */
        uint32_t mem_elem_per_gpu = 0;

        if( NULL == (gpu_device = (gpu_device_t*)dague_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( DAGUE_DEV_CUDA != gpu_device->super.type ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_data_register) cuCtxPushCurrent ", status,
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
               && (initial_free_mem - how_much_we_allocate)
               && !(mem_elem_per_gpu > (uint32_t)(nbelem/2*3)) ) {
            dague_gpu_data_copy_t* gpu_elem;
            CUdeviceptr device_ptr;
            cudaError_t cuda_status;
#if 0
            /* Enable to stress the GPU memory subsystem and the coherence protocol */
            if( mem_elem_per_gpu > 10 )
                break;
#endif
            gpu_elem = OBJ_NEW(dague_data_copy_t);

            cuda_status = (cudaError_t)cuMemAlloc( &device_ptr, eltsize);
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                    ({
#if CUDA_VERSION < 3020
                                        unsigned int _free_mem, _total_mem;
#else
                                        size_t _free_mem, _total_mem;
#endif  /* CUDA_VERSION < 3020 */
                                        cuMemGetInfo( &_free_mem, &_total_mem );
                                        WARNING(("Per context: free mem %zu total mem %zu\n",
                                                 _free_mem, _total_mem));
                                        free( gpu_elem );
                                        break;
                                     }) );
            gpu_elem->device_private = (void*)(long)device_ptr;
            gpu_elem->device_index = gpu_device->super.device_index;
            mem_elem_per_gpu++;
            dague_ulist_fifo_push( &gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem );
            cuMemGetInfo( &free_mem, &total_mem );
        }
        if( 0 == mem_elem_per_gpu ) {
            WARNING(("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!\n",
                     dague_context->my_rank, i));
            continue;
        }
        DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                              "GPU:\tAllocate %u tiles on the GPU memory\n", mem_elem_per_gpu ));
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
                continue;
            }
            DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                  "GPU:\tAllocate %u segment of size %d on the GPU memory\n",
                                  mem_elem_per_gpu, GPU_MALLOC_UNIT_SIZE ));
        }
#endif

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    return 0;
}

/**
 * This function release all copies of a data on all devices. It ensure
 * the most recent version is moved back in main memory, and then release
 * the corresponding data from all attached devices.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
int dague_gpu_data_unregister( dague_ddesc_t* ddesc )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i;

    for(i = 0; i < dague_nb_devices; i++) {
        if( NULL == (gpu_device = (gpu_device_t*)dague_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( DAGUE_DEV_CUDA != gpu_device->super.type ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_data_unregister) cuCtxPushCurrent ", status,
                                {continue;} );
        /* Free memory on GPU */
        DAGUE_ULIST_ITERATOR(&gpu_device->gpu_mem_lru, item, {
                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                dague_data_t* original = gpu_copy->original;
                DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                      "Considering suppresion of copy %p, attached to %p, in map %p",
                                      gpu_copy, original, ddesc));
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                cuMemFree( (CUdeviceptr)gpu_copy->device_private );
#else
                gpu_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
                item = dague_ulist_remove(&gpu_device->gpu_mem_lru, item);
                OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
            });
        DAGUE_ULIST_ITERATOR(&gpu_device->gpu_mem_owned_lru, item, {
                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                dague_data_t* original = gpu_copy->original;
                DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                      "Considering suppresion of owned copy %p, attached to %p, in map %p",
                                      gpu_copy, original, ddesc));
                if( DATA_COHERENCY_OWNED == gpu_copy->coherency_state ) {
                    WARNING(("GPU[%d] still OWNS the master memory copy for data %d and it is discarding it!\n",
                             i, original->key));
                }
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                cuMemFree( (CUdeviceptr)gpu_copy->device_private );
#else
                gpu_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
                item = dague_ulist_remove(&gpu_device->gpu_mem_lru, item);
                OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
            });

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        gpu_malloc_fini( gpu_device->memory );
        free( gpu_device->memory );
#endif

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

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
                                         int  move_data_count )
{
    dague_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    dague_data_t* master;
    int eltsize = 0, i, j;
    (void)eltsize;

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

        temp_loc[i] = NULL;
        master = this_task->data[i].data->original;
        gpu_elem = dague_data_get_copy(master, gpu_device->super.device_index);
        /* There is already a copy on the device */
        if( NULL != gpu_elem ) continue;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        gpu_elem = OBJ_NEW(dague_data_copy_t);

        eltsize = master->nb_elts;
        eltsize = (eltsize + GPU_MALLOC_UNIT_SIZE - 1) / GPU_MALLOC_UNIT_SIZE;

    malloc_data:
        gpu_elem->device_private = gpu_malloc( gpu_device->memory, eltsize );
        if( NULL == gpu_elem->device_private ) {
#endif

        find_another_data:
            lru_gpu_elem = (dague_gpu_data_copy_t*)dague_ulist_fifo_pop(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* Make sure all remaining temporary locations are set to NULL */
                for( ;  NULL != this_task->data[i].data; temp_loc[i++] = NULL );
                break;  /* Go and cleanup */
            }
            DAGUE_LIST_ITEM_SINGLETON(lru_gpu_elem);

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the dague_gpu_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                goto find_another_data;
            }
            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
            assert(NULL != lru_gpu_elem->original);
#endif
            if( master != lru_gpu_elem->original ) {
                if( NULL != lru_gpu_elem->original ) {
                    dague_data_t* oldmaster = lru_gpu_elem->original;
                    /* Let's check we're not trying to steal one of our own data */
                    for( j = 0; NULL != this_task->data[j].data; j++ ) {
                        if( this_task->data[j].data->original == oldmaster ) {
                            temp_loc[j] = lru_gpu_elem;
                            goto find_another_data;
                        }
                    }

                    dague_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
                    DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                          "GPU[%d]:\tRepurpose copy %p to mirror block %p (in task %s:i) instead of %p\n",
                                          gpu_device->cuda_index, lru_gpu_elem, master, this_task->function->name, i, oldmaster));

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                    gpu_free( gpu_device->memory, (void*)(lru_gpu_elem->device_private) );
                    free(lru_gpu_elem);
                    goto malloc_data;
#endif
                }
            }
            gpu_elem = lru_gpu_elem;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        }
#endif
        assert( 0 == gpu_elem->readers );
        gpu_elem->coherency_state = DATA_COHERENCY_INVALID;
        gpu_elem->version = 0;
        dague_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        move_data_count--;
        temp_loc[i] = gpu_elem;
        dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    }
    if( 0 != move_data_count ) {
        WARNING(("GPU:\tRequest space on GPU failed for %d out of %d data\n",
                 move_data_count, this_task->function->nb_parameters));
        /* We can't find enough room on the GPU. Insert the tiles in the begining of
         * the LRU (in order to be reused asap) and return without scheduling the task.
         */
        for( i = 0; NULL != this_task->data[i].data; i++ ) {
            if( NULL == temp_loc[i] ) continue;
            dague_ulist_lifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)temp_loc[i]);
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
                             CUstream stream )
{
    dague_data_copy_t* in_elem = task_data->data;
    dague_data_t* original = in_elem->original;
    dague_gpu_data_copy_t* gpu_elem = original->device_copies[gpu_device->super.device_index];
    int transfer_required = 0;

    /* If the data will be accessed in write mode, remove it from any lists
     * until the task is completed.
     */
    if( ACCESS_WRITE & type ) {
        dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
        DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
    }

    transfer_required = dague_data_copy_ownership_to_device(original, gpu_device->super.device_index, (uint8_t)type);
    gpu_device->super.required_data_in += original->nb_elts;
    if( transfer_required ) {
        cudaError_t status;

        DAGUE_OUTPUT_VERBOSE((5, dague_cuda_output_stream,
                              "GPU:\tMove data %x (%p:%p) to GPU %d requested\n",
                              original->key, in_elem->device_private, (void*)gpu_elem->device_private, gpu_device->cuda_index));
        /* Push data into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( (CUdeviceptr)gpu_elem->device_private,
                                                 in_elem->device_private, original->nb_elts, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device ", status,
                                { WARNING(("<<%p>> -> <<%p>> [%d]\n", in_elem->device_private, gpu_elem->device_private, original->nb_elts));
                                    return -1; } );
        gpu_device->super.transferred_data_in += original->nb_elts;
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
        DAGUE_OUTPUT_VERBOSE((2, dague_cuda_output_stream,
                              "GPU: Reschedule %s(task %p) priority %d: no room available on the GPU for data\n",
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
        DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                              "GPU: Event for task %s(task %p) priority %d requested\n",
                              task->ec->function->name, (void*)task->ec, task->ec->priority ));
    }
    task = NULL;

 check_completion:
    if( (NULL == *out_task) && (NULL != exec_stream->tasks[exec_stream->end]) ) {
        rc = cuEventQuery(exec_stream->events[exec_stream->end]);
        if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            task = *out_task = exec_stream->tasks[exec_stream->end];
            DAGUE_OUTPUT_VERBOSE((3, dague_cuda_output_stream,
                                  "GPU: Event for task %s(task %p) encountered\n", task->ec->function->name, (void*)task ));
            exec_stream->tasks[exec_stream->end] = NULL;
            exec_stream->end = (exec_stream->end + 1) % exec_stream->max_events;
            DAGUE_TASK_PROF_TRACE_IF(exec_stream->prof_event_track_enable,
                                     gpu_device->super.profiling,
                                     (-1 == exec_stream->prof_event_key_end ?
                                      DAGUE_PROF_FUNC_KEY_END(task->ec->dague_handle,
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
    }
    return saved_rc;
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
    printf("Device %d:%d (%p)\n", gpu_device->cuda_index, gpu_device->super.device_index, gpu_device);
    printf("\tpeer mask %x executed tasks %llu max streams %d\n",
           gpu_device->peer_access_mask, gpu_device->super.executed_tasks, gpu_device->max_exec_streams);
    printf("\tstats transferred [in %llu out %llu] required [in %llu out %llu]\n",
           gpu_device->super.transferred_data_in, gpu_device->super.transferred_data_out,
           gpu_device->super.required_data_in, gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dump_exec_stream(&gpu_device->exec_stream[i]);
    }
    if( !dague_ulist_is_empty(&gpu_device->gpu_mem_lru) ) {
        printf("#\n# LRU list\n#\n");
        i = 0;
        DAGUE_LIST_ITERATOR(&gpu_device->gpu_mem_lru, item,
                            {
                                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                dague_dump_data_copy(gpu_copy);
                                i++;
                            });
    };
    if( !dague_ulist_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        printf("#\n# Owned LRU list\n#\n");
        i = 0;
        DAGUE_LIST_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                            {
                                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                dague_dump_data_copy(gpu_copy);
                                i++;
                            });
    };
    printf("\n\n");
}

#endif /* HAVE_CUDA */
