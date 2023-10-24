/*
 * Copyright (c) 2010-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/scheduling.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/class/fifo.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

/**
 * According to
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
 * and
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
 * we should limit the list of supported architectures to more recent setups.
 */
static int cuda_legal_compute_capabilities[] = {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 90};

/* look up how many FMA per cycle in single/double, per cuda MP
 * precision.
 * The following table provides updated values for future archs
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
 */
static int parsec_cuda_device_lookup_cudamp_floprate(const struct cudaDeviceProp* prop, int *drate, int *srate, int *trate, int *hrate)
{
    /* Some sane defaults for unknown architectures */
    *srate = 8;
    *drate = 1;
    *hrate = *trate = 0;  /* not supported */

#if !defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    int major = prop->major;
    int minor = prop->minor;
    if ((major == 3 && minor == 0) ||
        (major == 3 && minor == 2)) {
        *srate = 192;
        *drate = 8;
    } else if ((major == 3 && minor == 5) ||
               (major == 3 && minor == 7)) {
        *srate = 192;
        *drate = 64;
    } else if ((major == 5 && minor == 0) ||
               (major == 5 && minor == 2)) {
        *srate = 128;
        *drate = 4;
    } else if (major == 5 && minor == 3) {
        *hrate = 256;
        *srate = 128;
        *drate = 4;
    } else if (major == 6 && minor == 0) {
        *hrate = 128;
        *srate = 64;
        *drate = 32;
    } else if (major == 6 && minor == 1) {
        *hrate = 2;
        *srate = 128;
        *drate = 4;
    } else if (major == 6 && minor == 2) {
        *hrate = 256;
        *srate = 128;
        *drate = 4;
    } else if (major == 7 && minor == 0) {
        *hrate = 128;
        *trate = 512;
        *srate = 64;
        *drate = 32;
    } else if (major == 7 && minor == 5) {
        *hrate = 128;
        *trate = 512;
        *srate = 64;
        *drate = 2;
    } else if (major == 8 && minor == 0) {
        *hrate = 256;
        *trate = 512;
        *srate = 64;
        *drate = 32;
    } else if (major == 8 && minor == 6) {
        *hrate = 256;
        *trate = 512;
        *srate = 128;
        *drate = 2;
    } else if (major == 8 && minor == 9) {
        *hrate = 256;
        *trate = 512;
        *srate = 128;
        *drate = 2;
    } else if (major == 9 && minor == 0) {
        *hrate = 3712;
        *trate = 1856;
        *srate = 128;
        *drate = 64;
    } else { /* Unknown device */
        return PARSEC_ERR_NOT_IMPLEMENTED;
    }
#else
    /* AMD devices all report the same major/minor so we need to use the arch number
     * https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html
     *
     * https://docs.amd.com/en/latest/understand/gpu_arch contains the FMA/cycle for
     * recent architectures. This list will assume MFMA if available for the type.
     * divide by 2 the numbers because they already double count FMAs in that source.
     */
    const char *name = prop->gcnArchName;
    if(0 == strncasecmp("gfx90a", name, 6)) {
        *hrate = 512;
        *srate = 128;
        *drate = 128;
    } else if(0 == strncasecmp("gfx908", name, 6)) {
        *hrate = 512;
        *srate = 128;
        *drate = 32;
    } else if(0 == strncasecmp("gfx906", name, 6)) {
        *hrate = 128;
        *srate = 64;
        *drate = 32;
    } else { /* unknown device */
        return PARSEC_ERR_NOT_IMPLEMENTED;
    }
#endif
    return PARSEC_SUCCESS;
}

static int
parsec_cuda_memory_register(parsec_device_module_t* device, parsec_data_collection_t* desc,
                            void* ptr, size_t length)
{
    cudaError_t status;
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with CUDA. */
    if (desc->memory_registration_status == PARSEC_MEMORY_STATUS_REGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to register the memory
     * as another thread might be submiting tasks at the same time
     * (cuda_scheduling.h), and we do not set a device since we register it for
     * all devices.
     */
    status = cudaHostRegister(ptr, length, cudaHostRegisterPortable );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_register) cudaHostRegister ", status,
                            { goto restore_and_return; } );

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = PARSEC_MEMORY_STATUS_REGISTERED;

  restore_and_return:
    (void)device;
    return rc;
}

static int parsec_cuda_memory_unregister(parsec_device_module_t* device, parsec_data_collection_t* desc, void* ptr)
{
    cudaError_t status;
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with CUDA. One registration = one deregistration */
    if (desc->memory_registration_status == PARSEC_MEMORY_STATUS_UNREGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to unregister the memory
     * as another thread might be submitting tasks at the same time (cuda_scheduling.h)
     */
    status = cudaHostUnregister(ptr);
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_unregister) cudaHostUnregister ", status,
                            {continue;} );

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = PARSEC_MEMORY_STATUS_UNREGISTERED;

    (void)device;
    return rc;
}

static void* parsec_cuda_find_incarnation(parsec_device_gpu_module_t* gpu_device,
                                          const char* fname)
{
    char library_name[FILENAME_MAX], function_name[FILENAME_MAX], *env;
    parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;
    int index, capability = cuda_device->major * 10 + cuda_device->minor;
    int rc;
    void *fn = NULL;
    char** argv = NULL;

    rc = gpu_device->set_device(gpu_device);
    if( PARSEC_SUCCESS != rc )
        return NULL;

    for( index = 0; index < (int)(sizeof(cuda_legal_compute_capabilities)/sizeof(int)); index++ ) {
        if(cuda_legal_compute_capabilities[index] >= capability) {
            break;
        }
    }
    /* Current position is either equal or larger than the current capability,
     * so we need to move one position back for the next iteration, such that
     * the first test is with the device capability and then we fall back on
     * the list of supported architectures.
     */
    index--;

    /**
     * Prepare the list of PATH or FILE to be searched for a CUDA shared library.
     * In any case this list might be a list of ; separated possible targets,
     * where each target can be either a directory or a specific file.
     */
    env = getenv("PARSEC_CUCORES_LIB");
    if( NULL != env ) {
        argv = parsec_argv_split(env, ';');
    } else if( NULL != parsec_cuda_lib_path ) {
        argv = parsec_argv_split(parsec_cuda_lib_path, ';');
    }

    snprintf(function_name, FILENAME_MAX, "%s_sm%2d", fname, capability);
  retry_lesser_sm_version:
    /* Try this by default, if not present its fine. CUCORES_LIB above can
     * contain a path to the fully named library if this default does not make
     * sense. */
    snprintf(library_name, FILENAME_MAX, "%s.so", fname);

    fn = parsec_device_find_function(function_name, library_name, (const char**)argv);
    if( NULL == fn ) {  /* look for the function with lesser capabilities */
        parsec_debug_verbose(10, parsec_gpu_output_stream,
                            "No function %s found for CUDA device %s",
                            function_name, gpu_device->super.name);
        if( -1 <= index ) {  /* we bail out at -2 */
            if( -1 == index ) {
                snprintf(function_name, FILENAME_MAX, "%s", fname);
            } else {
               capability = cuda_legal_compute_capabilities[index];
               snprintf(function_name, FILENAME_MAX, "%s_sm%2d", fname, capability);
            }
            index--;
            goto retry_lesser_sm_version;
        }
    }

    if( NULL != argv )
        parsec_argv_free(argv);

    return fn;
}

static int parsec_cuda_set_device(parsec_device_gpu_module_t *gpu)
{
    cudaError_t cudaStatus;
    parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t *)gpu;

    cudaStatus = cudaSetDevice(cuda_device->cuda_index);
    PARSEC_CUDA_CHECK_ERROR( "cudaSetDevice ", cudaStatus, {return PARSEC_ERROR;} );
    return PARSEC_SUCCESS;
}

static int parsec_cuda_memcpy_async(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream,
                                        void *dest, void *source, size_t bytes, parsec_device_transfer_direction_t direction)
{
    cudaError_t cudaStatus;
    enum cudaMemcpyKind kind;
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;

    (void)gpu;

    switch(direction) {
    case parsec_device_gpu_transfer_direction_d2d:
        kind = cudaMemcpyDeviceToDevice;
        break;
    case parsec_device_gpu_transfer_direction_d2h:
        kind = cudaMemcpyDeviceToHost;
        break;
    case parsec_device_gpu_transfer_direction_h2d:
        kind = cudaMemcpyHostToDevice;
        break;
    default:
        PARSEC_CUDA_CHECK_ERROR( "Translate parsec_device_transfer_direction_t to cudaMemcpyKind ", cudaErrorInvalidValue, {return PARSEC_ERROR;} );
    }

    cudaStatus =  cudaMemcpyAsync( dest, source, bytes, kind, cuda_stream->cuda_stream );
    PARSEC_CUDA_CHECK_ERROR( "cudaMemcpyAsync ", cudaStatus, {return PARSEC_ERROR;} );
    return PARSEC_SUCCESS;
}

static int parsec_cuda_event_record(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
    cudaError_t cudaStatus;
    (void)gpu;

    cudaStatus = cudaEventRecord(cuda_stream->events[event_idx], cuda_stream->cuda_stream);
    PARSEC_CUDA_CHECK_ERROR( "cudaEventRecord ", cudaStatus, {return PARSEC_ERROR;} );
    return PARSEC_SUCCESS;
}

static int parsec_cuda_event_query(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
    cudaError_t cudaStatus;
    (void)gpu;

    cudaStatus = cudaEventQuery(cuda_stream->events[event_idx]);
    if(cudaSuccess == cudaStatus) {
        return 1;
    }
    if(cudaErrorNotReady == cudaStatus) {
        return 0;
    }
    PARSEC_CUDA_CHECK_ERROR( "cudaEventQuery ", cudaStatus, {return PARSEC_ERROR;} );
    return PARSEC_ERROR; /* should be unreachable */
}

static int parsec_cuda_memory_info(struct parsec_device_gpu_module_s *gpu, size_t *free_mem, size_t *total_mem)
{
    cudaError_t cudaStatus;
    (void)gpu;
    cudaStatus = cudaMemGetInfo(free_mem, total_mem);
    PARSEC_CUDA_CHECK_ERROR( "cudaMemGetInfo", cudaStatus, {return PARSEC_ERROR;});
    return PARSEC_SUCCESS;
}

static int parsec_cuda_memory_allocate(struct parsec_device_gpu_module_s *gpu, size_t bytes, void **addr)
{
    cudaError_t cudaStatus;
    (void)gpu;
    cudaStatus = cudaMalloc(addr, bytes);
    PARSEC_CUDA_CHECK_ERROR( "cudaMalloc", cudaStatus, {return PARSEC_ERROR;});
    return PARSEC_SUCCESS;
}

static int parsec_cuda_memory_free(struct parsec_device_gpu_module_s *gpu, void *addr)
{
    cudaError_t cudaStatus;
    (void)gpu;
    cudaStatus = cudaFree(addr);
    PARSEC_CUDA_CHECK_ERROR( "cudaFree", cudaStatus, {return PARSEC_ERROR;});
    return PARSEC_SUCCESS;
}

int
parsec_cuda_module_init( int dev_id, parsec_device_module_t** module )
{
    int major, minor, concurrency, computemode, streaming_multiprocessor, drate, srate, trate, hrate, len;
    parsec_device_cuda_module_t* cuda_device;
    parsec_device_gpu_module_t* gpu_device;
    parsec_device_module_t* device;
    cudaError_t cudastatus;
    int show_caps_index, show_caps = 0, j, k;
    char *szName;
    uint64_t freqHz;
    double fp16, fp32, fp64, tf32;
    struct cudaDeviceProp prop;

    show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities"); 
    if(0 < show_caps_index) {
        parsec_mca_param_lookup_int(show_caps_index, &show_caps);
    }

    *module = NULL;
    cudastatus = cudaSetDevice( dev_id );
    PARSEC_CUDA_CHECK_ERROR( "cudaSetDevice ", cudastatus, {return PARSEC_ERROR;} );
    cudastatus = cudaGetDeviceProperties( &prop, dev_id );
    PARSEC_CUDA_CHECK_ERROR( "cudaGetDeviceProperties ", cudastatus, {return PARSEC_ERROR;} );

    szName    = prop.name;
    major     = prop.major;
    minor     = prop.minor;
    freqHz    = prop.clockRate * 1000;  /* clockRate is in KHz */
    concurrency = prop.concurrentKernels;
    streaming_multiprocessor = prop.multiProcessorCount;
    computemode = prop.computeMode;

    // We use calloc because we need some fields to be zero-initialized to ensure graceful handling of errors
    cuda_device = (parsec_device_cuda_module_t*)calloc(1, sizeof(parsec_device_cuda_module_t));
    gpu_device = &cuda_device->super;
    device = &gpu_device->super;
    PARSEC_OBJ_CONSTRUCT(cuda_device, parsec_device_cuda_module_t);
    cuda_device->cuda_index = (uint8_t)dev_id;
    cuda_device->major      = (uint8_t)major;
    cuda_device->minor      = (uint8_t)minor;
    len = asprintf(&gpu_device->super.name, "cuda(%d)", dev_id);
    if(-1 == len) { gpu_device->super.name = NULL; goto release_device; }
    gpu_device->data_avail_epoch = 0;

    gpu_device->max_exec_streams = parsec_cuda_max_streams;
    gpu_device->exec_stream =
        (parsec_gpu_exec_stream_t**)malloc(gpu_device->max_exec_streams * sizeof(parsec_gpu_exec_stream_t*));
    // To reduce the number of separate malloc, we allocate all the streams in a single block, stored in exec_stream[0]
    // Because the gpu_device structure does not know the size of cuda_stream or other GPU streams, it needs to keep
    // separate pointers for the beginning of each exec_stream
    // We use calloc because we need some fields to be zero-initialized to ensure graceful handling of errors
    gpu_device->exec_stream[0] = (parsec_gpu_exec_stream_t*)calloc(gpu_device->max_exec_streams,
                                                                   sizeof(parsec_cuda_exec_stream_t));
    for( j = 1; j < gpu_device->max_exec_streams; j++ ) {
        gpu_device->exec_stream[j] = (parsec_gpu_exec_stream_t*)(
                (parsec_cuda_exec_stream_t*)gpu_device->exec_stream[0] + j);
    }
    for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
        parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)gpu_device->exec_stream[j];
        parsec_gpu_exec_stream_t* exec_stream = &cuda_stream->super;

        /* We will have to release up to this stream in case of error */
        gpu_device->num_exec_streams++;

        /* Allocate the stream */
        cudastatus = cudaStreamCreate( &(cuda_stream->cuda_stream) );
        PARSEC_CUDA_CHECK_ERROR( "cudaStreamCreate ", cudastatus,
                                 {goto release_device;} );
        exec_stream->workspace    = NULL;
        PARSEC_OBJ_CONSTRUCT(&exec_stream->infos, parsec_info_object_array_t);
        parsec_info_object_array_init(&exec_stream->infos, &parsec_per_stream_infos, exec_stream);
        exec_stream->max_events   = PARSEC_MAX_EVENTS_PER_STREAM;
        exec_stream->executed     = 0;
        exec_stream->start        = 0;
        exec_stream->end          = 0;
        exec_stream->name         = NULL;
        exec_stream->fifo_pending = (parsec_list_t*)PARSEC_OBJ_NEW(parsec_list_t);
        PARSEC_OBJ_CONSTRUCT(exec_stream->fifo_pending, parsec_list_t);
        exec_stream->tasks    = (parsec_gpu_task_t**)malloc(exec_stream->max_events
                                                            * sizeof(parsec_gpu_task_t*));
        cuda_stream->events   = (cudaEvent_t*)malloc(exec_stream->max_events * sizeof(cudaEvent_t));
        /* and the corresponding events */
        for( k = 0; k < exec_stream->max_events; k++ ) {
            cuda_stream->events[k]   = NULL;
            exec_stream->tasks[k]    = NULL;
            cudastatus = cudaEventCreateWithFlags(&(cuda_stream->events[k]), cudaEventDisableTiming);
            PARSEC_CUDA_CHECK_ERROR( "(INIT) cudaEventCreateWithFlags ", (cudaError_t)cudastatus,
                                     {goto release_device;} );
        }
        if(j == 0) {
            len = asprintf(&exec_stream->name, "h2d_cuda(%d)", j);
        } else if(j == 1) {
            len = asprintf(&exec_stream->name, "d2h_cuda(%d)", j);
        } else {
            len = asprintf(&exec_stream->name, "cuda(%d)", j);
        }
        if(-1 == len) { exec_stream->name = NULL; goto release_device; }
#if defined(PARSEC_PROF_TRACE)
        /* Each 'exec' stream gets its own profiling stream, except IN and OUT stream that share it.
         * It's good to separate the exec streams to know what was submitted to what stream
         * We don't have this issue for the IN and OUT streams because types of event discriminate
         * what happens where, and separating them consumes memory and increases the number of 
         * events that needs to be matched between streams because we cannot differentiate some
         * ends between IN or OUT, so they are all logged on the same stream. */
        if(j == 0 || (parsec_device_gpu_one_profiling_stream_per_gpu_stream == 1 && j != 1))
            exec_stream->profiling = parsec_profiling_stream_init( 2*1024*1024, PARSEC_PROFILE_STREAM_STR, dev_id, j );
        else
            exec_stream->profiling = gpu_device->exec_stream[0]->profiling;
        if(j == 0) {
            exec_stream->prof_event_track_enable = parsec_gpu_trackable_events & ( PARSEC_PROFILE_GPU_TRACK_DATA_IN | PARSEC_PROFILE_GPU_TRACK_MEM_USE );
        } else if(j == 1) {
            exec_stream->prof_event_track_enable = parsec_gpu_trackable_events & ( PARSEC_PROFILE_GPU_TRACK_DATA_OUT | PARSEC_PROFILE_GPU_TRACK_MEM_USE );
        } else {
            exec_stream->prof_event_track_enable = parsec_gpu_trackable_events & ( PARSEC_PROFILE_GPU_TRACK_EXEC | PARSEC_PROFILE_GPU_TRACK_MEM_USE );
        }
#endif  /* defined(PARSEC_PROF_TRACE) */
    }

    device->type                 = PARSEC_DEV_CUDA;
    device->executed_tasks       = 0;
    device->data_in_array_size   = 0;     // We'll let the modules_attach allocate the array of the right size for us
    device->data_in_from_device  = NULL;
    device->data_out_to_host     = 0;
    device->required_data_in     = 0;
    device->required_data_out    = 0;

    device->attach              = parsec_device_attach;
    device->detach              = parsec_device_detach;
    device->taskpool_register   = parsec_device_taskpool_register;
    device->taskpool_unregister = parsec_device_taskpool_unregister;
    device->data_advise         = parsec_device_data_advise;
    device->memory_release      = parsec_device_flush_lru;

    if (parsec_cuda_device_lookup_cudamp_floprate(&prop, &drate, &srate, &trate, &hrate) == PARSEC_ERR_NOT_IMPLEMENTED ) {
        parsec_debug_verbose(0, parsec_gpu_output_stream, "Unknown device %s (%s) [capabilities %d.%d]: Gflops rate is a random guess and load balancing (performance) may be reduced.",
                        szName, gpu_device->super.name, major, minor );
        device->gflops_guess = true;
    }
    /* We compute gflops based on FMA rate */
    device->gflops_fp16 = fp16 = 2.f * hrate * streaming_multiprocessor * freqHz * 1e-9f;
    device->gflops_tf32 = tf32 = 2.f * trate * streaming_multiprocessor * freqHz * 1e-9f;
    device->gflops_fp32 = fp32 = 2.f * srate * streaming_multiprocessor * freqHz * 1e-9f;
    device->gflops_fp64 = fp64 = 2.f * drate * streaming_multiprocessor * freqHz * 1e-9f;
    /* don't assert fp16, tf32, maybe they actually do not exist on the architecture */
    assert(device->gflops_fp32 > 0);
    assert(device->gflops_fp64 > 0);
    device->device_load = 0;

    /* Initialize internal lists */
    PARSEC_OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&gpu_device->pending,           parsec_fifo_t);

    gpu_device->sort_starting_p = NULL;
    gpu_device->peer_access_mask = 0;  /* No GPU to GPU direct transfer by default */

    device->memory_register          = parsec_cuda_memory_register;
    device->memory_unregister        = parsec_cuda_memory_unregister;
    gpu_device->set_device       = parsec_cuda_set_device;
    gpu_device->memcpy_async     = parsec_cuda_memcpy_async;
    gpu_device->event_record     = parsec_cuda_event_record;
    gpu_device->event_query      = parsec_cuda_event_query;
    gpu_device->memory_info      = parsec_cuda_memory_info;
    gpu_device->memory_allocate  = parsec_cuda_memory_allocate;
    gpu_device->memory_free      = parsec_cuda_memory_free;
    gpu_device->find_incarnation = parsec_cuda_find_incarnation;

    if( PARSEC_SUCCESS != parsec_device_memory_reserve(gpu_device,
                                                           parsec_cuda_memory_percentage,
                                                           parsec_cuda_memory_number_of_blocks,
                                                           parsec_cuda_memory_block_size) ) {
        goto release_device;
    }

    if( show_caps ) {
        parsec_inform("GPU Device %-8s: %s [capability %d.%d] %s\n"
                      "\tLocation (PCI Bus/Device/Domain): %x:%x.%x\n"
                      "\tSM                 : %d\n"
                      "\tFrequency (GHz)    : %f\n"
                      "\tpeak Tflop/s       : %4.2f fp64,\t%4.2f fp32,\t%4.2f tf32,\t%4.2f fp16\n"
                      "\tPeak Mem Bw (GB/s) : %.2f [Clock Rate (Ghz) %.2f | Bus Width (bits) %d]\n"
                      "\tconcurrency        : %s\n"
                      "\tcomputeMode        : %d\n",
                      device->name, szName, cuda_device->major, cuda_device->minor,
                      device->gflops_guess? "(GUESSED Peak Tflop/s; load imbalance may RECUDE PERFORMANCE)": "",
                      prop.pciBusID, prop.pciDeviceID, prop.pciDomainID,
                      streaming_multiprocessor,
                      freqHz*1e-9f,
                      fp64, fp32, tf32, fp16,
                      2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6, prop.memoryClockRate*1e-6, prop.memoryBusWidth,
                      (concurrency == 1)? "yes": "no",
                      computemode);
    }

    *module = device;
    return PARSEC_SUCCESS;

 release_device:
    if( NULL != gpu_device->exec_stream) {
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_device->exec_stream[j];
            if(NULL == cuda_stream) continue;
            parsec_gpu_exec_stream_t* exec_stream = &cuda_stream->super;

            if( NULL != exec_stream->fifo_pending ) {
                PARSEC_OBJ_RELEASE(exec_stream->fifo_pending);
            }
            if( NULL != exec_stream->tasks ) {
                free(exec_stream->tasks); exec_stream->tasks = NULL;
            }
            if( NULL != cuda_stream->events ) {
                for( k = 0; k < exec_stream->max_events; k++ ) {
                    if( NULL != cuda_stream->events[k] ) {
                        (void)cudaEventDestroy(cuda_stream->events[k]);
                    }
                }
                free(cuda_stream->events); cuda_stream->events = NULL;
            }
            if( NULL != exec_stream->name ) {
                free(exec_stream->name); exec_stream->name = NULL;
            }
#if defined(PARSEC_PROF_TRACE)
            if( NULL != exec_stream->profiling ) {
                /* No function to clean the profiling stream. If one is introduced
                 * some day, remember that exec streams 0 and 1 always share the same 
                 * ->profiling stream, and that all of them share the same
                 * ->profiling stream if parsec_device_cuda_one_profiling_stream_per_cuda_stream == 0 */
            }
#endif  /* defined(PARSEC_PROF_TRACE) */
        }
        // All exec streams are stored in a single malloc block at exec_stream[0]
        free(gpu_device->exec_stream[0]);
        free(gpu_device->exec_stream);
        gpu_device->exec_stream = NULL;
    }
    free(gpu_device);
    return PARSEC_ERROR;
}

int
parsec_cuda_module_fini(parsec_device_module_t* device)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)device;
    parsec_device_cuda_module_t* cuda_device = (parsec_device_cuda_module_t*)device;
    cudaError_t status;
    int j, k;

    status = cudaSetDevice( cuda_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_device_fini) cudaSetDevice ", status,
                            {continue;} );

    /* Release the registered memory */
    parsec_device_memory_release(gpu_device);

    /* Release pending queue */
    PARSEC_OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
    for( j = 0; j < gpu_device->num_exec_streams; j++ ) {
        parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)gpu_device->exec_stream[j];
        parsec_gpu_exec_stream_t* exec_stream = &cuda_stream->super;

        exec_stream->executed = 0;
        exec_stream->start    = 0;
        exec_stream->end      = 0;

        for( k = 0; k < exec_stream->max_events; k++ ) {
            assert( NULL == exec_stream->tasks[k] );
            status = cudaEventDestroy(cuda_stream->events[k]);
            PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_device_fini) cudaEventDestroy ", status,
                                    {continue;} );
        }
        exec_stream->max_events = 0;
        free(cuda_stream->events); cuda_stream->events = NULL;
        free(exec_stream->tasks); exec_stream->tasks = NULL;
        free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;
        /* Release the stream */
        cudaStreamDestroy( cuda_stream->cuda_stream );
        free(exec_stream->name);

        /* Release Info object array */
        PARSEC_OBJ_DESTRUCT(&exec_stream->infos);
    }
    // All exec streams are stored in a single malloc block at exec_stream[0]
    free(gpu_device->exec_stream[0]);
    free(gpu_device->exec_stream);
    gpu_device->exec_stream = NULL;

    cuda_device->cuda_index = -1;

    /* Cleanup the GPU memory. */
    PARSEC_OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    PARSEC_OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    return PARSEC_SUCCESS;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */
