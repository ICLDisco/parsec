/*
 * Copyright (c) 2023-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

#if defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT)
#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/mca/device/level_zero/device_level_zero_internal.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/scheduling.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/class/fifo.h"
#include "parsec/mca/device/level_zero/device_level_zero_dpcpp.h"

#include <level_zero/ze_api.h>

static void* parsec_level_zero_find_incarnation(parsec_device_gpu_module_t* gpu_device,
                                                const char* fname)
{
    char library_name[FILENAME_MAX], *env;
    void *fn = NULL;
    char** argv = NULL;

    /**
     * Prepare the list of PATH or FILE to be searched for a LEVEL_ZERO shared library.
     * In any case this list might be a list of ; separated possible targets,
     * where each target can be either a directory or a specific file.
     */
    env = getenv("PARSEC_LZCORES_LIB");
    if( NULL != env ) {
        argv = parsec_argv_split(env, ';');
    } else if( NULL != parsec_level_zero_lib_path ) {
        argv = parsec_argv_split(parsec_level_zero_lib_path, ';');
    }

    fn = parsec_device_find_function(fname, library_name, (const char**)argv);
    if( NULL == fn ) {  /* look for the function with lesser capabilities */
        parsec_warning("No function '%s' found for LEVEL_ZERO device %s", fname, gpu_device->super.name);
    }

    if( NULL != argv )
        parsec_argv_free(argv);

    return fn;
}

static int parsec_level_zero_set_device(parsec_device_gpu_module_t *gpu)
{
    /* level_zero does not have a default device, all calls pass explicitly which device */
    (void)gpu;
    return PARSEC_SUCCESS;
}

static int parsec_level_zero_memcpy_async(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream,
                                          void *dest, void *source, size_t bytes, parsec_device_transfer_direction_t direction)
{
    ze_result_t ret;
    parsec_level_zero_exec_stream_t *level_zero_stream = (parsec_level_zero_exec_stream_t *)gpu_stream;

    (void)gpu;
    (void)direction; /* level_zero does not need to specify if source or destination is host or device */

    ret = (ze_result_t)zeCommandListAppendMemoryCopy(level_zero_stream->command_lists[gpu_stream->start],
                                                     dest,
                                                     source,
                                                     bytes,
                                                     NULL,
                                                     0, NULL);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandListAppendMemoryCopy ", ret, { return PARSEC_ERROR; } );

    return PARSEC_SUCCESS;
}

static int parsec_level_zero_event_record(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx)
{
    ze_result_t ze_rc;
    parsec_level_zero_exec_stream_t *level_zero_stream = (parsec_level_zero_exec_stream_t *)gpu_stream;

    (void)gpu;

    ze_rc = zeCommandListClose(level_zero_stream->command_lists[event_idx]);
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandListClose ", ze_rc, { return PARSEC_ERROR; } );
    ze_rc = zeCommandQueueExecuteCommandLists(level_zero_stream->level_zero_cq, 1,
                                              &level_zero_stream->command_lists[event_idx],
                                              level_zero_stream->fences[event_idx]);
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandQueueExecuteCommandLists ", ze_rc, { return PARSEC_ERROR; } );
    return PARSEC_SUCCESS;
}

static int parsec_level_zero_event_query(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx)
{
    ze_result_t ze_rc;
    parsec_level_zero_exec_stream_t *level_zero_stream = (parsec_level_zero_exec_stream_t *)gpu_stream;

    (void)gpu;
    ze_rc = zeFenceQueryStatus(level_zero_stream->fences[event_idx]);
    if(ZE_RESULT_SUCCESS == ze_rc) {
        ze_rc = zeCommandListReset(level_zero_stream->command_lists[event_idx]);
        PARSEC_LEVEL_ZERO_CHECK_ERROR("zeCommandListReset", ze_rc, { return PARSEC_ERROR; } );
        return 1;
    }
    if(ZE_RESULT_NOT_READY == ze_rc) {
        return 0;
    }
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeFenceQueryStatus ", ze_rc, { return PARSEC_ERROR; } );
    return PARSEC_ERROR; /* should be unreachable */
}

static int parsec_level_zero_memory_info(struct parsec_device_gpu_module_s *gpu, size_t *free_mem, size_t *total_mem)
{
    ze_result_t status;
    parsec_device_level_zero_module_t *level_zero_device = (parsec_device_level_zero_module_t*)gpu;
    ze_device_properties_t devProperties;
    ze_device_memory_properties_t *devMemProperties;
    ze_device_memory_access_properties_t memAccessProperties;
    uint32_t count = 0;
    int memIndex = -1;

    status = zeDeviceGetMemoryAccessProperties(level_zero_device->ze_device, &memAccessProperties);
    PARSEC_LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryAccessProperties ", status, {
        return PARSEC_ERROR;
    });
    if( 0 == (ZE_MEMORY_ACCESS_CAP_FLAG_RW & memAccessProperties.deviceAllocCapabilities) ) {
        parsec_warning("%s:%d -- Device %s does not have memory allocation capabilities with RW access\n",
                       __FILE__, __LINE__, gpu->super.name);
        return PARSEC_ERROR;
    }
    status = zeDeviceGetProperties(level_zero_device->ze_device, &devProperties);
    PARSEC_LEVEL_ZERO_CHECK_ERROR("zeDeviceGetProperties ", status, {
       return PARSEC_ERROR;
    });
    status = zeDeviceGetMemoryProperties(level_zero_device->ze_device, &count, NULL);
    PARSEC_LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryProperties ", status, {
        return PARSEC_ERROR;
    });
    devMemProperties = (ze_device_memory_properties_t*)malloc(count * sizeof(ze_device_memory_properties_t));
    status = zeDeviceGetMemoryProperties(level_zero_device->ze_device, &count, devMemProperties);
    PARSEC_LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryProperties ", status, {
        free(devMemProperties);
        return PARSEC_ERROR;
    });
    for(int i = 0; i < (int)count; i++) {
        // TODO: better approach would be to keep a list of pointers?
        //   for now we just take the memory that has the highest amount of memory available
        if( memIndex == -1 || devMemProperties[memIndex].totalSize < devMemProperties[i].totalSize)
            memIndex = i;
    }

    level_zero_device->memory_index = memIndex;
    *free_mem = devProperties.maxMemAllocSize < devMemProperties[memIndex].totalSize ?
            devProperties.maxMemAllocSize : devMemProperties[memIndex].totalSize;
    *total_mem = *free_mem; /* It is unclear if level_zero shows the amount of physical memory available, as it manages virtual memory */
    free(devMemProperties); devMemProperties = NULL;

    return PARSEC_SUCCESS;
}

static int parsec_level_zero_memory_allocate(struct parsec_device_gpu_module_s *gpu, size_t bytes, void **addr)
{
    ze_result_t status;
    size_t  alignment = 1 << 3;
    parsec_device_level_zero_module_t *level_zero_device = (parsec_device_level_zero_module_t*)gpu;
    ze_device_mem_alloc_desc_t memAllocDesc = {
            .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            .pNext = NULL,
            .flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED,
            .ordinal = level_zero_device->memory_index
    };
    if(-1 == level_zero_device->memory_index) {
        /* In usual cases, memory_info has been called before memory_allocate, but we do it to set the memory_index if it has not been done */
        size_t initial_mem;
        size_t total_mem;
        parsec_level_zero_memory_info(gpu, &initial_mem, &total_mem);
        memAllocDesc.ordinal = level_zero_device->memory_index;
    }
    assert(-1 != level_zero_device->memory_index);
    status = zeMemAllocDevice(level_zero_device->driver->ze_context, &memAllocDesc, bytes, alignment,
                              level_zero_device->ze_device, addr);
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeMemAllocDevice ", status, { return PARSEC_ERROR; } );
    return PARSEC_SUCCESS;
}

static int parsec_level_zero_memory_free(struct parsec_device_gpu_module_s *gpu, void *addr)
{
    ze_result_t status;
    parsec_device_level_zero_module_t *level_zero_device = (parsec_device_level_zero_module_t*)gpu;
    status = zeMemFree(level_zero_device->driver->ze_context, addr);
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeMemFree ", status, { return PARSEC_ERROR; } );
    return PARSEC_SUCCESS;
}

int parsec_level_zero_module_init( int dev_id, parsec_device_level_zero_driver_t *driver, ze_device_handle_t ze_device,
                                   ze_device_properties_t *prop, parsec_device_module_t** module )
{
    int sm, len;
    parsec_device_level_zero_module_t* level_zero_device;
    parsec_device_gpu_module_t* gpu_device;
    parsec_device_module_t* device;
    ze_result_t ze_rc;
    int show_caps_index, show_caps = 0, j, k;
    char *szName;
    float freqGHz;

    show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        parsec_mca_param_lookup_int(show_caps_index, &show_caps);
    }

    *module = NULL;

    szName    = prop->name;
    freqGHz   = prop->coreClockRate/1e3f;
    sm        = prop->numThreadsPerEU;

    level_zero_device = (parsec_device_level_zero_module_t*)calloc(1, sizeof(parsec_device_level_zero_module_t));
    gpu_device = &level_zero_device->super;
    device = &gpu_device->super;
    PARSEC_OBJ_CONSTRUCT(level_zero_device, parsec_device_level_zero_module_t);
    level_zero_device->level_zero_index = (uint8_t)dev_id;
    level_zero_device->driver = driver;
    level_zero_device->ze_device = ze_device;
    level_zero_device->memory_index = -1; /* Will be initialized during call to mem_info */

    len = asprintf(&gpu_device->super.name, "ze(%d)", dev_id);
    if(-1 == len) {
        gpu_device->super.name = NULL;
        goto release_device;
    }
    gpu_device->data_avail_epoch = 0;

    gpu_device->max_exec_streams = parsec_level_zero_max_streams;
    gpu_device->exec_stream =
        (parsec_gpu_exec_stream_t**)malloc(gpu_device->max_exec_streams * sizeof(parsec_gpu_exec_stream_t*));
    // To reduce the number of separate malloc, we allocate all the streams in a single block, stored in exec_stream[0]
    // Because the gpu_device structure does not know the size of level_zero_stream or other GPU streams, it needs to keep
    // separate pointers for the beginning of each exec_stream
    // We use calloc because we need some fields to be zero-initialized to ensure graceful handling of errors
    gpu_device->exec_stream[0] =
            (parsec_gpu_exec_stream_t*)malloc( gpu_device->max_exec_streams * sizeof(parsec_level_zero_exec_stream_t));
    for(j = 1; j < gpu_device->max_exec_streams; j++)
        gpu_device->exec_stream[j] = (parsec_gpu_exec_stream_t*)(
                (parsec_level_zero_exec_stream_t*)gpu_device->exec_stream[0]+j);

    // Discover all command queue groups
    uint32_t cmdqueueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(level_zero_device->ze_device, &cmdqueueGroupCount, NULL);

    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)
            malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
    zeDeviceGetCommandQueueGroupProperties(level_zero_device->ze_device, &cmdqueueGroupCount, cmdqueueGroupProperties);
    // Find a command queue type that support compute
    uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
    uint32_t copyQueueGroupOrdinal = cmdqueueGroupCount;
    for( uint32_t i = 0;
         i < cmdqueueGroupCount &&
                 (computeQueueGroupOrdinal == cmdqueueGroupCount ||
                  copyQueueGroupOrdinal == cmdqueueGroupCount);
         ++i ) {
        if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
            computeQueueGroupOrdinal = i;
        }
        if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY ) {
            copyQueueGroupOrdinal = i;
        }
    }
    //TODO: it might be more in line with the design to create different command queues for copy
    //      and compute than using the existing queues.
    if( computeQueueGroupOrdinal == cmdqueueGroupCount ) {
        parsec_warning( "level zero device: unable to find a Queue Group with COMPUTE flag");
        goto release_device;
    }
    if( copyQueueGroupOrdinal == cmdqueueGroupCount ) {
        parsec_warning( "level zero device: unable to find a Queue Group with COMPUTE flag");
        goto release_device;
    }

    for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
        parsec_level_zero_exec_stream_t* level_zero_stream =
                (parsec_level_zero_exec_stream_t*)gpu_device->exec_stream[j];
        parsec_gpu_exec_stream_t* exec_stream = &level_zero_stream->super;
        ze_command_queue_desc_t commandQueueDesc = {
                ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                NULL,
                (uint32_t)-1,
                0, // index
                0, // flags
                ZE_COMMAND_QUEUE_MODE_DEFAULT,
                ZE_COMMAND_QUEUE_PRIORITY_NORMAL
        };
        /* CommandListImmediate would seem better for I/O, but mixing CLImmediate and CL+CQ
         * seems to create some synchronization issues. */
        commandQueueDesc.ordinal = computeQueueGroupOrdinal;
        ze_rc = zeCommandQueueCreate(level_zero_device->driver->ze_context, level_zero_device->ze_device,
                                     &commandQueueDesc, &level_zero_stream->level_zero_cq);
        PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandQueueCreate ", ze_rc, {goto release_device;} );
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
        level_zero_stream->fences   = (ze_fence_handle_t*)malloc(exec_stream->max_events * sizeof(ze_fence_handle_t));
        level_zero_stream->command_lists = (ze_command_list_handle_t*)malloc(exec_stream->max_events*sizeof(ze_command_list_handle_t));

        /* create the fences and command lists */
        for( k = 0; k < exec_stream->max_events; k++ ) {
            ze_command_list_desc_t commandListDesc = {
                    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                    NULL,
                    computeQueueGroupOrdinal,
                    0 // flags
            };
            ze_fence_desc_t fence_desc = {
                .stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
                .pNext = NULL,
                .flags = 0
            };
            level_zero_stream->fences[k]   = NULL;
            exec_stream->tasks[k]    = NULL;
            ze_rc = zeCommandListCreate(level_zero_device->driver->ze_context, level_zero_device->ze_device,
                                        &commandListDesc, &level_zero_stream->command_lists[k]);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandListCreate ", ze_rc, {goto release_device;} );
            ze_rc = zeFenceCreate(level_zero_stream->level_zero_cq, &fence_desc, &(level_zero_stream->fences[k]));
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeFenceCreate ", ze_rc, {goto release_device;} );
        }
        if(j == 0) {
            len = asprintf(&exec_stream->name, "h2d(%d)", j);
            if(-1 == len)
                exec_stream->name = "h2d";
        } else if(j == 1) {
            len = asprintf(&exec_stream->name, "d2h(%d)", j);
            if(-1 == len)
                exec_stream->name = "d2h";
        } else {
            len = asprintf(&exec_stream->name, "level_zero(%d)", j);
            if(-1 == len)
                exec_stream->name = "level_zero";
        }
#if defined(PARSEC_PROF_TRACE)
        gpu_device->trackable_events = PARSEC_PROFILE_GPU_TRACK_EXEC | PARSEC_PROFILE_GPU_TRACK_DATA_OUT
                                    | PARSEC_PROFILE_GPU_TRACK_DATA_IN | PARSEC_PROFILE_GPU_TRACK_OWN | PARSEC_PROFILE_GPU_TRACK_MEM_USE
                                    | PARSEC_PROFILE_GPU_TRACK_PREFETCH;
        /* Each 'exec' stream gets its own profiling stream, except IN and OUT stream that share it.
         * It's good to separate the exec streams to know what was submitted to what stream
         * We don't have this issue for the IN and OUT streams because types of event discriminate
         * what happens where, and separating them consumes memory and increases the number of
         * events that needs to be matched between streams because we cannot differentiate some
         * ends between IN or OUT, so they are all logged on the same stream. */
        if(j == 0 || (parsec_device_level_zero_one_profiling_stream_per_gpu_stream == 1 && j != 1))
            exec_stream->profiling = parsec_profiling_stream_init( 2*1024*1024, PARSEC_PROFILE_STREAM_STR, dev_id, j );
        else
            exec_stream->profiling = gpu_device->exec_stream[0]->profiling;
        if(j == 0) {
            exec_stream->prof_event_track_enable = gpu_device->trackable_events & ( PARSEC_PROFILE_LEVEL_ZERO_TRACK_DATA_IN | PARSEC_PROFILE_LEVEL_ZERO_TRACK_MEM_USE );
        } else if(j == 1) {
            exec_stream->prof_event_track_enable = gpu_device->trackable_events & ( PARSEC_PROFILE_LEVEL_ZERO_TRACK_DATA_OUT | PARSEC_PROFILE_LEVEL_ZERO_TRACK_MEM_USE );
        } else {
            exec_stream->prof_event_track_enable = gpu_device->trackable_events & ( PARSEC_PROFILE_LEVEL_ZERO_TRACK_EXEC | PARSEC_PROFILE_LEVEL_ZERO_TRACK_MEM_USE );
        }
#endif  /* defined(PARSEC_PROF_TRACE) */
    }

    device->type                 = PARSEC_DEV_LEVEL_ZERO;
    device->executed_tasks       = 0;
    device->data_in_array_size   = 0;     // We'll let the modules_attach allocate the array of the right size for us
    device->data_in_from_device  = NULL;
    device->data_out_to_host     = 0;
    device->required_data_in     = 0;
    device->required_data_out    = 0;
    device->nb_evictions         = 0;

    device->attach              = parsec_device_attach;
    device->detach              = parsec_device_detach;
    device->taskpool_register   = parsec_device_taskpool_register;
    device->taskpool_unregister = parsec_device_taskpool_unregister;
    device->data_advise         = parsec_device_data_advise;
    device->memory_release      = parsec_device_flush_lru;
    device->kernel_scheduler    = parsec_device_kernel_scheduler;

    /*TODO NOT IMPLEMENTED: We compute gflops based on FMA rate on Ponte
     * Vecchio (e.g., Aurora) */
    device->gflops_guess = true;
    double fp16, fp32, fp64, tf32;
    device->gflops_fp16 = fp16 = 2.f * 4096.0 * sm * freqGHz;
    device->gflops_tf32 = tf32 = 2.f * 2048.0 * sm * freqGHz;
    device->gflops_fp32 = fp32 = 2.f * 256.0 * sm * freqGHz;
    device->gflops_fp64 = fp64 = 2.f * 256.0 * sm * freqGHz;
    assert(device->gflops_fp32 > 0);
    assert(device->gflops_fp64 > 0);
    if(device->gflops_guess)
        fp16 = tf32 = fp32 = fp64 = 0; /* don't report anything if we don't know */
    device->device_load = 0;

    /* Initialize internal lists */
    PARSEC_OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&gpu_device->pending,           parsec_fifo_t);

    gpu_device->sort_starting_p = NULL;
    gpu_device->peer_access_mask = 0;  /* No GPU to GPU direct transfer by default */

    device->memory_register          = NULL; // TODO there seem to be no memory pinning in level_zero?
    device->memory_unregister        = NULL; // TODO there seem to be no memory pinning in level_zero?
    gpu_device->set_device       = parsec_level_zero_set_device;
    gpu_device->memcpy_async     = parsec_level_zero_memcpy_async;
    gpu_device->event_record     = parsec_level_zero_event_record;
    gpu_device->event_query      = parsec_level_zero_event_query;
    gpu_device->memory_info      = parsec_level_zero_memory_info;
    gpu_device->memory_allocate  = parsec_level_zero_memory_allocate;
    gpu_device->memory_free      = parsec_level_zero_memory_free;
    gpu_device->find_incarnation = parsec_level_zero_find_incarnation;

    if( PARSEC_SUCCESS != parsec_device_memory_reserve(gpu_device,
                                                       parsec_level_zero_memory_percentage,
                                                       parsec_level_zero_memory_number_of_blocks,
                                                       parsec_level_zero_memory_block_size) ) {
        goto release_device;
    }

    if( show_caps ) {
        parsec_inform("Dev GPU %10s : %s [pci %x:%x.%x]\n"
                      "\tFrequency (GHz)    : %.2f\t[Threads: %d | Slices: %u.%u.%u | SimdWidth: %u | MaxMemAlloc: %.1f GB]\n"
                      "\tPeak Tflop/s %-5s : fp64: %-8.3f fp32: %-8.3f fp16: %-8.3f tf32: %-8.3f\n",
                      device->name, szName, prop->deviceId, prop->subdeviceId, prop->vendorId,
                      freqGHz, sm, prop->numSlices, prop->numSubslicesPerSlice, prop->numEUsPerSubslice, prop->physicalEUSimdWidth, prop->maxMemAllocSize/1024.f/1024.f/1024.f,
                      device->gflops_guess? "GUESS": "", fp64*1e-3, fp32*1e-3, fp16*1e-3, tf32*1e-3);
        //TODO: show memory capacity and reserved allocation
    }
    *module = device;
    return PARSEC_SUCCESS;

 release_device:
    if( NULL != gpu_device->exec_stream) {
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            parsec_level_zero_exec_stream_t *level_zero_stream =
                    (parsec_level_zero_exec_stream_t*)gpu_device->exec_stream[j];
            parsec_gpu_exec_stream_t* exec_stream = &level_zero_stream->super;

            if( NULL != exec_stream->fifo_pending ) {
                PARSEC_OBJ_RELEASE(exec_stream->fifo_pending);
            }
            if( NULL != exec_stream->tasks ) {
                free(exec_stream->tasks); exec_stream->tasks = NULL;
            }
            if( NULL != level_zero_stream->fences ) {
                for( k = 0; k < exec_stream->max_events; k++ ) {
                    if( NULL != level_zero_stream->fences[k] ) {
                        (void)zeFenceDestroy(level_zero_stream->fences[k]);
                    }
                }
                free(level_zero_stream->fences); level_zero_stream->fences = NULL;
            }
            if( NULL != exec_stream->name ) {
                free(exec_stream->name); exec_stream->name = NULL;
            }
#if defined(PARSEC_PROF_TRACE)
            if( NULL != exec_stream->profiling ) {
                /* No function to clean the profiling stream. If one is introduced
                 * some day, remember that exec streams 0 and 1 always share the same
                 * ->profiling stream, and that all of them share the same
                 * ->profiling stream if parsec_device_level_zero_one_profiling_stream_per_gpu_stream == 0 */
            }
#endif  /* defined(PARSEC_PROF_TRACE) */
        }
        // All exec_stream_t are allocated in a single malloc in gpu_device->exec_stream[0]
        free(gpu_device->exec_stream[0]);
        free(gpu_device->exec_stream);
        gpu_device->exec_stream = NULL;
    }
    free(gpu_device);
    return PARSEC_ERROR;
}

int
parsec_level_zero_module_fini(parsec_device_module_t* device)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)device;
    parsec_device_level_zero_module_t* level_zero_device = (parsec_device_level_zero_module_t*)device;
    ze_result_t status;
    int j, k;

    /* Release the registered memory */
    parsec_device_memory_release(gpu_device);

    /* Release pending queue */
    PARSEC_OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
    for( j = 0; j < gpu_device->num_exec_streams; j++ ) {
        parsec_level_zero_exec_stream_t* level_zero_stream = (parsec_level_zero_exec_stream_t*)gpu_device->exec_stream[j];
        parsec_gpu_exec_stream_t* exec_stream = &level_zero_stream->super;

        exec_stream->executed = 0;
        exec_stream->start    = 0;
        exec_stream->end      = 0;

        for( k = 0; k < exec_stream->max_events; k++ ) {
            assert( NULL == exec_stream->tasks[k] );
            status = zeFenceDestroy(level_zero_stream->fences[k]);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "(parsec_level_zero_device_fini) zeFenceDestroy ", status, {} );
            status = zeCommandListDestroy(level_zero_stream->command_lists[k]);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeCommandListDestroy ", status, {} );
        }
        exec_stream->max_events = 0;
        free(level_zero_stream->fences); level_zero_stream->fences = NULL;
        free(exec_stream->tasks); exec_stream->tasks = NULL;
        free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;
        /* Deleting the sycl queue wrapper and/or the command queue conflicts with the
         *   cleaning procedure of the Level Zero runtime... Didn't find a way to do it
         *   cleanly. Don't cleanup for now... */
        //parsec_sycl_wrapper_queue_destroy(level_zero_stream->swq);
        level_zero_stream->swq = NULL;
        free(exec_stream->name);

        /* Release Info object array */
        PARSEC_OBJ_DESTRUCT(&exec_stream->infos);
    }
    // All exec_stream_t are allocated in a single malloc in gpu_device->exec_stream[0]
    free(gpu_device->exec_stream[0]);
    free(gpu_device->exec_stream);
    gpu_device->exec_stream = NULL;
    level_zero_device->level_zero_index = -1;
    parsec_sycl_wrapper_device_destroy(level_zero_device->swd);
    level_zero_device->swd = NULL;

    /* Cleanup the GPU memory. */
    PARSEC_OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    PARSEC_OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    return PARSEC_SUCCESS;
}

#endif /* PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT */
