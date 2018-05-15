/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

#if defined(PARSEC_HAVE_OPENMP)
#include "parsec.h"
#include "parsec/data_internal.h"
#include "parsec/devices/openmp/dev_omp.h"
#include "parsec/devices/device.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/argv.h"
#include "parsec/scheduling.h"

#include <omp.h>
#include <errno.h>
#include <dlfcn.h>
#include <sys/stat.h>

#define PARSEC_GPU_USE_PRIORITIES     1
//#define PARSEC_OPENMP_OMP_ALLOC_PER_TILE 1

#define PARSEC_MAX_STREAMS            6
#define PARSEC_MAX_EVENTS_PER_STREAM  4

#if defined(PARSEC_PROF_TRACE)
/* Accepted values are: PARSEC_PROFILE_OMP_TRACK_DATA_IN | PARSEC_PROFILE_OMP_TRACK_DATA_OUT |
 *                      PARSEC_PROFILE_OMP_TRACK_OWN | PARSEC_PROFILE_OMP_TRACK_EXEC
 */
int parsec_omp_trackable_events = PARSEC_PROFILE_OMP_TRACK_EXEC | PARSEC_PROFILE_OMP_TRACK_DATA_OUT
    | PARSEC_PROFILE_OMP_TRACK_DATA_IN | PARSEC_PROFILE_OMP_TRACK_OWN;
int parsec_omp_movein_key_start;
int parsec_omp_movein_key_end;
int parsec_omp_moveout_key_start;
int parsec_omp_moveout_key_end;
int parsec_omp_own_GPU_key_start;
int parsec_omp_own_GPU_key_end;
#endif  /* defined(PROFILING) */

int parsec_openmp_output_stream = -1;
static char* omp_lib_path = NULL;

static int
parsec_omp_memory_reserve( parsec_openmp_device_t* gpu_device,
                           int           memory_percentage,
                           int           number_of_elements,
                           size_t        eltsize );
static int
parsec_omp_memory_release( parsec_openmp_device_t* gpu_device );


/* look up how many FMA per cycle in single/double, per omp MP
 * precision.
 * The following table provides updated values for future archs
 * http://docs.nvidia.com/omp/omp-c-programming-guide/index.html#arithmetic-instructions
 */
static int parsec_omp_device_lookup_ompmp_floprate(int major, int minor, int *drate, int *srate, int *trate, int *hrate) {
    //TODO find a way to compute this if possible
    *hrate = *trate = *srate = *drate = 1;
    return PARSEC_SUCCESS;
}


static int parsec_omp_device_fini(parsec_device_t* device)
{
    parsec_openmp_device_t* gpu_device = (parsec_openmp_device_t*)device;
    int j, k;

    /* Release the registered memory */
    parsec_omp_memory_release(gpu_device);

    /* Release pending queue */
    OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
    for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
        parsec_openmp_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

        exec_stream->max_events   = PARSEC_MAX_EVENTS_PER_STREAM;
        exec_stream->executed     = 0;
        exec_stream->start        = 0;
        exec_stream->end          = 0;

        for( k = 0; k < exec_stream->max_events; k++ ) {
            assert( NULL == exec_stream->tasks[k] );
            assert( 0 == exec_stream->events[k] );
        }
        free(exec_stream->events); exec_stream->events = NULL;
        free(exec_stream->tasks); exec_stream->tasks = NULL;
        free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;
        /* Release the stream */
        //TODO: omp-team-destroy? on exec_stream->omp_stream
    }
    free(gpu_device->exec_stream); gpu_device->exec_stream = NULL;

    gpu_device->omp_index = -1;

    /* Cleanup the GPU memory. */
    OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    return PARSEC_SUCCESS;
}

static int parsec_omp_memory_register(parsec_device_t* device, parsec_data_collection_t* desc, void* ptr, size_t length)
{
    int rc = PARSEC_ERROR;

    if (desc->memory_registration_status == MEMORY_STATUS_REGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the OMP interface to register the memory
     * as another thread might be submiting tasks at the same time
     * (omp_scheduling.h), and we do not set a device since we register it for
     * all devices.
     */
#if 0
    status = ompHostRegister(ptr, length, ompHostRegisterPortable );
    PARSEC_OMP_CHECK_ERROR( "(parsec_omp_memory_register) ompHostRegister ", status,
                            { goto restore_and_return; } );
#endif
    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_REGISTERED;

  restore_and_return:
    (void)device;
    return rc;
}

static int parsec_omp_memory_unregister(parsec_device_t* device, parsec_data_collection_t* desc, void* ptr)
{
    int rc = PARSEC_ERROR;

    if (desc->memory_registration_status == MEMORY_STATUS_UNREGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the OMP interface to unregister the memory
     * as another thread might be submiting tasks at the same time (omp_scheduling.h)
     */
#if 0
    status = ompHostUnregister(ptr);
    PARSEC_OMP_CHECK_ERROR( "(parsec_omp_memory_ununregister) ompHostUnregister ", status,
                            {continue;} );
#endif
    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_UNREGISTERED;

    (void)device;
    return rc;
}


void* omp_find_incarnation(parsec_openmp_device_t* gpu_device,
                                     const char* fname)
{
    char library_name[FILENAME_MAX], function_name[FILENAME_MAX], *env;
    int index = -1, capability = gpu_device->major * 10 + gpu_device->minor;
    void *fn = NULL, *dlh = NULL;
    char** argv = NULL, **target;

    /**
     * Prepare the list of PATH or FILE to be searched for a OMP shared library.
     * In any case this list might be a list of ; separated possible targets,
     * where each target can be either a directory or a specific file.
     */
    env = getenv("PARSEC_OMPCORES_LIB");
    if( NULL != env ) {
        argv = parsec_argv_split(env, ';');
    } else if( NULL != omp_lib_path ) {
        argv = parsec_argv_split(omp_lib_path, ';');
    }

  retry_lesser_version:
    if( -1 == index ) {
        capability = 0;
        snprintf(function_name, FILENAME_MAX, "%s", fname);
    }
    else {
        snprintf(function_name, FILENAME_MAX, "%s_%2d", fname, capability);
    }

    for( target = argv; (NULL != target) && (NULL != *target); target++ ) {
        struct stat status;
        if( 0 != stat(*target, &status) ) {
            parsec_debug_verbose(10, parsec_openmp_output_stream,
                                "Could not stat the %s path (%s)", *target, strerror(errno));
            continue;
        }
        if( S_ISDIR(status.st_mode) ) {
            if( capability )
                snprintf(library_name,  FILENAME_MAX, "%s/libdplasma_cores_omp%d.so", *target, capability);
            else
                snprintf(library_name,  FILENAME_MAX, "%s/libdplasma_cores_omp.so", *target);
        } else {
            snprintf(library_name,  FILENAME_MAX, "%s", *target);
        }

        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            parsec_debug_verbose(10, parsec_openmp_output_stream,
                                "Could not find %s dynamic library (%s)", library_name, dlerror());
            continue;
        }
        fn = dlsym(dlh, function_name);
        dlclose(dlh);
        if( NULL != fn ) {
            parsec_debug_verbose(4, parsec_openmp_output_stream,
                                "Function %s found in shared library %s",
                                function_name, library_name);
            break;  /* we got one, stop here */
        }
    }
    /* Couldn't load from named dynamic libs, try linked/static */
    if(NULL == fn) {
        parsec_output_verbose(10, parsec_openmp_output_stream,
                             "No dynamic function %s found, trying from compile time linked in\n",
                             function_name);
        dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
        if(NULL != dlh) {
            fn = dlsym(dlh, function_name);
            if(NULL != fn) {
                parsec_debug_verbose(4, parsec_openmp_output_stream,
                                    "Function %s found in the application symbols",
                                    function_name);
            }
            dlclose(dlh);
        }
    }

    /* Still not found?? skip this GPU */
    if(NULL == fn) {
        parsec_debug_verbose(10, parsec_openmp_output_stream,
                            "No function %s found for OMP device %s",
                            function_name, gpu_device->super.name);
        index--;
        if(-1 <= index)
            goto retry_lesser_version;
    }

    if( NULL != argv )
        parsec_argv_free(argv);

    return fn;
}

static int
parsec_omp_taskpool_register(parsec_device_t* device, parsec_taskpool_t* tp)
{
    parsec_openmp_device_t* gpu_device = (parsec_openmp_device_t*)device;
    uint32_t i, j, dev_mask = 0x0;
    int32_t rc = PARSEC_ERR_NOT_FOUND;

    /**
     * Let's suppose it is not our job to detect if a particular body can
     * run or not. We will need to add some properties that will allow the
     * user to write the code to assess this.
     */
    assert(PARSEC_DEV_OPENMP == device->type);
    for( i = 0; NULL != tp->task_classes_array[i]; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( dev_mask = j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type == device->type ) {
                if ( NULL == chores[j].dyld ) {
                    /* No dynamic load for this kernel */
                    chores[gpu_device->omp_index].dyld_fn = NULL;
                    rc = PARSEC_SUCCESS;
                    dev_mask |= chores[j].type;
                }
                else {
                    void* devf = omp_find_incarnation(gpu_device, chores[j].dyld);
                    if( NULL != devf ) {
                        chores[gpu_device->omp_index].dyld_fn = devf;
                        rc = PARSEC_SUCCESS;
                        dev_mask |= chores[j].type;
                    }
                }
            }
        }
    }
    /* Not a single chore supports this device, there is no reason to check anything further */
    if(PARSEC_SUCCESS != rc) {
        tp->devices_mask &= ~(device->device_index);
    }

    return rc;
}

static int
parsec_omp_taskpool_unregister(parsec_device_t* device, parsec_taskpool_t* tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

int parsec_omp_init(parsec_context_t *parsec_context)
{
    int omp_memory_block_size, omp_memory_percentage, omp_memory_number_of_blocks = -1;
    int show_caps_index, show_caps = 0;
    int use_omp_index, use_omp;
    int omp_mask, omp_verbosity;
    int ndevices, i, j, k;

    use_omp_index = parsec_mca_param_reg_int_name("device_omp", "enabled",
                                                   "The number of OMP device to enable for the next PaRSEC context",
                                                   false, false, -1, &use_omp);
    parsec_mca_param_reg_int_name("device_omp", "mask",
                                  "The bitwise mask of OMP devices to be enabled (default all)",
                                  false, false, 0xffffffff, &omp_mask);
    (void)parsec_mca_param_reg_int_name("device_omp", "verbose",
                                        "Set the verbosity level of the OMP device (negative value: use debug verbosity), higher is less verbose)\n",
                                        false, false, -1, &omp_verbosity);
    (void)parsec_mca_param_reg_string_name("device_omp", "path",
                                           "Path to the shared library files containing the OMP version of the hooks. It is a ;-separated list of either directories or .so files.\n",
                                           false, false, PARSEC_LIB_OPENMP_PREFIX, &omp_lib_path);
    (void)parsec_mca_param_reg_int_name("device_omp", "memory_block_size",
                                        "The OMP memory page for PaRSEC internal management.",
                                        false, false, 32*1024, &omp_memory_block_size);
    (void)parsec_mca_param_reg_int_name("device_omp", "memory_use",
                                        "The percentage of the total GPU memory to be used by this PaRSEC context",
                                        false, false, 95, &omp_memory_percentage);
    (void)parsec_mca_param_reg_int_name("device_omp", "memory_number_of_blocks",
                                        "Alternative to device_omp_memory_use: sets exactly the number of blocks to allocate (-1 means to use a percentage of the available memory)",
                                        false, false, -1, &omp_memory_number_of_blocks);
    if( 0 == use_omp ) {
        return -1;  /* Nothing to do around here */
    }

    if( omp_verbosity >= 0 ) {
        parsec_openmp_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_openmp_output_stream, omp_verbosity);
    } else {
        parsec_openmp_output_stream = parsec_debug_output;
    }

    assert(omp_is_initial_device());
    ndevices = omp_get_num_devices();

    if( -1 == use_omp ) {  /* use all OMP devices */
        parsec_mca_param_set_int(use_omp_index, ndevices);
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "device_omp_enabled updated to the total number of devices (%d)\n", ndevices);
    } else {
        if( ndevices > use_omp ) {
            if( 0 < use_omp_index ) {
                ndevices = use_omp;
            }
        } else if (ndevices < use_omp ) {
            if( 0 < use_omp_index ) {
                parsec_warning("User requested %d GPUs, but only %d are available in this machine. PaRSEC will enable all of them.",
                               use_omp, ndevices);
                parsec_mca_param_set_int(use_omp_index, ndevices);
            }
        }
    }

    /* Update the number of GPU for the upper layer */
    use_omp = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

    show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        parsec_mca_param_lookup_int(show_caps_index, &show_caps);
    }
#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                             sizeof(intptr_t), "pointer{int64_t}",
                                             &parsec_omp_movein_key_start, &parsec_omp_movein_key_end);
    parsec_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                             sizeof(intptr_t), "pointer{int64_t}",
                                             &parsec_omp_moveout_key_start, &parsec_omp_moveout_key_end);
    parsec_profiling_add_dictionary_keyword( "omp", "fill:#66ff66",
                                             0, NULL,
                                             &parsec_omp_own_GPU_key_start, &parsec_omp_own_GPU_key_end);
#endif  /* defined(PROFILING) */

    for( i = 0; i < ndevices; i++ ) {
        parsec_openmp_device_t* gpu_device;
        char *szName;
        int major, minor, host_emu, streaming_multiprocessor, drate, srate, trate, hrate;
        float clockRate;

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & omp_mask) ) continue;

#pragma omp target device(i) map(from: streaming_multiprocessor, host_emu)
{
        streaming_multiprocessor = omp_get_num_procs();
        host_emu = omp_is_initial_device();
}
        major     = 4; /* omp target version */
        minor     = 5;
        //TODO: fill this with real values
        clockRate = 2e3; /* in MHz */
        asprintf(&szName, "OMP device %d (%s)", i, host_emu?"host emulation": "offload");
        
        gpu_device = (parsec_openmp_device_t*)calloc(1, sizeof(parsec_openmp_device_t));
        OBJ_CONSTRUCT(gpu_device, parsec_list_item_t);
        gpu_device->omp_index = (uint8_t)i;
        gpu_device->major      = (uint8_t)major;
        gpu_device->minor      = (uint8_t)minor;
        gpu_device->super.name = strdup(szName);

        gpu_device->max_exec_streams = PARSEC_MAX_STREAMS;
        gpu_device->exec_stream =
            (parsec_openmp_exec_stream_t*)malloc(gpu_device->max_exec_streams
                                              * sizeof(parsec_openmp_exec_stream_t));
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            parsec_openmp_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

#if 0
            //TODO: create an OMP team to async the memcpy
            /* Allocate the stream */
            ompstatus = ompStreamCreate( &(exec_stream->omp_stream) );
            PARSEC_OMP_CHECK_ERROR( "ompStreamCreate ", ompstatus,
                                     {break;} );
#else
            exec_stream->omp_stream = j;
#endif
            exec_stream->workspace    = NULL;
            exec_stream->max_events   = PARSEC_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;
            exec_stream->fifo_pending = (parsec_list_t*)OBJ_NEW(parsec_list_t);
            OBJ_CONSTRUCT(exec_stream->fifo_pending, parsec_list_t);
            exec_stream->tasks  = (parsec_openmp_context_t**)malloc(exec_stream->max_events
                                                                 * sizeof(parsec_openmp_context_t*));
            exec_stream->events = (int*)malloc(exec_stream->max_events * sizeof(int));
            /* and the corresponding events */
            for( k = 0; k < exec_stream->max_events; k++ ) {
                exec_stream->events[k] = 0;
                exec_stream->tasks[k]  = NULL;
            }
#if defined(PARSEC_PROF_TRACE)
            exec_stream->profiling = parsec_profiling_thread_init( 2*1024*1024, PARSEC_PROFILE_STREAM_STR, i, j );
            if(j == 0) {
                exec_stream->prof_event_track_enable = parsec_omp_trackable_events & PARSEC_PROFILE_OMP_TRACK_DATA_IN;
                exec_stream->prof_event_key_start    = parsec_omp_movein_key_start;
                exec_stream->prof_event_key_end      = parsec_omp_movein_key_end;
            } else if(j == 1) {
                exec_stream->prof_event_track_enable = parsec_omp_trackable_events & PARSEC_PROFILE_OMP_TRACK_DATA_OUT;
                exec_stream->prof_event_key_start    = parsec_omp_moveout_key_start;
                exec_stream->prof_event_key_end      = parsec_omp_moveout_key_end;
            } else {
                exec_stream->prof_event_track_enable = parsec_omp_trackable_events & PARSEC_PROFILE_OMP_TRACK_EXEC;
                exec_stream->prof_event_key_start    = -1;
                exec_stream->prof_event_key_end      = -1;
            }
#endif  /* defined(PARSEC_PROF_TRACE) */
        }

        gpu_device->super.type                 = PARSEC_DEV_OPENMP;
        gpu_device->super.executed_tasks       = 0;
        gpu_device->super.transferred_data_in  = 0;
        gpu_device->super.transferred_data_out = 0;
        gpu_device->super.required_data_in     = 0;
        gpu_device->super.required_data_out    = 0;

        gpu_device->super.device_fini                = parsec_omp_device_fini;
        gpu_device->super.device_memory_register     = parsec_omp_memory_register;
        gpu_device->super.device_memory_unregister   = parsec_omp_memory_unregister;
        gpu_device->super.device_taskpool_register   = parsec_omp_taskpool_register;
        gpu_device->super.device_taskpool_unregister = parsec_omp_taskpool_unregister;

        if (parsec_omp_device_lookup_ompmp_floprate(major, minor, &drate, &srate, &trate, &hrate) == PARSEC_ERROR ) {
            return -1;
        }
        gpu_device->super.device_hweight = (float)streaming_multiprocessor * (float)hrate * (float)clockRate * 4e-3f;
        gpu_device->super.device_tweight = (float)streaming_multiprocessor * (float)trate * (float)clockRate * 2e-3f;
        gpu_device->super.device_sweight = (float)streaming_multiprocessor * (float)srate * (float)clockRate * 4e-3f;
        gpu_device->super.device_dweight = (float)streaming_multiprocessor * (float)drate * (float)clockRate * 2e-3f;

        if( show_caps ) {
            parsec_inform("OMP Device %d (capability %d.%d): %s\n"
                          "\tSM                 : %d\n"
                          "\tclockRate (GHz)    : %2.2f\n"
                          "\tpeak Gflops         : double %2.4f, single %2.4f tensor %2.4f half %2.4f",
                          i, major, minor, szName,
                          clockRate*1e-3,
                          streaming_multiprocessor,
                          gpu_device->super.device_dweight, gpu_device->super.device_sweight, gpu_device->super.device_tweight, gpu_device->super.device_hweight);
        }

        if( PARSEC_SUCCESS != parsec_omp_memory_reserve(gpu_device,
                                                         omp_memory_percentage,
                                                         omp_memory_number_of_blocks,
                                                         omp_memory_block_size) ) {
            free(gpu_device);
            continue;
        }

        /* Initialize internal lists */
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       parsec_list_t);
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, parsec_list_t);
        OBJ_CONSTRUCT(&gpu_device->pending,           parsec_list_t);

        gpu_device->sort_starting_p = NULL;
        parsec_devices_add(parsec_context, &(gpu_device->super));
    }

#if 1 || defined(PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    /* In OpenMP all devices have peer access, it may be emulated if
     * needed */
    for( i = 0; i < ndevices; i++ ) {
        parsec_openmp_device_t *source_gpu, *target_gpu;

        if( NULL == (source_gpu = (parsec_openmp_device_t*)parsec_devices_get(i)) ) continue;
        /* Skip all non OMP devices */
        if( PARSEC_DEV_OPENMP != source_gpu->super.type ) continue;

        source_gpu->peer_access_mask = 0;

        for( j = 0; j < ndevices; j++ ) {
            if( (NULL == (target_gpu = (parsec_openmp_device_t*)parsec_devices_get(j))) || (i == j) ) continue;
            /* Skip all non OMP devices */
            if( PARSEC_DEV_OPENMP != target_gpu->super.type ) continue;

            /* Communication mask */
            /* In OpenMP all devices have peer access, it may be emulated if
             * needed */
            source_gpu->peer_access_mask = (int16_t)(source_gpu->peer_access_mask | (int16_t)(1 << target_gpu->omp_index));
        }
    }
#endif

    return 0;
}

int parsec_omp_fini(void)
{
    parsec_openmp_device_t* gpu_device;
    int i;

    for(i = 0; i < parsec_devices_enabled(); i++) {
        if( NULL == (gpu_device = (parsec_openmp_device_t*)parsec_devices_get(i)) ) continue;
        if(PARSEC_DEV_OPENMP != gpu_device->super.type) continue;
        parsec_omp_device_fini((parsec_device_t*)gpu_device);
        parsec_devices_remove((parsec_device_t*)gpu_device);
        free(gpu_device);
    }

    if( parsec_debug_output != parsec_openmp_output_stream ) parsec_output_close(parsec_openmp_output_stream);
    parsec_openmp_output_stream = -1;

    if ( omp_lib_path ) {
        free(omp_lib_path);
    }

    return PARSEC_SUCCESS;
}

/**
 * This function reserve the memory_percentage of the total device memory for PaRSEC.
 * This memory will be managed in chuncks of size eltsize. However, multiple chuncks
 * can be reserved in a single allocation.
 */
static int
parsec_omp_memory_reserve( parsec_openmp_device_t* gpu_device,
                           int           memory_percentage,
                           int           number_blocks,
                           size_t        eltsize )
{
    (void)eltsize;

    size_t how_much_we_allocate;
    size_t total_mem, initial_free_mem;
    uint32_t mem_elem_per_gpu = 0;

    /* Determine how much memory we can allocate */
    //ompMemGetInfo( &initial_free_mem, &total_mem );
    //TODO: figure out a way to not have to hardcode this and autodetect
    total_mem = initial_free_mem = 1024*1024*1024*(size_t)2;
    if( number_blocks != -1 ) {
        if( number_blocks == 0 ) {
            parsec_warning("OMP: Invalid argument: requesting 0 bytes of memory on OMP device %s", gpu_device->super.name);
            return PARSEC_ERROR;
        } else {
            how_much_we_allocate = number_blocks * eltsize;
        }
    } else {
        /** number_blocks == -1 means memory_percentage is used */
        how_much_we_allocate = (memory_percentage * initial_free_mem) / 100;
    }
    if( how_much_we_allocate > initial_free_mem ) {
        /** Handle the case of jokers who require more than 100% of memory,
         *  and eleventh case of computer scientists who don't know how
         *  to divide a number by another
         */
        parsec_warning("OMP: Requested %d bytes on OMP device %s, but only %d bytes are available -- reducing allocation to max available",
                      how_much_we_allocate, initial_free_mem);
        how_much_we_allocate = initial_free_mem;
    }
    if( how_much_we_allocate < eltsize ) {
        /** Handle another kind of jokers entirely, and cases of
         *  not enough memory on the device
         */
        parsec_warning("OMP: Cannot allocate at least one element on OMP device %s", gpu_device->super.name);
        return PARSEC_ERROR;
    }

#if defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
    size_t free_mem = initial_free_mem;
    /*
     * We allocate a bunch of tiles that will be used
     * during the computations
     */
    while( (free_mem > eltsize )
           && ((total_mem - free_mem) < how_much_we_allocate) ) {
        parsec_openmp_data_copy_t* gpu_elem;
        void *device_ptr;

        device_ptr = omp_target_alloc(eltsize, gpu_device->omp_index);
        if( NULL == device_ptr ) {
            size_t _free_mem = 0, _total_mem = 0;
            //ompMemGetInfo( &_free_mem, &_total_mem );
            parsec_inform("Per context: free mem %zu total mem %zu (allocated tiles %u)",
                          _free_mem, _total_mem, mem_elem_per_gpu);
        }
        gpu_elem = OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "Allocate OMP copy %p [ref_count %d] for data [%p]",
                            gpu_elem, gpu_elem->super.super.obj_reference_count, NULL);
        gpu_elem->device_private = (void*)(long)device_ptr;
        gpu_elem->device_index = gpu_device->super.device_index;
        mem_elem_per_gpu++;
        OBJ_RETAIN(gpu_elem);
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "Retain and insert OMP copy %p [ref_count %d] in LRU",
                            gpu_elem, gpu_elem->super.super.obj_reference_count);
        parsec_list_nolock_fifo_push( &gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem );
        //ompMemGetInfo( &free_mem, &total_mem );
        free_mem -= eltsize;
    }
    if( 0 == mem_elem_per_gpu && parsec_list_nolock_is_empty( &gpu_device->gpu_mem_lru ) ) {
        parsec_warning("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!",
                      gpu_device->super.context->my_rank, gpu_device->omp_index);
    }
    else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,  "GPU:\tAllocate %u tiles on the GPU memory", mem_elem_per_gpu );
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                        "GPU:\tAllocate %u tiles on the GPU memory", mem_elem_per_gpu);
#else
    if( NULL == gpu_device->memory ) {
        void* base_ptr;
        /* We allocate all the memory on the GPU and we use our memory management */
        mem_elem_per_gpu = (how_much_we_allocate + eltsize - 1 ) / eltsize;
        size_t total_size = (size_t)mem_elem_per_gpu * eltsize;
        base_ptr = omp_target_alloc(total_size, gpu_device->omp_index);
        if( NULL == base_ptr ) {
            parsec_warning("Allocating memory on the GPU device failed");
        }
        gpu_device->memory = zone_malloc_init( base_ptr, mem_elem_per_gpu, eltsize );

        if( gpu_device->memory == NULL ) {
            parsec_warning("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!",
                          gpu_device->super.context->my_rank, gpu_device->omp_index);
            return PARSEC_ERROR;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "GPU:\tAllocate %u segment of size %d on the GPU memory",
                            mem_elem_per_gpu, eltsize );
    }
#endif

    return PARSEC_SUCCESS;
}

static void parsec_omp_memory_release_list(parsec_openmp_device_t* gpu_device,
                                           parsec_list_t* list)
{
    parsec_list_item_t* item;

    while(NULL != (item = parsec_list_nolock_fifo_pop(list)) ) {
        parsec_openmp_data_copy_t* gpu_copy = (parsec_openmp_data_copy_t*)item;
        parsec_data_t* original = gpu_copy->original;

        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "Release OMP copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p, in map %p",
                            gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                            original, (NULL != original ? original->dc : NULL));
        assert( gpu_copy->device_index == gpu_device->super.device_index );
        if( DATA_COHERENCY_OWNED == gpu_copy->coherency_state ) {
            parsec_warning("GPU[%d] still OWNS the master memory copy for data %d and it is discarding it!",
                          gpu_device->omp_index, original->key);
        }
#if defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
        omp_target_free( gpu_copy->device_private, gpu_device->omp_index );
#else
        zone_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
        gpu_copy->device_private = NULL;

        /* At this point the data copies should have no attachement to a data_t. Thus,
         * before we get here (aka below parsec_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
    }
}

/**
 * This function release the OMP memory reserved for this device.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
static int
parsec_omp_memory_release( parsec_openmp_device_t* gpu_device )
{

#if 1
    dump_omp_device(gpu_device); // debug only
#endif
    /* Free all memory on GPU */
    parsec_omp_memory_release_list(gpu_device, &gpu_device->gpu_mem_lru);
    parsec_omp_memory_release_list(gpu_device, &gpu_device->gpu_mem_owned_lru);

#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
    assert( NULL != gpu_device->memory );
    void* ptr = zone_malloc_fini(&gpu_device->memory);
    omp_target_free(ptr, gpu_device->omp_index);
#endif

    return PARSEC_SUCCESS;
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
static inline int
parsec_openmp_data_reserve_device_space( parsec_openmp_device_t* gpu_device,
                                     parsec_openmp_context_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_openmp_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    parsec_data_t* master, *oldmaster;
    const parsec_flow_t *flow;
    int i, j;

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(!(flow->flow_flags)) continue;

        temp_loc[i] = NULL;
        master   = this_task->data[i].data_in->original;
        gpu_elem = PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;

        /* There is already a copy on the device */
        if( NULL != gpu_elem ) continue;

#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
        gpu_elem = OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "Allocate OMP copy %p [ref_count %d] for data %p",
                            gpu_elem, gpu_elem->super.super.obj_reference_count, master);
      malloc_data:
        gpu_elem->device_private = zone_malloc(gpu_device->memory, master->nb_elts);
        if( NULL == gpu_elem->device_private ) {
#endif

          find_another_data:
            /* Look for a data_copy to free */
            lru_gpu_elem = (parsec_openmp_data_copy_t*)parsec_list_nolock_fifo_pop(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the begining of
                 * the LRU (in order to be reused asap) and return without scheduling the task.
                 */
                parsec_warning("GPU:\tRequest space on GPU failed for %d out of %d data",
                              this_task->task_class->nb_flows - i,
                              this_task->task_class->nb_flows);
                for( j = 0; j < i; j++ ) {
                    if( NULL != temp_loc[j] ) {
                        parsec_list_nolock_lifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
                    }
                }
#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
                OBJ_RELEASE(gpu_elem);
#endif
                return -2;
            }

            PARSEC_LIST_ITEM_SINGLETON(lru_gpu_elem);
            PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                                "Release LRU-retrieved OMP copy %p [ref_count %d]",
                                lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            assert( NULL != lru_gpu_elem );

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the parsec_openmp_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                goto find_another_data; // TODO: add an assert of some sort to check for leaks here?
            }

            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
            assert( NULL   != lru_gpu_elem->original );

            /* Let's check we're not trying to steal one of our own data */
            oldmaster = lru_gpu_elem->original;
            for( j = 0; j < this_task->task_class->nb_flows; j++ ) {
                if( NULL == this_task->data[j].data_in ) continue;
                if( this_task->data[j].data_in->original == oldmaster ) {
                    goto find_another_data;
                }
            }

            /* The data is not used, and it's not one of ours: we can free it or reuse it */
            PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                                "GPU[%d]:\tRepurpose copy %p to mirror block %p (in task %s:i) instead of %p",
                                gpu_device->omp_index, lru_gpu_elem, master, this_task->task_class->name, i, oldmaster);
            parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);

#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
            /* Let's free this space, and try again to malloc some space */
            zone_free( gpu_device->memory, (void*)(lru_gpu_elem->device_private) );
            PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                                "Release LRU-retrieved OMP copy %p [ref_count %d: must be 0]",
                                lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            OBJ_RELEASE(lru_gpu_elem); assert( NULL == lru_gpu_elem );
            goto malloc_data;
        }
#else
        gpu_elem = lru_gpu_elem;
#endif
        assert( 0 == gpu_elem->readers );
        gpu_elem->coherency_state = DATA_COHERENCY_INVALID;
        gpu_elem->version = 0;
        parsec_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;
        temp_loc[i] = gpu_elem;
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "Retain and insert OMP copy %p [ref_count %d] in LRU",
                            gpu_elem, gpu_elem->super.super.obj_reference_count);
        parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem);
    }
    return 0;
}

/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns:
 *    0: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   -1: A copy cannot be issued due to OMP.
 */
static inline int
parsec_openmp_data_stage_in( parsec_openmp_device_t* gpu_device,
                         int32_t type,
                         parsec_data_pair_t* task_data,
                         parsec_openmp_context_t *gpu_task,
                         parsec_openmp_exec_stream_t *gpu_stream )
{
    parsec_data_copy_t* in_elem = task_data->data_in;
    parsec_data_t* original = in_elem->original;
    parsec_openmp_data_copy_t* gpu_elem = task_data->data_out;
    int transfer_from = -1;

    /**
     * If the data will be accessed in write mode, remove it from any GPU data management
     * lists until the task is completed.
     */
    if( FLOW_ACCESS_WRITE & type ) {
        if (gpu_elem->readers > 0 ) {
            parsec_warning("GPU[%d]:\tWrite access to data with existing readers (Possible anti-dependency, "
                          "or concurrent accesses), please prevent that with CTL dependencies\n", gpu_device->omp_index);
            return -86;
        }
        parsec_list_item_ring_chop((parsec_list_item_t*)gpu_elem);
        PARSEC_LIST_ITEM_SINGLETON(gpu_elem);
    }

    /* DtoD copy, if data is read only, then we go back to CPU copy, and fetch data from CPU (HtoD) */
    if( (in_elem != original->device_copies[0]) &&
        (in_elem->version == original->device_copies[0]->version) ) {
        /* We should never enter here, as long as we don't foward the GPU data to the input */
        assert(!((in_elem != original->device_copies[0]) &&
                 (in_elem->version == original->device_copies[0]->version)));
        parsec_data_copy_release(in_elem);  /* release the copy in GPU1 */
        task_data->data_in = original->device_copies[0];
        in_elem = task_data->data_in;
        OBJ_RETAIN(in_elem);  /* retain the corresponding CPU copy */
    }

    transfer_from = parsec_data_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);
    gpu_device->super.required_data_in += original->nb_elts;
    if( -1 != transfer_from ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                            "GPU[%d]:\t\tMove H2D data %x of %d bytes\t(Host: v:%d, ptr:%p / Device: v:%d, ptr:%p)",
                            gpu_device->omp_index, original->key, original->nb_elts,
                            in_elem->version, in_elem->device_private,
                            gpu_elem->version, (void*)gpu_elem->device_private);

        assert( gpu_elem->version <= in_elem->version );
        assert((gpu_elem->version != in_elem->version) || (gpu_elem->data_transfer_status == DATA_STATUS_NOT_TRANSFER));

#if defined(PARSEC_PROF_TRACE)
        if( gpu_stream->prof_event_track_enable ) {
            parsec_task_t *this_task = gpu_task->ec;

            assert(-1 != gpu_stream->prof_event_key_start);
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   gpu_stream->prof_event_key_start,
                                   this_task->task_class->key_functions->key_hash(this_task->task_class->make_key(this_task->taskpool, this_task->locals), 64, NULL),
                                   this_task->taskpool->taskpool_id,
                                   &original);
        }
#endif

        /* update the data version in GPU immediately, and mark the data under transfer */
        assert((gpu_elem->version != in_elem->version) || (gpu_elem->data_transfer_status == DATA_STATUS_NOT_TRANSFER));
        gpu_elem->version = in_elem->version;
        gpu_elem->data_transfer_status = DATA_STATUS_UNDER_TRANSFER;
        gpu_elem->push_task = gpu_task->ec;  /* only the task who does the transfer can modify the data status later. */
        /* TODO: take ownership of the data */
        
        /* Push data into the GPU */
        int* event = &gpu_stream->events[gpu_stream->start];
        (*event)++;
//#pragma omp target update nowait fromdepend(out: event[0]) device(gpu_device->omp_index)
#pragma omp task depend(out: event[0]) firstprivate(gpu_device, original, in_elem, gpu_elem)
{ //firstprivate(original, gpu_elem, in_elem, gpu_device)
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream, 
                        "GPU[%d]: starting omp_target_memcpy h2d from %p to %p on event %p (thread %d)",
                        gpu_device->omp_index, in_elem->device_private, gpu_elem->device_private, event, omp_get_thread_num());
    int status = omp_target_memcpy(gpu_elem->device_private, in_elem->device_private,
                                   original->nb_elts, 0, 0,
                                   gpu_device->omp_index, omp_get_initial_device());
    if( 0 != status ) {
        parsec_warning("<<%p>> -> <<%p>> [%d]", in_elem->device_private, gpu_elem->device_private, original->nb_elts);
        //return -1; 
    }
}
        gpu_device->super.transferred_data_in += original->nb_elts;
        return 1;
    }
    else {
        assert( gpu_elem->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER ||
                gpu_elem->data_transfer_status == DATA_STATUS_UNDER_TRANSFER);

        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                            "GPU[%d]:\t\tNO Move H2D for data %x of %d bytes\n",
                            gpu_device->omp_index, original->key, original->nb_elts);
    }
    /* TODO: data keeps the same coherence flags as before */
    return 0;
}

void* parsec_openmp_pop_workspace(parsec_openmp_device_t* gpu_device, parsec_openmp_exec_stream_t* gpu_stream, size_t size)
{
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (parsec_openmp_workspace_t *)malloc(sizeof(parsec_openmp_workspace_t));
        gpu_stream->workspace->total_workspace = PARSEC_OPENMP_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = PARSEC_OPENMP_MAX_WORKSPACE - 1;

        for( int i = 0; i < PARSEC_OPENMP_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
        }
    }
    assert (gpu_stream->workspace->stack_head >= 0);
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE) */
    return work;
}

int parsec_openmp_push_workspace(parsec_openmp_device_t* gpu_device, parsec_openmp_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_OPENMP_MAX_WORKSPACE);
#endif /* !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE) */
    return 0;
}

int parsec_openmp_free_workspace(parsec_openmp_device_t * gpu_device)
{
    (void)gpu_device;
#if !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        parsec_openmp_exec_stream_t *gpu_stream = &(gpu_device->exec_stream[i]);
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(PARSEC_OPENMP_OMP_ALLOC_PER_TILE) */
    return 0;
}

static inline int parsec_openmp_check_space_needed(parsec_openmp_device_t *gpu_device, parsec_openmp_context_t *gpu_task)
{
    int i;
    int space_needed = 0;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_data_t *original;
    parsec_data_copy_t *data;
    const parsec_flow_t *flow;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        if(!(flow->flow_flags)) continue;

        data = this_task->data[i].data_in;
        original = data->original;
        if( NULL != PARSEC_DATA_GET_COPY(original, gpu_device->super.device_index) ) {
            continue;
        }
        if(flow->flow_flags & FLOW_ACCESS_READ)
            space_needed++;
    }
    return space_needed;
}


static int parsec_openmp_sort_pending_list(parsec_openmp_device_t *gpu_device)
{
    //parsec_list_t *sort_list = &(gpu_device->pending);
    parsec_list_t *sort_list = gpu_device->exec_stream[0].fifo_pending;
#if 0
    if ( !parsec_atomic_trylock(&(sort_list->atomic_lock)) ) {
        return 0;
    }
#endif
    if (parsec_list_nolock_is_empty(sort_list) ) { /* list is empty */
#if 0
        parsec_atomic_unlock(&(sort_list->atomic_lock));
#endif
        return 0;
    }

    if (gpu_device->sort_starting_p == NULL || !parsec_list_nolock_contains(sort_list, gpu_device->sort_starting_p) ) {
        gpu_device->sort_starting_p = (parsec_list_item_t*)sort_list->ghost_element.list_next;
    }

    /* p is head */
    parsec_list_item_t *p = gpu_device->sort_starting_p;
    int i, j, NB_SORT = 10, space_q, space_min;

    parsec_list_item_t *q, *prev_p, *min_p;
    for (i = 0; i < NB_SORT; i++) {
        if ( p == &(sort_list->ghost_element) ) {
            break;
        }
        min_p = p; /* assume the minimum one is the first one p */
        q = (parsec_list_item_t*)min_p->list_next;
        space_min = parsec_openmp_check_space_needed(gpu_device, (parsec_openmp_context_t*)min_p);
        for (j = i+1; j < NB_SORT; j++) {
            if ( q == &(sort_list->ghost_element) ) {
                break;
            }
            space_q = parsec_openmp_check_space_needed(gpu_device, (parsec_openmp_context_t*)q);
            if ( space_min > space_q ) {
                min_p = q;
                space_min = space_q;
            }
            q = (parsec_list_item_t*)q->list_next;

        }
        if (min_p != p) { /* minimum is not the first one, let's insert min_p before p */
            /* take min_p out */
            parsec_list_item_ring_chop(min_p);
            PARSEC_LIST_ITEM_SINGLETON(min_p);
            prev_p = (parsec_list_item_t*)p->list_prev;

            /* insert min_p after prev_p */
            parsec_list_nolock_add_after( sort_list, prev_p, min_p);
        }
        p = (parsec_list_item_t*)min_p->list_next;
    }

#if 0
        parsec_atomic_unlock(&(sort_list->atomic_lock));
#endif
    return 0;
}

/**
 * Transfer at most the PARSEC_OPENMP_W2R_NB_MOVE_OUT oldest data from the GPU back
 * to main memory. Create a single task to move them all out, then switch the
 * GPU data copy in shared mode.
 */
static parsec_openmp_context_t* parsec_openmp_create_W2R_task(parsec_openmp_device_t *gpu_device, parsec_execution_stream_t *es)
{
    parsec_openmp_context_t *w2r_task = NULL;
    parsec_task_t *ec = NULL;
    parsec_openmp_data_copy_t *gpu_copy;
    parsec_data_t* original;
    parsec_list_item_t* item = (parsec_list_item_t*)gpu_device->gpu_mem_owned_lru.ghost_element.list_next;
    int nb_cleaned = 0;

    /* Find a data copy that has no pending users on the GPU, and can be
     * safely moved back on the main memory */
    while(nb_cleaned < PARSEC_OPENMP_W2R_NB_MOVE_OUT) {
        /* Break at the end of the list */
        if( item == &(gpu_device->gpu_mem_owned_lru.ghost_element) ) {
            break;
        }
        gpu_copy = (parsec_openmp_data_copy_t*)item;
        original = gpu_copy->original;
        if( (0 != gpu_copy->readers) || (0 != original->device_copies[0]->readers) ) {
            item = (parsec_list_item_t*)item->list_next;  /* conversion needed for volatile */
            continue;
        }
        if( NULL == ec ) {  /* allocate on-demand */
            ec = (parsec_task_t*)parsec_thread_mempool_allocate(es->context_mempool);
            if( NULL == ec )  /* we're running out of memory. Bail out. */
                break;
            ec->status = PARSEC_TASK_STATUS_NONE;
        }
        parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
        PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
        gpu_copy->readers++;
        ec->data[nb_cleaned].data_out = gpu_copy;
        nb_cleaned++;
    }

    if( 0 == nb_cleaned )
        return NULL;

    w2r_task = (parsec_openmp_context_t *)malloc(sizeof(parsec_openmp_context_t));
    OBJ_CONSTRUCT(w2r_task, parsec_list_item_t);
    ec->priority = INT32_MAX;
    ec->task_class = NULL;
    w2r_task->ec = ec;
    w2r_task->task_type = GPU_TASK_TYPE_D2HTRANSFER;
    return w2r_task;
}

/**
 * Complete a data copy transfer originated from the engine.
 */
int parsec_openmp_W2R_task_fini(parsec_openmp_device_t *gpu_device,
                            parsec_openmp_context_t *w2r_task,
                            parsec_execution_stream_t *es)
{
    parsec_openmp_data_copy_t *gpu_copy, *cpu_copy;
    parsec_task_t *ec = w2r_task->ec;
    parsec_data_t* original;
    int i;

    assert(w2r_task->task_type == GPU_TASK_TYPE_D2HTRANSFER);
    for( i = 0; (i < PARSEC_OPENMP_W2R_NB_MOVE_OUT) && (NULL != (gpu_copy = ec->data[i].data_out)); i++ ) {
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];
        cpu_copy->coherency_state =  DATA_COHERENCY_SHARED;
        cpu_copy->version = gpu_copy->version;
        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                            "Mirror on CPU and move OMP copy %p [ref_count %d] in LRU",
                            gpu_copy, gpu_copy->super.super.obj_reference_count);
        parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        gpu_copy->readers--;
        assert(gpu_copy->readers >= 0);
    }
    parsec_thread_mempool_free(es->context_mempool, w2r_task->ec);
    free(w2r_task);
    return 0;
}

#if PARSEC_GPU_USE_PRIORITIES
static inline parsec_list_item_t* parsec_fifo_push_ordered( parsec_list_t* fifo,
                                                          parsec_list_item_t* elem )
{
    parsec_list_nolock_push_sorted(fifo, elem, parsec_execution_context_priority_comparator);
    return elem;
}
#define PARSEC_FIFO_PUSH  parsec_fifo_push_ordered
#else
#define PARSEC_FIFO_PUSH  parsec_list_nolock_fifo_push
#endif

static inline int
progress_stream( parsec_openmp_device_t* gpu_device,
                 parsec_openmp_exec_stream_t* exec_stream,
                 advance_openmp_task_function_t progress_fct,
                 parsec_openmp_context_t* task,
                 parsec_openmp_context_t** out_task )
{
    int saved_rc = 0, rc, i;
    *out_task = NULL;
    parsec_task_t *this_task;
    const parsec_flow_t *flow;

    if( NULL != task ) {
        PARSEC_FIFO_PUSH(exec_stream->fifo_pending, (parsec_list_item_t*)task);
        task = NULL;
    }
  grab_a_task:
    if( NULL == exec_stream->tasks[exec_stream->start] ) {
        /* get the best task */
        task = (parsec_openmp_context_t*)parsec_list_nolock_fifo_pop(exec_stream->fifo_pending);
    }
    if( NULL == task ) {
        /* No more room on the event list or no tasks. Keep moving */
        goto check_completion;
    }
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)task);

    assert( NULL == exec_stream->tasks[exec_stream->start] );
    /**
     * In case the task is succesfully progressed, the corresponding profiling
     * event is triggered.
     */
    if ( NULL == progress_fct ) {
        /* Grab the submit function */
        progress_fct = task->submit;
#if defined(PARSEC_DEBUG_PARANOID)
        for( i = 0; i < task->ec->task_class->nb_flows; i++ ) {
            flow = task->flow[i];
            assert( flow );
            assert( flow->flow_index == i );
            if(!flow->flow_flags) continue;
            assert(task->ec->data[i].data_out->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER);
        }
#endif /* defined(PARSEC_DEBUG_PARANOID) */
    }
    assert( NULL != progress_fct );
    rc = progress_fct( gpu_device, task, exec_stream );
    if( 0 > rc ) {
        if( PARSEC_HOOK_RETURN_AGAIN != rc) {
            *out_task = task;
            return rc;
        }

        PARSEC_FIFO_PUSH(exec_stream->fifo_pending, (parsec_list_item_t*)task);
        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                             "GPU[%d]: Reschedule %s(task %p) priority %d: no room available on the GPU for data",
                             gpu_device->omp_index, task->ec->task_class->name, (void*)task->ec, task->ec->priority);
    } else {
        /**
         * Do not skip the omp event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to incorrect results.
         */
        int* event = &exec_stream->events[exec_stream->start];
        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                             "GPU[%d]:  %s(task %p) target completion waited on event %p, posted %d actions in that event",
                             gpu_device->omp_index, task->ec->task_class->name, (void*)task->ec, event, *event);
#pragma omp task depend(inout: event[0])
{
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream, 
                        "GPU[%d]: decreasing event counter %p from %d to 0",
                        gpu_device->omp_index, event, *event);
    assert(0 != *event);
    (*event)=0;
}
        exec_stream->tasks[exec_stream->start] = task;
        exec_stream->start = (exec_stream->start + 1) % exec_stream->max_events;
#if defined(PARSEC_DEBUG_NOISIER)
        if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                                "GPU[%d]: Submitted Transfer(task %p) on stream %d",
                                gpu_device->omp_index, (void*)task->ec,
                                exec_stream->omp_stream);
        }
        else {
            PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                                "GPU[%d]: Submitted %s(task %p) priority %d on stream %d",
                                gpu_device->omp_index,
                                task->ec->task_class->name, (void*)task->ec, task->ec->priority,
                                exec_stream->omp_stream);
        }
#endif
    }
    task = NULL;

  check_completion:
    if( (NULL == *out_task) && (NULL != exec_stream->tasks[exec_stream->end]) ) {
#pragma omp taskyield
        if(0 == exec_stream->events[exec_stream->end]) {
            /**
             * Even though omp event return success, the PUSH may not be
             * completed if no PUSH is required by this task and the PUSH is
             * actually done by another task, so we need to check if the data is
             * actually ready to use
             */
            if (exec_stream == &(gpu_device->exec_stream[0])) {  /* exec_stream[0] is the PUSH stream */
                parsec_openmp_context_t *gtask = exec_stream->tasks[exec_stream->end];
                this_task = gtask->ec;
                for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
                    flow = gtask->flow[i];
                    assert( flow );
                    assert( flow->flow_index == i );
                    if(!flow->flow_flags) continue;
                    if (this_task->data[i].data_out->push_task == this_task) {   /* only the task who did this PUSH can modify the status */
                        this_task->data[i].data_out->data_transfer_status = DATA_STATUS_COMPLETE_TRANSFER;
                        continue;
                    }
                    assert(this_task->data[i].data_out->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER);
                    if (this_task->data[i].data_out->data_transfer_status != DATA_STATUS_COMPLETE_TRANSFER) {  /* data is not ready */
                        /**
                         * As long as we have only one stream to push the data
                         * to the GPU, we should never end up in this case.
                         * Remove previous assert if changed.
                         */
                        return saved_rc;
                    }
                }
            }

            /* Save the task for the next step */
            task = *out_task = exec_stream->tasks[exec_stream->end];
#if defined(PARSEC_DEBUG_NOISIER)
            if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
                PARSEC_DEBUG_VERBOSE(19, parsec_openmp_output_stream,
                                    "GPU[%d]: Completed Transfer(task %p) on stream %d",
                                    gpu_device->omp_index,
                                    (void*)task->ec,
                                    exec_stream->omp_stream);
            }
            else {
                PARSEC_DEBUG_VERBOSE(19, parsec_openmp_output_stream,
                                    "GPU[%d]: Completed %s(task %p) priority %d on stream %d",
                                    gpu_device->omp_index,
                                    task->ec->task_class->name, (void*)task->ec, task->ec->priority,
                                    exec_stream->omp_stream);
            }
#endif
            exec_stream->tasks[exec_stream->end] = NULL;
            exec_stream->end = (exec_stream->end + 1) % exec_stream->max_events;
#if defined(PARSEC_PROF_TRACE)
            if( exec_stream->prof_event_track_enable ) {
                if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
                    assert( exec_stream->prof_event_key_end == parsec_omp_moveout_key_end );
                    PARSEC_PROFILING_TRACE(exec_stream->profiling,
                                          exec_stream->prof_event_key_end,
                                          -1, 0, NULL);
                } else {
                    PARSEC_TASK_PROF_TRACE(exec_stream->profiling,
                                          (-1 == exec_stream->prof_event_key_end ?
                                           PARSEC_PROF_FUNC_KEY_END(task->ec->taskpool,
                                                                    task->ec->task_class->task_class_id) :
                                           exec_stream->prof_event_key_end),
                                          task->ec);
                }
            }
#endif /* (PARSEC_PROF_TRACE) */
            task = NULL;  /* Try to schedule another task */
            goto grab_a_task;
        }
    }
    return saved_rc;
}

void dump_omp_stream(parsec_openmp_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    parsec_debug_verbose(0, parsec_openmp_output_stream,
                        "Dump GPU exec stream %p [events = %d, start = %d, end = %d, executed = %d]",
                        exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
                        exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        parsec_debug_verbose(0, parsec_openmp_output_stream,
                            "    %d: %s", i, parsec_task_snprintf(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void dump_omp_device(parsec_openmp_device_t* gpu_device)
{
    int i;

    parsec_output(parsec_openmp_output_stream, "\n\n");
    parsec_output(parsec_openmp_output_stream, "Device %d:%d (%p)\n", gpu_device->omp_index, gpu_device->super.device_index, gpu_device);
    parsec_output(parsec_openmp_output_stream, "\tpeer mask %x executed tasks %llu max streams %d\n",
                 gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->max_exec_streams);
    parsec_output(parsec_openmp_output_stream, "\tstats transferred [in %llu out %llu] required [in %llu out %llu]\n",
                 (unsigned long long)gpu_device->super.transferred_data_in, (unsigned long long)gpu_device->super.transferred_data_out,
                 (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dump_omp_stream(&gpu_device->exec_stream[i]);
    }
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_lru) ) {
        parsec_output(parsec_openmp_output_stream, "#\n# LRU list\n#\n");
        i = 0;
        PARSEC_LIST_NOLOCK_ITERATOR(&gpu_device->gpu_mem_lru, item,
                             {
                                 parsec_openmp_data_copy_t* gpu_copy = (parsec_openmp_data_copy_t*)item;
                                 parsec_output(parsec_openmp_output_stream, "  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    };
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        parsec_output(parsec_openmp_output_stream, "#\n# Owned LRU list\n#\n");
        i = 0;
        PARSEC_LIST_NOLOCK_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                             {
                                 parsec_openmp_data_copy_t* gpu_copy = (parsec_openmp_data_copy_t*)item;
                                 parsec_output(parsec_openmp_output_stream, "  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    };
    parsec_output(parsec_openmp_output_stream, "\n\n");
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
int
parsec_openmp_kernel_push( parsec_openmp_device_t            *gpu_device,
                       parsec_openmp_context_t     *gpu_task,
                       parsec_openmp_exec_stream_t *gpu_stream)
{
    int i, ret = 0;
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t        *flow;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Try to Push %s",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    /**
     * First, let's reserve enough space on the device to transfer the data on the GPU.
     */
    ret = parsec_openmp_data_reserve_device_space( gpu_device,
                                               gpu_task );
    if( ret < 0 ) {
        return ret;
    }

    /**
     * Second, We have enough space, let's schedule the required transfer
     */
    PARSEC_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_stream->profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                        this_task->task_class->task_class_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        /* Skip CTL flows */
        if(!(flow->flow_flags)) continue;

        assert( NULL != parsec_data_copy_get_ptr(this_task->data[i].data_in) );

        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                            "GPU[%1d]:\t\tIN  Data of %s <%x> on GPU\n",
                            gpu_device->omp_index, flow->name,
                            this_task->data[i].data_out->original->key);
        ret = parsec_openmp_data_stage_in( gpu_device, flow->flow_flags,
                                       &(this_task->data[i]), gpu_task, gpu_stream );
        if( ret < 0 ) {
            return ret;
        }
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Push task %s DONE",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    return ret;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
int
parsec_openmp_kernel_pop( parsec_openmp_device_t            *gpu_device,
                      parsec_openmp_context_t     *gpu_task,
                      parsec_openmp_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_openmp_data_copy_t     *gpu_copy;
    parsec_data_t              *original;
    const parsec_flow_t        *flow;
    int return_code = 0, how_many = 0, i;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < 1; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            original = gpu_copy->original;
            int* event = &gpu_stream->events[gpu_stream->start];
            (*event)++;
#pragma omp task depend(out: event[0]) firstprivate(this_task, gpu_device, original, gpu_copy)
{ //firstprivate(original, gpu_elem, in_elem, gpu_device)
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream, 
                        "GPU[%d]: starting omp_target_memcpy d2h from %p to %p on event %p (thread %d)",
                        gpu_device->omp_index, gpu_copy->device_private, original->device_copies[0]->device_private, event, omp_get_thread_num());
    int status = omp_target_memcpy(original->device_copies[0]->device_private, 
                                   gpu_copy->device_private,
                                   original->nb_elts, 0, 0,
                                   omp_get_initial_device(), gpu_device->omp_index);
    if( 0 != status ) {
        parsec_warning("data %s <<%p>> -> <<%p>> [%d]", this_task->task_class->out[i]->name, 
                                                        gpu_copy->device_private,
                                                        original->device_copies[0]->device_private,
                                                        original->nb_elts);
    }
}
        }
        return return_code;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Try to Pop %s",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;

        flow = gpu_task->flow[i];

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(original == this_task->data[i].data_in->original);

        if( !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[i].data_out = original->device_copies[0];
        }

        if( flow->flow_flags & FLOW_ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
                parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
                PARSEC_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
                continue;  /* done with this element, go for the next one */
            }
        }
        if( flow->flow_flags & FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == parsec_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((parsec_list_item_t*)gpu_copy)->list_next == (parsec_list_item_t*)gpu_copy );
            assert( ((parsec_list_item_t*)gpu_copy)->list_prev == (parsec_list_item_t*)gpu_copy );

            PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                                "GPU[%1d]:\tOUT Data of %s\n", gpu_device->omp_index, flow->name);

            if( gpu_task->pushout[i] ) {
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                                    "GPU[%d]:\tMove D2H data <%s:%x> D:%p -> H:%p requested\n",
                                    gpu_device->omp_index, flow->name, original->key,
                                    (void*)gpu_copy->device_private, original->device_copies[0]->device_private);
                PARSEC_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_stream->profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                                    this_task->task_class->task_class_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* Move the data back into main memory */
                int* event = &gpu_stream->events[gpu_stream->start];
                (*event)++;
#pragma omp task depend(out: event[0]) firstprivate(this_task, gpu_device, original, gpu_copy)
{ //firstprivate(original, gpu_elem, in_elem, gpu_device)
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream, 
                        "GPU[%d]: starting omp_target_memcpy d2h from %p to %p on event %p (thread %d)",
                        gpu_device->omp_index, gpu_copy->device_private, original->device_copies[0]->device_private, event, omp_get_thread_num());
    int status = omp_target_memcpy(original->device_copies[0]->device_private, 
                                   gpu_copy->device_private,
                                   original->nb_elts, 0, 0,
                                   omp_get_initial_device(), gpu_device->omp_index);
    if( 0 != status ) {
        parsec_warning("data %s <<%p>> -> <<%p>> [%d]", this_task->task_class->out[i]->name, 
                                                        gpu_copy->device_private,
                                                        original->device_copies[0]->device_private,
                                                        original->nb_elts);
    }
}
                gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            }
        }
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Pop %s DONE",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

  release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
int
parsec_openmp_kernel_epilog( parsec_openmp_device_t        *gpu_device,
                         parsec_openmp_context_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_openmp_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Epilog of %s",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;

        if( !(gpu_task->flow[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        /**
         * There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         *
         * WARNING: For now we always forward the cpu_copy to the next task, to
         * do that, we lie to the engine by updating the CPU copy to the same
         * status than the GPU copy without updating the data itself. Thus, the
         * cpu copy is really invalid. this is related to Issue #88, and the
         * fact that:
         *      - we don't forward the gpu copy as output
         *      - we always take a cpu copy as input, so it has to be in the
         *        same state as the GPU to prevent an extra data movement.
         */
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state = DATA_COHERENCY_SHARED;

        /**
         *  The cpu_copy will be updated in the completion, and at that moment
         *  the two versions will be identical.
         */
        cpu_copy->version = gpu_copy->version;
        gpu_copy->version++;  /* on to the next version */

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;

        if( gpu_task->pushout[i] ) {
            parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
            PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                                "OMP copy %p [ref_count %d] moved to the read LRU in %s\n",
                                gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
        } else {
            parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_owned_lru, (parsec_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

/** @brief Release the OMP copies of the data used in WRITE mode.
 *
 * @details This function can be used when the OMP task didn't run
 *          to completion on the device (either due to an error, or
 *          simply because the body requested a reexecution on a
 *          different location). It releases the OMP copies of the
 *          output data, allowing them to be reused by the runtime.
 *          This function has the drawback of kicking in too late,
 *          after all data transfers have been completed toward the
 *          device.
 *
 * @param [IN] gpu_device, the GPU device the the task has been
 *             supposed to execute.
 * @param [IN] gpu_task, the task that has been cancelled, and which
 *             needs it's data returned to the runtime.
 * @return Currently only success.
 */
int
parsec_openmp_kernel_cleanout( parsec_openmp_device_t        *gpu_device,
                            parsec_openmp_context_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_openmp_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                        "GPU[%d]: Cleanup of %s",
                        gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        if( !(gpu_task->flow[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(gpu_copy->super.super.obj_reference_count > 1);
        /* Issue #134 */
        parsec_data_copy_detach(original, gpu_copy, gpu_device->super.device_index);
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy = original->device_copies[0];

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;
        parsec_list_nolock_fifo_push(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        PARSEC_DEBUG_VERBOSE(20, parsec_openmp_output_stream,
                             "OMP copy %p [ref_count %d] moved to the read LRU in %s\n",
                             gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
    }
    return 0;
}

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on OMP events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_openmp_kernel_scheduler( parsec_execution_stream_t *es,
                            parsec_openmp_context_t    *gpu_task,
                            int which_gpu )
{
    parsec_openmp_device_t* gpu_device;
    int rc, exec_stream = 0;
    parsec_openmp_context_t *progress_task, *out_task_push, *out_task_submit = NULL, *out_task_pop = NULL;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    gpu_device = (parsec_openmp_device_t*)parsec_devices_get(which_gpu);

#if defined(PARSEC_PROF_TRACE)
    PARSEC_PROFILING_TRACE_FLAGS( es->es_profile,
                                  PARSEC_PROF_FUNC_KEY_END(gpu_task->ec->taskpool,
                                                           gpu_task->ec->task_class->task_class_id),
                                  gpu_task->ec->task_class->key_functions->key_hash(gpu_task->ec->task_class->make_key(gpu_task->ec->taskpool, gpu_task->ec->locals), 64, NULL),
                                  gpu_task->ec->taskpool->taskpool_id, NULL,
                                  PARSEC_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(PARSEC_PROF_TRACE) */

    /* Check the GPU status */
    rc = parsec_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        parsec_fifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }

#if defined(PARSEC_PROF_TRACE)
    if( parsec_omp_trackable_events & PARSEC_PROFILE_OMP_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_omp_own_GPU_key_start,
                               (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

    omp_set_default_device( gpu_device->omp_index );

#pragma omp parallel num_threads(gpu_device->max_exec_streams) proc_bind(master)
#pragma omp master
{
  check_in_deps:
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_openmp_output_stream,
                            "GPU[%1d]:\tUpload data (if any) for %s priority %d",
                            gpu_device->omp_index,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[0]),
                          parsec_openmp_kernel_push,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    gpu_task = progress_task;
    out_task_push = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "GPU[%1d]:\tExecute %s priority %d", gpu_device->omp_index,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[2+exec_stream]),
                          NULL,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( PARSEC_HOOK_RETURN_DISABLE == rc )
            goto disable_gpu;
        if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
            /* Reschedule the task. As the chore_id has been modified,
               another incarnation of the task will be executed. */
            if( NULL != progress_task ) {
                __parsec_reschedule(es, progress_task->ec);
                parsec_openmp_kernel_cleanout(gpu_device, progress_task);
                gpu_task = progress_task;
                progress_task = NULL;
                goto remove_gpu_task;
            }
            gpu_task = NULL;
            goto fetch_task_from_shared_queue;
        }
        progress_task = NULL;
    }
    gpu_task = progress_task;
    out_task_submit = progress_task;

    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "GPU[%1d]:\tRetrieve data (if any) for %s priority %d", gpu_device->omp_index,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    if (out_task_submit == NULL && out_task_push == NULL) {
        gpu_task = parsec_openmp_create_W2R_task(gpu_device, es);
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[1]),
                          parsec_openmp_kernel_pop,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a succesfully completed task. However, it is not gpu_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        gpu_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    gpu_task = progress_task;
    out_task_pop = progress_task;

  fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    if (out_task_submit == NULL && out_task_pop == NULL) {
        parsec_openmp_sort_pending_list(gpu_device);
    }
    gpu_task = (parsec_openmp_context_t*)parsec_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "GPU[%1d]:\tGet from shared queue %s priority %d", gpu_device->omp_index,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    goto check_in_deps;

  complete_task:
    assert( NULL != gpu_task );
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "GPU[%1d]:\tComplete %s priority %d", gpu_device->omp_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                        gpu_task->ec->priority );
    /* Everything went fine so far, the result is correct and back in the main memory */
    PARSEC_LIST_ITEM_SINGLETON(gpu_task);
    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        parsec_openmp_W2R_task_fini(gpu_device, gpu_task, es);
        gpu_task = progress_task;
        goto fetch_task_from_shared_queue;
    }
    parsec_openmp_kernel_epilog( gpu_device, gpu_task );
    __parsec_complete_execution( es, gpu_task->ec );
    gpu_device->super.executed_tasks++;
  remove_gpu_task:
    parsec_device_load[gpu_device->super.device_index] -= parsec_device_sweight[gpu_device->super.device_index];
    free( gpu_task );
    rc = parsec_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(PARSEC_PROF_TRACE)
        if( parsec_omp_trackable_events & PARSEC_PROFILE_OMP_TRACK_OWN )
            PARSEC_PROFILING_TRACE( es->es_profile, parsec_omp_own_GPU_key_end,
                                   (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

        rc = PARSEC_HOOK_RETURN_ASYNC;
        goto fini;
    }
    gpu_task = progress_task;
    goto fetch_task_from_shared_queue;

  disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    parsec_warning("Critical issue related to the GPU discovered. Giving up\n");
    rc = PARSEC_HOOK_RETURN_DISABLE;
  fini:
    (void)rc;
} // omp parallel
    return rc;
}

#endif /* PARSEC_HAVE_OPENMP */
