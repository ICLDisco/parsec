/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/mca/device/cuda/device_cuda.h"
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

static int device_cuda_component_open(void);
static int device_cuda_component_close(void);
static int device_cuda_component_query(mca_base_module_2_0_0_t **module, int *priority);
static int device_cuda_component_register(void);

int use_cuda_index, use_cuda;
int cuda_mask, cuda_verbosity;
int cuda_memory_block_size, cuda_memory_percentage, cuda_memory_number_of_blocks;
int parsec_cuda_output_stream = -1;
int32_t parsec_CUDA_sort_pending_list = 0;

int32_t parsec_CUDA_d2h_max_flows;
char* cuda_lib_path = NULL;

#if defined(PARSEC_PROF_TRACE)
/* Accepted values are: PARSEC_PROFILE_CUDA_TRACK_DATA_IN | PARSEC_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      PARSEC_PROFILE_CUDA_TRACK_OWN | PARSEC_PROFILE_CUDA_TRACK_EXEC |
 *                      PARSEC_PROFILE_CUDA_TRACK_MEM_USE | PARSEC_PROFILE_CUDA_TRACK_PREFETCH
 */
int parsec_cuda_trackable_events = PARSEC_PROFILE_CUDA_TRACK_EXEC | PARSEC_PROFILE_CUDA_TRACK_DATA_OUT
    | PARSEC_PROFILE_CUDA_TRACK_DATA_IN | PARSEC_PROFILE_CUDA_TRACK_OWN | PARSEC_PROFILE_CUDA_TRACK_MEM_USE
    | PARSEC_PROFILE_CUDA_TRACK_PREFETCH;
int parsec_cuda_movein_key_start;
int parsec_cuda_movein_key_end;
int parsec_cuda_moveout_key_start;
int parsec_cuda_moveout_key_end;
int parsec_cuda_own_GPU_key_start;
int parsec_cuda_own_GPU_key_end;
int parsec_cuda_allocate_memory_key;
int parsec_cuda_free_memory_key;
int parsec_cuda_use_memory_key_start;
int parsec_cuda_use_memory_key_end;
int parsec_cuda_prefetch_key_start;
int parsec_cuda_prefetch_key_end;
#endif  /* defined(PROFILING) */

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
parsec_device_base_component_t parsec_device_cuda_component = {
    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_DEVICE_BASE_VERSION_2_0_0,

        /* Component name and version */
        "cuda",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        device_cuda_component_open,
        device_cuda_component_close,
        device_cuda_component_query,
        /*< specific query to return the module and add it to the list of available modules */
        device_cuda_component_register,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    },
    NULL
};
 
mca_base_component_t * device_cuda_static_component(void)
{
    return (mca_base_component_t *)&parsec_device_cuda_component;
}
 
static int device_cuda_component_query(mca_base_module_t **module, int *priority)
{
    int i, j, rc;

    *module = NULL;
    *priority = 0;
    if( 0 == use_cuda ) {
        return MCA_SUCCESS;
    }
#if defined(PARSEC_PROF_TRACE)
      parsec_profiling_add_dictionary_keyword( "cuda", "fill:#66ff66",
                                               0, NULL,
                                               &parsec_cuda_own_GPU_key_start, &parsec_cuda_own_GPU_key_end);
      parsec_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                               sizeof(parsec_profile_data_collection_info_t), PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                               &parsec_cuda_movein_key_start, &parsec_cuda_movein_key_end);
     parsec_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                              sizeof(parsec_profile_data_collection_info_t), PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                              &parsec_cuda_moveout_key_start, &parsec_cuda_moveout_key_end);
    parsec_profiling_add_dictionary_keyword( "prefetch", "fill:#66ff66",
                                             sizeof(parsec_profile_data_collection_info_t), PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                             &parsec_cuda_prefetch_key_start, &parsec_cuda_prefetch_key_end);
    parsec_profiling_add_dictionary_keyword( "cuda_mem_alloc", "fill:#FF66FF",
                                             sizeof(int64_t), "size{int64_t}",
                                             &parsec_cuda_allocate_memory_key, &parsec_cuda_free_memory_key);
    parsec_profiling_add_dictionary_keyword( "cuda_mem_use", "fill:#FF66FF",
                                             sizeof(int64_t), "size{int64_t}",
                                             &parsec_cuda_use_memory_key_start, &parsec_cuda_use_memory_key_end);
#endif  /* defined(PROFILING) */

    parsec_device_cuda_component.modules = (parsec_device_module_t**)calloc(use_cuda + 1, sizeof(parsec_device_module_t*));

    for( i = j = 0; i < use_cuda; i++ ) {

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & cuda_mask) ) continue;

        rc = parsec_cuda_module_init(i, &parsec_device_cuda_component.modules[j]);
        if( PARSEC_SUCCESS != rc ) {
            assert( NULL == parsec_device_cuda_component.modules[j] );
            continue;
        }
        parsec_device_cuda_component.modules[j]->component = &parsec_device_cuda_component;
        j++;  /* next available spot */
        parsec_device_cuda_component.modules[j] = NULL;
    }

#if defined(PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    parsec_device_cuda_module_t *source_gpu, *target_gpu;
    cudaError_t cudastatus;

    for( i = 0; NULL != (source_gpu = (parsec_device_cuda_module_t*)parsec_device_cuda_component.modules[i]); i++ ) {
        int canAccessPeer;

        for( j = 0; NULL != (target_gpu = (parsec_device_cuda_module_t*)parsec_device_cuda_component.modules[j]); j++ ) {
            if( i == j ) continue;

            /* Communication mask */
            cudastatus = cudaDeviceCanAccessPeer( &canAccessPeer, source_gpu->cuda_index, target_gpu->cuda_index );
            PARSEC_CUDA_CHECK_ERROR( "(parsec_device_cuda_component_query) cudaDeviceCanAccessPeer ", cudastatus,
                                     {continue;} );
            if( 1 == canAccessPeer ) {
                cudastatus = cudaDeviceEnablePeerAccess( target_gpu->cuda_index, 0 );
                PARSEC_CUDA_CHECK_ERROR( "(parsec_device_cuda_ciomponent_query) cuCtxEnablePeerAccess ", cudastatus,
                                         {continue;} );
                source_gpu->peer_access_mask = (int16_t)(source_gpu->peer_access_mask | (int16_t)(1 << target_gpu->cuda_index));
            }
        }
    }
#endif

    /* module type should be: const mca_base_module_t ** */
    void *ptr = parsec_device_cuda_component.modules;
    *priority = 10;
    *module = (mca_base_module_t *)ptr;

    return MCA_SUCCESS;
}

static int device_cuda_component_register(void)
{
    use_cuda_index = parsec_mca_param_reg_int_name("device_cuda", "enabled",
                                                   "The number of CUDA device to enable for the next PaRSEC context (-1 for all available)",
                                                   false, false, -1, &use_cuda);
    (void)parsec_mca_param_reg_int_name("device_cuda", "mask",
                                        "The bitwise mask of CUDA devices to be enabled (default all)",
                                        false, false, 0xffffffff, &cuda_mask);
    (void)parsec_mca_param_reg_int_name("device_cuda", "verbose",
                                        "Set the verbosity level of the CUDA device (negative value: use debug verbosity), higher is less verbose)\n",
                                        false, false, -1, &cuda_verbosity);
    (void)parsec_mca_param_reg_string_name("device_cuda", "path",
                                           "Path to the shared library files containing the CUDA version of the hooks. It is a ;-separated list of either directories or .so files.\n",
                                           false, false, PARSEC_LIB_CUDA_PREFIX, &cuda_lib_path);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_block_size",
                                        "The CUDA memory page for PaRSEC internal management (in bytes).",
                                        false, false, 512*1024, &cuda_memory_block_size);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_use",
                                        "The percentage of the total GPU memory to be used by this PaRSEC context",
                                        false, false, 95, &cuda_memory_percentage);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_number_of_blocks",
                                        "Alternative to device_cuda_memory_use: sets exactly the number of blocks to allocate (-1 means to use a percentage of the available memory)",
                                        false, false, -1, &cuda_memory_number_of_blocks);
    (void)parsec_mca_param_reg_int_name("device_cuda", "max_number_of_ejected_data",
                                        "Sets up the maximum number of blocks that can be ejected from GPU memory",
                                        false, false, MAX_PARAM_COUNT, &parsec_CUDA_d2h_max_flows);
    (void)parsec_mca_param_reg_int_name("device_cuda", "sort_pending_tasks",
                                        "Boolean to let the GPU engine sort the first pending tasks stored in the list",
                                        false, false, 0, &parsec_CUDA_sort_pending_list);

    /* If CUDA was not requested avoid initializing the devices */
    return (0 == use_cuda ? MCA_ERROR : MCA_SUCCESS);
}

/**
 * Open CUDA and check that devices are available and ready to be used. This operation should
 * only be done once during the initialization, and the devices should from there on be managed
 * by PaRSEC.
 */
static int device_cuda_component_open(void)
{
    cudaError_t cudastatus;
    int ndevices;

    if( 0 == use_cuda ) {
        return MCA_ERROR;  /* Nothing to do around here */
    }

    parsec_cuda_output_stream = parsec_device_output;
    if( cuda_verbosity >= 0 ) {
        parsec_cuda_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_cuda_output_stream, cuda_verbosity);
    }

    cudastatus = cudaGetDeviceCount( &ndevices );
    if( cudaErrorNoDevice == (cudaError_t) cudastatus ) {
        ndevices = 0;
        /* This is normal on machines with no GPUs, let it flow
         * to do the normal checks vis-a-vis the number of requested
         * devices and issue a warning only when not fulfilling
         * the user demands
         */
    }
    else {
        PARSEC_CUDA_CHECK_ERROR( "cudaGetDeviceCount ", cudastatus,
                             {
                                parsec_mca_param_set_int(use_cuda_index, 0);
                                return MCA_ERROR;
                             } );
    }

    if( ndevices > use_cuda ) {
        if( 0 < use_cuda_index ) {
            ndevices = use_cuda;
        }
    } else if (ndevices < use_cuda ) {
        if( 0 < use_cuda_index ) {
            parsec_warning("User requested %d CUDA devices, but only %d are available on %s\n."
                           " PaRSEC will enable all %d of them.",
                           use_cuda, ndevices, parsec_hostname, ndevices);
            parsec_mca_param_set_int(use_cuda_index, ndevices);
        }
    }

    /* Update the number of GPU for the upper layer */
    use_cuda = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

    return MCA_SUCCESS;
}

/**
 * Remove all CUDA devices from the PaRSEC available devices, and turn them off.
 * At the end of this function all CUDA initialization in the context of PaRSEC
 * should be undone, and pending tasks either completed or transferred to another
 * chore (if available), and all CUDA resources (events, streams and memory)
 * released.
 */
static int device_cuda_component_close(void)
{
    parsec_device_cuda_module_t* cdev;
    int i, rc;

    if( NULL == parsec_device_cuda_component.modules ) {  /* No devices */
        return PARSEC_SUCCESS;
    }

    for( i = 0; NULL != (cdev = (parsec_device_cuda_module_t*)parsec_device_cuda_component.modules[i]); i++ ) {
        parsec_device_cuda_component.modules[i] = NULL;

        rc = parsec_cuda_module_fini((parsec_device_module_t*)cdev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_cuda_output_stream,
                                 "GPU[%d] Failed to release resources on CUDA device\n", 
                                 cdev->cuda_index);
        }

        /* unregister the device from PaRSEC */
        rc = parsec_mca_device_remove((parsec_device_module_t*)cdev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_cuda_output_stream,
                                 "GPU[%d] Failed to unregister CUDA device %d\n", 
                                 cdev->cuda_index, cdev->cuda_index);
        }

        free(cdev);
    }

#if defined(PARSEC_DEBUG_NOISIER)
    /* Check that no CUDA devices are still registered with PaRSEC */
    for(i = 0; i < parsec_mca_device_enabled(); i++) {
        if( NULL == (cdev = (parsec_device_cuda_module_t*)parsec_mca_device_get(i)) ) continue;
        if(PARSEC_DEV_CUDA != cdev->super.type) continue;

        PARSEC_DEBUG_VERBOSE(0, parsec_cuda_output_stream,
                             "GPU[%d] CUDA device still registered with PaRSEC at the end of CUDA finalize.\n"
                             " Please contact the developers or fill an issue.\n", 
                             cdev->cuda_index);
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    if( parsec_device_output != parsec_cuda_output_stream )
        parsec_output_close(parsec_cuda_output_stream);
    parsec_cuda_output_stream = parsec_device_output;

    if ( cuda_lib_path ) {
        free(cuda_lib_path);
    }

    return PARSEC_SUCCESS;
}
