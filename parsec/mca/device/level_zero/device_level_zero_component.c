/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

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

#include <level_zero/ze_api.h>

PARSEC_OBJ_CLASS_INSTANCE(parsec_device_level_zero_module_t, parsec_device_module_t, NULL, NULL);

static int device_level_zero_component_open(void);
static int device_level_zero_component_close(void);
static int device_level_zero_component_query(mca_base_module_2_0_0_t **module, int *priority);
static int device_level_zero_component_register(void);

int parsec_device_level_zero_index, parsec_device_level_zero_enabled;
int parsec_level_zero_max_streams = PARSEC_GPU_MAX_STREAMS;
int parsec_level_zero_memory_block_size, parsec_level_zero_memory_percentage, parsec_level_zero_memory_number_of_blocks;
char* parsec_level_zero_lib_path = NULL;

static int level_zero_mask, level_zero_nvlink_mask;
static int parsec_level_zero_sort_pending;

#if defined(PARSEC_PROF_TRACE)
int parsec_device_level_zero_one_profiling_stream_per_gpu_stream = 0;
#endif /* PARSEC_PROF_TRACE */

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
parsec_device_base_component_t parsec_device_level_zero_component = {
    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_DEVICE_BASE_VERSION_2_0_0,

        /* Component name and version */
        "level_zero",
        /* Component options */
#if defined(PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS)
        "+peer_access"
#endif
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        device_level_zero_component_open,
        device_level_zero_component_close,
        device_level_zero_component_query,
        /*< specific query to return the module and add it to the list of available modules */
        device_level_zero_component_register,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    },
    NULL
};
 
mca_base_component_t * device_level_zero_static_component(void)
{
    return (mca_base_component_t *)&parsec_device_level_zero_component;
}
 
static int device_level_zero_component_query(mca_base_module_t **module, int *priority)
{
    int i, j, rc;
    ze_result_t ze_rc;
    ze_device_handle_t *devices = NULL;
    ze_driver_handle_t *allDrivers = NULL;

    *module = NULL;
    *priority = 0;
    if( 0 == parsec_device_level_zero_enabled ) {
        return MCA_SUCCESS;
    }
#if defined(PARSEC_PROF_TRACE)
    parsec_device_init_profiling();
#endif  /* defined(PROFILING) */

    if( parsec_device_level_zero_enabled >= 1) {
        uint32_t driverCount = 0;
        uint32_t totalDeviceCount = 0, maxDeviceCount = 0;

        // Discover all the driver instances
        ze_rc = zeDriverGet(&driverCount, NULL);
        PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDriverGet ", ze_rc, { return MCA_ERR_NOT_AVAILABLE; } );

        allDrivers = malloc(driverCount * sizeof(ze_driver_handle_t));
        ze_rc = zeDriverGet(&driverCount, allDrivers);
        PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDriverGet ", ze_rc, { free(allDrivers); return MCA_ERR_NOT_AVAILABLE; } );

        for(uint32_t did = 0; did < driverCount; ++did ) {
            uint32_t deviceCount = 0;
            zeDeviceGet(allDrivers[did], &deviceCount, NULL);
            totalDeviceCount += deviceCount;
            if(maxDeviceCount < deviceCount)
                maxDeviceCount = deviceCount;
        }

        parsec_device_level_zero_enabled = totalDeviceCount < (uint32_t)parsec_device_level_zero_enabled ? (int)totalDeviceCount : parsec_device_level_zero_enabled;

        if(parsec_device_level_zero_enabled > 0) {
            parsec_device_level_zero_component.modules =
                    (parsec_device_module_t **)calloc(parsec_device_level_zero_enabled + 1,
                                                      sizeof(parsec_device_module_t *));
            devices = (ze_device_handle_t *)malloc(maxDeviceCount * sizeof(ze_device_handle_t));

            i = j = 0;
            for(uint32_t did = 0; i < parsec_device_level_zero_enabled && did < driverCount; ++did ) {
                uint32_t deviceCount = maxDeviceCount;
                parsec_device_level_zero_driver_t *driver = malloc(sizeof(parsec_device_level_zero_driver_t));
                
                driver->ze_driver = allDrivers[did];
                driver->ref_count = 0;
                driver->swp = NULL;

                // Create context
                ze_context_desc_t ctxtDesc = {
                    ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                    NULL,
                    0
                };
                ze_result_t ze_rc = zeContextCreate(driver->ze_driver, &ctxtDesc, &driver->ze_context);
                PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeContextCreate ", ze_rc, { free(driver); continue; } );

                ze_rc = zeDeviceGet(allDrivers[did], &deviceCount, devices);
                PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDeviceGet ", ze_rc, { zeContextDestroy(driver->ze_context); free(driver); continue; } );

                for(uint32_t devid = 0; i < parsec_device_level_zero_enabled && devid < deviceCount; devid++) {
                    ze_device_properties_t device_properties;
                    ze_rc = zeDeviceGetProperties(devices[devid], &device_properties);
                    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDeviceGetProperties ", ze_rc, { continue; } );
                    if( ZE_DEVICE_TYPE_GPU != device_properties.type) { continue; }
                    if( !((1 << i) & level_zero_mask) ) { i++; continue; }
                    rc = parsec_level_zero_module_init(i, driver, devices[devid], &device_properties,
                                                       &parsec_device_level_zero_component.modules[j]);
                    if( PARSEC_SUCCESS != rc ) {
                        assert( NULL == parsec_device_level_zero_component.modules[j] );
                        continue;
                    }
                    driver->ref_count++;
                    parsec_device_level_zero_component.modules[j]->component = &parsec_device_level_zero_component;
                    if(parsec_device_level_zero_sort_pending) {
                        parsec_device_level_zero_component.modules[j]->sort_pending_list = parsec_device_sort_pending_list;
                    }
                    j++;  /* next available spot */
                    parsec_device_level_zero_component.modules[j] = NULL;
                    i++;
                }

                if( driver->ref_count == 0 ) {
                    zeContextDestroy(driver->ze_context);
                    free(driver);
                }
            }
            parsec_device_level_zero_enabled = j;
            if(0 == parsec_device_level_zero_enabled) {
                free( parsec_device_level_zero_component.modules ); 
            }
        }
    } 
    
    if(0 == parsec_device_level_zero_enabled )
        parsec_device_level_zero_component.modules = NULL;

    if(NULL != devices)
        free(devices);
    if(NULL != allDrivers)
        free(allDrivers);

    /* SYCL wrappers must be built once all devices and queues and contexts and drivers are up,
     * and they must be built in the following order: driver wrapper, then device wrappers for this
     * driver, then context wrapper for this driver, then queue wrappers for each queue.
     * Also, it MUST be one to one: a single device wrapper per device, a single context wrapper
     * per context etc... */
    for(int did = 0; did < parsec_device_level_zero_enabled; did++) {
        if(NULL == parsec_device_level_zero_component.modules[did])
            continue;
        parsec_device_level_zero_module_t *module = (parsec_device_level_zero_module_t *)parsec_device_level_zero_component.modules[did];
        if(NULL != module->driver->swp) 
            continue;

        module->driver->swp = parsec_sycl_wrapper_platform_create(module->driver->ze_driver);
        int nbdev = 1;
        for(int i = 1; did + i < parsec_device_level_zero_enabled; i++) {
            if( ((parsec_device_level_zero_module_t *)parsec_device_level_zero_component.modules[did+i])->driver == module->driver )
                nbdev++;
        }
        parsec_sycl_wrapper_device_t **devices = (parsec_sycl_wrapper_device_t **)malloc(nbdev*sizeof(parsec_sycl_wrapper_device_t*));
        for(int i = 0, j = 0; j < nbdev; i++) {
            parsec_device_level_zero_module_t *mod2 = (parsec_device_level_zero_module_t *)parsec_device_level_zero_component.modules[did+i];
            if( mod2->driver == module->driver ) {
                mod2->swd = parsec_sycl_wrapper_device_create(mod2->ze_device);
                devices[j++] = mod2->swd;
            }
        }
        parsec_sycl_wrapper_platform_add_context(module->driver->swp, module->driver->ze_context, devices, nbdev);
	    free(devices);
        for(int i = 0, j = 0; j < nbdev; i++) {
            parsec_device_level_zero_module_t *mod2 = (parsec_device_level_zero_module_t *)parsec_device_level_zero_component.modules[did+i];
            if( mod2->driver == module->driver ) {
                j++;
                for(int s = 0; s < mod2->super.max_exec_streams; s++) {
                    parsec_level_zero_exec_stream_t *exec_stream = (parsec_level_zero_exec_stream_t *)mod2->super.exec_stream[s];
		            if(NULL != exec_stream->level_zero_cq)
                        exec_stream->swq = parsec_sycl_wrapper_queue_create(module->driver->swp, mod2->swd, exec_stream->level_zero_cq);
                }
            }
        }
    }

    parsec_device_enable_debug();

    /* module type should be: const mca_base_module_t ** */
    void *ptr = parsec_device_level_zero_component.modules;
    *priority = 10;
    *module = (mca_base_module_t *)ptr;
    
    return MCA_SUCCESS;
}

static int device_level_zero_component_register(void)
{
    parsec_device_level_zero_index = parsec_mca_param_reg_int_name("device_level_zero", "enabled",
                                                   "The number of LEVEL_ZERO device to enable for the next PaRSEC context (-1 for all available)",
                                                   false, false, -1, &parsec_device_level_zero_enabled);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "mask",
                                        "The bitwise mask of LEVEL_ZERO devices to be enabled (default all)",
                                        false, false, 0xffffffff, &level_zero_mask);
     (void)parsec_mca_param_reg_int_name("device_level_zero", "nvlink_mask",
                                        "What devices are allowed to use NVLINK if available (default all)",
                                        false, false, 0xffffffff, &level_zero_nvlink_mask);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "verbose",
                                        "Set the verbosity level of the LEVEL_ZERO device (negative value: use debug verbosity), higher is less verbose)\n",
                                        false, false, -1, &parsec_gpu_verbosity);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "memory_block_size",
                                        "The LEVEL_ZERO memory page for PaRSEC internal management (in bytes).",
                                        false, false, 512*1024, &parsec_level_zero_memory_block_size);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "memory_use",
                                        "The percentage of the total GPU memory to be used by this PaRSEC context",
                                        false, false, 95, &parsec_level_zero_memory_percentage);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "memory_number_of_blocks",
                                        "Alternative to device_level_zero_memory_use: sets exactly the number of blocks to allocate (-1 means to use a percentage of the available memory)",
                                        false, false, -1, &parsec_level_zero_memory_number_of_blocks);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "max_number_of_ejected_data",
                                        "Sets up the maximum number of blocks that can be ejected from GPU memory",
                                        false, false, MAX_PARAM_COUNT, &parsec_gpu_d2h_max_flows);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "max_streams",
                                        "Maximum number of Streams to use for the GPU engine; 2 streams are used for communication between host and device, so the minimum is 3",
                                        false, false, PARSEC_GPU_MAX_STREAMS, &parsec_level_zero_max_streams);
    (void)parsec_mca_param_reg_int_name("device_level_zero", "sort_pending_tasks",
                                        "Boolean to let the GPU engine sort the first pending tasks stored in the list",
                                        false, false, 0, &parsec_level_zero_sort_pending);
#if defined(PARSEC_PROF_TRACE)
    (void)parsec_mca_param_reg_int_name("device_level_zero", "one_profiling_stream_per_level_zero_stream",
                                        "Boolean to separate the profiling of each level_zero stream into a single profiling stream",
                                        false, false, 0, &parsec_device_level_zero_one_profiling_stream_per_gpu_stream);
#endif
    /* If LEVEL_ZERO was not requested avoid initializing the devices */
    return (0 == parsec_device_level_zero_enabled ? MCA_ERROR : MCA_SUCCESS);
}

/**
 * Open LEVEL_ZERO and check that devices are available and ready to be used. This operation should
 * only be done once during the initialization, and the devices should from there on be managed
 * by PaRSEC.
 */
static int device_level_zero_component_open(void)
{
    ze_result_t ze_rc;
    ze_driver_handle_t *allDrivers = NULL;
    int ndevices = 0;
    uint32_t driverCount = 0;

    if( 0 == parsec_device_level_zero_enabled ) {
        return MCA_ERROR;  /* Nothing to do around here */
    }

    ze_rc = zeInit( 0 );
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeInit ", ze_rc,
                                   {
                                       parsec_mca_param_set_int(parsec_device_level_zero_index, 0);
                                       return MCA_ERROR;
                                   } );

    // Discover all the driver instances
    ze_rc = zeDriverGet(&driverCount, NULL);
    PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDriverGet ", ze_rc,
                                   {
                                       parsec_mca_param_set_int(parsec_device_level_zero_index, 0);
                                       return MCA_ERROR;
                                   } );
    if(driverCount > 0) {
        allDrivers = malloc(driverCount * sizeof(ze_driver_handle_t));
        ze_rc = zeDriverGet(&driverCount, allDrivers);
        PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDriverGet ", ze_rc,
                                       {
                                           free(allDrivers);
                                           parsec_mca_param_set_int(parsec_device_level_zero_index, 0);
                                           return MCA_ERROR;
                                       } );

        for( uint32_t did = 0; did < driverCount; ++did ) {
            uint32_t deviceCount = 0;
            ze_rc = zeDeviceGet(allDrivers[did], &deviceCount, NULL);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeDeviceGet ", ze_rc,
                                           {
                                               free(allDrivers);
                                               parsec_mca_param_set_int(parsec_device_level_zero_index, 0);
                                               return MCA_ERROR;
                                           } );
            ndevices += (int)deviceCount;
        }

        free(allDrivers);
    }


    if( ndevices > parsec_device_level_zero_enabled ) {
        if( 0 < parsec_device_level_zero_index ) {
            ndevices = parsec_device_level_zero_enabled;
        }
    } else if (ndevices < parsec_device_level_zero_enabled ) {
        if( 0 < parsec_device_level_zero_index ) {
            parsec_warning("User requested %d LEVEL_ZERO devices, but only %d are available on %s\n."
                           " PaRSEC will enable all %d of them.",
                           parsec_device_level_zero_enabled, ndevices, parsec_hostname, ndevices);
            parsec_mca_param_set_int(parsec_device_level_zero_index, ndevices);
        }
    }

    /* Update the number of GPU for the upper layer */
    parsec_device_level_zero_enabled = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

    return MCA_SUCCESS;
}

/**
 * Remove all LEVEL_ZERO devices from the PaRSEC available devices, and turn them off.
 * At the end of this function all LEVEL_ZERO initialization in the context of PaRSEC
 * should be undone, and pending tasks either completed or transferred to another
 * chore (if available), and all LEVEL_ZERO resources (fences, streams and memory)
 * released.
 */
static int device_level_zero_component_close(void)
{
    parsec_device_level_zero_module_t* cdev;
    int i, rc;

    if( NULL == parsec_device_level_zero_component.modules ) {  /* No devices */
        return PARSEC_SUCCESS;
    }

    for( i = 0; NULL != (cdev = (parsec_device_level_zero_module_t*)parsec_device_level_zero_component.modules[i]); i++ ) {
        parsec_device_level_zero_component.modules[i] = NULL;

        rc = parsec_level_zero_module_fini((parsec_device_module_t*)cdev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_gpu_output_stream,
                                 "GPU[%d] Failed to release resources on LEVEL_ZERO device\n", 
                                 cdev->level_zero_index);
        }
        if(0 == --cdev->driver->ref_count) {
            ze_result_t ze_rc = zeContextDestroy(cdev->driver->ze_context);
            PARSEC_LEVEL_ZERO_CHECK_ERROR( "zeContextDestroy ", ze_rc, {});
            parsec_sycl_wrapper_platform_destroy(cdev->driver->swp);
        }

        /* unregister the device from PaRSEC */
        rc = parsec_mca_device_remove((parsec_device_module_t*)cdev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_gpu_output_stream,
                                 "GPU[%d] Failed to unregister LEVEL_ZERO device %d\n", 
                                 cdev->level_zero_index, cdev->level_zero_index);
        }

        free(cdev);
    }

#if defined(PARSEC_DEBUG_NOISIER)
    /* Check that no LEVEL_ZERO devices are still registered with PaRSEC */
    for(i = 0; i < parsec_mca_device_enabled(); i++) {
        if( NULL == (cdev = (parsec_device_level_zero_module_t*)parsec_mca_device_get(i)) ) continue;
        if(PARSEC_DEV_LEVEL_ZERO != cdev->super.super.type) continue;

        PARSEC_DEBUG_VERBOSE(0, parsec_gpu_output_stream,
                             "GPU[%d] LEVEL_ZERO device still registered with PaRSEC at the end of LEVEL_ZERO finalize.\n"
                             " Please contact the developers or fill an issue.\n", 
                             cdev->level_zero_index);
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    if( parsec_device_output != parsec_gpu_output_stream )
        parsec_output_close(parsec_gpu_output_stream);
    parsec_gpu_output_stream = parsec_device_output;

    if ( parsec_level_zero_lib_path ) {
        free(parsec_level_zero_lib_path);
    }

    return PARSEC_SUCCESS;
}
