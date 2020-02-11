/*
 * Copyright (c) 2019      The University of Tennessee and The University
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
#include "parsec/mca/device/template/device_template.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/scheduling.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/class/fifo.h"

static int device_template_component_open(void);
static int device_template_component_close(void);
static int device_template_component_query(mca_base_module_2_0_0_t **module, int *priority);
static int device_template_component_register(void);

int use_template_index, use_template;
int template_verbosity;
int parsec_template_output_stream = -1;

#if defined(PARSEC_PROF_TRACE)
int parsec_template_key_start;
int parsec_template_key_end;
#endif  /* defined(PROFILING) */

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
parsec_device_base_component_t parsec_device_template_component = {
    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_DEVICE_BASE_VERSION_2_0_0,

        /* Component name and version */
        "template",
#if defined(PARSEC_SOME_SPECIAL_BUILD_OPTIONS)
        "+build_option"
#endif
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        device_template_component_open,
        device_template_component_close,
        device_template_component_query,
        /*< specific query to return the module and add it to the list of available modules */
        device_template_component_register,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    },
    NULL
};
 
mca_base_component_t * device_template_static_component(void)
{
    return (mca_base_component_t *)&parsec_device_template_component;
}
 
static int
device_template_component_query(mca_base_module_t **module, int *priority)
{
    int i, j, rc;

#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_add_dictionary_keyword( "template", "fill:#66ff66",
                                             0, NULL,
                                             &parsec_template_key_start, &parsec_template_key_end);
#endif  /* defined(PROFILING) */

    parsec_device_template_component.modules = (parsec_device_module_t**)calloc(use_template + 1, sizeof(parsec_device_module_t*));

    for( i = j = 0; i < use_template; i++ ) {

        rc = parsec_device_template_module_init(i, &parsec_device_template_component.modules[j]);
        if( PARSEC_SUCCESS != rc ) {
            assert( NULL == parsec_device_template_component.modules[j] );
            continue;
        }
        parsec_device_template_component.modules[j]->component = &parsec_device_template_component;
        j++;  /* next available spot */
        parsec_device_template_component.modules[j] = NULL;
    }

    /* module type should be: const mca_base_module_t ** */
    void *ptr = parsec_device_template_component.modules;
    *priority = 10;
    *module = (mca_base_module_t *)ptr;

    return MCA_SUCCESS;
}

static int device_template_component_register(void)
{
    use_template_index = parsec_mca_param_reg_int_name("device_template", "enabled",
                                                       "The number of TEMPLATE device to enable for the next PaRSEC context (-1 for all available)",
                                                       false, false, -1, &use_template);
    (void)parsec_mca_param_reg_int_name("device_template", "verbose",
                                        "Set the verbosity level of the TEMPLATE device (negative value: use debug verbosity), higher is less verbose)\n",
                                        false, false, -1, &template_verbosity);

    /* If TEMPLATE was not requested avoid initializing the devices */
    return (0 == use_template ? MCA_ERROR : MCA_SUCCESS);
}

/**
 * Open TEMPLATE and check that devices are available and ready to be used. This operation should
 * only be done once during the initialization, and the devices should from there on be managed
 * by PaRSEC.
 */
static int device_template_component_open(void)
{
    if( 0 == use_template ) {
        return MCA_ERROR;  /* Nothing to do around here */
    }

    parsec_template_output_stream = parsec_device_output;
    if( template_verbosity >= 0 ) {
        parsec_template_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_template_output_stream, template_verbosity);
    }

    return MCA_SUCCESS;
}

/**
 * Remove all TEMPLATE devices from the PaRSEC available devices, and turn them off.
 * At the end of this function all TEMPLATE initialization in the context of PaRSEC
 * should be undone, and pending tasks either completed or transferred to another
 * chore (if available), and all resources released.
 */
static int device_template_component_close(void)
{
    parsec_device_template_module_t* dev;
    int i, rc;

    if( NULL == parsec_device_template_component.modules ) {  /* No devices */
        return PARSEC_SUCCESS;
    }

    for( i = 0; NULL != (dev = (parsec_device_template_module_t*)parsec_device_template_component.modules[i]); i++ ) {
        parsec_device_template_component.modules[i] = NULL;

        /* unregister the device from PaRSEC */
        rc = parsec_mca_device_remove((parsec_device_module_t*)dev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_template_output_stream,
                                 "T[%d] Failed to unregister TEMPLATE device %d\n", 
                                 dev->device_index, dev->device_index);
        }

        rc = parsec_device_template_module_fini((parsec_device_module_t*)dev);
        if( PARSEC_SUCCESS != rc ) {
            PARSEC_DEBUG_VERBOSE(0, parsec_template_output_stream,
                                 "T[%d] Failed to release resources on TEMPLATE device\n", 
                                 dev->device_index);
        }
        free(dev);
    }

    if( parsec_device_output != parsec_template_output_stream )
        parsec_output_close(parsec_template_output_stream);
    parsec_template_output_stream = parsec_device_output;

    return PARSEC_SUCCESS;
}
