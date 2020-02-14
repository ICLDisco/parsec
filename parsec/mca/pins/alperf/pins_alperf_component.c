/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include <errno.h>
#include <stdio.h>
#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/mca/pins/pins.h"
#include "pins_alperf.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"
#include <string.h>
#include "parsec/sys/atomic.h"

#include "parsec/mca/pins/alperf/pins_alperf.h"
#include "parsec/dictionary.h"

/*
 * Local function
 */
static int pins_alperf_component_query(mca_base_module_t **module, int *priority);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_pins_base_component_t parsec_pins_alperf_component = {

    /* First, the mca_component_t struct containing meta information
     about the component itself */

    {
        PARSEC_PINS_BASE_VERSION_2_0_0,

        /* Component name and version */
        "alperf",
        "", /* options */
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL,
        NULL,
        pins_alperf_component_query,
        /*< specific query to return the module and add it to the list of available modules */
        NULL,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};

mca_base_component_t * pins_alperf_static_component(void)
{
    return (mca_base_component_t *)&parsec_pins_alperf_component;
}

static int pins_alperf_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_pins_alperf_module;
    *priority = 50;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}

