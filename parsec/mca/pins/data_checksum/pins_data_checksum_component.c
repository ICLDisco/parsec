/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
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
#include "parsec/parsec_config.h"
#include "parsec/runtime.h"

#include "parsec/mca/pins/pins.h"
#include "parsec/mca/pins/data_checksum/pins_data_checksum.h"
#include "parsec/utils/mca_param.h"

/*
 * Local function
 */
static int pins_data_checksum_component_query(mca_base_module_t **module, int *priority);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_pins_base_component_t parsec_pins_data_checksum_component = {

    /* First, the mca_component_t struct containing meta information
     about the component itself */

    {
        PARSEC_PINS_BASE_VERSION_2_0_0,

        /* Component name and version */
        "data_checksum",
        "", /* options */
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL,
        NULL,
        pins_data_checksum_component_query,
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

mca_base_component_t * pins_data_checksum_static_component(void)
{
    return (mca_base_component_t *)&parsec_pins_data_checksum_component;
}

int parsec_pins_data_checksum_trace = 0;

static int pins_data_checksum_component_query(mca_base_module_t **module, int *priority)
{
    parsec_mca_param_reg_int_name("pins", "data_checksum_trace",
                                  "What to do beyond verifying read-only flows across the "
                                  "body, as a sum of:\n"
                                  "  1 -- print a fingerprint of every tile each task reads "
                                  "and writes. Two runs of a deterministic program can then "
                                  "be compared to find the first task that saw different "
                                  "data, including the purely local ones no transfer log "
                                  "can show.\n"
                                  "  2 -- verify read-only flows a second time when the task "
                                  "completes, which covers the window between the body "
                                  "returning and the dependencies being released. For a task "
                                  "that ran on a device that window holds the whole "
                                  "asynchronous execution and its stage-out.\n",
                                  false, false,
                                  parsec_pins_data_checksum_trace,
                                  &parsec_pins_data_checksum_trace);

    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_pins_data_checksum_module;
    *priority = 50;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}
