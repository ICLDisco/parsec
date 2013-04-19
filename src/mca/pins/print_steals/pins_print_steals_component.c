/*
 * Copyright (c) 2013      The University of Tennessee and The University
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

#include "dague_config.h"
#include "dague.h"

#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/print_steals/pins_print_steals.h"

/*
 * Local function
 */
static int pins_print_steals_component_query(mca_base_module_t **module, int *priority);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const dague_pins_base_component_t dague_pins_print_steals_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        DAGUE_PINS_BASE_VERSION_2_0_0,

        /* Component name and version */
        "print_steals",
        DAGUE_VERSION_MAJOR,
        DAGUE_VERSION_MINOR,

        /* Component open and close functions */
        NULL, 
        NULL, 
        pins_print_steals_component_query, 
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
mca_base_component_t * pins_print_steals_static_component(void)
{
    return (mca_base_component_t *)&dague_pins_print_steals_component;
}

static int pins_print_steals_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&dague_pins_print_steals_module;
    *priority = 6;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}

