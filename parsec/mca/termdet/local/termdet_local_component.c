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
 * semantics.  See general comments on MCA components.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/parsec_internal.h"

#include "parsec/mca/termdet/termdet.h"
#include "parsec/mca/termdet/local/termdet_local.h"

/*
 * Local function
 */
static int termdet_local_component_query(mca_base_module_t **module, int *priority);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_termdet_base_component_t parsec_termdet_local_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_TERMDET_BASE_VERSION_2_0_0,

        /* Component name, options and version */
        "local",
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL, /*< No open: termdet_local is always available, no need to check at runtime */
        NULL, /*< No close: open did not allocate any resource, no need to release them */
        termdet_local_component_query, 
        /*< specific query to return the module and add it to the list of available modules */
        NULL, /*< No register: no parameters to the absolute priority component */
        "", /*< no reserve */
    },
    {
        /* The component has no metada */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};

mca_base_component_t *termdet_local_static_component(void)
{
    return (mca_base_component_t *)&parsec_termdet_local_component;
}

static int termdet_local_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_termdet_local_module;
    *priority = 1;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}

