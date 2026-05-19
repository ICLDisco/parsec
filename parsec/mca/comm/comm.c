/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/comm/comm.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/utils/debug.h"
#include <assert.h>

static parsec_comm_base_component_t *parsec_comm_selected_component = NULL;

parsec_comm_engine_t *
parsec_comm_engine_component_init(parsec_context_t *context)
{
    mca_base_component_t **components;
    mca_base_module_t *selected_module = NULL;
    mca_base_component_t *selected_component = NULL;
    parsec_comm_engine_t *ce;

    assert(NULL == parsec_comm_selected_component);

    /*
     * Query all compiled and user-enabled comm components, close every component
     * that was not selected, and keep the selected component open until
     * parsec_comm_engine_component_fini().
     */
    components = mca_components_open_bytype("comm");
    mca_components_query(components, &selected_module, &selected_component);
    mca_components_close(components);

    if( NULL == selected_module ) {
        parsec_warning("No communication engine component could be selected");
        return NULL;
    }

    parsec_comm_selected_component = (parsec_comm_base_component_t *)selected_component;

    parsec_debug_verbose(4, parsec_debug_output, "Installing communication engine %s",
                         parsec_comm_selected_component->base_version.mca_component_name);

    ce = ((parsec_comm_module_t *)selected_module)->module.init(context);
    if( NULL == ce ) {
        mca_component_close((mca_base_component_t *)parsec_comm_selected_component);
        parsec_comm_selected_component = NULL;
    }
    return ce;
}

int
parsec_comm_engine_component_fini(void)
{
    if( NULL != parsec_comm_selected_component ) {
        mca_component_close((mca_base_component_t *)parsec_comm_selected_component);
        parsec_comm_selected_component = NULL;
    }
    return PARSEC_SUCCESS;
}
