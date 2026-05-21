/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/comm/ucx/comm_ucx.h"

static int comm_ucx_component_query(mca_base_module_t **module, int *priority);

static parsec_comm_module_t parsec_comm_ucx_module = {
    .component = &parsec_comm_ucx_component,
    .module = {
        .init = comm_ucx_init,
    },
    .datatype = &parsec_datatype_basic_module,
};

const parsec_comm_base_component_t parsec_comm_ucx_component = {
    {
        PARSEC_COMM_BASE_VERSION_2_0_0,

        "ucx",
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        NULL,
        NULL,
        comm_ucx_component_query,
        NULL,
        "",
    },
    {
        MCA_BASE_METADATA_PARAM_NONE,
        "",
    }
};

mca_base_component_t *
comm_ucx_static_component(void)
{
    return (mca_base_component_t *)&parsec_comm_ucx_component;
}

static int
comm_ucx_component_query(mca_base_module_t **module, int *priority)
{
    /*
     * Keep MPI as the default when both backends eventually become buildable
     * together.  UCX can be selected explicitly with the comm MCA parameter.
     */
    *priority = 50;
    *module = (mca_base_module_t *)&parsec_comm_ucx_module;
    return MCA_SUCCESS;
}
