/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/comm/mpi/comm_mpi.h"
#include "comm_mpi_funnelled.h"

static int comm_mpi_component_query(mca_base_module_t **module, int *priority);

/*
 * The MPI component currently wraps the pre-existing funnelled MPI engine.
 * The module init function returns the global parsec_ce populated with MPI
 * callbacks; no extra selected-component accessor is needed after init.
 */
static parsec_comm_module_t parsec_comm_mpi_module = {
    .component = &parsec_comm_mpi_component,
    .module = {
        .init = mpi_funnelled_init,
    },
};

const parsec_comm_base_component_t parsec_comm_mpi_component = {
    {
        PARSEC_COMM_BASE_VERSION_2_0_0,

        "mpi",
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        NULL,
        NULL,
        comm_mpi_component_query,
        NULL,
        "",
    },
    {
        MCA_BASE_METADATA_PARAM_NONE,
        "",
    }
};

mca_base_component_t *
comm_mpi_static_component(void)
{
    return (mca_base_component_t *)&parsec_comm_mpi_component;
}

static int
comm_mpi_component_query(mca_base_module_t **module, int *priority)
{
    /*
     * MPI is the only comm component in this first componentization step, so it
     * keeps a high fixed priority and remains the default backend.
     */
    *priority = 100;
    *module = (mca_base_module_t *)&parsec_comm_mpi_module;
    return MCA_SUCCESS;
}
