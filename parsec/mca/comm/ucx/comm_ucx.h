/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * UCX communication engine MCA component declaration.
 *
 * The UCX backend uses PMIx only for process bootstrap and UCX worker-address
 * exchange.  Runtime data movement is done with UCX active messages and CPU
 * RMA operations.
 */
#ifndef PARSEC_COMM_UCX_H_HAS_BEEN_INCLUDED
#define PARSEC_COMM_UCX_H_HAS_BEEN_INCLUDED

#include "parsec/mca/comm/comm.h"
#include "parsec/datatype_module.h"
#include <ucp/api/ucp.h>

BEGIN_C_DECLS

PARSEC_DECLSPEC extern const parsec_comm_base_component_t parsec_comm_ucx_component;
PARSEC_DECLSPEC extern const parsec_datatype_module_t parsec_datatype_basic_module;

/**
 * UCX state supplied by an application that initializes UCX itself.
 *
 * PaRSEC does not take ownership of either handle. The application must keep
 * both alive until the PaRSEC context using this communication engine has been
 * finalized. PaRSEC still performs the late runtime setup: worker-address
 * publication through PMIx, endpoint creation, and active-message handler
 * registration.
 */
typedef struct parsec_comm_ucx_external_worker_s {
    ucp_context_h context;
    ucp_worker_h worker;
} parsec_comm_ucx_external_worker_t;

PARSEC_DECLSPEC parsec_comm_engine_t *comm_ucx_init(parsec_context_t *context);
PARSEC_DECLSPEC mca_base_component_t *comm_ucx_static_component(void);

END_C_DECLS

#endif  /* PARSEC_COMM_UCX_H_HAS_BEEN_INCLUDED */
