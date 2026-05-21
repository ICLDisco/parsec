/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <assert.h>
#include "parsec/parsec_config.h"
#include "parsec/mca/comm/comm.h"
#include "parsec/remote_dep.h"

parsec_comm_engine_t parsec_ce;

#if defined(DISTRIBUTED)

/* Select and initialize the distributed communication backend. */
parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    parsec_comm_engine_t *ce = parsec_comm_engine_component_init(parsec_context);

    assert(NULL != ce);
    assert(ce->capabilites.sided > 0 && ce->capabilites.sided < 3);
    return ce;
}

extern int remote_dep_ce_fini(parsec_context_t* context);

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    (void) parsec_remote_dep_fini(comm_engine->parsec_context);
    remote_dep_ce_fini(comm_engine->parsec_context);
    /* Finalize the backend engine before releasing the selected MCA component. */
    parsec_ce.fini(&parsec_ce);
    parsec_comm_engine_component_fini();
    return PARSEC_SUCCESS;
}

#else

parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    /* Local builds keep the in-process engine and do not select a comm component. */
    parsec_ce.parsec_context = parsec_context;
    parsec_ce.capabilites.sided = 0;
    parsec_ce.capabilites.supports_noncontiguous_datatype = 0;
    parsec_ce.capabilites.multithreaded = 0;
    parsec_ce.taskpool_sync_ids = NULL;
    return &parsec_ce;
}

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    (void)comm_engine;
    return PARSEC_SUCCESS;
}

#endif  /* defined(DISTRIBUTED) */
