/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <assert.h>
#include "parsec/parsec_mpi_funnelled.h"
#include "parsec/remote_dep.h"

parsec_comm_engine_t parsec_ce;

/* This function will be called by the runtime */
parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    /* call the selected module init */
    parsec_comm_engine_t *ce = mpi_funnelled_init(parsec_context);

    assert(ce->capabilites.sided > 0 && ce->capabilites.sided < 3);
    return ce;
}

extern int remote_dep_ce_fini(parsec_context_t* context);

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    (void) parsec_remote_dep_fini(comm_engine->parsec_context);
    remote_dep_ce_fini(comm_engine->parsec_context);
    /* call the selected module fini */
    parsec_ce.fini(&parsec_ce);
    return PARSEC_SUCCESS;
}
