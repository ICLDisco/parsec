/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dplasma.h"

#define SIZEOF_FW_MASK(eu_context) (((eu_context)->master_context->nb_nodes + sizeof(char) - 1) / sizeof(char))

int dplasma_remote_dep_init(dplasma_context_t* context);
int dplasma_remote_dep_fini(dplasma_context_t* context);

int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                dplasma_execution_context_t* exec_context,
                                const param_t* dest_param );

int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context);

void dplasma_remote_dep_reset_forwarded( dplasma_execution_unit_t* eu_context );

#endif /* __USE_REMOTE_DEP_H__ */

