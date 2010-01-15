/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dplasma.h"
#include <string.h>

int dplasma_remote_dep_init(dplasma_context_t* context);
int dplasma_remote_dep_fini(dplasma_context_t* context);

/* Activate all the dependencies of origin on the rank hosting exec_context
 */
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                dplasma_execution_context_t* exec_context,
                                const param_t* dest_param );

/* Poll for remote completion of tasks that would enable some work locally */
int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context);

/* Compute the flat rank of the node hosting exec_context in the process grid */
int dplasma_remote_dep_compute_grid_rank(dplasma_execution_unit_t* eu_context,
                                         const dplasma_execution_context_t* origin,
                                         dplasma_execution_context_t* exec_context);

/* Clear the already forwarded remote dependency matrix */
static inline void dplasma_remote_dep_reset_forwarded( dplasma_execution_unit_t* eu_context )
{
    int rfwsize = eu_context->master_context->remote_dep_fw_mask_sizeof;

    if(rfwsize)
    {
        DEBUG(("fw reset\tcontext %p", (void*) eu_context));
        memset(eu_context->remote_dep_fw_mask, 0, rfwsize);        
    }
}
/* Mark a rank as already forwarded the termination of the current task */
void dplasma_remote_dep_mark_forwarded( dplasma_execution_unit_t* eu_context, int rank);
/* Check if rank has already been forwarded the termination of the current task */
int dplasma_remote_dep_is_forwarded( dplasma_execution_unit_t* eu_context, int rank);

#endif /* __USE_REMOTE_DEP_H__ */

