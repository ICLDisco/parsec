/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dplasma.h"

#if defined(USE_MPI)
#   define ALLOW_REMOTE_DEP
#else
#   undef ALLOW_REMOTE_DEP
#endif


int dplasma_remote_dep_init(dplasma_context_t* context);
int dplasma_remote_dep_fini(dplasma_context_t* context);

/* Activate all the dependencies of origin on the rank hosting exec_context
 */
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                const dplasma_execution_context_t* exec_context,
                                const param_t* dest_param );

int dplasma_remote_dep_activate_rank(dplasma_execution_unit_t* eu_context, 
                                     const dplasma_execution_context_t* origin, 
                                     const param_t* origin_param, 
                                     int rank, void** data);

/* Gives pointers to expr_t allowing for evaluation of GRID predicates */
int dplasma_remote_dep_get_rank_preds(const expr_t **predicates,
                                      expr_t **rowpred,
                                      expr_t **colpred, 
                                      symbol_t **rowsize,
                                      symbol_t **colsize);

/* Compute the flat rank of the node hosting exec_context in the process grid */
int dplasma_remote_dep_compute_grid_rank(dplasma_execution_unit_t* eu_context,
                                         const dplasma_execution_context_t* origin,
                                         const dplasma_execution_context_t* exec_context);
#if defined(ALLOW_REMOTE_DEP)

/* Poll for remote completion of tasks that would enable some work locally */
int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context);

#include <string.h>

/* Clear the already forwarded remote dependency matrix */
static inline void dplasma_remote_dep_reset_forwarded( dplasma_execution_unit_t* eu_context )
{
    DEBUG(("fw reset\tcontext %p\n", (void*) eu_context));
    memset(eu_context->remote_dep_fw_mask, 0, eu_context->master_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void dplasma_remote_dep_mark_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint8_t mask = 1;
    
    DEBUG(("fw mark\tREMOTE rank %d\n", rank));
    boffset = rank / sizeof(uint8_t);
    mask = ((uint8_t)1) << (rank % sizeof(uint8_t));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    eu_context->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int dplasma_remote_dep_is_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint8_t mask = 1;
    
    boffset = rank / sizeof(uint8_t);
    mask = ((uint8_t)1) << (rank % sizeof(uint8_t));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    DEBUG(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (eu_context->remote_dep_fw_mask[boffset] & mask)));
    return (int) (eu_context->remote_dep_fw_mask[boffset] & mask);
}

#else 
#   define dplasma_remote_dep_progress(ctx) (0)
#   define dplasma_remote_dep_reset_forwarded(ctx)
#   define dplasma_remote_dep_mark_forwarded(ctx, rk)
#   define dplasma_remote_dep_is_forwarded(ctx, rk) (0)
#endif /* ALLOW_REMOTE_DEP */

#endif /* __USE_REMOTE_DEP_H__ */

