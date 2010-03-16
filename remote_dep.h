/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dplasma.h"
#include "execution_unit.h"

#if defined(USE_MPI)
# define DISTRIBUTED
#else
# undef DISTRIBUTED
#endif


#if defined(DISTRIBUTED) || defined(DPLASMA_DEBUG)
# ifdef DEPRECATED
/* Activate all the dependencies of origin on the rank hosting exec_context
 */
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                const dplasma_execution_context_t* exec_context,
                                const param_t* dest_param );
# endif

int dplasma_remote_dep_activate_rank(dplasma_execution_unit_t* eu_context, 
                                     const dplasma_execution_context_t* origin, 
                                     const param_t* origin_param, 
                                     int rank, gc_data_t** data);

#else
# ifdef DEPRECATED
#   define dplasma_remote_dep_activate(eu, o, op, e, ep) (0)
# endif
# ifndef DPLASMA_DEBUG
#   define dplasma_remote_dep_activate_rank(eu, o, op, r, d) (0)
# endif
#endif

/* Gives pointers to expr_t allowing for evaluation of GRID predicates, needed 
 * by the precompiler only */
int dplasma_remote_dep_get_rank_preds(const expr_t **predicates,
                                      expr_t **rowpred,
                                      expr_t **colpred, 
                                      symbol_t **rowsize,
                                      symbol_t **colsize);

#if defined(DISTRIBUTED)

int dplasma_remote_dep_init(dplasma_context_t* context);
int dplasma_remote_dep_fini(dplasma_context_t* context);
int dplasma_remote_dep_on(dplasma_context_t* context);
int dplasma_remote_dep_off(dplasma_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context);

#include <string.h>

/* Clear the already forwarded remote dependency matrix */
static inline void dplasma_remote_dep_reset_forwarded( dplasma_execution_unit_t* eu_context )
{
    /*DEBUG(("fw reset\tcontext %p\n", (void*) eu_context));*/
    memset(eu_context->remote_dep_fw_mask, 0, eu_context->master_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void dplasma_remote_dep_mark_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint32_t mask;
    
    /*DEBUG(("fw mark\tREMOTE rank %d\n", rank));*/
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    eu_context->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int dplasma_remote_dep_is_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint32_t mask;
    
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    /*DEBUG(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (eu_context->remote_dep_fw_mask[boffset] & mask)));*/
    return (int) ((eu_context->remote_dep_fw_mask[boffset] & mask) != 0);
}

#else 
# define dplasma_remote_dep_init(ctx) (1)
# define dplasma_remote_dep_fini(ctx) (0)
# define dplasma_remote_dep_on(ctx)   (0)
# define dplasma_remote_dep_off(ctx)  (0)
# define dplasma_remote_dep_progress(ctx) (0)
# define dplasma_remote_dep_reset_forwarded(ctx)
# define dplasma_remote_dep_mark_forwarded(ctx, rk)
# define dplasma_remote_dep_is_forwarded(ctx, rk) (0)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

