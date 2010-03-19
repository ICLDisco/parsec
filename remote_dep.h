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
#include <mpi.h>
#else
# undef DISTRIBUTED
#endif

#define DPLASMA_ACTION_RELEASE_REMOTE_DEPS 0x0100
#define DPLASMA_ACTION_DEPS_MASK           0x00FF

typedef struct dplasma_remote_deps_t {
    dplasma_list_item_t                       item;
    struct dplasma_atomic_lifo_t*             origin;
    const struct dplasma_execution_context_t* exec_context;
    struct {
        gc_data_t*                            data;
        uint32_t*                             rank_bits;
        uint32_t                              count;
        void*                                 type;
    } output[1];
} dplasma_remote_deps_t;

#if defined(DISTRIBUTED) || defined(DPLASMA_DEBUG)
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                dplasma_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count );
#else
# ifndef DPLASMA_DEBUG
#   define dplasma_remote_dep_activate(eu, o, op, e, ep) (0)
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

#if defined(USE_MPI)
void dplasma_remote_dep_memcpy(void *dst, gc_data_t *src, MPI_Datatype datatype);
void remote_dep_mpi_create_default_datatype(int tile_size, MPI_Datatype base);

extern MPI_Datatype DPLASMA_DEFAULT_DATA_TYPE;
#endif

#else 
# define dplasma_remote_dep_init(ctx) (1)
# define dplasma_remote_dep_fini(ctx) (0)
# define dplasma_remote_dep_on(ctx)   (0)
# define dplasma_remote_dep_off(ctx)  (0)
# define dplasma_remote_dep_progress(ctx) (0)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

