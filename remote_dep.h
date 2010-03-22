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
typedef MPI_Datatype dplasma_remote_dep_datatype_t;
#else
# undef DISTRIBUTED
typedef void dplasma_remote_dep_datatype_t;
#endif

#define DPLASMA_ACTION_INIT_REMOTE_DEPS    0x0100
#define DPLASMA_ACTION_SEND_REMOTE_DEPS    0x0200
#define DPLASMA_ACTION_RECV_REMOTE_DEPS    0x0400
#define DPLASMA_ACTION_RELEASE_REMOTE_DEPS (DPLASMA_ACTION_INIT_REMOTE_DEPS | DPLASMA_ACTION_SEND_REMOTE_DEPS)
#define DPLASMA_ACTION_GETDATA_REMOTE_DEPS (DPLASMA_ACTION_INIT_REMOTE_DEPS | DPLASMA_ACTION_RECV_REMOTE_DEPS)
#define DPLASMA_ACTION_DEPS_MASK           0x00FF

typedef unsigned long remote_dep_datakey_t;

typedef struct remote_dep_wire_activate_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t function;
    assignment_t locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
} remote_dep_wire_get_t;

typedef struct dplasma_remote_deps_t {
    dplasma_list_item_t                       item;
    struct dplasma_atomic_lifo_t*             origin;
    remote_dep_wire_activate_t                msg;
    struct {
        gc_data_t*                            data;
        uint32_t*                             rank_bits;
        uint32_t                              count;
        dplasma_remote_dep_datatype_t*        type;
    } output[1];
} dplasma_remote_deps_t;


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

/* Send remote dependencies to target processes */
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                dplasma_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count );

/* Memcopy a particular data using datatype specification */
void dplasma_remote_dep_memcpy(void *dst, gc_data_t *src, const dplasma_remote_dep_datatype_t datatype);

/* Create a default datatype */
void remote_dep_mpi_create_default_datatype(int tile_size, dplasma_remote_dep_datatype_t base);

extern dplasma_remote_dep_datatype_t DPLASMA_DEFAULT_DATA_TYPE;

#else 
# define dplasma_remote_dep_init(ctx) (1)
# define dplasma_remote_dep_fini(ctx) (0)
# define dplasma_remote_dep_on(ctx)   (0)
# define dplasma_remote_dep_off(ctx)  (0)
# define dplasma_remote_dep_progress(ctx) (0)
# define dplasma_remote_dep_activate(ctx, o, r, c) (-1)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

