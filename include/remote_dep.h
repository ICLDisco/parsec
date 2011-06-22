/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dague_config.h"

typedef unsigned long remote_dep_datakey_t;

#include "debug.h"

#if defined(HAVE_MPI)
#include <mpi.h>
typedef MPI_Datatype dague_remote_dep_datatype_t;
#else
typedef void* dague_remote_dep_datatype_t;
#endif

#include "assignment.h"
#include "lifo.h"
#include "execution_unit.h"
#include "dague.h"
#include "arena.h"
#include "datarepo.h"

#define DAGUE_ACTION_DEPS_MASK                  0x00FF
#define DAGUE_ACTION_RELEASE_LOCAL_DEPS         0x0100
#define DAGUE_ACTION_RELEASE_LOCAL_REFS         0x0200
#define DAGUE_ACTION_NO_PLACEHOLDER             0x0800
#define DAGUE_ACTION_SEND_INIT_REMOTE_DEPS      0x1000
#define DAGUE_ACTION_SEND_REMOTE_DEPS           0x2000
#define DAGUE_ACTION_RECV_INIT_REMOTE_DEPS      0x4000
#define DAGUE_ACTION_RELEASE_REMOTE_DEPS        (DAGUE_ACTION_SEND_INIT_REMOTE_DEPS | DAGUE_ACTION_SEND_REMOTE_DEPS)

typedef struct remote_dep_wire_activate_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    uint32_t             object_id;
    uint32_t             function_id;
    assignment_t locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_t
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    remote_dep_datakey_t tag;
} remote_dep_wire_get_t;

struct dague_remote_deps_t {
    dague_list_item_t                       item;
    struct dague_atomic_lifo_t*             origin;  /**< The memory arena where the data pointer is comming from */
    remote_dep_wire_activate_t              msg;     /**< A copy of the message control */
    int                                     root;    /**< The root of the control message */
    int                                     from;    /**< From whom we received the control */
    int                                     max_priority;
    uint32_t                                output_count;
    uint32_t                                output_sent_count;
    uint32_t*                               remote_dep_fw_mask;  /**< list of peers already notified about
                                                                  * the control sequence (only used for control messages) */
    struct { /** Never change this structure without understanding the 
              *   "subtle" relation with  remote_deps_allocation_init in remote_dep.c
              */
        void*    	                        data;
        struct dague_arena_t* 	            type;
        uint32_t*                           rank_bits;
        uint32_t                            count;
    } output[1];
};


/* Gives pointers to expr_t allowing for evaluation of GRID predicates, needed 
 * by the precompiler only */
int dague_remote_dep_get_rank_preds(const dague_object_t *dague_object,
                                    const expr_t **predicates,
                                    const expr_t **rowpred,
                                    const expr_t **colpred, 
                                    const symbol_t **rowsize,
                                    const symbol_t **colsize);

#if defined(DISTRIBUTED)

typedef struct {
    dague_atomic_lifo_t freelist;
    uint32_t            max_dep_count;
    uint32_t            max_nodes_number;
    uint32_t            elem_size;
} dague_remote_dep_context_t;

extern dague_remote_dep_context_t dague_remote_dep_context;

int remote_deps_allocation_init(int np, int max_deps);

static inline dague_remote_deps_t* remote_deps_allocation( dague_atomic_lifo_t* lifo )
{
    uint32_t i, rank_bit_size;
    char *ptr;
    dague_remote_deps_t* remote_deps = (dague_remote_deps_t*)dague_atomic_lifo_pop(lifo);
    if( NULL == remote_deps ) {
        remote_deps = (dague_remote_deps_t*)calloc(1, dague_remote_dep_context.elem_size);
        remote_deps->origin = lifo;
        ptr = (char*)(&(remote_deps->output[dague_remote_dep_context.max_dep_count]));
        rank_bit_size = sizeof(uint32_t) * ((dague_remote_dep_context.max_nodes_number + 31) / 32);
        for( i = 0; i < dague_remote_dep_context.max_dep_count; i++ ) {
            remote_deps->output[i].rank_bits = (uint32_t*)ptr;
            ptr += rank_bit_size;
        }
        /* fw_mask immediatly follows outputs */
        remote_deps->remote_dep_fw_mask = (uint32_t*) ptr;
        assert( (int)(ptr - (char*)remote_deps) <= (int)(dague_remote_dep_context.elem_size - sizeof(uint32_t) * (dague_remote_dep_context.max_nodes_number+31)/32) );
    }
    remote_deps->max_priority = 0xffffffff;
    return remote_deps;
}
#define DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(REMOTE_DEPS, EXEC_CONTEXT, COUNT) \
    if( NULL == (REMOTE_DEPS) ) { /* only once per function */                 \
        (REMOTE_DEPS) = (dague_remote_deps_t*)remote_deps_allocation(&dague_remote_dep_context.freelist); \
    }


int dague_remote_dep_init(dague_context_t* context);
int dague_remote_dep_fini(dague_context_t* context);
int dague_remote_dep_on(dague_context_t* context);
int dague_remote_dep_off(dague_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int dague_remote_dep_progress(dague_execution_unit_t* eu_context);

/* Send remote dependencies to target processes */
int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                                const dague_execution_context_t* origin,
                                dague_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count );

/* Memcopy a particular data using datatype specification */
void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context, 
        void *dst, dague_arena_chunk_t *src, 
        const dague_remote_dep_datatype_t datatype);

#else 
# define dague_remote_dep_init(ctx) (1)
# define dague_remote_dep_fini(ctx) (0)
# define dague_remote_dep_on(ctx)   (0)
# define dague_remote_dep_off(ctx)  (0)
# define dague_remote_dep_progress(ctx) (0)
# define dague_remote_dep_activate(ctx, o, r, c) (-1)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

