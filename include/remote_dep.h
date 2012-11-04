/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dague_config.h"
#include "dague_internal.h"

typedef unsigned long remote_dep_datakey_t;

#include "debug.h"
#include <string.h>


#if defined(HAVE_MPI)
#include <mpi.h>
typedef MPI_Datatype dague_remote_dep_datatype_t;
#else
typedef void* dague_remote_dep_datatype_t;
#endif

#include "dague_description_structures.h"
#include "lifo.h"

#define DAGUE_ACTION_DEPS_MASK                  0x00FF
#define DAGUE_ACTION_RELEASE_LOCAL_DEPS         0x0100
#define DAGUE_ACTION_RELEASE_LOCAL_REFS         0x0200
#define DAGUE_ACTION_SEND_INIT_REMOTE_DEPS      0x1000
#define DAGUE_ACTION_SEND_REMOTE_DEPS           0x2000
#define DAGUE_ACTION_RECV_INIT_REMOTE_DEPS      0x4000
#define DAGUE_ACTION_RELEASE_REMOTE_DEPS        (DAGUE_ACTION_SEND_INIT_REMOTE_DEPS | DAGUE_ACTION_SEND_REMOTE_DEPS)

typedef struct remote_dep_wire_activate_s
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    remote_dep_datakey_t tag;
    uint32_t             object_id;
    uint32_t             function_id;
    assignment_t locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_s
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t which;
    remote_dep_datakey_t tag;
} remote_dep_wire_get_t;

struct remote_dep_output_param_s {
/** Never change this structure without understanding the
  *   "subtle" relation with remote_deps_allocation_init in
  *  remote_dep.c
  */
    struct dague_data_s    *data;
    struct dague_arena_s   *type;
    uint32_t                 nbelt;
    uint32_t                 count;
    uint32_t               *rank_bits;
};

struct dague_remote_deps_s {
    dague_list_item_t           item;
    dague_lifo_t               *origin;  /**< The memory arena where the data pointer is comming from */
    struct dague_object_s      *dague_object;  /**< dague object generating this data transfer */
    remote_dep_wire_activate_t  msg;     /**< A copy of the message control */
    int                         root;    /**< The root of the control message */
    int                         from;    /**< From whom we received the control */
    int                         max_priority;
    uint32_t                    output_count;
    uint32_t                    output_sent_count;
    uint32_t                   *remote_dep_fw_mask;  /**< list of peers already notified about
                                                             * the control sequence (only used for control messages) */
    struct remote_dep_output_param_s  output[1];
};
/* { item .. remote_dep_fw_mask (points to fw_mask_bitfield),
 *   output[0] .. output[max_deps < MAX_PARAM_COUNT],
 *   (max_dep_count x (np+31)/32 uint32_t) rank_bits
 *   ((np+31)/32 x uint32_t) fw_mask_bitfield } */



#if defined(DISTRIBUTED)

typedef struct {
    dague_lifo_t freelist;
    uint32_t     max_dep_count;
    uint32_t     max_nodes_number;
    uint32_t     elem_size;
} dague_remote_dep_context_t;

extern dague_remote_dep_context_t dague_remote_dep_context;

void remote_deps_allocation_init(int np, int max_deps);
void remote_deps_allocation_fini(void);

static inline dague_remote_deps_t* remote_deps_allocate( dague_lifo_t* lifo )
{
    dague_remote_deps_t* remote_deps = (dague_remote_deps_t*)dague_lifo_pop(lifo);
    uint32_t i, rank_bit_size;

    if( NULL == remote_deps ) {
        char *ptr;
        remote_deps = (dague_remote_deps_t*)calloc(1, dague_remote_dep_context.elem_size);
        DAGUE_LIST_ITEM_CONSTRUCT(remote_deps);
        remote_deps->origin = lifo;
        ptr = (char*)(&(remote_deps->output[dague_remote_dep_context.max_dep_count]));
        rank_bit_size = sizeof(uint32_t) * ((dague_remote_dep_context.max_nodes_number + 31) / 32);
        for( i = 0; i < dague_remote_dep_context.max_dep_count; i++ ) {
            remote_deps->output[i].rank_bits = (uint32_t*)ptr;
            ptr += rank_bit_size;
        }
        /* fw_mask immediatly follows outputs */
        remote_deps->remote_dep_fw_mask = (uint32_t*) ptr;
        assert( (int)(ptr - (char*)remote_deps) == (int)(dague_remote_dep_context.elem_size - rank_bit_size));
    }
    remote_deps->max_priority = 0xffffffff;
    remote_deps->dague_object = NULL;
    remote_deps->root         = -1;
    return remote_deps;
}
#define DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(REMOTE_DEPS, EXEC_CONTEXT, COUNT) \
    if( NULL == (REMOTE_DEPS) ) { /* only once per function */                 \
        (REMOTE_DEPS) = (dague_remote_deps_t*)remote_deps_allocate(&dague_remote_dep_context.freelist); \
    }
/* This returns the deps to the freelist, no use counter */
static inline void remote_deps_free(dague_remote_deps_t* deps) {
    uint32_t k = 0, count = 0, a;
    while( count < deps->output_count ) {
        for(a = 0; a < (dague_remote_dep_context.max_nodes_number + 31)/32; a++)
            deps->output[k].rank_bits[a] = 0;
        count += deps->output[k].count;
        deps->output[k].count = 0;
#if defined(DAGUE_DEBUG)
        deps->output[k].data = NULL;
        deps->output[k].type = NULL;
        deps->output[k].nbelt = -1;
#endif
        k++;
        assert(k < MAX_PARAM_COUNT);
    }
    assert(count == deps->output_count);
#if defined(DAGUE_DEBUG)
    DEBUG(("remote_deps_free: sent_count=%u/%u\n", deps->output_sent_count, deps->output_count));
    memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
    deps->output_count = 0;
    deps->output_sent_count = 0;
    dague_lifo_push(deps->origin, (dague_list_item_t*)deps);
}

int dague_remote_dep_init(dague_context_t* context);
int dague_remote_dep_fini(dague_context_t* context);
int dague_remote_dep_on(dague_context_t* context);
int dague_remote_dep_off(dague_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int dague_remote_dep_progress(dague_execution_unit_t* eu_context);

/* Inform the communication engine from the creation of new objects */
int dague_remote_dep_new_object(dague_object_t* obj);

/* Send remote dependencies to target processes */
int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                                const dague_execution_context_t* origin,
                                dague_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count );

/* Memcopy a particular data using datatype specification */
void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context,
                             dague_object_t* dague_object,
                             struct dague_data_s *dst, struct dague_data_s *src,
                             const dague_remote_dep_datatype_t datatype,
                             int nbelt);

#else
# define dague_remote_dep_init(ctx) (1)
# define dague_remote_dep_fini(ctx) (0)
# define dague_remote_dep_on(ctx)   (0)
# define dague_remote_dep_off(ctx)  (0)
# define dague_remote_dep_progress(ctx) (0)
# define dague_remote_dep_activate(ctx, o, r, c) (-1)
# define dague_remote_dep_new_object(obj) (0)
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */

