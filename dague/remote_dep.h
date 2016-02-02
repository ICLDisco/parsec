/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dague_config.h"
#include "dague_internal.h"

typedef unsigned long remote_dep_datakey_t;

#include "dague/debug.h"
#include <string.h>

#include "dague/dague_description_structures.h"
#include "dague/class/lifo.h"

#define DAGUE_ACTION_DEPS_MASK                  0x00FFFFFF
#define DAGUE_ACTION_RELEASE_LOCAL_DEPS         0x01000000
#define DAGUE_ACTION_RELEASE_LOCAL_REFS         0x02000000
#define DAGUE_ACTION_GET_REPO_ENTRY             0x04000000
#define DAGUE_ACTION_SEND_INIT_REMOTE_DEPS      0x10000000
#define DAGUE_ACTION_SEND_REMOTE_DEPS           0x20000000
#define DAGUE_ACTION_RECV_INIT_REMOTE_DEPS      0x40000000
#define DAGUE_ACTION_RELEASE_REMOTE_DEPS        (DAGUE_ACTION_SEND_INIT_REMOTE_DEPS | DAGUE_ACTION_SEND_REMOTE_DEPS)

typedef struct remote_dep_wire_activate_s
{
    remote_dep_datakey_t deps;         /**< a pointer to the dep structure on the source */
    remote_dep_datakey_t output_mask;  /**< the mask of the output dependencies satisfied by this activation message */
    remote_dep_datakey_t tag;
    uint32_t             handle_id;
    uint32_t             function_id;
    uint32_t             length;
    assignment_t         locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_s
{
    remote_dep_datakey_t deps;
    remote_dep_datakey_t output_mask;
    remote_dep_datakey_t tag;
} remote_dep_wire_get_t;

/**
 * This structure holds the key information for any data mouvement. It contains the arena
 * where the data is allocated from, or will be allocated from. It also contains the
 * pointer to the buffer involved in the communication (or NULL if the data will be
 * allocated before the reception). Finally, it contains the triplet allowing a correct send
 * or receive operation: the memory layout, the number fo repetitions and the displacement
 * from the data pointer where the operation will start. If the memory layout is NULL the
 * one attached to the arena must be used instead.
 */
struct dague_dep_data_description_s {
    struct dague_arena_s     *arena;
    struct dague_data_copy_s *data;
    dague_datatype_t          layout;
    uint64_t                  count;
    int64_t                   displ;
};

struct remote_dep_output_param_s {
    /** Never change this structure without understanding the
     *   "subtle" relation with remote_deps_allocation_init in
     *  remote_dep.c
     */
    dague_list_item_t                    super;
    dague_remote_deps_t                 *parent;
    struct dague_dep_data_description_s  data;        /**< The data propagated by this message. */
    uint32_t                             deps_mask;   /**< A bitmask of all the output dependencies
                                                       propagated by this message. The bitmask uses
                                                       depedencies indexes not flow indexes. */
    int32_t                              priority;    /**< the priority of the message */
    uint32_t                             count_bits;  /**< The number of participants */
    uint32_t*                            rank_bits;   /**< The array of bits representing the propagation path */
};

struct dague_remote_deps_s {
    dague_list_item_t                super;
    dague_lifo_t                    *origin;  /**< The memory arena where the data pointer is comming from */
    struct dague_handle_s           *dague_handle;  /**< dague object generating this data transfer */
    int32_t                          pending_ack;  /**< Number of releases before completion */
    int32_t                          from;    /**< From whom we received the control */
    int32_t                          root;    /**< The root of the control message */
    uint32_t                         incoming_mask;  /**< track all incoming actions (receives) */
    uint32_t                         outgoing_mask;  /**< track all outgoing actions (send) */
    remote_dep_wire_activate_t       msg;     /**< A copy of the message control */
    int32_t                          max_priority;
    int32_t                          priority;
    uint32_t                        *remote_dep_fw_mask;  /**< list of peers already notified about
                                                            * the control sequence (only used for control messages) */
    struct data_repo_entry_s        *repo_entry;
    struct remote_dep_output_param_s output[1];
};
/* { item .. remote_dep_fw_mask (points to fw_mask_bitfield),
 *   output[0] .. output[max_deps < MAX_PARAM_COUNT],
 *   (max_dep_count x (np+31)/32 uint32_t) rank_bits
 *   ((np+31)/32 x uint32_t) fw_mask_bitfield } */

/* This int can take the following values:
 * - negative: no communication engine has been enabled
 * - 0: the communication engine is not turned on
 * - positive: the meaning is defined by the communication engine.
 */
extern int dague_communication_engine_up;

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
        DAGUE_LIFO_ITEM_ALLOC( lifo, remote_deps, dague_remote_dep_context.elem_size );
        remote_deps->origin = lifo;
        remote_deps->dague_handle = NULL;
        ptr = (char*)(&(remote_deps->output[dague_remote_dep_context.max_dep_count]));
        rank_bit_size = sizeof(uint32_t) * ((dague_remote_dep_context.max_nodes_number + 31) / 32);
        memset(ptr, 0, rank_bit_size * dague_remote_dep_context.max_dep_count);
        for( i = 0; i < dague_remote_dep_context.max_dep_count; i++ ) {
            OBJ_CONSTRUCT(&remote_deps->output[i].super, dague_list_item_t);
            remote_deps->output[i].parent     = remote_deps;
            remote_deps->output[i].rank_bits  = (uint32_t*)ptr;
            remote_deps->output[i].deps_mask  = 0;
            remote_deps->output[i].count_bits = 0;
            remote_deps->output[i].priority   = 0xffffffff;
            ptr += rank_bit_size;
        }
        /* fw_mask immediatly follows outputs */
        remote_deps->remote_dep_fw_mask = (uint32_t*) ptr;
        assert( (int)(ptr - (char*)remote_deps) ==
                (int)(dague_remote_dep_context.elem_size - rank_bit_size));
    }
    assert(NULL == remote_deps->dague_handle);
    remote_deps->max_priority    = 0xffffffff;
    remote_deps->root            = -1;
    remote_deps->pending_ack     = 0;
    remote_deps->incoming_mask   = 0;
    remote_deps->outgoing_mask   = 0;
    DAGUE_DEBUG_VERBOSE(30, dague_debug_output, "remote_deps_allocate: %p\n", remote_deps);
    return remote_deps;
}

#define DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(REMOTE_DEPS, EXEC_CONTEXT, COUNT) \
    if( NULL == (REMOTE_DEPS) ) { /* only once per function */                 \
        (REMOTE_DEPS) = (dague_remote_deps_t*)remote_deps_allocate(&dague_remote_dep_context.freelist); \
    }

/* This returns the deps to the freelist, no use counter */
static inline void remote_deps_free(dague_remote_deps_t* deps)
{
    uint32_t k, a;
    assert(0 == deps->pending_ack);
    assert(0 == deps->incoming_mask);
    assert(0 == deps->outgoing_mask);
    for( k = 0; k < dague_remote_dep_context.max_dep_count; k++ ) {
        if( 0 == deps->output[k].count_bits ) continue;
        for(a = 0; a < (dague_remote_dep_context.max_nodes_number + 31)/32; a++)
            deps->output[k].rank_bits[a] = 0;
        deps->output[k].count_bits = 0;
#if defined(DAGUE_DEBUG_PARANOID)
        deps->output[k].data.data   = NULL;
        deps->output[k].data.arena  = NULL;
        deps->output[k].data.layout = DAGUE_DATATYPE_NULL;
        deps->output[k].data.count  = -1;
        deps->output[k].data.displ  = 0xFFFFFFFF;
#endif
    }
    DAGUE_DEBUG_VERBOSE(30, dague_debug_output, "remote_deps_free: %p mask %x\n", deps, deps->outgoing_mask);
#if defined(DAGUE_DEBUG_PARANOID)
    memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
    deps->dague_handle      = NULL;
    dague_lifo_push(deps->origin, (dague_list_item_t*)deps);
}

int dague_remote_dep_init(dague_context_t* context);
int dague_remote_dep_fini(dague_context_t* context);
int dague_remote_dep_on(dague_context_t* context);
int dague_remote_dep_off(dague_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int dague_remote_dep_progress(dague_execution_unit_t* eu_context);

/* Inform the communication engine from the creation of new objects */
int dague_remote_dep_new_object(dague_handle_t* handle);

/* Send remote dependencies to target processes */
int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                              const dague_execution_context_t* origin,
                              dague_remote_deps_t* remote_deps,
                              uint32_t propagation_mask);

/* Memcopy a particular data using datatype specification */
void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context,
                             dague_handle_t* dague_handle,
                             dague_data_copy_t *dst,
                             dague_data_copy_t *src,
                             dague_dep_data_description_t* data);

#ifdef DAGUE_DIST_COLLECTIVES
/* Propagate an activation order from the current node down the original tree */
int dague_remote_dep_propagate(dague_execution_unit_t* eu_context,
                               const dague_execution_context_t* task,
                               dague_remote_deps_t* deps);
#endif

#else
#define dague_remote_dep_init(ctx)           1
#define dague_remote_dep_fini(ctx)           0
#define dague_remote_dep_on(ctx)             0
#define dague_remote_dep_off(ctx)            0
#define dague_remote_dep_progress(ctx)       0
#define dague_remote_dep_activate(ctx, o, r) -1
#define dague_remote_dep_new_object(ctx)     0
#endif /* DISTRIBUTED */

#endif /* __USE_REMOTE_DEP_H__ */
