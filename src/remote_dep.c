/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "remote_dep.h"
#include "scheduling.h"
#include "execution_unit.h"
#include <stdio.h>

#ifdef DISTRIBUTED
/* Clear the already forwarded remote dependency matrix */
static inline void remote_dep_reset_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps )
{
    DEBUG3(("fw reset\tcontext %p\n", (void*) eu_context));
    memset(rdeps->remote_dep_fw_mask, 0, eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void remote_dep_mark_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    unsigned int boffset;
    uint32_t mask;

    DEBUG3(("fw mark\tREMOTE rank %d\n", rank));
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
    (void)eu_context;
    rdeps->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int remote_dep_is_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    unsigned int boffset;
    uint32_t mask;

    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
    DEBUG3(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (rdeps->remote_dep_fw_mask[boffset] & mask)));
    (void)eu_context;
    return (int) ((rdeps->remote_dep_fw_mask[boffset] & mask) != 0);
}


/* make sure we don't leave before serving all data deps */
static inline void remote_dep_inc_flying_messages(dague_object_t *dague_object, dague_context_t* ctx)
{
    assert( dague_object->nb_local_tasks > 0 );
    dague_atomic_inc_32b( &(dague_object->nb_local_tasks) );
    (void)ctx;
}

/* allow for termination when all deps have been served */
static inline void remote_dep_dec_flying_messages(dague_object_t *dague_object, dague_context_t* ctx)
{
    __dague_complete_task(dague_object, ctx);
}

/* Mark that ncompleted of the remote deps are finished, and return the remote dep to
 * the free items queue if it is now done */
static inline void remote_dep_complete_and_cleanup(dague_remote_deps_t* deps, int ncompleted)
{
    deps->output_sent_count += ncompleted;
    assert( deps->output_sent_count <= deps->output_count );
    if( deps->output_count == deps->output_sent_count ) {
        remote_deps_free(deps);
    }
}

#endif

#ifdef HAVE_MPI
#include "remote_dep_mpi.c"

#else
#endif /* NO TRANSPORT */


#ifdef DISTRIBUTED
int dague_remote_dep_init(dague_context_t* context)
{
    (void)remote_dep_init(context);

    if(context->nb_nodes > 1)
    {
        context->remote_dep_fw_mask_sizeof = ((context->nb_nodes + 31) / 32) * sizeof(uint32_t);
    }
    else
    {
        context->remote_dep_fw_mask_sizeof = 0; /* hoping memset(0b) is fast */
    }
    return context->nb_nodes;
}

int dague_remote_dep_fini(dague_context_t* context)
{
    int rc = remote_dep_fini(context);
    remote_deps_allocation_fini();
    return rc;
}

int dague_remote_dep_on(dague_context_t* context)
{
    return remote_dep_on(context);
}

int dague_remote_dep_off(dague_context_t* context)
{
    return remote_dep_off(context);
}

int dague_remote_dep_progress(dague_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}


#ifdef DAGUE_DIST_COLLECTIVES
#define DAGUE_DIST_COLLECTIVES_TYPE_CHAINPIPELINE
#undef  DAGUE_DIST_COLLECTIVES_TYPE_BINOMIAL

# ifdef DAGUE_DIST_COLLECTIVES_TYPE_CHAINPIPELINE
static inline int remote_dep_bcast_chainpipeline_child(int me, int him)
{
    assert(him >= 0);
    if(me == -1) return 0;
    if(him == me+1) return 1;
    return 0;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_chainpipeline_child(me, him)

# elif defined(DAGUE_DIST_COLLECTIVES_TYPE_BINOMIAL)
static inline int remote_dep_bcast_binonial_child(int me, int him)
{
    int k, mask;

    /* flush out the easy cases first */
    assert(him >= 0);
    if(him == 0) return 0; /* root is child to nobody */
    if(me == -1) return 0; /* I don't even know who I am yet... */

    /* look for the leftmost 1 bit */
    for(k = sizeof(int) * 8 - 1; k >= 0; k--)
    {
        mask = 0x1<<k;
        if(him & mask)
        {
            him ^= mask;
            break;
        }
    }
    /* is the remainder suffix "me" ? */
    return him == me;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_binonial_child(me, him)

# else
#  error "INVALID COLLECTIVE TYPE. YOU MUST DEFINE ONE COLLECTIVE TYPE WHEN ENABLING COLLECTIVES"
# endif

#else
static inline int remote_dep_bcast_star_child(int me, int him)
{
    (void)him;
    if(me == 0) return 1;
    else return 0;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_star_child(me, him)
#endif

int dague_remote_dep_new_object(dague_object_t* obj) {
    return remote_dep_new_object(obj);
}

int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                              const dague_execution_context_t* exec_context,
                              dague_remote_deps_t* remote_deps,
                              uint32_t remote_deps_count )
{
    const dague_function_t* function = exec_context->function;
    int i, me, him, current_mask;
    int skipped_count = 0, flow_index;
    unsigned int array_index, count, bit_index;

    assert(eu_context->virtual_process->dague_context->nb_nodes > 1);
    assert(remote_deps_count);

#if defined(DAGUE_DEBUG)
    /* make valgrind happy */
    memset(&remote_deps->msg, 0, sizeof(remote_dep_wire_activate_t));
#endif
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    remote_dep_reset_forwarded(eu_context, remote_deps);
    remote_deps->dague_object = exec_context->dague_object;
    remote_deps->output_count = remote_deps_count;
    remote_deps->msg.deps = (uintptr_t) remote_deps;
    remote_deps->msg.object_id   = exec_context->dague_object->object_id;
    remote_deps->msg.function_id = function->function_id;
    for(i = 0; i < function->nb_locals; i++) {
        remote_deps->msg.locals[i] = exec_context->locals[i];
    }

    if(remote_deps->root == eu_context->virtual_process->dague_context->my_rank) me = 0;
    else me = -1;

    for( i = 0; remote_deps_count; i++) {
        if( 0 == remote_deps->output[i].count_bits ) continue;

        him = 0;
        for( array_index = count = 0; count < remote_deps->output[i].count_bits; array_index++ ) {
            current_mask = remote_deps->output[i].rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; (bit_index < (8 * sizeof(uint32_t))) && (current_mask != 0); bit_index++ ) {
                if( current_mask & (1 << bit_index) ) {
                    int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                    assert(rank >= 0);
                    assert(rank < eu_context->virtual_process->dague_context->nb_nodes);

                    current_mask ^= (1 << bit_index);
                    count++;
                    remote_deps_count--;

                    DEBUG3((" TOPO\t%s\troot=%d\t%d (d%d) -? %d (dna)\n",
                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context),
                            remote_deps->root, eu_context->virtual_process->dague_context->my_rank, me, rank));

                    /* root already knows but falsely appear in this bitfield */
                    if(rank == remote_deps->root) continue;

                    if((me == -1) && (rank >= eu_context->virtual_process->dague_context->my_rank))
                    {
                        /* the next bit points after me, so I know my dense rank now */
                        me = ++him;
                        if(rank == eu_context->virtual_process->dague_context->my_rank) {
                            skipped_count++;
                            continue;
                        }
                    }
                    him++;

                    if(remote_dep_bcast_child(me, him))
                    {
                        DEBUG2((" TOPO\t%s\troot=%d\t%d (d%d) -> %d (d%d)\n",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context),
                                remote_deps->root, eu_context->virtual_process->dague_context->my_rank, me, rank, him));
                        for( flow_index = 0; NULL != exec_context->function->out[flow_index]; flow_index++ )
                            if( exec_context->function->out[flow_index]->flow_mask & (1 << i) )
                                break;
                        assert( NULL != exec_context->function->out[flow_index] );
                        if(FLOW_ACCESS_NONE != (exec_context->function->out[flow_index]->flow_flags & FLOW_ACCESS_MASK))
                        {
                            AREF(remote_deps->output[flow_index].data.ptr);
                        }
                        if(remote_dep_is_forwarded(eu_context, remote_deps, rank))
                        {
                            continue;
                        }
                        remote_dep_inc_flying_messages(exec_context->dague_object, eu_context->virtual_process->dague_context);
                        remote_dep_mark_forwarded(eu_context, remote_deps, rank);
                        remote_dep_send(rank, remote_deps);
                    } else {
                        skipped_count++;
                        DEBUG2((" TOPO\t%s\troot=%d\t%d (d%d) ][ %d (d%d)\n",
                               dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context), remote_deps->root,
                               eu_context->virtual_process->dague_context->my_rank, me, rank, him));
                    }
                }
            }
        }
    }

    /* Only the thread doing bcast forwarding can enter the following line.
     * the same communication thread calls here and does the
     * sends that call complete_and_cleanup concurently.
     * This has to be done only if the receiver is a leaf in a broadcast.
     * The remote_deps has then been allocated in dague_release_dep_fct
     * when we didn't knew yet if we forward the data or not.
     */
    if( skipped_count ) {
        remote_dep_complete_and_cleanup(remote_deps, skipped_count);
    }

    return 0;
}

dague_remote_dep_context_t dague_remote_dep_context;
static int dague_remote_dep_inited = 0;

/* THIS FUNCTION MUST NOT BE CALLED WHILE REMOTE DEP IS ON.
 * NOT THREAD SAFE (AND SHOULD NOT BE) */
void remote_deps_allocation_init(int np, int max_output_deps)
{
    /* First, if we have already allocated the list but it is now too tight,
     * lets redo it at the right size */
    if( dague_remote_dep_inited && (max_output_deps > (int)dague_remote_dep_context.max_dep_count) )
    {
        remote_deps_allocation_fini();
    }

    if( 0 == dague_remote_dep_inited ) {
        /* compute the maximum size of the dependencies array */
        int rankbits_size = sizeof(uint32_t) * ((np + 31)/32);
        dague_remote_deps_t fake_rdep;

        dague_remote_dep_context.max_dep_count = max_output_deps;
        dague_remote_dep_context.max_nodes_number = np;
        dague_remote_dep_context.elem_size =
            /* sizeof(dague_remote_deps_t+outputs+padding) */
            ((intptr_t)&fake_rdep.output[dague_remote_dep_context.max_dep_count])-(intptr_t)&fake_rdep +
            /* One rankbits fw array per output param */
            dague_remote_dep_context.max_dep_count * rankbits_size +
            /* One extra rankbit to track the delivery of Activates */
            rankbits_size;
        dague_lifo_construct(&dague_remote_dep_context.freelist);
        dague_remote_dep_inited = 1;
    }

    assert( (int)dague_remote_dep_context.max_dep_count >= max_output_deps );
    assert( (int)dague_remote_dep_context.max_nodes_number >= np );
}


void remote_deps_allocation_fini(void)
{
    dague_remote_deps_t* rdeps;

    if(1 == dague_remote_dep_inited) {
        while(NULL != (rdeps = (dague_remote_deps_t*) dague_lifo_pop(&dague_remote_dep_context.freelist))) {
            free(rdeps);
        }
        dague_lifo_destruct(&dague_remote_dep_context.freelist);
    }
    dague_remote_dep_inited = 0;
}

#endif /* DISTRIBUTED */

