/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/remote_dep.h"
#include "dague/scheduling.h"
#include "dague/execution_unit.h"
#include "dague/data_internal.h"
#include "dague/arena.h"
#include <stdio.h>

/**
 * Indicator for the status of the communication engine. The following values are valid:
 * - -1: the engine is not initialized (e.g. MPI is not up and running)
 * -  0: any value > 0 indicate communication capabilities
 * -  1: the communication capabilities are enabled (internal engine is initialized)
 * -  2: communication thread is running
 * -  3: communication thread is up but sleeping
 */
int dague_communication_engine_up = -1;

#ifdef DISTRIBUTED

static int remote_dep_bind_thread(dague_context_t* context);

/* Clear the already forwarded remote dependency matrix */
static inline void
remote_dep_reset_forwarded(dague_execution_unit_t* eu_context,
                           dague_remote_deps_t* rdeps)
{
    DEBUG3(("fw reset\tcontext %p deps %p\n", (void*)eu_context, rdeps));
    memset(rdeps->remote_dep_fw_mask, 0,
           eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void
remote_dep_mark_forwarded(dague_execution_unit_t* eu_context,
                          dague_remote_deps_t* rdeps,
                          int rank)
{
    uint32_t boffset;

    DEBUG3(("fw mark\tREMOTE rank %d\n", rank));
    boffset = rank / (8 * sizeof(uint32_t));
    assert(boffset <= eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
    (void)eu_context;
    rdeps->remote_dep_fw_mask[boffset] |= ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int
remote_dep_is_forwarded(dague_execution_unit_t* eu_context,
                        dague_remote_deps_t* rdeps,
                        int rank)
{
    uint32_t boffset, mask;

    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->virtual_process->dague_context->remote_dep_fw_mask_sizeof);
    DEBUG3(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (rdeps->remote_dep_fw_mask[boffset] & mask)));
    (void)eu_context;
    return (int) ((rdeps->remote_dep_fw_mask[boffset] & mask) != 0);
}


/* make sure we don't leave before serving all data deps */
static inline void
remote_dep_inc_flying_messages(dague_handle_t *dague_handle,
                               dague_context_t* ctx)
{
    assert( dague_handle->nb_local_tasks > 0 );
    dague_atomic_inc_32b( &(dague_handle->nb_local_tasks) );
    (void)ctx;
}

/* allow for termination when all deps have been served */
static inline void
remote_dep_dec_flying_messages(dague_handle_t *dague_handle,
                               dague_context_t* ctx)
{
    __dague_complete_task(dague_handle, ctx);
}

/* Mark that ncompleted of the remote deps are finished, and return the remote dep to
 * the free items queue if it is now done */
static inline int
remote_dep_complete_and_cleanup(dague_remote_deps_t** deps,
                                int ncompleted,
                                dague_context_t* ctx)
{
    int32_t saved = dague_atomic_sub_32b((int32_t*)&(*deps)->pending_ack, ncompleted);
    DEBUG2(("Complete %d (%d left) outputs of dep %p%s\n",
            ncompleted, saved, *deps,
            (0 == saved ? " (decrease inflight)" : "")));
    if(0 == saved) {
        /**
         * Decrease the refcount of each output data once to mark the completion
         * of all communications related to the task. This is not optimal as it
         * increases the timespan of a data, but it is much easier to implement.
         */
        for( int i = 0; (*deps)->outgoing_mask >> i; i++ )
            if( (1U << i) & (*deps)->outgoing_mask ) {
                assert( (*deps)->output[i].count_bits );
                if( NULL != (*deps)->output[i].data.data )  /* if not CONTROL */
                    DAGUE_DATA_COPY_RELEASE((*deps)->output[i].data.data);
            }
        (*deps)->outgoing_mask = 0;
        if(ncompleted)
            remote_dep_dec_flying_messages((*deps)->dague_handle, ctx);
        remote_deps_free(*deps);
        *deps = NULL;
        return 1;
    }
    return 0;
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

    context->remote_dep_fw_mask_sizeof = 0;
    if(context->nb_nodes > 1)
        context->remote_dep_fw_mask_sizeof = ((context->nb_nodes + 31) / 32) * sizeof(uint32_t);

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

int dague_remote_dep_set_ctx( dague_context_t* context, void* opaque_comm_ctx )
{
    dague_remote_dep_fini( context );
    context->comm_ctx = opaque_comm_ctx;
    return dague_remote_dep_init( context );
}

int dague_remote_dep_progress(dague_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context->virtual_process[0].dague_context, 1);
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

int dague_remote_dep_new_object(dague_handle_t* obj) {
    return remote_dep_new_object(obj);
}

#ifdef DAGUE_DIST_COLLECTIVES
/**
 * This function is called from the task successor iterator in order to rebuilt
 * the information needed to propagate the collective in a meaningful way. In
 * other words it reconstruct the entire information as viewed by the root of
 * the collective. This information is stored in the corresponding output
 * structures. In addition, this function compute the set of data is currently
 * available locally and can be propagated to our predecessors.
 */
dague_ontask_iterate_t
dague_gather_collective_pattern(dague_execution_unit_t *eu,
                                const dague_execution_context_t *newcontext,
                                const dague_execution_context_t *oldcontext,
                                const dep_t* dep,
                                dague_dep_data_description_t* data,
                                int src_rank, int dst_rank, int dst_vpid,
                                void *param)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];
    const int _array_pos  = dst_rank / (8 * sizeof(uint32_t));
    const int _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));

    if( dst_rank == eu->virtual_process->dague_context->my_rank )
        deps->outgoing_mask |= (1 << dep->dep_datatype_index);

    if( !(output->rank_bits[_array_pos] & _array_mask) ) {  /* new participant */
        output->rank_bits[_array_pos] |= _array_mask;
        output->deps_mask |= (1 << dep->dep_index);
        output->count_bits++;
    }
    if(newcontext->priority > output->priority) {  /* select the priority */
        output->priority = newcontext->priority;
        if(newcontext->priority > deps->max_priority)
            deps->max_priority = newcontext->priority;
    }
    (void)eu; (void)oldcontext; (void)dst_vpid; (void)data; (void)src_rank;
    return DAGUE_ITERATE_CONTINUE;
}

/**
 * This is the local continuation of a collective pattern. Upon receiving an
 * activation from the predecessor the first thing is to retrieve all the data
 * needed locally (this is a super-set of the data to be propagated down
 * the collective tree). Thus, once all the data become available locally, this
 * function is called to start propagating the activation order and the data.
 */
int dague_remote_dep_propagate(dague_execution_unit_t* eu_context,
                               const dague_execution_context_t* task,
                               dague_remote_deps_t* deps)
{
    const dague_function_t* function = task->function;
    uint32_t dep_mask = 0;

    assert(deps->root != eu_context->virtual_process->dague_context->my_rank );
    /* If I am not the root of the collective I must rebuild the same
     * information as the root, i.e. the exact same propagation tree as the
     * initiator.
     */
    assert(0 == deps->outgoing_mask);

    /* We need to convert from a dep_datatype_index mask into a dep_index mask */
    for(int i = 0; NULL != function->out[i]; i++ )
        for(int j = 0; NULL != function->out[i]->dep_out[j]; j++ )
            if(deps->msg.output_mask & (1U << function->out[i]->dep_out[j]->dep_datatype_index))
                dep_mask |= (1U << function->out[i]->dep_out[j]->dep_index);

    function->iterate_successors(eu_context, task,
                                 dep_mask | DAGUE_ACTION_RELEASE_REMOTE_DEPS,
                                 dague_gather_collective_pattern,
                                 deps);

    return dague_remote_dep_activate(eu_context, task, deps, deps->msg.output_mask);
}
#endif

/**
 *
 */
int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                              const dague_execution_context_t* exec_context,
                              dague_remote_deps_t* remote_deps,
                              uint32_t propagation_mask)
{
    const dague_function_t* function = exec_context->function;
    int i, my_idx, idx, current_mask, keeper = 0;
    unsigned int array_index, count, bit_index;
    struct remote_dep_output_param_s* output;

    assert(eu_context->virtual_process->dague_context->nb_nodes > 1);

#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif

    remote_dep_reset_forwarded(eu_context, remote_deps);
    remote_deps->dague_handle    = exec_context->dague_handle;
    /* Safe-keep the propagation mask (it must be packed in the message) */
    remote_deps->msg.output_mask = propagation_mask;
    remote_deps->msg.deps        = (uintptr_t)remote_deps;
    remote_deps->msg.handle_id   = exec_context->dague_handle->handle_id;
    remote_deps->msg.function_id = function->function_id;
    for(i = 0; i < function->nb_locals; i++) {
        remote_deps->msg.locals[i] = exec_context->locals[i];
    }
#if defined(DAGUE_DEBUG_ENABLE)
    /* make valgrind happy */
    memset(&remote_deps->msg.locals[i], 0, (MAX_LOCAL_COUNT - i) * sizeof(int));
#endif

    /* Mark the root of the collective as rank 0 */
    remote_dep_mark_forwarded(eu_context, remote_deps, remote_deps->root);
    assert((propagation_mask & remote_deps->outgoing_mask) == remote_deps->outgoing_mask);

    for( i = 0; propagation_mask >> i; i++ ) {
        if( !((1U << i) & propagation_mask )) continue;
        output = &remote_deps->output[i];
        assert( 0 != output->count_bits );

        my_idx = (remote_deps->root == eu_context->virtual_process->dague_context->my_rank) ? 0 : -1;
        idx = 0;
        /**
         * Increase the refcount of each local output data once, to ensure the
         * data is protected during the entire execution of the communication,
         * independent on what is happening with the data outside of the
         * communication engine.
         */
        if( (remote_deps->outgoing_mask & (1U<<i)) && (NULL != output->data.data) ) {
            /* if propagated and not a CONTROL */
            assert(NULL != output->data.arena);
            assert(NULL != (void*)(intptr_t)(output->data.layout));
            OBJ_RETAIN(output->data.data);
        }

        for( array_index = count = 0; count < remote_deps->output[i].count_bits; array_index++ ) {
            current_mask = output->rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; current_mask != 0; bit_index++ ) {
                if( !(current_mask & (1 << bit_index)) ) continue;

                int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                assert((rank >= 0) && (rank < eu_context->virtual_process->dague_context->nb_nodes));

                current_mask ^= (1 << bit_index);
                count++;

                if(remote_dep_is_forwarded(eu_context, remote_deps, rank)) {  /* already in the counting */
                    DEBUG3(("[%d:%d] task %s my_idx %d idx %d rank %d -- skip (already done)\n",
                            remote_deps->root, i, tmp, my_idx, idx, rank));
                    continue;
                }
                idx++;
                if(my_idx == -1) {
                    DEBUG3(("[%d:%d] task %s my_idx %d idx %d rank %d -- skip\n",
                            remote_deps->root, i, tmp, my_idx, idx, rank));
                    if(rank == eu_context->virtual_process->dague_context->my_rank) {
                        my_idx = idx;
                    }
                    remote_dep_mark_forwarded(eu_context, remote_deps, rank);
                    continue;
                }
                DEBUG3((" TOPO\t%s\troot=%d\t%d (d%d) -? %d (dna)\n",
                        tmp, remote_deps->root, eu_context->virtual_process->dague_context->my_rank, my_idx, rank));

                if(remote_dep_bcast_child(my_idx, idx)) {
                    DEBUG3(("[%d:%d] task %s my_idx %d idx %d rank %d -- send (%x)\n",
                            remote_deps->root, i, tmp, my_idx, idx, rank, remote_deps->outgoing_mask));
                    assert(remote_deps->outgoing_mask & (1U<<i));
#if DAGUE_DEBUG_VERBOSE != 0
                    for(int flow_index = 0; NULL != exec_context->function->out[flow_index]; flow_index++) {
                        if( exec_context->function->out[flow_index]->flow_datatype_mask & (1<<i) ) {
                            assert( NULL != exec_context->function->out[flow_index] );
                            DEBUG2((" TOPO\t%s flow %s root=%d\t%d (d%d) -> %d (d%d)\n",
                                    tmp, exec_context->function->out[flow_index]->name, remote_deps->root,
                                    eu_context->virtual_process->dague_context->my_rank, my_idx, rank, idx));
                            break;
                        }
                    }
#endif  /* DAGUE_DEBUG_VERBOSE */
                    assert(output->parent->dague_handle == exec_context->dague_handle);
                    if( 1 == dague_atomic_add_32b(&remote_deps->pending_ack, 1) ) {
                        keeper = 1;
                        /* Let the engine know we're working to activate the dependencies remotely */
                        remote_dep_inc_flying_messages(exec_context->dague_handle,
                                                       eu_context->virtual_process->dague_context);
                        /* We need to increase the pending_ack to make the deps persistant until the
                         * end of this function.
                         */
                        dague_atomic_add_32b((int32_t*)&remote_deps->pending_ack, 1);
                    }
                    remote_dep_send(rank, remote_deps);
                } else {
                    DEBUG3(("[%d:%d] task %s my_idx %d idx %d rank %d -- skip (not my direct descendant)\n",
                            remote_deps->root, i, tmp, my_idx, idx, rank));
                }
                assert(!remote_dep_is_forwarded(eu_context, remote_deps, rank));
                remote_dep_mark_forwarded(eu_context, remote_deps, rank);
            }
        }
    }
    remote_dep_complete_and_cleanup(&remote_deps, (keeper ? 1 : 0),
                                    eu_context->virtual_process->dague_context);
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
    if( dague_remote_dep_inited &&
        (max_output_deps > (int)dague_remote_dep_context.max_dep_count) ) {
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
        OBJ_CONSTRUCT(&dague_remote_dep_context.freelist, dague_lifo_t);
        dague_remote_dep_inited = 1;
    }

    assert( (int)dague_remote_dep_context.max_dep_count >= max_output_deps );
    assert( (int)dague_remote_dep_context.max_nodes_number >= np );
}


void remote_deps_allocation_fini(void)
{
    if(1 == dague_remote_dep_inited) {
        dague_remote_deps_t* rdeps;
        while(NULL != (rdeps = (dague_remote_deps_t*) dague_lifo_pop(&dague_remote_dep_context.freelist))) {
            free(rdeps);
        }
        OBJ_DESTRUCT(&dague_remote_dep_context.freelist);
    }
    dague_remote_dep_inited = 0;
}

/* Bind the communication thread on an unused core if possible */
static int remote_dep_bind_thread(dague_context_t* context)
{
    do_nano = 1;

#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    char *str = NULL;
    if( context->comm_th_core >= 0 ) {
        /* Bind to the specified core */
        if(dague_bindthread(context->comm_th_core, -1) == context->comm_th_core) {
            STATUS(("Communication thread bound to physical core %d\n",  context->comm_th_core));

            /* Check if this core is not used by a computation thread */
            if( hwloc_bitmap_isset(context->index_core_free_mask, context->comm_th_core) )
                do_nano = 0;
        } else {
            /* There is no guarantee the thread doesn't share the core. Let do_nano to 1. */
            WARNING(("Request to bind the communication thread on core %d failed.\n", context->comm_th_core));
        }
    } else if( context->comm_th_core == -2 ) {
        /* Bind to the specified mask */
        hwloc_cpuset_t free_common_cores;

        /* reduce the mask to unused cores if any */
        free_common_cores=hwloc_bitmap_alloc();
        hwloc_bitmap_and(free_common_cores, context->index_core_free_mask, context->comm_th_index_mask);

        if( !hwloc_bitmap_iszero(free_common_cores) ) {
            hwloc_bitmap_copy(context->comm_th_index_mask, free_common_cores);

            do_nano = 0;
        }
        hwloc_bitmap_asprintf(&str, context->comm_th_index_mask);
        hwloc_bitmap_free(free_common_cores);
        if( dague_bindthread_mask(context->comm_th_index_mask) >= 0 ) {
            DEBUG(("Communication thread bound on the index mask %s\n", str));
        } else {
            WARNING(("Communication thread requested to be bound on the cpu mask %s \n", str));
            do_nano = 1;
        }
    } else {
        /* no binding specified
         * - bind on available cores if any,
         * - let float otherwise
         */

        if( !hwloc_bitmap_iszero(context->index_core_free_mask) ) {
            if( dague_bindthread_mask(context->index_core_free_mask) > -1 ){
                hwloc_bitmap_asprintf(&str, context->index_core_free_mask);
                DEBUG(("Communication thread bound on the cpu mask %s\n", str));
                free(str);
                do_nano = 0;
            }
        }
    }
#else /* NO HAVE_HWLOC */
    /* If we don't have hwloc, try to bind the thread on the core #nbcore as the
     * default strategy disributed the computation threads from core 0 to nbcore-1 */
    int p, nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }
    int boundto = dague_bindthread(nb_total_comp_threads, -1);
    if (boundto != nb_total_comp_threads) {
        DEBUG(("Communication thread floats\n"));
    } else {
        do_nano = 0;
        DEBUG(("Communication thread bound to physical core %d\n", boundto));
    }
#endif /* NO HAVE_HWLOC */
    return 0;
}

#endif /* DISTRIBUTED */

