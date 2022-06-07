/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/remote_dep.h"
#include "parsec/scheduling.h"
#include "parsec/execution_stream.h"
#include "parsec/data_internal.h"
#include "parsec/arena.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"
#include <stdio.h>

/**
 * Indicator for the status of the communication engine. The following values are valid:
 * - -1: the engine is not initialized (e.g. MPI is not up and running)
 * -  0: any value > 0 indicate communication capabilities
 * -  1: the communication capabilities are enabled (internal engine is initialized)
 * -  2: communication thread is running
 * -  3: communication thread is up but sleeping
 */
int parsec_communication_engine_up = -1;
int parsec_comm_output_stream = 0;
int parsec_comm_verbose = 0;

#ifdef DISTRIBUTED

/* comm_yield mode: see valid values in the corresponding mca_register */
int comm_yield = 1;
/* comm_yield_duration (ns) */
int comm_yield_ns = 5000;
/* comm_thread_multiple: see values in the corresponding mca_register */
int parsec_param_comm_thread_multiple = -1;

static int remote_dep_bcast_star_child(int me, int him);
#ifdef PARSEC_DIST_COLLECTIVES
/* comm_coll_bcast: see values in the corresponding mca_register */
static int parsec_param_comm_coll_bcast = 1;
static int remote_dep_bcast_chainpipeline_child(int me, int him);
static int remote_dep_bcast_binomial_child(int me, int him);
static int (*remote_dep_bcast_child)(int me, int him) = remote_dep_bcast_chainpipeline_child;
#else
#define remote_dep_bcast_child(me, him) remote_dep_bcast_start_child(me, him)
#endif

int remote_dep_bind_thread(parsec_context_t* context);

/* Clear the already forwarded remote dependency matrix */
static inline void
remote_dep_reset_forwarded(parsec_execution_stream_t* es,
                           parsec_remote_deps_t* rdeps)
{
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "fw reset\tcontext %p deps %p", (void*)es, rdeps);
    memset(rdeps->remote_dep_fw_mask, 0,
           es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void
remote_dep_mark_forwarded(parsec_execution_stream_t* es,
                          parsec_remote_deps_t* rdeps,
                          int rank)
{
    uint32_t boffset;

    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "fw mark\tREMOTE rank %d", rank);
    boffset = rank / (8 * sizeof(uint32_t));
    assert(boffset <= es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
    (void)es;
    rdeps->remote_dep_fw_mask[boffset] |= ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int
remote_dep_is_forwarded(parsec_execution_stream_t* es,
                        parsec_remote_deps_t* rdeps,
                        int rank)
{
    uint32_t boffset, mask;

    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "fw test\tREMOTE rank %d (value=%x)", rank, (int) (rdeps->remote_dep_fw_mask[boffset] & mask));
    (void)es;
    return (int) ((rdeps->remote_dep_fw_mask[boffset] & mask) != 0);
}

#if 0
/* make sure we don't leave before serving all data deps */
static inline void
remote_dep_inc_flying_messages(parsec_taskpool_t* handle)
{
    assert( handle->nb_pending_actions > 0 );
    (void)parsec_atomic_fetch_inc_int32( &(handle->nb_pending_actions) );
}

/* allow for termination when all deps have been served */
static inline void
remote_dep_dec_flying_messages(parsec_taskpool_t *handle)
{
    (void)parsec_taskpool_update_runtime_nbtask(handle, -1);
}
#endif

/* Mark that ncompleted of the remote deps are finished, and return the remote dep to
 * the free items queue if it is now done */
int
remote_dep_complete_and_cleanup(parsec_remote_deps_t** deps,
                                int ncompleted)
{
    int32_t saved = parsec_atomic_fetch_sub_int32(&(*deps)->pending_ack, ncompleted) - ncompleted;
    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "Complete %d (%d left) outputs of dep %p%s",
            ncompleted, saved, *deps,
            (0 == saved ? " (decrease inflight)" : ""));
    if(0 == saved) {
        /**
         * Decrease the refcount of each output data once to mark the completion
         * of all communications related to the task. This is not optimal as it
         * increases the timespan of a data, but it is much easier to implement.
         */
        for( int i = 0; (*deps)->outgoing_mask >> i; i++ )
            if( (1U << i) & (*deps)->outgoing_mask ) {
                assert( (*deps)->output[i].count_bits );
                if( NULL != (*deps)->output[i].data.data ) { /* if not CONTROL */
                    if( PARSEC_TASKPOOL_TYPE_DTD == (*deps)->taskpool->taskpool_type ) {
                        (void)parsec_atomic_fetch_dec_int32(&(*deps)->output[i].data.data->readers);
                    }
                    PARSEC_DATA_COPY_RELEASE((*deps)->output[i].data.data);
                }
            }
        (*deps)->outgoing_mask = 0;
        if(ncompleted)
            remote_dep_dec_flying_messages((*deps)->taskpool);
        remote_deps_free(*deps);
        *deps = NULL;
        return 1;
    }
    return 0;
}

parsec_remote_deps_t* remote_deps_allocate( parsec_lifo_t* lifo )
{
    parsec_remote_deps_t* remote_deps = (parsec_remote_deps_t*)parsec_lifo_pop(lifo);
    uint32_t i, rank_bit_size;

    if( NULL == remote_deps ) {
        char *ptr;
        remote_deps = (parsec_remote_deps_t*)parsec_lifo_item_alloc( lifo, parsec_remote_dep_context.elem_size );
        PARSEC_VALGRIND_MEMPOOL_ALLOC(lifo,
                                      ((unsigned char *)remote_deps)+sizeof(parsec_list_item_t),
                                      parsec_remote_dep_context.elem_size - sizeof(parsec_list_item_t));
        remote_deps->origin = lifo;
        remote_deps->taskpool = NULL;
        ptr = (char*)(&(remote_deps->output[parsec_remote_dep_context.max_dep_count]));
        rank_bit_size = sizeof(uint32_t) * ((parsec_remote_dep_context.max_nodes_number + 31) / 32);
        memset(ptr, 0, rank_bit_size * parsec_remote_dep_context.max_dep_count);
        for( i = 0; i < parsec_remote_dep_context.max_dep_count; i++ ) {
            PARSEC_OBJ_CONSTRUCT(&remote_deps->output[i].super, parsec_list_item_t);
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
                (int)(parsec_remote_dep_context.elem_size - rank_bit_size));
    } else {
        PARSEC_VALGRIND_MEMPOOL_ALLOC(lifo,
                                      ((unsigned char *)remote_deps)+sizeof(parsec_list_item_t),
                                      parsec_remote_dep_context.elem_size - sizeof(parsec_list_item_t));
    }
    assert(NULL == remote_deps->taskpool);
    remote_deps->max_priority    = 0xffffffff;
    remote_deps->root            = -1;
    remote_deps->pending_ack     = 0;
    remote_deps->incoming_mask   = 0;
    remote_deps->outgoing_mask   = 0;
    PARSEC_DEBUG_VERBOSE(30, parsec_comm_output_stream, "remote_deps_allocate: %p", remote_deps);
    return remote_deps;
}

inline void remote_deps_free(parsec_remote_deps_t* deps)
{
    uint32_t k, a;
    assert(0 == deps->pending_ack);
    assert(0 == deps->incoming_mask);
    assert(0 == deps->outgoing_mask);
    for( k = 0; k < parsec_remote_dep_context.max_dep_count; k++ ) {
        if( 0 == deps->output[k].count_bits ) continue;
        for(a = 0; a < (parsec_remote_dep_context.max_nodes_number + 31)/32; a++)
            deps->output[k].rank_bits[a] = 0;
        deps->output[k].count_bits = 0;
#if defined(PARSEC_DEBUG_PARANOID)
        deps->output[k].data.data   = NULL;
        deps->output[k].data.local.arena  = NULL;
        deps->output[k].data.local.src_displ = deps->output[k].data.local.dst_displ = 0xFFFFFFFF;
        deps->output[k].data.local.src_datatype = deps->output[k].data.local.dst_datatype = PARSEC_DATATYPE_NULL;
        deps->output[k].data.local.src_count = deps->output[k].data.local.dst_count = -1;
        deps->output[k].data.remote.arena  = NULL;
        deps->output[k].data.remote.src_displ = deps->output[k].data.remote.dst_displ = 0xFFFFFFFF;
        deps->output[k].data.remote.src_datatype = deps->output[k].data.remote.dst_datatype = PARSEC_DATATYPE_NULL;
        deps->output[k].data.remote.src_count = deps->output[k].data.remote.dst_count = -1;
#endif
    }
    PARSEC_DEBUG_VERBOSE(30, parsec_comm_output_stream, "remote_deps_free: %p mask %x", deps, deps->outgoing_mask);
#if defined(PARSEC_DEBUG_PARANOID)
    memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
    deps->taskpool      = NULL;
    parsec_lifo_push(deps->origin, (parsec_list_item_t*)deps);
    PARSEC_VALGRIND_MEMPOOL_FREE(deps->origin, ((unsigned char *)deps)+sizeof(parsec_list_item_t));
}

#endif

#if 0
#ifdef PARSEC_HAVE_MPI
#include "remote_dep_mpi.c"

#else
#endif /* NO TRANSPORT */

#endif

#ifdef DISTRIBUTED

#include "parsec/utils/mca_param.h"

int parsec_remote_dep_init(parsec_context_t* context)
{
    parsec_mca_param_reg_int_name("runtime", "comm_thread_yield", "Controls the yielding behavior of the communication thread (if applicable).\n"
                                                                  "  0: the communication thread never yield.\n"
                                                                  "  1: the communication thread remain active when communication are pending.\n"
                                                                  "  2: the communication thread yields as soon as it idles.",
                                 false, false, comm_yield, &comm_yield);
    parsec_mca_param_reg_int_name("runtime", "comm_thread_yield_duration", "Controls how long (in nanoseconds) the communication thread yields (if applicable).",
                                  false, false, comm_yield_ns, &comm_yield_ns);
    parsec_mca_param_reg_int_name("runtime", "comm_thread_multiple", "Controls the threaded access to the communication thread.\n"
            " -1: the communication thread access is automatically selected based on transport capabilities (e.g., MPI_THREAD_MULTIPLE).\n"
            "  0: the communication thread access is serialized.\n"
            "  1: the communication thread access is multiple (if the underlying transports allows (e.g., MPI_THREAD_MULTIPLE).",
                                  false, false, parsec_param_comm_thread_multiple, &parsec_param_comm_thread_multiple);
    parsec_mca_param_reg_int_name("comm", "verbose",
                                  "Set the output level for the communication engine messages"
                                  ", 0: Errors only"
                                  ", 1: Warnings (minimum recommended)"
                                  ", 2: Info (default)"
                                  ", 3-4: User Debug"
                                  ", 5-9: Devel Debug"
                                  ", >=10: Chatterbox Debug"
#if !defined(PARSEC_DEBUG_PARANOID) || !defined(PARSEC_DEBUG_NOISIER) || !defined(PARSEC_DEBUG_HISTORY)
                                  " (heaviest debug output available only when compiling with PARSEC_DEBUG_PARANOID, PARSEC_DEBUG_NOISIER and/or PARSEC_DEBUG_HISTORY in ccmake)"
#endif
                                  , false, false, 1, &parsec_comm_verbose);
    if( parsec_comm_verbose >= 0 ) {
        parsec_comm_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_comm_output_stream, parsec_comm_verbose);
    } else {
        parsec_comm_output_stream = parsec_debug_output;
    }

#ifdef PARSEC_DIST_COLLECTIVES
    parsec_mca_param_reg_int_name("runtime", "comm_coll_bcast", "Controls the default broadcast algorithm topology.\n"
                                                                "  0: star topology (direct one to all).\n"
                                                                "  1: chain topology.\n"
                                                                "  2: binomial topology.\n",
                                  false, false, parsec_param_comm_coll_bcast, &parsec_param_comm_coll_bcast);
    switch(parsec_param_comm_coll_bcast) {
    case 0:
        remote_dep_bcast_child = remote_dep_bcast_star_child;
        break;
    case 1:
        remote_dep_bcast_child = remote_dep_bcast_chainpipeline_child;
        break;
    case 2:
        remote_dep_bcast_child = remote_dep_bcast_binomial_child;
        break;
    default:
        parsec_warning("Invalid collective type requested %d; using star topology.", parsec_param_comm_coll_bcast);
        remote_dep_bcast_child = remote_dep_bcast_star_child;
        break;
    }
#endif

    (void)remote_dep_init(context);

    context->remote_dep_fw_mask_sizeof = 0;
    if(context->nb_nodes > 1)
        context->remote_dep_fw_mask_sizeof = ((context->nb_nodes + 31) / 32) * sizeof(uint32_t);

    return context->nb_nodes;
}

int parsec_remote_dep_fini(parsec_context_t* context)
{
    int rc = remote_dep_fini(context);
    remote_deps_allocation_fini();
    return rc;
}

int parsec_remote_dep_on(parsec_context_t* context)
{
    return remote_dep_on(context);
}

int parsec_remote_dep_off(parsec_context_t* context)
{
    return remote_dep_off(context);
}

int parsec_remote_dep_set_ctx( parsec_context_t* context, intptr_t opaque_comm_ctx )
{
    return remote_dep_set_ctx( context, opaque_comm_ctx );
}

int parsec_remote_dep_progress(parsec_execution_stream_t* es)
{
    return remote_dep_progress(es, 1);
}

int parsec_remote_dep_new_taskpool(parsec_taskpool_t* tp) {
    return remote_dep_new_taskpool(tp);
}

static int remote_dep_bcast_star_child(int me, int him)
{
    (void)him;
    if(me == 0) return 1;
    else return 0;
}

#ifdef PARSEC_DIST_COLLECTIVES

static int remote_dep_bcast_chainpipeline_child(int me, int him)
{
    assert(him >= 0);
    if(me == -1) return 0;
    if(him == me+1) return 1;
    return 0;
}

static int remote_dep_bcast_binomial_child(int me, int him)
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

/**
 * This function is called from the successor iterator in order to rebuilt
 * the information needed to propagate the collective in a meaningful way. In
 * other words it reconstruct the entire information as viewed by the root of
 * the collective. This information is stored in the corresponding output
 * structures. In addition, this function compute the set of data currently
 * available locally and can be propagated to our predecessors.
 */
parsec_ontask_iterate_t
parsec_gather_collective_pattern(parsec_execution_stream_t *es,
                                 const parsec_task_t *newcontext,
                                 const parsec_task_t *oldcontext,
                                 const parsec_dep_t* dep,
                                 parsec_dep_data_description_t* data,
                                 int src_rank, int dst_rank, int dst_vpid,
                                 data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                                 void *param)
{
    (void)successor_repo; (void) successor_repo_key;
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];
    const int _array_pos  = dst_rank / (8 * sizeof(uint32_t));
    const int _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));

    if( dst_rank == es->virtual_process->parsec_context->my_rank )
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
    (void)oldcontext; (void)dst_vpid; (void)data; (void)src_rank;
    return PARSEC_ITERATE_CONTINUE;
}

/**
 * This is the local continuation of a collective pattern. Upon receiving an
 * activation from the predecessor the first thing is to retrieve all the data
 * needed locally (this is a super-set of the data to be propagated down
 * the collective tree). Thus, once all the data become available locally, this
 * function is called to start propagating the activation order and the data.
 */
int parsec_remote_dep_propagate(parsec_execution_stream_t* es,
                                const parsec_task_t* task,
                                parsec_remote_deps_t* deps)
{
    const parsec_task_class_t* tc = task->task_class;
    uint32_t dep_mask = 0;

    assert(deps->root != es->virtual_process->parsec_context->my_rank );
    /* If I am not the root of the collective I must rebuild the same
     * information as the root, i.e. the exact same propagation tree as the
     * initiator.
     */
    assert(0 == deps->outgoing_mask);

    /* We need to convert from a dep_datatype_index mask into a dep_index mask */
    for(int i = 0; NULL != tc->out[i]; i++ )
        for(int j = 0; NULL != tc->out[i]->dep_out[j]; j++ )
            if(deps->msg.output_mask & (1U << tc->out[i]->dep_out[j]->dep_datatype_index))
                dep_mask |= (1U << tc->out[i]->dep_out[j]->dep_index);

    tc->iterate_successors(es, task,
                           dep_mask | PARSEC_ACTION_RELEASE_REMOTE_DEPS,
                           parsec_gather_collective_pattern,
                           deps);

    return parsec_remote_dep_activate(es, task, deps, deps->msg.output_mask);
}
#endif

/**
 *
 */
int parsec_remote_dep_activate(parsec_execution_stream_t* es,
                               const parsec_task_t* task,
                               parsec_remote_deps_t* remote_deps,
                               uint32_t propagation_mask)
{
    const parsec_task_class_t* tc = task->task_class;
    int i, my_idx, idx, current_mask, keeper = 0;
    unsigned int array_index, count, bit_index;
    struct remote_dep_output_param_s* output;

    assert(es->virtual_process->parsec_context->nb_nodes > 1);

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task);
#endif

    remote_dep_reset_forwarded(es, remote_deps);
    remote_deps->taskpool    = task->taskpool;
    /* Safe-keep the propagation mask (it must be packed in the message) */
    remote_deps->msg.output_mask = propagation_mask;
    remote_deps->msg.deps        = (uintptr_t)remote_deps;
    remote_deps->msg.taskpool_id   = task->taskpool->taskpool_id;
    remote_deps->msg.task_class_id = tc->task_class_id;
    for(i = 0; i < tc->nb_locals; i++) {
        remote_deps->msg.locals[i] = task->locals[i];
    }
#if defined(PARSEC_DEBUG_PARANOID)
    /* make valgrind happy */
    memset(&remote_deps->msg.locals[i], 0, (MAX_LOCAL_COUNT - i) * sizeof(int));
#endif

    /* Mark the root of the collective as rank 0 */
    remote_dep_mark_forwarded(es, remote_deps, remote_deps->root);
    assert((propagation_mask & remote_deps->outgoing_mask) == remote_deps->outgoing_mask);

    for( i = 0; propagation_mask >> i; i++ ) {
        if( !((1U << i) & propagation_mask )) continue;
        output = &remote_deps->output[i];
        assert( 0 != output->count_bits );

        my_idx = (remote_deps->root == es->virtual_process->parsec_context->my_rank) ? 0 : -1;
        idx = 0;
        /**
         * Increase the refcount of each local output data once, to ensure the
         * data is protected during the entire execution of the communication,
         * independent on what is happening with the data outside of the
         * communication engine.
         */
        if( (remote_deps->outgoing_mask & (1U<<i)) && (NULL != output->data.data) ) {
            /* if propagated and not a CONTROL */
            /* This assert is not correct anymore, we don't need and arena to send to a remote
             * assert(NULL != output->data.remote.arena);*/
            assert( !parsec_is_CTL_dep(output->data) );
            PARSEC_OBJ_RETAIN(output->data.data);
        }

        for( array_index = count = 0; count < remote_deps->output[i].count_bits; array_index++ ) {
            current_mask = output->rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; current_mask != 0; bit_index++ ) {
                if( !(current_mask & (1 << bit_index)) ) continue;

                int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                assert((rank >= 0) && (rank < es->virtual_process->parsec_context->nb_nodes));

                current_mask ^= (1 << bit_index);
                count++;

                if(remote_dep_is_forwarded(es, remote_deps, rank)) {  /* already in the counting */
                    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "[%d:%d] task %s my_idx %d idx %d rank %d -- skip (already done)",
                            remote_deps->root, i, tmp, my_idx, idx, rank);
                    continue;
                }
                idx++;
                if(my_idx == -1) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "[%d:%d] task %s my_idx %d idx %d rank %d -- skip",
                            remote_deps->root, i, tmp, my_idx, idx, rank);
                    if(rank == es->virtual_process->parsec_context->my_rank) {
                        my_idx = idx;
                    }
                    remote_dep_mark_forwarded(es, remote_deps, rank);
                    continue;
                }
                PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, " TOPO\t%s\troot=%d\t%d (d%d) -? %d (dna)",
                        tmp, remote_deps->root, es->virtual_process->parsec_context->my_rank, my_idx, rank);

                int remote_dep_bcast_child_permits = 0;
                /* Right now DTD only supports a star broadcast topology */
                if( PARSEC_TASKPOOL_TYPE_DTD == task->taskpool->taskpool_type ) {
                    remote_dep_bcast_child_permits = remote_dep_bcast_star_child(my_idx, idx);
                } else {
#ifdef PARSEC_DIST_COLLECTIVES
                    remote_dep_bcast_child_permits = remote_dep_bcast_child(my_idx, idx);
#else
                    remote_dep_bcast_child_permits = remote_dep_bcast_star_child(my_idx, idx);
#endif  /* PARSEC_DIST_COLLECTIVES */
                }

                if(remote_dep_bcast_child_permits) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "[%d:%d] task %s my_idx %d idx %d rank %d -- send (%x)",
                            remote_deps->root, i, tmp, my_idx, idx, rank, remote_deps->outgoing_mask);
                    assert(remote_deps->outgoing_mask & (1U<<i));
#if defined(PARSEC_DEBUG_NOISIER)
                    for(int flow_index = 0; NULL != task->task_class->out[flow_index]; flow_index++) {
                        if( task->task_class->out[flow_index]->flow_datatype_mask & (1<<i) ) {
                            assert( NULL != task->task_class->out[flow_index] );
                            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, " TOPO\t%s flow %s root=%d\t%d (d%d) -> %d (d%d)",
                                    tmp, task->task_class->out[flow_index]->name, remote_deps->root,
                                    es->virtual_process->parsec_context->my_rank, my_idx, rank, idx);
                            break;
                        }
                    }
#endif  /* PARSEC_DEBUG_NOISIER */
                    assert(output->parent->taskpool == task->taskpool);
                    if( 0 == parsec_atomic_fetch_inc_int32(&remote_deps->pending_ack) ) {
                        keeper = 1;
                        /* Let the engine know we're working to activate the dependencies remotely */
                        remote_dep_inc_flying_messages(task->taskpool);
                        /* We need to increase the pending_ack to make the deps persistant until the
                         * end of this function.
                         */
                        (void)parsec_atomic_fetch_inc_int32(&remote_deps->pending_ack);
                    }
                    remote_dep_send(es, rank, remote_deps);
                } else {
                    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "[%d:%d] task %s my_idx %d idx %d rank %d -- skip (not my direct descendant)",
                            remote_deps->root, i, tmp, my_idx, idx, rank);
                }
                assert(!remote_dep_is_forwarded(es, remote_deps, rank));
                remote_dep_mark_forwarded(es, remote_deps, rank);
            }
        }
    }
    remote_dep_complete_and_cleanup(&remote_deps, (keeper ? 1 : 0));
    return 0;
}

parsec_remote_dep_context_t parsec_remote_dep_context;
int parsec_remote_dep_inited = 0;

/* THIS FUNCTION MUST NOT BE CALLED WHILE REMOTE DEP IS ON.
 * NOT THREAD SAFE (AND SHOULD NOT BE) */
void remote_deps_allocation_init(int np, int max_output_deps)
{
    /* First, if we have already allocated the list but it is now too tight,
     * lets redo it at the right size */
    if( parsec_remote_dep_inited &&
        (max_output_deps > (int)parsec_remote_dep_context.max_dep_count) ) {
        remote_deps_allocation_fini();
    }

    if( 0 == parsec_remote_dep_inited ) {
        /* compute the maximum size of the dependencies array */
        int rankbits_size = sizeof(uint32_t) * ((np + 31)/32);
        parsec_remote_deps_t fake_rdep;

        parsec_remote_dep_context.max_dep_count = max_output_deps;
        parsec_remote_dep_context.max_nodes_number = np;
        parsec_remote_dep_context.elem_size =
            /* sizeof(parsec_remote_deps_t+outputs+padding) */
            ((intptr_t)&fake_rdep.output[parsec_remote_dep_context.max_dep_count])-(intptr_t)&fake_rdep +
            /* One rankbits fw array per output param */
            parsec_remote_dep_context.max_dep_count * rankbits_size +
            /* One extra rankbit to track the delivery of Activates */
            rankbits_size;
        PARSEC_OBJ_CONSTRUCT(&parsec_remote_dep_context.freelist, parsec_lifo_t);
        PARSEC_VALGRIND_CREATE_MEMPOOL(&parsec_remote_dep_context.freelist, 0, 1);
        parsec_remote_dep_inited = 1;
    }

    assert( (int)parsec_remote_dep_context.max_dep_count >= max_output_deps );
    assert( (int)parsec_remote_dep_context.max_nodes_number >= np );
}


void remote_deps_allocation_fini(void)
{
    if(1 == parsec_remote_dep_inited) {
        parsec_remote_deps_t* rdeps;
        while(NULL != (rdeps = (parsec_remote_deps_t*) parsec_lifo_pop(&parsec_remote_dep_context.freelist))) {
            free(rdeps);
        }
        PARSEC_OBJ_DESTRUCT(&parsec_remote_dep_context.freelist);
        PARSEC_VALGRIND_DESTROY_MEMPOOL(&parsec_remote_dep_context.freelist);
    }
    parsec_remote_dep_inited = 0;
}

/* Bind the communication thread on an unused core if possible */
int remote_dep_bind_thread(parsec_context_t* context)
{
#if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    char *str = NULL;
    if( context->comm_th_core >= 0 ) {
        /* Bind to the specified core */
        if(parsec_bindthread(context->comm_th_core, -1) == context->comm_th_core) {

            /* Check if this core is not used by a computation thread */
            if( hwloc_bitmap_isset(context->cpuset_free_mask, context->comm_th_core) ) {
                /* The thread enjoys an exclusive core. Force disable comm_yield. */
                parsec_debug_verbose(4, parsec_comm_output_stream, "Communication thread bound to physical core %d (without yield back-off)",  context->comm_th_core);
                comm_yield = 0;
            } else {
                /* The thread shares the core. Let comm_yield as user-set. */
                parsec_debug_verbose(4, parsec_comm_output_stream, "Communication thread is bound to core %d (with%s yield back-off); This core is also hosting a compute execution unit", context->comm_th_core, comm_yield? "": "out");
            }
        } else {
            /* There is no guarantee the thread doesn't share the core. Let comm_yield as user-set. */
            parsec_warning("Request to bind the communication thread on core %d failed.", context->comm_th_core);
        }
    } else {
        /* bind the communication thread to any available core (which means described by the
         * binding scheme but not used by a computational thread), or if no such core exists
         * as a floating thread on all computational cores.
         */
        if( !hwloc_bitmap_iszero(context->cpuset_free_mask) ) {
            if( parsec_bindthread_mask(context->cpuset_free_mask) > -1 ) {
                hwloc_bitmap_asprintf(&str, context->cpuset_free_mask);
                /* The thread enjoys an exclusive core. Force disable comm_yield. */
                comm_yield = 0;
            }
        } else {
            if( parsec_bindthread_mask(context->cpuset_allowed_mask) > -1 ) {
                hwloc_bitmap_asprintf(&str, context->cpuset_allowed_mask);
                /* There is no guarantee the thread doesn't share the core. Let comm_yield as user-set. */
            }
        }
        parsec_debug_verbose(4, parsec_comm_output_stream,
                            "Communication thread bound on the cpu mask %s (with%s yield back-off)",
                            str? str: "NOT BOUND", (comm_yield ? "" : "out"));
        if(str) free(str);
    }
#else /* NO PARSEC_HAVE_HWLOC */
    /* If we don't have hwloc, try to bind the thread on the core #nbcore as the
     * default strategy disributed the computation threads from core 0 to nbcore-1 */
    int p, nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }
    int boundto = parsec_bindthread(nb_total_comp_threads, -1);
    if (boundto != nb_total_comp_threads) {
        parsec_debug_verbose(4, parsec_comm_output_stream, "Communication thread floats");
    } else {
        parsec_debug_verbose(4, parsec_comm_output_stream, "Communication thread bound to physical core %d", boundto);
        /* The thread (presumably) enjoys an exclusive core. Force disable comm_yield. */
        comm_yield = 0;
    }
#endif /* NO PARSEC_HAVE_HWLOC */
    return 0;
}

#endif /* DISTRIBUTED */

