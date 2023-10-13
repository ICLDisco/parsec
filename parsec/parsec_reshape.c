/**
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"

#include "parsec/utils/output.h"
#include "parsec/data_internal.h"
#include "parsec/class/list.h"
#include "parsec/remote_dep.h"
#include "parsec/datarepo.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/data_distribution.h"

#define PARSEC_UNFULFILLED_RESHAPE_PROMISE 0
#define PARSEC_FULFILLED_RESHAPE_PROMISE   1


/**
 *
 * Callback routine to clean up a reshape promise.
 *
 * @param[in] future  datacopy future.
 */
void parsec_cleanup_reshape_promise(parsec_base_future_t *future)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    if(d_fut->cb_fulfill_data_in != NULL){
        parsec_reshape_promise_description_t *future_in_data =
                ((parsec_reshape_promise_description_t *)d_fut->cb_fulfill_data_in);
        /* Release the input datacopy for the reshape, it will only be release
         * once all successors have consumed the future, in case it is needed
         * as an input for nested futures.
         */
        if(d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_TRIGGERED){
            PARSEC_DATA_COPY_RELEASE(future_in_data->data);
        }
        if(future_in_data->local != NULL){
            free(future_in_data->local);
        }
        free(future_in_data);
    }
    if(d_fut->cb_match_data_in != NULL){
        parsec_datatype_t * match_data = (parsec_datatype_t*)d_fut->cb_match_data_in;
        free(match_data);
    }
    if(d_fut->super.tracked_data != NULL){
        parsec_data_copy_t * data = (parsec_data_copy_t*) d_fut->super.tracked_data;
        PARSEC_DATA_COPY_RELEASE(data);
    }
}

/**
 *
 * Callback routine to check if the data tracked by the future matches the
 * the JDF requested shape.
 *
 * @param[in] future datacopy future.
 * @param[in] t1 matching data set up when initializing the future. In this case,
 * parsec_datatype_t [2] of the tracked datacopy.
 * @param[in] t2 parsec_dep_data_description_t passed during get_or_trigger
 * invocation.
 * @return 1 if the tracked data matches the requested shape.
 */
int
parsec_reshape_check_match_datatypes(parsec_base_future_t* f,
                                     void *t1, void *t2)
{
    (void)f;
    parsec_datatype_t *tracked_data_match = (parsec_datatype_t *)t1;
    parsec_dep_data_description_t *target = (parsec_dep_data_description_t *)t2;

    return ( (( parsec_type_match(tracked_data_match[0], target->local.src_datatype) == PARSEC_SUCCESS ) /*Same reshaping*/
               && ( parsec_type_match(tracked_data_match[1], target->local.dst_datatype) == PARSEC_SUCCESS ))
           || ((PARSEC_DATATYPE_NULL == target->local.src_datatype) /*Default tracked data*/
               && (PARSEC_DATATYPE_NULL == target->local.dst_datatype)) );
}

/**
 *
 * Auxiliary routine to create a reshape promise with the given specifications.
 *
 * @param[in] data parsec_dep_data_description_t containing the input datacopy and
 * the reshaping datatypes.
 * @param[in] type indicating PARSEC_UNFULFILLED_RESHAPE_PROMISE or
 * PARSEC_FULFILLED_RESHAPE_PROMISE.
 * @param[in] pred_repo predecessor repo from which consume when the reshape has
 * been fulfilled.
 * @param[in] pred_repo_key on the predecessor repo that is consume when the
 * reshape promise has been fulfilled.
 * @return  parsec_datacopy_future_t* new datacopy_future.
 */
static parsec_datacopy_future_t *
parsec_new_reshape_promise(parsec_dep_data_description_t* data,
                           int type)
{
    parsec_reshape_promise_description_t *future_in_data;
    parsec_datacopy_future_t *data_future;

    data_future = PARSEC_OBJ_NEW(parsec_datacopy_future_t);
    future_in_data = (parsec_reshape_promise_description_t*)malloc(
                            sizeof(parsec_reshape_promise_description_t));
    parsec_datatype_t * match_data = (parsec_datatype_t*)malloc(
                            sizeof(parsec_datatype_t)*2);

    future_in_data->data = data->data;
    future_in_data->local = (parsec_dep_type_description_t*)malloc(
                                   sizeof(parsec_dep_type_description_t));
    future_in_data->local->arena        = data->local.arena;
    future_in_data->local->src_datatype = data->local.src_datatype;
    future_in_data->local->src_count    = data->local.src_count;
    future_in_data->local->src_displ    = data->local.src_displ;
    future_in_data->local->dst_datatype = data->local.dst_datatype;
    future_in_data->local->dst_count    = data->local.dst_count;
    future_in_data->local->dst_displ    = data->local.dst_displ;
    if(type == PARSEC_UNFULFILLED_RESHAPE_PROMISE) {
        match_data[0]                       = data->local.src_datatype;
        match_data[1]                       = data->local.dst_datatype;
    } else {
        match_data[0] = match_data[1]       = future_in_data->data->dtt;
        /* JDF generated code set up src & dst to PARSEC_DATATYPE_NULL
         * to indicate no type on dependency. Correct for possible nested reshapes. */
        future_in_data->local->src_datatype = future_in_data->local->dst_datatype = future_in_data->data->dtt;
    }

#ifdef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
    future_in_data->remote_send_guard            = 0;
#endif
    future_in_data->remote_recv_guard            = 0;

    parsec_future_init(data_future, parsec_local_reshape, future_in_data,
                       parsec_reshape_check_match_datatypes, match_data,
                       parsec_cleanup_reshape_promise);
    /* We have to retain the data to count the first successor who is
     * going to consume the original data->data in order to reshape it, and
     * all other successors will use directly the reshaped data instead.
     */
    PARSEC_OBJ_RETAIN( future_in_data->data );

    return data_future;
}

/**
 *
 * Callback routine to set up a nested reshape future.
 *
 * @param[inout] future nested datacopy_future that is being created new future
 * will reshape from old_future.src_datatype to cb_data_in.dst_datatype.
 * @param[in]    old future from which derive.
 * @param[in]    cb_data_in pointer to the data parsec_dep_data_description_t that
 * used to get_or_trigger the future. Contains the specifications of the reshaping.
 */
void parsec_setup_nested_future(parsec_datacopy_future_t** future,
                                parsec_datacopy_future_t*  parent_future,
                                void * cb_data_in)
{
    parsec_dep_data_description_t *data = (parsec_dep_data_description_t *)cb_data_in;
    parsec_reshape_promise_description_t *old_cb_data_in = (parsec_reshape_promise_description_t *)parent_future->cb_fulfill_data_in;
    parsec_data_copy_t *data_src = old_cb_data_in->data;// = (parsec_data_copy_t *)parent_future->super.tracked_data;
    parsec_data_copy_t *tmp = data->data;
    data->data = data_src; /* set the input datacopy for the reshape */

    /* Save specs from JDF to update matched data */
    parsec_datatype_t match_d0 = data->local.src_datatype;
    parsec_datatype_t match_d1 = data->local.dst_datatype;

    /* Update src type with the one from the previous output dep */
    data->local.src_datatype = old_cb_data_in->local->src_datatype;
    data->local.src_count    = old_cb_data_in->local->src_count;
    data->local.src_displ    = old_cb_data_in->local->src_displ;
    *future = parsec_new_reshape_promise(data, PARSEC_UNFULFILLED_RESHAPE_PROMISE);

    /*However the match data has to match the one pass as arg */
    ((parsec_datatype_t*)(*future)->cb_match_data_in)[0] = match_d0;
    ((parsec_datatype_t*)(*future)->cb_match_data_in)[1] = match_d1;

    data->data = tmp; /* restore the contents of the dep_data_description */

#if (defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)) && defined(PARSEC_HAVE_MPI)
    char type_string[MAX_TASK_STRLEN]="UNFULFILLED";
    char orig_string[MAX_TASK_STRLEN]="NESTED";
    char type_name_src[MAX_TASK_STRLEN] = "NULL";
    char type_name_dst[MAX_TASK_STRLEN] = "NULL";
    int len;
    if(data->local.src_datatype != PARSEC_DATATYPE_NULL)
        MPI_Type_get_name(data->local.src_datatype, type_name_src, &len);
    if(data->local.dst_datatype != PARSEC_DATATYPE_NULL)
        MPI_Type_get_name(data->local.dst_datatype, type_name_dst, &len);
    PARSEC_DEBUG_VERBOSE(12, parsec_debug_output,
                         "RESHAPE_PROMISE CREATE %s %s [%p:..:%p -> ..:%p] fut %p dtt %s -> %s",
                         type_string, orig_string, data_src, data_src->dtt,
                         data->local.dst_datatype,
                         *future, type_name_src, type_name_dst);
#endif  /* (defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)) && defined(PARSEC_HAVE_MPI) */
}


/**
 *
 * Auxiliary routine to create a fulfilled/unfulfilled reshape promise if one was not
 * already created for this data description.
 * The reshape promised can be set up:
 * - on the predecessor repo: in case it is empty
 * - on the successor repo: once the predecessor repo is full.
 * NOTE: flow dependencies are ordered by type & type_remote and
 * type=UNDEFINED (no reshape, fulfilled promise) are place as the first dependency.
 * If there was no fulfilled promise set up on the predecessor before this
 * unfulfilled one, there won't be one.
 *
 * @param[in] es parsec_execution_stream_t.
 * @param[inout] data contains the description of the reshape and holds the
 * new future that is created on the first invocation (when data.data_future==NULL).
 * @param[in] src_rank, dst_rank used to check if this reshape will be used
 * for a remote communication. In that case, it will be set up and retained only
 * for the first remote successor per dep_datatype_index (set for the first one
 * for each different type_remote).
 * @param[in] dep_datatype_index on the predecessor indicating different type_remote.
 *
 * @param[in] predecessor_dep_flow_index predecessor repo info.
 * @param[inout] predecessor_repo predecessor repo info.
 * @param[in] predecessor_repo_entry predecessor repo info.
 *
 * @param[in] successor_dep_flow_index successor repo info.
 * @param[inout] successor_repo successor repo info.
 * @param[in] successor_repo_key successor repo info.
 *
 * @param[out] setup_repo repo on which the reshape has been set up.
 * @param[out] setup_repo_key key on repo on which the reshape has been set up.
 * @param[inout] ouput_usage counter for the predecessor repo usage.
 *
 * @param[in] promise_type fulfilled or unfulfilled reshape promise.
 */
static void
parsec_create_reshape_promise(parsec_execution_stream_t *es,
                              parsec_dep_data_description_t *data,
                              int src_rank, int dst_rank,
                              uint8_t dep_datatype_index,
                              uint8_t predecessor_dep_flow_index,
                              data_repo_t *predecessor_repo,
                              data_repo_entry_t *predecessor_repo_entry,
                              uint8_t successor_dep_flow_index,
                              data_repo_t *successor_repo,
                              parsec_key_t successor_repo_key,
                              data_repo_t **setup_repo,
                              parsec_key_t *setup_repo_key,
                              uint32_t *output_usage,
                              int promise_type)
{
    parsec_reshape_promise_description_t *future_in_data;
    uint8_t setup_flow_index;
    data_repo_entry_t *setup_repo_entry = NULL;
    int new_future = 0;
    (void)src_rank; (void)dst_rank; (void)dep_datatype_index; (void)future_in_data;

    /* This routine relies on iterate_succcessors running on outputs ordered by
     * reshape type:
     * - first, the dependencies with an undefined reshape type.
     * - subsequent dependencies will be in order of reshape type.
     * This enables the usage of data->data_future to pass along the reshape promise
     * shared among output dependencies with the same reshape type.
     * When the reshape type changes, data->data_future must be set to NULL.
     */

    assert( predecessor_repo_entry != NULL ); /* pred_entry is now created at the beginning of release_deps_of */
    *setup_repo = predecessor_repo;
    *setup_repo_key = predecessor_repo_entry->ht_item.key;
    setup_flow_index = predecessor_dep_flow_index;
    setup_repo_entry = predecessor_repo_entry;

    if ( predecessor_repo_entry->data[predecessor_dep_flow_index] != NULL ) {
        if(promise_type == PARSEC_UNFULFILLED_RESHAPE_PROMISE) {
            /* New unfulfilled reshape promises are set up on the succcessor repo
             * in case the predecessor repo is already occupied. */
            *setup_repo = successor_repo;
            *setup_repo_key = successor_repo_key;
            setup_flow_index = successor_dep_flow_index;
            setup_repo_entry = data_repo_lookup_entry_and_create(es, successor_repo,
                                                               successor_repo_key);
        } else {
            data->data_future = (parsec_datacopy_future_t*)predecessor_repo_entry->data[predecessor_dep_flow_index];
            /* New fulfilled promises are set up on the successor repo in case
             * they track a data different to the one tracked by the predecessor repo. */
            if(data->data != parsec_future_get_or_trigger(data->data_future, NULL, NULL, NULL, NULL)) {
                /* This case happens when a predecessor sends multiple copies with
                 * different shapes (type_remote) on the same output flow to a set
                 * of successors on the same remote destination node.
                 * Release_deps is invoked with the correct copy and by setting it up
                 * on the successors repo, we work around having only one pred_entry on
                 * a flow generating multiple remote outputs.
                 * (only case a fulfilled reshape promised is set up on the successors
                 * repo; no reshaping deps are the first ones on the dependency list
                 * and they fill the pred repo if they exist).
                 */
                *setup_repo = successor_repo;
                *setup_repo_key = successor_repo_key;
                setup_flow_index = successor_dep_flow_index;
                setup_repo_entry = data_repo_lookup_entry_and_create(es, successor_repo,
                                                                     successor_repo_key);
                data->data_future = NULL; /* Force the generation of a new reshape. */
            }
        }
    }


    if ( data->data_future == NULL ) {
        /* Create a new future in case one is not already available. */
        data->data_future = parsec_new_reshape_promise(data, promise_type);
        new_future = 1;

        if(promise_type == PARSEC_FULFILLED_RESHAPE_PROMISE) {
            parsec_future_set(data->data_future, data->data);
        }
    }

    future_in_data = ((parsec_reshape_promise_description_t *)data->data_future->cb_fulfill_data_in);
    assert( data->data == future_in_data->data );

#ifdef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
    if ( dst_rank != src_rank ) {
        if ( future_in_data->remote_send_guard & (1 << dep_datatype_index) ) return;
        future_in_data->remote_send_guard |= (1 << dep_datatype_index);
    }
#endif

    /* retain the future if it's being reuse. */
    if ( !new_future ) PARSEC_OBJ_RETAIN(data->data_future);

    /* Set up the reshape promise. */
    setup_repo_entry->data[setup_flow_index] = (parsec_data_copy_t *)data->data_future;
    /* Increase the usage count of the repo where the reshape has been set up. */
    if ( (*setup_repo == successor_repo) && (( *setup_repo_key == successor_repo_key )) ) {
        /* If the reshape promise is set up on the successor repo, usage limit
         * is increase now. */
        data_repo_entry_addto_usage_limit(*setup_repo, *setup_repo_key, 1);
    } else {
        /* If the reshape promise is set up on the predecessor repo, output_usage
         * is increase now to track the usage, and the limit is set on release_deps
         * later on.
         * Note, when creating inline reshape promise reading from desc, output_usage
         * will be null.
         */
        if( NULL != output_usage ) (*output_usage)++;
    }
}

/**
 *
 * Routine to set up a reshape promise invoke during the first call of
 * iterate_successors_of on release_deps_of (also includes running
 * release_deps_of after receiving data from a remote predecessor).
 *
 * Setting up reshape promises shared among local or remote successors.
 * Two scenarios:
 * - No reshaping needed: fulfilled promise set up on predecessor repo.
 * - Reshaping needed: unfullfilled promise set up on the predecessor if
 * the repo is free, otherwise on the successors repo.
 *
 * * (data->local.*_count == 0) corresponds to CTL flow.
 * * (data->local.*_datatype == PARSEC_DATATYPE_NULL) corresponds with no reshaping type.
 *
 * During release_deps_of a fake remote predecessor (from which this node has
 * received data) this routine detects PARSEC_DATATYPE_PACKED and generates
 * the appropriate reshape promises for the successsors reception datatypes.
 *
 * NOTE: flow dependencies are ordered by type & type_remote and
 * type=UNDEFINED (no reshape, fulfilled promise) are placed as the first
 * dependencies.
 *
 * @param[in] es parsec_execution_stream_t.
 * @param[in] newcontext successor parsec_task_t.
 * @param[in] oldcontext current parsec_task_t.
 * @param[in] dep parsec_dep_t.
 * @param[inout] data contains the description of the reshape and holds the
 * future that is created on the first invocation. This enables passing the future
 * along the different successors on the same dependency.
 * @param[in] src_rank source rank.
 * @param[in] dst_rank destination rank.
 * @param[in] dst_vpid
 * @param[in] successor_repo successor repo for unfulfilled reshape promises.
 * @param[in] successor_repo_key key on successor repo for the successor task.
 * @param[in] param parsec_release_dep_fct_arg_t
 * @return PARSEC_ITERATE_CONTINUE to iterate over all successors.
 */

parsec_ontask_iterate_t
parsec_set_up_reshape_promise(parsec_execution_stream_t *es,
                              const parsec_task_t *newcontext,
                              const parsec_task_t *oldcontext,
                              const parsec_dep_t* dep,
                              parsec_dep_data_description_t* data,
                              int src_rank, int dst_rank, int dst_vpid,
                              data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                              void *param)
{

    (void)dst_vpid;(void)oldcontext;
    data_repo_t *setup_repo;
    parsec_key_t setup_repo_key;
    int promise_type;

    parsec_release_dep_fct_arg_t *arg = (parsec_release_dep_fct_arg_t *)param;

    if( (data->data == NULL) /* NULL data, can this happened apart from the tests?*/
       || (data->local.src_count == 0) /* CTL FLOW*/){
        return PARSEC_ITERATE_CONTINUE;
    }

    /* Check we have a correct type on the data. Otherwise no reshaping is requested.
     * (Don't force user to defined a datatype if no distributed run.
     *  e.g. tests/interfaces/ptg/compiler_checks/write_check ) */
    assert( (data->data->dtt != PARSEC_DATATYPE_NULL)
            || ((data->local.dst_datatype == data->data->dtt)
               && (data->local.src_datatype == data->data->dtt)));

#ifndef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
    if (dst_rank != es->virtual_process->parsec_context->my_rank){
        /* avoid setting up reshape for remotes */
        return PARSEC_ITERATE_CONTINUE;
    }
#endif

    if(arg->action_mask & PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE){
        /* Faking predecessor to release local dependencies. */
        if (dst_rank != es->virtual_process->parsec_context->my_rank){
            /* if this dep is not for this rank we don't care */
            return PARSEC_ITERATE_CONTINUE;
        }
        src_rank = es->virtual_process->parsec_context->my_rank;
        if( parsec_type_match(data->data->dtt, PARSEC_DATATYPE_PACKED) != PARSEC_SUCCESS){
            /* Data has been received with the expected remote type of the
             * successor contained on data->data->dtt. */
            data->local.dst_datatype = data->local.src_datatype = data->data->dtt;
        }else{
            /* Packed data because multiple unpacking alternatives at reception. */
            const parsec_task_class_t* fct = newcontext->task_class;
            uint32_t flow_mask = (1U << dep->flow->flow_index) | 0x80000000;  /* in flow */
            int dsize;
            parsec_dep_data_description_t aux_data;

            if ( PARSEC_HOOK_RETURN_DONE == fct->get_datatype(es, newcontext, &flow_mask, &aux_data)){
                parsec_fatal("Unable to find unpacking datatype.");
            }
            data->local = aux_data.remote;
            data->local.src_datatype = PARSEC_DATATYPE_PACKED;
            MPI_Pack_size(aux_data.remote.dst_count, aux_data.remote.dst_datatype , MPI_COMM_WORLD, &dsize);
            data->local.src_count = dsize;

            /* Check if the previous future set up on iterate successor is tracking the same
             * data with the same reshaping. This can not be the case when after a reception,
             * as we may generate different reshapings from PACKED to successors remote_type.
             * (data->data_future is only clean up during iterate_successors when the predecessor
             * remote type changes, there's no info about the successor remote type).
             */
            if(data->data_future != NULL) {
                if( 0 == parsec_reshape_check_match_datatypes((parsec_base_future_t*)data->data_future,
                                                              data->data_future->cb_match_data_in, data)){
                    data->data_future = NULL;
                }
            }
        }
    }


    if(    ( parsec_type_match(data->local.dst_datatype, PARSEC_DATATYPE_NULL) == PARSEC_SUCCESS) /* No reshape dtt on dep: fulfilled reshape promise */
        || ( parsec_type_match(data->local.dst_datatype, data->data->dtt) == PARSEC_SUCCESS) )     /* Same dtt: fulfilled reshape promise*/
    {
        promise_type = PARSEC_FULFILLED_RESHAPE_PROMISE;
    }else{
        promise_type = PARSEC_UNFULFILLED_RESHAPE_PROMISE;
    }
    parsec_create_reshape_promise(es,
                                  data,
                                  src_rank, dst_rank,
                                  dep->dep_datatype_index,
                                  dep->belongs_to->flow_index, arg->output_repo, arg->output_entry,  /*Current task*/
                                  dep->flow->flow_index, successor_repo, successor_repo_key, /* Successor task */
                                  &setup_repo,
                                  &setup_repo_key,
                                  &arg->output_usage,
                                  promise_type);

    if(arg->action_mask & PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE){
        /* Mark this future as originated after a reception
         * to be able to avoid re-reshaping on input during data_lookup.
         */
        parsec_reshape_promise_description_t* data_in =
                (parsec_reshape_promise_description_t*)data->data_future->cb_fulfill_data_in;
        data_in->remote_recv_guard = PARSEC_AVOID_RESHAPE_AFTER_RECEPTION;
    }

#if defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)
    char tmpo[MAX_TASK_STRLEN], tmpt[MAX_TASK_STRLEN];
    char type_string[MAX_TASK_STRLEN]="UNFULFILLED";
    parsec_task_snprintf(tmpo, MAX_TASK_STRLEN, oldcontext);
    parsec_task_snprintf(tmpt, MAX_TASK_STRLEN, newcontext);
    if( promise_type == PARSEC_FULFILLED_RESHAPE_PROMISE){
        type_string[0]=type_string[1]='*';
    }
    char orig_string[MAX_TASK_STRLEN]="REMOTE";
    if( dst_rank == src_rank ) {
        sprintf(orig_string, "LOCAL");
    }
    char side_string[MAX_TASK_STRLEN]="TO";
    if(arg->action_mask & PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE){
        sprintf(side_string, "FOR");
    }

    char type_name_src[MAX_TASK_STRLEN] = "NULL";
    char type_name_dst[MAX_TASK_STRLEN] = "NULL";
    char type_name_data[MAX_TASK_STRLEN] = "NULL";
    int len;
    if(data->local.src_datatype!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->local.src_datatype, type_name_src, &len);
    if(data->local.dst_datatype!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->local.dst_datatype, type_name_dst, &len);
    if(data->data->dtt!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->data->dtt, type_name_data, &len);

    PARSEC_DEBUG_VERBOSE(12, parsec_debug_output,
                         "th%d RESHAPE_PROMISE CREATE %s %s %s [%p:%s:%p -> %s:%p] flow_idx %u fut %p on %s(%p) k%d dtt %s -> %s [data %s]",
                         es->th_id, type_string, side_string, orig_string, data->data, tmpo, data->data->dtt,
                         tmpt, data->local.dst_datatype,
                         dep->belongs_to->flow_index,
                         data->data_future,
                         (setup_repo == successor_repo)? "SUCC_REPO" : "PRED_REPO", setup_repo, setup_repo_key, type_name_src, type_name_dst, type_name_data);

#endif
    assert(data->data_future != NULL);

    return PARSEC_ITERATE_CONTINUE;
}

/**
 *
 * Auxiliary routine to create an inline reshape promise, i.e., creating and
 * fulfilling a local future promise (only the current task instance is involved).
 * Each thread accessing the same original tile on the matrix will
 * create a new reshaped copy.
 * The reshape promise is stored on the task instance repo entry
 * (for reshape promises) while it's been triggered and completed.
 *
 * @param[in] es parsec_execution_stream_t.
 * @param[in] tp parsec_taskpool_t.
 * @param[in] task current parsec_task_t.
 * @param[in] dep_flow_index index of the flow in the task
 * @param[in] reshape_repo repo for unfulfilled reshape promises.
 * @param[in] reshape_entry_key key on repo for unfulfilled reshape promises.
 * @param[in] data contains the description of the reshape.
 * @param[out] reshape the new reshape datacopy.
 * @return PARSEC_HOOK_RETURN_AGAIN when the reshaping has been trigger and
 * will be performed by the communication thread; PARSEC_HOOK_RETURN_RESHAPE_DONE
 * once the reshape has been completed and datacopy set up on reshape.
 */
static int
parsec_get_copy_reshape_inline(parsec_execution_stream_t *es,
                               parsec_taskpool_t* tp,
                               parsec_task_t *task,
                               uint8_t dep_flow_index,
                               data_repo_t *reshape_repo,
                               parsec_key_t reshape_entry_key,
                               parsec_dep_data_description_t *data,
                               parsec_data_copy_t**reshape)
{
    (void) tp;
    data_repo_entry_t *reshape_repo_entry = NULL;
    data_repo_t *setup_repo;
    parsec_key_t setup_repo_key;

    /* Set up the reshaping promise */
    reshape_repo_entry = data_repo_lookup_entry(reshape_repo, reshape_entry_key);
    assert( reshape_repo_entry != NULL ); /* repo has been created at the begining of data_lookup*/
    if (reshape_repo_entry != NULL){ /* reshape promise already set up on the repo */
        data->data_future = (parsec_datacopy_future_t*)reshape_repo_entry->data[dep_flow_index];
    }

    if(data->data_future == NULL){
        parsec_create_reshape_promise(es,
                                      data,
                                      0, -1, -1, /* no src dst rank */
                                      dep_flow_index, reshape_repo, reshape_repo_entry, /* Current task */
                                      0xFF, NULL, -1, /* Successor task */
                                      &setup_repo,
                                      &setup_repo_key,
                                      NULL,
                                      PARSEC_UNFULFILLED_RESHAPE_PROMISE);


#if defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)
        char task_name[MAX_TASK_STRLEN];
        parsec_task_snprintf(task_name, MAX_TASK_STRLEN, task);

        char type_string[MAX_TASK_STRLEN]="UNFULFILLED";
        char orig_string[MAX_TASK_STRLEN]="LOCAL INLINE";

        char type_name_src[MAX_TASK_STRLEN] = "NULL";
        char type_name_dst[MAX_TASK_STRLEN] = "NULL";
        char type_name_data[MAX_TASK_STRLEN] = "NULL";
        int len;
        if(data->local.src_datatype!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->local.src_datatype, type_name_src, &len);
        if(data->local.dst_datatype!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->local.dst_datatype, type_name_dst, &len);
        if(data->data->dtt!=PARSEC_DATATYPE_NULL) MPI_Type_get_name(data->data->dtt, type_name_data, &len);

        PARSEC_DEBUG_VERBOSE(12, parsec_debug_output,
                             "th%d RESHAPE_PROMISE CREATE %s %s [%s:%p:%p -> %p] flow_idx %u fut %p on %s(%p) k%d dtt %s -> %s [data %s]",
                             es->th_id, type_string, orig_string, task_name, data->data, data->data->dtt,
                             data->local.dst_datatype,
                             dep_flow_index,
                             data->data_future,
                             "CURR_REPO", setup_repo, setup_repo_key, type_name_src, type_name_dst, type_name_data);

#endif
    }

    /* Trigger and obtain reshape data */
    assert(data->data_future!= NULL);
    *reshape = parsec_future_get_or_trigger(data->data_future, NULL, NULL, es, task);
    if(*reshape == NULL){
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    PARSEC_DEBUG_VERBOSE(12, parsec_debug_output,
                         "th%d RESHAPE_PROMISE OBTAINED [%p:%p] fut %p",
                         es->th_id, *reshape, (*reshape)->dtt, data->data_future);

    /* reshape completed */
    PARSEC_OBJ_RETAIN(*reshape);
    PARSEC_OBJ_RELEASE(data->data_future);
    /* Clean up the old stuff on the repo used temporarily to hold
     * the inline reshape promise.
     */
    reshape_repo_entry->data[dep_flow_index] = NULL;

    return PARSEC_HOOK_RETURN_RESHAPE_DONE;
}

/**
 *
 * Routine to obtain a reshaped copy matching the specifications when reading
 * a tile from the datacollection.
 * If a reshape needs to be performed, it is done using an inline reshape
 * promise, i.e., creating and fulfilling a local future promise (only
 * the current task instance is involved). Each thread accessing the same
 * original tile on the matrix will create a new reshaped copy.
 * Used in data_lookup_of for the second-level reshaping (the local one, not
 * set up by predecessors)
 * The reshape promise is stored on the task instance repo entry
 * while it's been trigger and completed.
 *
 * @param[in] es parsec_execution_stream_t.
 * @param[in] tp parsec_taskpool_t.
 * @param[in] task current parsec_task_t.
 * @param[in] dep_flow_index index of the flow in the task.
 * @param[in] reshape_repo repo to hold the reshape promise.
 * @param[in] reshape_entry_key key on the repo to hold the reshape promise.
 * @param[in] data contains the description of the reshape.
 * @param[out] reshape the new reshape datacopy.
 * @return PARSEC_HOOK_RETURN_AGAIN when the reshaping has been trigger and
 * will be performed by the communication thread; PARSEC_HOOK_RETURN_RESHAPE_DONE
 * once the reshape has been completed and datacopy set up on reshape.
 */
int
parsec_get_copy_reshape_from_desc(parsec_execution_stream_t *es,
                                  parsec_taskpool_t* tp,
                                  parsec_task_t *task,
                                  uint8_t dep_flow_index,
                                  data_repo_t *reshape_repo,
                                  parsec_key_t reshape_entry_key,
                                  parsec_dep_data_description_t *data,
                                  parsec_data_copy_t**reshape)
{
    /* In this case, we have read from a local descA and we need to reshape
     * it to the correct type.
     * Data is always fill with correct information.
     */
    int ret;
    if(data->data == NULL) /* NULL data, can this happen apart from the tests?*/
    {
        return PARSEC_HOOK_RETURN_DONE_NO_RESHAPE;
    }

    if( ( data->local.arena == NULL ) /*No reshaping dtt on dep -> default*/
       || ( parsec_type_match(data->local.dst_datatype, data->data->dtt) == PARSEC_SUCCESS) )/* Same dtt: no reshaping */
    {
        return PARSEC_HOOK_RETURN_DONE_NO_RESHAPE;
    }
    assert(data->data->dtt != PARSEC_DATATYPE_NULL); /* If this happens probably the datacollection has not a default dtt */

    ret = parsec_get_copy_reshape_inline(es, tp, task,
                                             dep_flow_index,
                                             reshape_repo, reshape_entry_key,
                                             data, reshape);

    return ret;
}


/**
 *
 * Routine to obtain a reshaped copy matching the specifications on an input
 * dependency from a predecessor.
 * The reshape promise set up by the predecessor (fulfilled or unfulfilled) is
 * passed on the parsec_dep_data_description_t data_future.
 * If the input dependency hasn't a reshape type, the based copy tracked by the
 * future will be returned, otherwise, if a different type from the based
 * data tracked by the future is requested, a new nested future will be set up internally
 * by the future.
 *
 * @param[in] es parsec_execution_stream_t.
 * @param[in] tp parsec_taskpool_t.
 * @param[in] task current parsec_task_t.
 * @param[in] dep_flow_index index of the flow in the task.
 * @param[in] reshape_repo repo to hold the reshape promise. Not used.
 * @param[in] reshape_entry_key key on the repo to hold the reshape promise. Not used.
 * @param[in] data contains the description of the reshape.
 * @param[out] reshape the new reshape datacopy.
 * @return PARSEC_HOOK_RETURN_AGAIN when the reshaping has been trigger and
 * will be performed by the communication thread; PARSEC_HOOK_RETURN_RESHAPE_DONE
 * once the reshape has been completed and datacopy set up on reshape.
 */
int
parsec_get_copy_reshape_from_dep(parsec_execution_stream_t *es,
                                 parsec_taskpool_t* tp,
                                 parsec_task_t *task,
                                 uint8_t dep_flow_index,
                                 data_repo_t *reshape_repo,
                                 parsec_key_t reshape_entry_key,
                                 parsec_dep_data_description_t *data,
                                 parsec_data_copy_t**reshape)
{
    (void)dep_flow_index;
    (void)reshape_repo; (void)reshape_entry_key;
    (void)tp;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    char task_string[MAX_TASK_STRLEN];
    (void)parsec_task_snprintf(task_string, MAX_TASK_STRLEN, task);
#endif

    /* There's always a reshape promise as input, whether from:
     * - Predecessor repo
     * - Current task promises repo
     * We need to get the base reshaped copy or trigger a new nested reshape.
     */

    assert(data->data_future != NULL);

    parsec_reshape_promise_description_t* data_in =
            (parsec_reshape_promise_description_t*)data->data_future->cb_fulfill_data_in;

    if(data_in->remote_recv_guard & PARSEC_AVOID_RESHAPE_AFTER_RECEPTION){
        /* avoid re-reshaping with this data is originated after the reception
         * from a remote.
         */
        *reshape = parsec_future_get_or_trigger(data->data_future, NULL, NULL, es, task);
    }else{
        *reshape = parsec_future_get_or_trigger(data->data_future,
                                parsec_setup_nested_future, data, es, task);
    }
    if(*reshape == NULL){
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    PARSEC_DEBUG_VERBOSE(12, parsec_debug_output,
                         "th%d RESHAPE_PROMISE OBTAINED [%p:%p] for %s fut %p",
                         es->th_id, *reshape, (*reshape)->dtt, task_string, data->data_future);

    PARSEC_OBJ_RETAIN(*reshape);
    PARSEC_OBJ_RELEASE(data->data_future);

    return PARSEC_HOOK_RETURN_RESHAPE_DONE;
}
