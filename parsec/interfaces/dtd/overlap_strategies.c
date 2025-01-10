/**
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/* **************************************************************************** */
/**
 * @file overlap_strategies.c
 *
 */

#include "parsec/runtime.h"
#include "parsec/parsec_internal.h"

#include "parsec/data_distribution.h"
#include "parsec/remote_dep.h"
#include "parsec/execution_stream.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

/***************************************************************************//**
 *
 * This function makes sure that nextinline descendant is really NULL
 *
 * @param[in]   current_task
 *                  Task which is trying to release ownership
 * @param[in]   flow_index
 *                  Flow index of the task for which the task is releasing
 * @return
 *              Returns 1 if successfully released ownership, 0 otherwise
 *
 ******************************************************************************/
static int
made_sure_nextinline_is_null(parsec_dtd_task_t *current_task, int flow_index)
{
    parsec_dtd_tile_t *tile = FLOW_OF(current_task, flow_index)->tile;

    parsec_dtd_last_user_lock( &(tile->last_user) );
    parsec_mfence(); /* Write */

    /* If this_task is still the owner of the data we remove this_task */
    if( tile->last_user.task == current_task ) {
        /* indicating that the input chain has been activated */
        tile->last_user.alive = TASK_IS_NOT_ALIVE;

        /* Setting the iterated flag of parent as YES */
        if( !((FLOW_OF((PARENT_OF(current_task, flow_index))->task, (PARENT_OF(current_task, flow_index))->flow_index))->flags & SUCCESSOR_ITERATED) ) {
            (FLOW_OF((PARENT_OF(current_task, flow_index))->task, (PARENT_OF(current_task, flow_index))->flow_index))->flags |= SUCCESSOR_ITERATED;
        }

        parsec_dtd_last_user_unlock( &(tile->last_user) );
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 1;
    } else {
        /* if we can not atomically swap the last user of the tile to NULL then we
         * wait until we find our successor.
         */
        int unit_waited = 0;
        while(NULL == (DESC_OF(current_task, flow_index))->task) {
            unit_waited++;
            if (1000 == unit_waited) {
                unit_waited = 0;
                usleep(1);
            }
            parsec_mfence();  /* force the local update of the cache */
        }
        parsec_dtd_last_user_unlock( &(tile->last_user) );
        return 0;
    }
}

/***************************************************************************//**
 *
 * This function releases the ownership of a data for a task
 *
 * @param[in]   current_task
 *                  Task which is trying to release ownership
 * @param[in]   flow_index
 *                  Flow index of the task for which the task is releasing
 * @return
 *              Returns 1 if successfully released ownership, 0 otherwise
 *
 ******************************************************************************/
static int
release_ownership_of_data(parsec_dtd_task_t *current_task, int flow_index)
{
    parsec_dtd_tile_t *tile = FLOW_OF(current_task, flow_index)->tile;

    parsec_dtd_last_user_lock( &(tile->last_user) );
    parsec_mfence(); /* Write */

    /* If this_task is still the owner of the data we remove this_task */
    if( tile->last_user.task == current_task ) {
        tile->last_user.alive = TASK_IS_NOT_ALIVE;

        FLOW_OF(current_task, flow_index)->flags |= SUCCESSOR_ITERATED;

        parsec_dtd_last_user_unlock( &(tile->last_user) );
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 1;
    } else {
        /* if we can not atomically swap the last user of the tile to NULL then we
         * wait until we find our successor.
         */
        int unit_waited = 0;
        while(NULL == (DESC_OF(current_task, flow_index))->task) {
            unit_waited++;
            if (1000 == unit_waited) {
                unit_waited = 0;
                usleep(1);
            }
            parsec_mfence();  /* force the local update of the cache */
        }
        parsec_dtd_last_user_unlock( &(tile->last_user) );
        return 0;
    }
}

/***************************************************************************//**
 *
 * This function implements the iterate successors taking
 * anti dependence into consideration. At first we go through all
 * the descendant of a task for each flow and put them in a list.
 * This is helpful in terms of creating chains of PARSEC_INPUT tasks.
 * INPUT tasks are activated if they are found in successions,
 * and they are treated the same way.
 *
 * @param[in]   es
 *                  Execution unit
 * @param[in]   this_task
 *                  We will iterate thorugh the successors of this task
 * @param[in]   action_mask,ontask_arg
 * @param[in]   ontask
 *                  Function pointer to function that activates successsor
 *
 ******************************************************************************/
void
parsec_dtd_ordering_correctly( parsec_execution_stream_t *es,
                               const parsec_task_t *this_task,
                               uint32_t action_mask,
                               parsec_ontask_function_t *ontask,
                               void *ontask_arg )
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    int current_dep;
    parsec_dtd_task_t *current_desc = NULL;
    int op_type_on_current_flow, desc_op_type, desc_flow_index;
    parsec_dtd_tile_t *tile;
    parsec_arena_datatype_t *adt;

    parsec_dep_t deps;
    parsec_release_dep_fct_arg_t *arg = (parsec_release_dep_fct_arg_t *)ontask_arg;
    parsec_dep_data_description_t data;
    int rank_src = 0, rank_dst = 0, vpid_dst=0;

    /* finding for which flow we need to iterate successors of */
    int flow_mask = action_mask;

    rank_src = current_task->rank;
    for( current_dep = 0; current_dep < current_task->super.task_class->nb_flows; current_dep++ ) {
        if( (flow_mask & (1<<current_dep)) ) {
            current_desc = (DESC_OF(current_task, current_dep))->task;
            op_type_on_current_flow = (FLOW_OF(current_task, current_dep)->op_type & PARSEC_GET_OP_TYPE);
            tile = FLOW_OF(current_task, current_dep)->tile;

            if( NULL == tile ) {
                continue;
            }

            if( FLOW_OF(current_task, current_dep)->op_type & PARSEC_DONT_TRACK ) {
                /* User has instructed us not to track this data */
                continue;
            }

            if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                if( PARSEC_INPUT == op_type_on_current_flow ) {
                    if(parsec_dtd_task_is_local(current_task)){
                       (void)parsec_atomic_fetch_dec_int32( &current_task->super.data[current_dep].data_out->readers );
                    }
                }
            }

            /**
             * In case the same data is used for multiple flows, only the last reference
             * points to the potential successors. Every other use of the data will point
             * to ourself.
             */
            if(current_task == current_desc) {
                if(parsec_dtd_task_is_local(current_desc)) {
                    current_desc->super.data[(DESC_OF(current_task, current_dep))->flow_index].data_in = current_task->super.data[current_dep].data_out;
                    current_desc->super.data[(DESC_OF(current_task, current_dep))->flow_index].data_out = current_task->super.data[current_dep].data_out;
                    parsec_dtd_retain_data_copy(current_task->super.data[current_dep].data_out);
                }
                continue;
            }

            if( NULL == current_desc ) {
                if( PARSEC_INOUT == op_type_on_current_flow ||
                    PARSEC_OUTPUT == op_type_on_current_flow ) {
                    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                        if(release_ownership_of_data(current_task, current_dep)) { /* trying to release ownership */
                            continue;  /* no descendent for this data */
                        } else {
                            current_desc = (DESC_OF(current_task, current_dep))->task;
                        } /* Current task has a descendant hence we must activate her */
                    }
                } else {
                    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                        if( FLOW_OF(current_task, current_dep)->flags & RELEASE_OWNERSHIP_SPECIAL ){
                            made_sure_nextinline_is_null(current_task, current_dep);
                        }
                    }
                    continue;
                }
            }

#if defined(PARSEC_DEBUG_ENABLE)
            assert(current_desc != NULL);
#endif

            /* setting data */
            data.data   = current_task->super.data[current_dep].data_out;
            adt = parsec_dtd_get_arena_datatype(this_task->taskpool->context,
                                                FLOW_OF(current_task, current_dep)->arena_index);
            if( NULL != adt ) {
                // Not all programs need to define types for each flow: they might not create a communication.
                data.remote.arena = adt->arena;
                data.remote.src_datatype = data.remote.dst_datatype = adt->opaque_dtt;
            }
            data.remote.src_count = data.remote.dst_count = 1;
            data.remote.src_displ = data.remote.dst_displ = 0;

            data.data_future = NULL;
            data.local.arena  = NULL;
            data.local.src_datatype = data.local.dst_datatype = PARSEC_DATATYPE_NULL;
            data.local.src_count = data.local.dst_count = data.local.src_displ = data.local.dst_displ = 0;

            desc_op_type = ((DESC_OF(current_task, current_dep))->op_type & PARSEC_GET_OP_TYPE);
            desc_flow_index = (DESC_OF(current_task, current_dep))->flow_index;

            int get_out = 0, tmp_desc_flow_index, release_parent = 0;
            parsec_dtd_task_t *nextinline = current_desc;

            do {
                tmp_desc_flow_index = desc_flow_index;
                current_desc = nextinline;
                assert(NULL != current_desc);
                /* Forward the data to each successor */
                if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                    if(parsec_dtd_task_is_local(current_desc)) {
                       /* printf("[iterate_sucessors] %s -> %s, data_in = %p, data_out = %p\n", current_task->super.task_class->name, current_desc->super.task_class->name, current_desc->super.data[desc_flow_index].data_in, current_task->super.data[current_dep].data_out); */
                       current_desc->super.data[desc_flow_index].data_in = current_task->super.data[current_dep].data_out;
                        /* We retain local, remote data for each successor */
                        parsec_dtd_retain_data_copy(current_task->super.data[current_dep].data_out);
                    }
                }

                get_out = 1;  /* by default escape */
                if( !(PARSEC_OUTPUT == desc_op_type || PARSEC_INOUT == desc_op_type) ) {

                  look_for_next:
                    nextinline = (DESC_OF(current_desc, desc_flow_index))->task;
                    if( NULL != nextinline ) {
                        desc_op_type    = ((DESC_OF(current_desc, desc_flow_index))->op_type & PARSEC_GET_OP_TYPE);
                        desc_flow_index =  (DESC_OF(current_desc, desc_flow_index))->flow_index;
                        get_out = 0;  /* We have a successor, keep going */
                        if( nextinline == current_desc ) {
                            /* We have same descendant using same data in multiple flows
                             * So we activate the successor once and skip the other times
                             */
                            if( parsec_dtd_task_is_remote(current_desc) ) {
                                /* releasing remote read task that is in the chain */
                                parsec_dtd_remote_task_release( current_desc );
                            }
                            continue;
                        } else {
                            if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS)
                                (DESC_OF(current_desc, tmp_desc_flow_index))->task = NULL;
                        }
                    } else {
                        if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                            /* Make sure there is no nextinline */
                            if( made_sure_nextinline_is_null(current_desc, desc_flow_index) ) {
                            } else {
                                goto look_for_next;
                            }

                        }
                    }

                    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                        if(parsec_dtd_task_is_local(current_desc)){
                           (void)parsec_atomic_fetch_inc_int32( &current_task->super.data[current_dep].data_out->readers );
                        }
                    }
                    /* Each reader increments the ref count of the data_copy
                     * We should have a function to retain data copies like
                     * PARSEC_DATA_COPY_RELEASE
                     */
                } else {
                    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                        if( !(FLOW_OF(current_task, current_dep)->flags & SUCCESSOR_ITERATED) ){
                            FLOW_OF(current_task, current_dep)->flags |= SUCCESSOR_ITERATED;
                        }
                        /* Found next owner of data, decrement count on current local owner */
                        if( parsec_dtd_task_is_local(current_task) ) {
                            release_parent = 1;
                        }
                    }
                }

                PARSEC_DEBUG_VERBOSE(parsec_dtd_dump_traversal_info, parsec_dtd_debug_output,
                                     "------\nsuccessor of: %s \t %lld rank %d --> %s \t %lld rank: %d\nTotal flow: %d  flow_count:"
                                     "%d\n----- for pred flow: %d and desc flow: %d\n", current_task->super.task_class->name,
                                     current_task->ht_item.key, current_task->rank, current_desc->super.task_class->name,
                                     current_desc->ht_item.key, current_desc->rank, current_desc->super.task_class->nb_flows,
                                     current_desc->flow_count, current_dep, tmp_desc_flow_index);

                deps.cond            = NULL;
                deps.ctl_gather_nb   = NULL;
                deps.task_class_id   = current_desc->super.task_class->task_class_id;
                deps.flow            = current_desc->super.task_class->in[tmp_desc_flow_index];
                deps.dep_index       = 0; /* it will not be used anywhere for DTD, so whatever */
                deps.belongs_to      = current_task->super.task_class->out[current_dep];
                deps.direct_data     = NULL;
                deps.dep_datatype_index = current_dep;

                rank_dst = current_desc->rank;

                ontask( es, (parsec_task_t *)current_desc, (parsec_task_t *)current_task,
                        &deps, &data, rank_src, rank_dst, vpid_dst, NULL, 0, ontask_arg );
                vpid_dst = (vpid_dst+1) % current_task->super.taskpool->context->nb_vp;

                /* releasing remote tasks that is a descendant of a local task */
                if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                    if( parsec_dtd_task_is_remote(current_desc) && parsec_dtd_task_is_local(current_task) ) {
                        parsec_dtd_remote_task_release( current_desc );
                    }
                    if(release_parent) {
                        if( parsec_dtd_task_is_local(current_task) ) {
                            parsec_dtd_release_local_task( current_task );
                        }
                    }
                }
            } while (0 == get_out);

#if defined(DISTRIBUTED)
            if( (action_mask & PARSEC_ACTION_COMPLETE_LOCAL_TASK) && (NULL != arg->remote_deps) ) {
                (void)parsec_atomic_fetch_inc_int32(&current_task->super.data[current_dep].data_out->readers);
                parsec_remote_dep_activate(es, (parsec_task_t *)current_task, arg->remote_deps, arg->remote_deps->outgoing_mask);
                arg->remote_deps = NULL;
            }
#endif
        }
    }
}
