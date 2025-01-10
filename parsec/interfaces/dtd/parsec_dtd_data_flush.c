/**
 * Copyright (c) 2013-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* **************************************************************************** */
/**
 * @file parsec_dtd_data_flush.c
 *
 */

#include "parsec/runtime.h"
#include "parsec/parsec_internal.h"
#include "parsec/remote_dep.h"
#include "parsec/scheduling.h"
#include "parsec/execution_stream.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

/**
 * @addtogroup DTD_INTERFACE_INTERNAL
 */

/**
 * This is the body of the specialized data_flush task.
 * We use this body for both the send and receive task
 * in the case where the last writer of the data is in
 * remote node.
 */
int
parsec_dtd_data_flush_sndrcv(parsec_execution_stream_t *es,
                             parsec_task_t *this_task)
{
    (void)es;
    parsec_arena_datatype_t *adt;
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_tile_t *tile = (FLOW_OF(current_task, 0))->tile;

    assert(tile != NULL);

#if defined(DISTRIBUTED)
    if(tile->rank == current_task->rank) { /* this is a receive task*/
        if( current_task->super.data[0].data_in != tile->data_copy ) {
            int16_t index = (FLOW_OF(current_task, 0))->arena_index;
            parsec_dep_data_description_t data;
            data.data   = current_task->super.data[0].data_in;
            adt = parsec_dtd_get_arena_datatype(this_task->taskpool->context, index);
            data.local.arena = adt->arena;
            data.local.src_datatype = data.local.dst_datatype = adt->opaque_dtt;
            data.local.src_count = data.local.dst_count = 1;
            data.local.src_displ = data.local.dst_displ = 0;
            parsec_remote_dep_memcpy(es, this_task->taskpool,
                         tile->data_copy, current_task->super.data[0].data_in, &data);
        }
    }
#endif

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * For general tasks we set the dependencies between
 * the task classes in a generic way. Data flush tasks
 * are special, where we know the kind of relationships
 * each flow of this task class will have.
 *
 * In this method we set the deps of each flow of
 * data_flush task class. The deps of each flow is
 * extremely simple, where the task class has only one
 * flow and for that flow there is one dep
 * (from send task -> receive task )
 */
int
set_deps_for_flush_task(const parsec_task_class_t *tc)
{
    parsec_dep_t *desc_dep = (parsec_dep_t *) malloc(sizeof(parsec_dep_t));
    parsec_dep_t *parent_dep = (parsec_dep_t *) malloc(sizeof(parsec_dep_t));

    parent_dep->cond            = NULL;
    parent_dep->ctl_gather_nb   = NULL;
    parent_dep->task_class_id   = tc->task_class_id;
    parent_dep->flow            = tc->in[0];
    parent_dep->dep_index       = ((parsec_dtd_task_class_t*)tc)->dep_out_index++;
    parent_dep->belongs_to      = tc->out[0];
    parent_dep->direct_data     = NULL;
    parent_dep->dep_datatype_index = tc->out[0]->flow_index;

    desc_dep->cond              = NULL;
    desc_dep->ctl_gather_nb     = NULL;
    desc_dep->task_class_id     = tc->task_class_id;
    desc_dep->flow              = tc->out[0];
    desc_dep->dep_index         = ((parsec_dtd_task_class_t*)tc)->dep_in_index++;
    desc_dep->belongs_to        = tc->in[0];
    desc_dep->direct_data       = NULL;
    desc_dep->dep_datatype_index = tc->in[0]->flow_index;


    parsec_flow_t **parent_out = (parsec_flow_t **)&(tc->out[0]);
    (*parent_out)->dep_out[0] = (parsec_dep_t *)parent_dep;
    (*parent_out)->flow_datatype_mask |= (1U << parent_dep->dep_datatype_index);

    parsec_flow_t **desc_in = (parsec_flow_t **)&(tc->in[0]);
    (*desc_in)->dep_in[0]  = (parsec_dep_t *)desc_dep;

    return 1;
}

/**
 * Function inserting special data flush task in the runtime.
 * This is a very simple form of the generic insert function,
 * where we know the exact number of flow this type of task
 * has and the behavior of that flow.
 */
int
parsec_insert_dtd_flush_task(parsec_dtd_task_t *this_task, parsec_dtd_tile_t *tile)
{
    const parsec_task_class_t *tc          =  this_task->super.task_class;
    parsec_dtd_taskpool_t *dtd_tp = (parsec_dtd_taskpool_t *)this_task->super.taskpool;

    int flow_index = 0;
    int satisfied_flow = 0, tile_op_type = PARSEC_INOUT;
    static int vpid = 0;

    if( NULL == tile ) {
        assert(0);
    }

    /* Retaining every remote_task */
    if( parsec_dtd_task_is_remote( this_task ) ) {
        parsec_dtd_remote_task_retain( this_task );
    }

    parsec_dtd_tile_user_t last_user, last_writer;
    if(0 == dtd_tp->flow_set_flag[tc->task_class_id]) {
        /* Setting flow in function structure */
        parsec_dtd_set_flow_in_function(dtd_tp, this_task, tile_op_type, flow_index);
        set_deps_for_flush_task(tc);
    }

    (FLOW_OF(this_task, flow_index))->arena_index = tile->arena_index;

    parsec_dtd_last_user_lock(&(tile->last_user));

    READ_FROM_TILE(last_user, tile->last_user);
    READ_FROM_TILE(last_writer, tile->last_writer);

#if defined(PARSEC_PROF_TRACE)
    this_task->super.prof_info.desc = NULL;
    this_task->super.prof_info.priority = 0;
    this_task->super.prof_info.data_id = tile->key;
    this_task->super.prof_info.task_class_id = tc->task_class_id;
#endif

    /* Setting the last_user info with info of this_task */
    tile->last_writer.task        = this_task;
    tile->last_writer.flow_index  = flow_index;
    tile->last_writer.op_type     = tile_op_type;
    tile->last_writer.alive       = TASK_IS_ALIVE;

    /* Setting the last_user info with info of this_task */
    tile->last_user.task       = this_task;
    tile->last_user.flow_index = flow_index;
    tile->last_user.op_type    = tile_op_type;
    tile->last_user.alive      = TASK_IS_ALIVE;

    if( parsec_dtd_task_is_remote( this_task ) ) {
        if( parsec_dtd_task_is_local( last_writer.task ) ) {
            /* everytime we have a remote_task as descendant of a local task */
            parsec_dtd_remote_task_retain( this_task );
        }
    }

    parsec_dtd_last_user_unlock(&(tile->last_user));

    if(TASK_IS_ALIVE == last_user.alive) {
            assert( NULL != last_user.task );
            parsec_dtd_set_parent(last_writer.task, last_writer.flow_index,
                                  this_task, flow_index, last_writer.op_type,
                                  tile_op_type);

            parsec_dtd_set_descendant(last_user.task, last_user.flow_index,
                                      this_task, flow_index, last_user.op_type,
                                      tile_op_type, last_user.alive);

    } else {
        parsec_dtd_set_parent(last_writer.task, last_writer.flow_index,
                              this_task, flow_index, last_writer.op_type,
                              tile_op_type);
        parsec_dtd_set_descendant((PARENT_OF(this_task, flow_index))->task, (PARENT_OF(this_task, flow_index))->flow_index,
                                  this_task, flow_index, (PARENT_OF(this_task, flow_index))->op_type,
                                  tile_op_type, last_user.alive);


        parsec_dtd_task_t *parent_task = (PARENT_OF(this_task, flow_index))->task;
        if( parsec_dtd_task_is_local(parent_task) || parsec_dtd_task_is_local(this_task) ) {
            int action_mask = 0;
            action_mask |= (1<<(PARENT_OF(this_task, flow_index))->flow_index);

            parsec_execution_stream_t *es = parsec_my_execution_stream();

            if( parsec_dtd_task_is_local(parent_task) && parsec_dtd_task_is_remote(this_task) ) {
                /* To make sure we do not release any remote data held by this task */
                parsec_dtd_remote_task_retain(parent_task);
            }
            this_task->super.task_class->release_deps(es,
                                       (parsec_task_t *)(PARENT_OF(this_task, flow_index))->task,
                                       action_mask                         |
                                       PARSEC_ACTION_SEND_REMOTE_DEPS      |
                                       PARSEC_ACTION_SEND_INIT_REMOTE_DEPS |
                                       PARSEC_ACTION_RELEASE_REMOTE_DEPS   |
                                       PARSEC_ACTION_COMPLETE_LOCAL_TASK   |
                                       PARSEC_ACTION_RELEASE_LOCAL_DEPS , NULL);
            if( parsec_dtd_task_is_local(parent_task) && parsec_dtd_task_is_remote(this_task) ) {
                parsec_dtd_release_local_task( parent_task );
            }
        }
    }

    if( parsec_dtd_task_is_remote( last_writer.task ) ) {
        /* releasing last writer every time writer is changed */
        parsec_dtd_remote_task_release( last_writer.task );
    }

    dtd_tp->flow_set_flag[tc->task_class_id] = 1;

    if( parsec_dtd_task_is_local(this_task) ) {/* Task is local */
        dtd_tp->super.tdm.module->taskpool_addto_nb_tasks(&dtd_tp->super, 1);
        dtd_tp->local_task_inserted++;
        PARSEC_DEBUG_VERBOSE(parsec_dtd_dump_traversal_info, parsec_dtd_debug_output,
                             "Task generated -> %s %d rank %d\n", this_task->super.task_class->name, this_task->ht_item.key, this_task->rank);
    }

    /* Releasing every remote_task */
    if( parsec_dtd_task_is_remote( this_task ) ) {
        parsec_dtd_remote_task_release( this_task );
    }


    /* Increase the count of satisfied flows to counter-balance the increase in the
     * number of expected flows done during the task creation.  */
    satisfied_flow++;

    if( parsec_dtd_task_is_local(this_task) ) {
        parsec_dtd_schedule_task_if_ready(satisfied_flow, this_task,
                                          dtd_tp, &vpid);
    }

    parsec_dtd_block_if_threshold_reached(dtd_tp, parsec_dtd_threshold_size);

    return 1;
}

/**
 * This function inserts a flushing task to the taskpool. It is used to flush a data
 * using parsec_dtd_insert_flush_task_pair
 */
int
parsec_dtd_insert_flush_task(parsec_taskpool_t *tp, parsec_dtd_tile_t *tile, int task_rank,
                             int priority, bool is_first_flush_task)
{
    parsec_dtd_taskpool_t *dtd_tp = (parsec_dtd_taskpool_t *)tp;

    const parsec_task_class_t *tc = dtd_tp->super.task_classes_array[PARSEC_DTD_FLUSH_TC_ID];

    parsec_dtd_task_t *this_task = parsec_dtd_create_and_initialize_task(dtd_tp,
                                            (parsec_task_class_t *)tc, task_rank);
    this_task->super.priority = priority;
    this_task->super.chore_mask = 1; /* data flush task class has only one incarnation */
    int flow_index = 0;
    parsec_dtd_set_params_of_task(this_task, tile, PARSEC_INOUT, &flow_index, NULL, NULL, PASSED_BY_REF);

    parsec_object_t *object = (parsec_object_t *)this_task;
    /* this task will vanish as we insert the next receive task */
    if(parsec_dtd_task_is_local(this_task)) {
        /* retaining the local task as many write flows as
         * it has and one to indicate when we have executed the task */
        (void)parsec_atomic_fetch_add_int32(&object->obj_reference_count, 2);
    } else {
        (void)parsec_atomic_fetch_inc_int32(&object->obj_reference_count);
        if (is_first_flush_task) {
            /* add another reference to prevent a release of the task
             * as the last_writer, it will be released once the dependency is released */
            parsec_dtd_remote_task_retain(this_task);
        }
    }

    parsec_insert_dtd_flush_task(this_task, tile);

    return 1;
}

/**
 * This is the method called from the exposed interface to flush a data.
 * Here, we insert two tasks 1. Send task, 2. Receive task, given the last
 * writer of the data we are flushing does not reside on the rank of the
 * original owner of the data.
 * In the other case, where the last writer of the data is in the same
 * rank as the owner, we insert only one task to make sure we copy data
 * back to the matrix from a floating data copy, if there was a remote
 * writer in between(from the beginning of the execution to the point of
 * the data flush).
 */
int
parsec_dtd_insert_flush_task_pair(parsec_taskpool_t *tp, parsec_dtd_tile_t *tile)
{
    parsec_dtd_tile_user_t last_writer;

    parsec_dtd_last_user_lock(&(tile->last_user));
    READ_FROM_TILE(last_writer, tile->last_writer);
    parsec_dtd_last_user_unlock(&(tile->last_user));

    if(last_writer.task != NULL) {
        /* Otherwise it is a no-op, this tile has never
         * been used, so it is in the same state as a
         * flush would result in.
         */

        /* We at first insert a task in the rank of the last_writer of the tile
         * to clean up the last_writer.
         */
        if(last_writer.task->rank != tile->rank) {
            /* We only need a pair if these ranks are not same */
            parsec_dtd_tile_retain(tile);
            parsec_dtd_insert_flush_task(tp, tile, last_writer.task->rank, 0, true);
        }


        /* We insert a second task in the rank of the original owner of the tile
         * this task will be used to receive the data.
         */
        parsec_dtd_tile_retain(tile);
        parsec_dtd_insert_flush_task(tp, tile, tile->rank, 0, false);
    }

    return 1;
}

/**
 * The internal function to flush the data, aka bring the most updated version of
 * a data back to its owner. This call is asynchronous, if necessary it inserts a
 * transfer task into the associated taskpool, but does not wait until this transfer
 * task completes. Thus, to ensure correct behavior the application is expected to
 * wait on the taskpool completion before generating additional users of this data.
 */
static inline void
parsec_internal_dtd_data_flush(parsec_dtd_tile_t *tile, parsec_taskpool_t *tp)
{
    assert(tile->flushed == NOT_FLUSHED);
    parsec_dtd_tile_retain(tile);

    parsec_dtd_insert_flush_task_pair(tp, tile);

    tile->flushed = FLUSHED;
    parsec_dtd_tile_remove( tile->dc, tile->key );
    parsec_dtd_tile_release( tile );
}

/**
 * User facing function to flush a data. This will cause the data to move
 * back to the original owner. This data should not be reused task before
 * the flush is complete. To ensure consistent and correct behavior, users
 * must wait on the taskpool before inserting new tasks using this data.
 * This function is non-blocking.
 */
int
parsec_dtd_data_flush(parsec_taskpool_t *tp, parsec_dtd_tile_t *tile)
{
    parsec_internal_dtd_data_flush(tile, tp);
    return PARSEC_SUCCESS; /* TODO: internal_dtd_data_flush should care for error codepaths */
}

/**
 * This function will flush all the data DTD has seen so far
 * pertaining to the data collection passed. The same constraints
 * hold for this function as for flushing individual tasks.
 * This function is non-blocking and will comeback as soon as
 * flush tasks are inserted. Users have to wait on the taskpool
 * before reusing this data collection.
 */
int
parsec_dtd_data_flush_all(parsec_taskpool_t *tp, parsec_data_collection_t *dc)
{
    parsec_hash_table_t *hash_table   = (parsec_hash_table_t *)dc->tile_h_table;
    parsec_execution_stream_t *es = parsec_my_execution_stream();

    PARSEC_PINS(es, DATA_FLUSH_BEGIN, NULL);

    parsec_hash_table_for_all( hash_table, (parsec_hash_elem_fct_t)parsec_internal_dtd_data_flush, tp);

    PARSEC_PINS(es, DATA_FLUSH_END, NULL);
    return PARSEC_SUCCESS; /* TODO: internal_dtd_data_flush should care for error codepaths */
}
