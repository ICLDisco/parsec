/**
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/* **************************************************************************** */
/**
 * @file overlap_strategies.c
 *
 * @version 2.0.0
 * @author Reazul Hoque
 *
 */

#include "dague_config.h"
#include "dague/dague_internal.h"

#include <stdio.h>
#include "dague/data_distribution.h"
#include "dague/remote_dep.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

extern int dump_traversal_info; /**< For printing traversal info */

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
int
release_ownership_of_data_1(const dague_dtd_task_t *current_task, int flow_index)
{
    dague_dtd_tile_t *tile = current_task->flow[flow_index].tile;

    dague_dtd_last_user_lock( &(tile->last_user) );
    dague_mfence(); /* Write */

    /* If this_task is still the owner of the data we remove this_task */
    if( tile->last_user.task == current_task ) {
        tile->last_user.alive       = TASK_IS_NOT_ALIVE;
        dague_dtd_last_user_unlock( &(tile->last_user) );
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 1;
    } else {
        /* if we can not atomically swap the last user of the tile to NULL then we
         * wait until we find our successor.
         */
        int unit_waited = 0;
        while(NULL == current_task->desc[flow_index].task) {
            unit_waited++;
            if (1000 == unit_waited) {
                unit_waited = 0;
                usleep(1);
            }
            dague_mfence();  /* force the local update of the cache */
        }
        dague_dtd_last_user_unlock( &(tile->last_user) );
        return 0;
    }
}

/***************************************************************************//**
 *
 * This function implements the iterate successors taking
 * anti dependence into consideration. At first we go through all
 * the descendant of a task for each flow and put them in a list.
 * This is helpful in terms of creating chains of INPUT tasks.
 * INPUT tasks are activated if they are found in successions,
 * and they are treated the same way.
 *
 * @param[in]   eu
 *                  Execution unit
 * @param[in]   this_task
 *                  We will iterate thorugh the successors of this task
 * @param[in]   action_mask,ontask_arg
 * @param[in]   ontask
 *                  Function pointer to function that activates successsor
 *
 ******************************************************************************/
void
ordering_correctly_1(dague_execution_unit_t *eu,
                     const dague_execution_context_t *this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t *ontask,
                     void *ontask_arg)
{
    dague_dtd_task_t *current_task = (dague_dtd_task_t *)this_task;
    int current_dep;
    dague_dtd_task_t *current_desc = NULL;
    int op_type_on_current_flow, desc_op_type, desc_flow_index;
    dague_dtd_tile_t *tile;

    dep_t deps;
    dague_dep_data_description_t data;
    int rank_src = 0, rank_dst = 0, vpid_dst=0;
    (void)action_mask;

    for( current_dep = 0; current_dep < current_task->super.function->nb_flows; current_dep++ ) {
#if defined(DAGUE_PROF_GRAPHER)
        deps.dep_index = current_dep;
#endif
        current_desc = current_task->desc[current_dep].task;
        op_type_on_current_flow = (current_task->flow[current_dep].op_type & GET_OP_TYPE);
        tile = current_task->flow[current_dep].tile;

        if( NULL == tile ) {
            continue;
        }

        if( INPUT == op_type_on_current_flow ) {
            dague_atomic_add_32b( (int *)&(current_task->super.data[current_dep].data_out->readers), -1 );
            //DAGUE_DATA_COPY_RELEASE(current_task->super.data[current_dep].data_out);
        }

        /**
         * In case the same data is used for multiple flows, only the last reference
         * points to the potential successors. Every other use of the data will point
         * to ourself.
         */
        if(current_task == current_desc) {
            dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
            continue;
        }

        if( NULL == current_desc ) {
             if( INOUT == op_type_on_current_flow ||
                 OUTPUT == op_type_on_current_flow ||
                (current_task->dont_skip_releasing_data[current_dep])) {
#if defined (OVERLAP)
                if(release_ownership_of_data_1(current_task, current_dep)) { /* trying to release ownership */
#endif
                    dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
                    continue;  /* no descendent for this data */
#if defined (OVERLAP)
                } else {
                    current_desc = current_task->desc[current_dep].task;
                } /* Current task has a descendant hence we must activate her */
#endif
            } else {
                dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
                continue;
            }
        }

#if defined(DAGUE_DEBUG_ENABLE)
        assert(current_desc != NULL);
#endif

        desc_op_type = (current_task->desc[current_dep].op_type & GET_OP_TYPE);
        desc_flow_index = current_task->desc[current_dep].flow_index;

        int get_out = 0, tmp_desc_flow_index;
        dague_dtd_task_t *nextinline = current_desc;

        do {
            tmp_desc_flow_index = desc_flow_index;
            current_desc = nextinline;

            /* Forward the data to each successor */
            current_desc->super.data[desc_flow_index].data_in = current_task->super.data[current_dep].data_out;

            get_out = 1;  /* by default escape */
            if( !(OUTPUT == desc_op_type || INOUT == desc_op_type) ) {

                nextinline = current_desc->desc[desc_flow_index].task;
                if( NULL != nextinline ) {
                    desc_op_type    = (current_desc->desc[desc_flow_index].op_type & GET_OP_TYPE);
                    desc_flow_index =  current_desc->desc[desc_flow_index].flow_index;
                    get_out = 0;  /* We have a successor, keep going */
                    if( nextinline == current_desc ) {
                        /* We have same descendant using same data in multiple flows
                         * So we activate the successor once and skip the other times
                         */
                        continue;
                    } else {
                        current_desc->desc[tmp_desc_flow_index].task = NULL;
                    }
                } else {
                    /* Mark it specially as it is a task that performs INPUT type of operation
                     * on the data.
                     */
                    current_desc->dont_skip_releasing_data[desc_flow_index] = 1;
                }

                dague_atomic_add_32b( (int *)&(current_task->super.data[current_dep].data_out->readers), 1 );
                /* Each reader increments the ref count of the data_copy
                 * We should have a function to retain data copies like
                 * DAGUE_DATA_COPY_RELEASE
                 */
                //OBJ_RETAIN(current_task->super.data[current_dep].data_out);

            }

            if(dump_traversal_info) {
                dague_output(dague_debug_output, "------\nsuccessor: %s \t %lld\nTotal flow: %d  flow_count:"
                       "%d\n-----for pred flow: %d and desc flow: %d\n", current_desc->super.function->name, current_desc->super.super.key,
                       current_desc->super.function->nb_flows, current_desc->flow_count, current_dep, tmp_desc_flow_index);
            }

            ontask( eu, (dague_execution_context_t *)current_desc, (dague_execution_context_t *)current_task,
                    &deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg );
            vpid_dst = (vpid_dst+1) % current_task->super.dague_handle->context->nb_vp;

        } while (0 == get_out);
        dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
    }
}
