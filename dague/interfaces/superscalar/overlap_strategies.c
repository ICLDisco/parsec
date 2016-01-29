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
release_ownership_of_data(const dague_dtd_task_t *current_task, int flow_index)
{
    dague_dtd_task_t *task_pointer = (dague_dtd_task_t *)((uintptr_t)current_task|flow_index);
    dague_dtd_tile_t* tile = current_task->desc[flow_index].tile;

    dague_mfence(); /* Write */
    if( dague_atomic_cas(&(tile->last_user.task), task_pointer, NULL) ) {
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 1;
    }
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
    return 0;
}

/***************************************************************************//**
 *
 * This function tries to make a Fake task with operation type of OUTPUT
 * on the data as the last user of the tile. If this try fails we make
 * sure we get the last_user of the tile updated for us to read.
 *
 * @param[in]   last_read
 *                  The last reader of the tile
 * @param[in]   fake_writer
 *                  Fake task we are trying to set as last user
 *                  of the data(tile)
 * @param[in]   last_read_flow_index
 *                  Flow index of the last reader of the tile
 * @return
 *              1 if successfully set the fake writer as the last user,
 *              0 otherwise
 *
 ******************************************************************************/
int
put_fake_writer_as_last_user( dague_dtd_task_t *last_read,
                              dague_dtd_task_t *fake_writer,
                              int last_read_flow_index )
{
    dague_dtd_tile_t* tile = last_read->desc[last_read_flow_index].tile;
    dague_dtd_task_t *task_pointer = (dague_dtd_task_t *)((uintptr_t)last_read|last_read_flow_index);

    dague_mfence(); /* Write */
    if( dague_atomic_cas(&(tile->last_user.task), task_pointer, fake_writer) ) {
        tile->last_user.op_type = OUTPUT;
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 1;
    }
    /* if we can not atomically swap the last user of the tile to NULL then we
     * wait until we find our successor.
     */
    int unit_waited = 0;
    while(NULL == last_read->desc[last_read_flow_index].task) {
        unit_waited++;
        if (1000 == unit_waited) {
            unit_waited = 0;
            usleep(1);
        }
        dague_mfence();  /* force the local update of the cache, Read */
    }
    return 0;
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
ordering_correctly_2(dague_execution_unit_t *eu,
                     const dague_execution_context_t *this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t *ontask,
                     void *ontask_arg)
{
    dague_dtd_task_t *current_task = (dague_dtd_task_t *)this_task;
    int current_dep, count;
    dague_dtd_task_t *last_read, *out_task;
    dague_dtd_task_t *current_desc = NULL;
    int op_type_on_current_flow, desc_op_type, desc_flow_index, out_task_flow_index;
    dague_dtd_tile_t *tile;

    dep_t deps;
    dague_dep_data_description_t data;
    int rank_src = 0, rank_dst = 0, vpid_dst=0;
    (void)action_mask;

    for( current_dep = 0; current_dep < current_task->super.function->nb_flows; current_dep++ ) {
#if defined(DAGUE_PROF_GRAPHER)
        deps.dep_index = current_dep;
#endif
        last_read = NULL; out_task = NULL;
        count = 0;
        current_desc = current_task->desc[current_dep].task;
        op_type_on_current_flow = (current_task->desc[current_dep].op_type_parent & GET_OP_TYPE);
        tile = current_task->desc[current_dep].tile;

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
                if(release_ownership_of_data(current_task, current_dep)) { /* trying to release ownership */
#endif
                    dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
                    continue;  /* no descendent for this data */
#if defined (OVERLAP)
                } else {
                    current_desc = current_task->desc[current_dep].task;
                } /* Current task has a descendant hence we must activate her */
#endif
            } else {
                continue;
            }
        }

#if defined(DAGUE_DEBUG_ENABLE)
        assert(current_desc != NULL);
#endif
        desc_op_type = (current_task->desc[current_dep].op_type & GET_OP_TYPE);
        desc_flow_index = current_task->desc[current_dep].flow_index;

        int tmp_desc_flow_index = 254;
        int8_t keep_fake_writer = 0;

        /* Create Fake output_task */
        dague_dtd_task_t *fake_writer = create_fake_writer_task( (dague_dtd_handle_t *)current_task->super.dague_handle , tile);
        out_task = fake_writer;
        out_task_flow_index = 0;

        while( NULL != current_desc ) {
            /* Check to make sure we don't overcount in case task uses same data in multiple flows */
            if( current_desc == last_read ) {
                count--;
            }
            if( OUTPUT == desc_op_type || INOUT == desc_op_type ) {
                out_task = current_desc;
                out_task_flow_index = desc_flow_index;
                break; /* We have found our last_out_task, lets get out */
            }
            count++;

            tmp_desc_flow_index =  desc_flow_index;
            last_read           =  current_desc;
            current_desc        =  current_desc->desc[tmp_desc_flow_index].task;
            desc_flow_index     =  last_read->desc[tmp_desc_flow_index].flow_index;
            desc_op_type        = (last_read->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);

            if( current_desc == NULL ) {
                if( !(keep_fake_writer = put_fake_writer_as_last_user(last_read, fake_writer, tmp_desc_flow_index)) ) {
                    current_desc    = last_read->desc[tmp_desc_flow_index].task;
                    desc_flow_index = last_read->desc[tmp_desc_flow_index].flow_index;
                    desc_op_type    = (last_read->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);
                }
            }
        }

        if( !keep_fake_writer ) {
           fake_writer->super.function->release_task(eu, (dague_execution_context_t*)fake_writer);
        } else {
            OBJ_RETAIN(tile); /* Recreating the effect of inserting a real task using the tile */
            dague_atomic_add_32b((int *)&(current_task->super.dague_handle->nb_tasks), 1);
#if defined(DEBUG_HEAVY)
            dague_dtd_task_insert( (dague_dtd_handle_t *)current_task->super.dague_handle, fake_writer );
#endif
        }

        /* Looping through the chain and assigning the out_task as a descendant of all the
         * INPUT tasks in the chain.
         */
        current_desc = current_task->desc[current_dep].task;
        desc_op_type = (current_task->desc[current_dep].op_type & GET_OP_TYPE);
        desc_flow_index = current_task->desc[current_dep].flow_index;

        dague_dtd_task_t *tmp_desc;
        dague_atomic_add_32b((int *) &(out_task->flow_count), count);
        while( out_task != current_desc && NULL != current_desc ) {
            tmp_desc = current_desc;
            tmp_desc_flow_index =  desc_flow_index;

            desc_flow_index     =  current_desc->desc[desc_flow_index].flow_index;
            current_desc        =  current_desc->desc[tmp_desc_flow_index].task;

            if( tmp_desc != current_desc ) {
                tmp_desc->desc[tmp_desc_flow_index].task = out_task;
                tmp_desc->desc[tmp_desc_flow_index].flow_index = out_task_flow_index;
                tmp_desc->desc[tmp_desc_flow_index].op_type = out_task->desc[out_task_flow_index].op_type_parent;
            }
            ontask( eu, (dague_execution_context_t *)tmp_desc, (dague_execution_context_t *)current_task,
                    &deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);
        }

        dague_dtd_tile_release( (dague_dtd_handle_t *)current_task->super.dague_handle, tile);
        ontask( eu, (dague_execution_context_t *)out_task, (dague_execution_context_t *)current_task,
                    &deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);

        vpid_dst = (vpid_dst+1)%current_task->super.dague_handle->context->nb_vp;
    }
}
