/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague_config.h"
#include "dague/dague_internal.h"

#include <stdio.h>
#include "dague/data_distribution.h"
#include "dague/remote_dep.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

/* This function releases the ownership of data once the task is done with it
 * Arguments:   - the task who wants to release the data (dague_dtd_task_t *)
 - the index of the flow for which this data was being used (int)
 * Returns:     - 0 if the task successfully released data /
 1 if the task got a descendant before releasing ownership (int)
 */
int
multithread_dag_build_1(const dague_dtd_task_t* current_task, int flow_index)
{
    dague_dtd_task_t *task_pointer = (dague_dtd_task_t *)((uintptr_t)current_task|flow_index);
    dague_dtd_tile_t* tile = current_task->desc[flow_index].tile;

    dague_mfence(); /* Write */
    //if( dague_atomic_cas(&(tile->last_user.task), current_task, NULL) ) {
    if( dague_atomic_cas(&(tile->last_user.task), task_pointer, NULL) ) {
        /* we are successful, we do not need to wait and there is no successor yet*/
        return 0;
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
    return 1;
}

/* This function implements the iterate successors taking anti dependence into consideration.
 * At first we go through all the descendant of a task for each flow and put them in a list.
 * This is helpful in terms of creating chains of INPUT and ATOMIC_WRITE tasks.
 * INPUT and atomic writes are all activated if they are found in successions, and they are treated the
 * same way.
 */
void
ordering_correctly_1(dague_execution_unit_t * eu,
                     const dague_execution_context_t * this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t * ontask,
                     void *ontask_arg)
{
    dague_dep_data_description_t data;
    dague_dtd_task_t *current_task = (dague_dtd_task_t*) this_task;
    dague_dtd_task_t *current_desc_task, *tmp_task;
    dague_dtd_task_t *out_task = NULL; /* last task that will be the descendant of every READ-ONLY task */
    /* both are initialized to high numbers to indicate garbage values */
    int flow_index_out_task = 99, op_type_out_task = 99;
    dep_t* deps;

    dtd_successor_list_t *head_succ = NULL, *current_succ = NULL, *tmp_succ = NULL;

    uint32_t rank_src=0, rank_dst=0;
    int vpid_dst = 0, i;
    uint8_t tmp_flow_index, last_iterate_flag=0;
    uint8_t atomic_write_found;

    (void)action_mask;

    /* Traversing through the successors, for each flow to build list of Read-Only tasks */
    for(i=0; i<current_task->super.function->nb_flows; i++) {
        atomic_write_found = 0;
        head_succ = NULL;
        out_task = NULL;
        tmp_task = current_task;
        last_iterate_flag = 0;

        /**
         * In case the same data is used for multiple flows, only the last reference
         * points to the potential successors. Every other use of the data will point
         * to ourself.
         */
        if(current_task == current_task->desc[i].task)
            continue;

        if(NULL == current_task->desc[i].task) {
            /* if the operation type for this flow is not INPUT or ATOMIC_WRITE
             * we release the ownership as we found there's no successor for this flow
             */
            if (INOUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) ||
                OUTPUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) ||
                (current_task->dont_skip_releasing_data[i])) {
#if defined (OVERLAP)
                if(!multithread_dag_build_1(current_task, i)) { /* trying to release ownership */
#endif
                    continue;  /* no descendent for this data */
#if defined (OVERLAP)
                }
                /* Current task has a descendant hence we must activate her */
#endif
            } else {
                continue;
            }
        }

        if ((current_task->desc[i].op_type & GET_OP_TYPE) == INOUT ||
            (current_task->desc[i].op_type & GET_OP_TYPE) == OUTPUT) {
            last_iterate_flag = 1;
        }

        tmp_flow_index  = i;
        current_desc_task = current_task->desc[i].task;

        /* ATOMIC_WRITE are considered as INPUT for this purpose. So unleashed them all */
        atomic_write_found = ((current_task->desc[i].op_type & ATOMIC_WRITE) == ATOMIC_WRITE);

        if(!(current_desc_task->desc[current_task->desc[i].flow_index].task == NULL
             || current_desc_task == current_desc_task->desc[current_task->desc[i].flow_index].task)) {
            if(atomic_write_found &&
               ((current_desc_task->desc[current_task->desc[i].flow_index].op_type & GET_OP_TYPE) != ATOMIC_WRITE)) {
                last_iterate_flag = 1;
            }
        }

        while(NULL != current_desc_task) {
            if (dump_traversal_info) {
                printf("Current successor: %s \t %d\nTotal flow: %d  flow_satisfied: %d\n",
                       current_desc_task->super.function->name, current_desc_task->super.super.key,
                       current_desc_task->flow_count, current_desc_task->flow_satisfied);
            }

            if (NULL != out_task) {
                break;
            }

            deps = (dep_t*) malloc (sizeof(dep_t));
            dague_flow_t* dst_flow = (dague_flow_t*) malloc(sizeof(dague_flow_t));

            deps->dep_index = i; /* src_flow_index */

            tmp_succ = (dtd_successor_list_t *) malloc(sizeof(dtd_successor_list_t));

            dst_flow->flow_index = tmp_task->desc[tmp_flow_index].flow_index;
            tmp_flow_index = dst_flow->flow_index;

            rank_dst   = 0;
            deps->flow = dst_flow;

            tmp_succ->task = current_desc_task;
            tmp_succ->deps = deps;
            tmp_succ->flow_index = dst_flow->flow_index;
            tmp_succ->next = NULL;

            if(NULL == head_succ) {
                head_succ = tmp_succ;
            } else {
                current_succ->next = tmp_succ;
            }
            current_succ = tmp_succ;

            if(last_iterate_flag) {
                break;
            }

            tmp_task = current_desc_task;

            if( NULL != current_desc_task->desc[dst_flow->flow_index].task &&
                current_desc_task->desc[dst_flow->flow_index].task != current_desc_task ) {
                /* Check to stop building chain of ATOMIC_WRITE tasks when we find any task with
                 other type of operation like INPUT, INOUT or OUTPUT */
                if(atomic_write_found && ((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) != ATOMIC_WRITE)) {
                    last_iterate_flag = 1;
                    op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                    flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index;
                    out_task = current_desc_task->desc[dst_flow->flow_index].task;
#if defined (OVERLAP)
                    dague_atomic_add_32b((int *) &(out_task->flow_count),-1);
#else
                    out_task->flow_count--;
#endif
                }

                /* check to deal with ATOMIC_WRITE when building chain of INPUT tasks */
                if(!atomic_write_found && ((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) != INPUT)) {
                    last_iterate_flag = 1;
                    /**
                     * Checking if the last task in the chain of INPUT task has any overlapping region or not.
                     * If yes we treat that as the task that needs to wait for the whole chain to finish.
                     * Otherwise we treat as another INPUT task in the chain
                     */
                    if((current_desc_task->desc[dst_flow->flow_index].op_type_parent & GET_REGION_INFO) & (current_desc_task->desc[dst_flow->flow_index].op_type & GET_REGION_INFO)) {
                        op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                        flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index;
                        out_task = current_desc_task->desc[dst_flow->flow_index].task;
#if defined (OVERLAP)
                        dague_atomic_add_32b((int *) &(out_task->flow_count),-1);
#else
                        out_task->flow_count--;
#endif
                    }
                }
            }
            current_desc_task = tmp_task->desc[dst_flow->flow_index].task;
        }

        /* Activating all successors for each flow and setting the last OUT task as the descendant */
        current_succ = head_succ;
        int task_is_ready; /* TODO: What the point of this variable ?*/
        while(NULL != current_succ) {
            /* If there's a OUT task after at least one INPUT task we assign the OUT task as
             * the descendant for that flow for each of the other INPUT task(s) before it
             */
            if(NULL != out_task) {
                current_succ->task->desc[current_succ->flow_index].op_type = op_type_out_task;
                current_succ->task->desc[current_succ->flow_index].flow_index = flow_index_out_task;
                current_succ->task->desc[current_succ->flow_index].task = out_task;
#if defined (OVERLAP)
                dague_atomic_add_32b((int *) &(out_task->flow_count),1);
#else
                out_task->flow_count++;
#endif
            } else {
                if(INPUT == (current_succ->task->desc[current_succ->flow_index].op_type_parent & GET_OP_TYPE)
                   || ATOMIC_WRITE == (current_succ->task->desc[current_succ->flow_index].op_type_parent & GET_OP_TYPE)) {
                    /* treating the last INPUT and ATOMIC_WRITE task specially
                     * as this task needs to release ownership of data
                     */
                    if (current_succ->next == NULL) {
                        current_succ->task->dont_skip_releasing_data[current_succ->flow_index] = 1;
                    } else {
                        current_succ->task->desc[current_succ->flow_index].task = NULL;
                    }
                }
            }

            task_is_ready = 0;
            task_is_ready = ontask(eu, (dague_execution_context_t*)current_succ->task, (dague_execution_context_t*)current_task,
                                   current_succ->deps, &data, rank_src, rank_dst,
                                   vpid_dst, ontask_arg);

            vpid_dst = (vpid_dst+1)%current_task->super.dague_handle->context->nb_vp;
            tmp_succ = current_succ;
            current_succ = current_succ->next;
            free((dague_flow_t *)(tmp_succ->deps->flow));
            free(tmp_succ->deps);
            free(tmp_succ);
            (void)task_is_ready;
        }
    }
}

int
put_fake_writer_as_last_user( dague_dtd_task_t *last_read,
                              dague_dtd_task_t *fake_writer,
                              int last_read_flow_index )
{
    assert(last_read_flow_index!=255);
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


void
ordering_correctly_2(dague_execution_unit_t * eu,
                     const dague_execution_context_t *this_task,
                     uint32_t action_mask,
                     dague_ontask_function_t * ontask,
                     void *ontask_arg)
{
    dague_dtd_task_t *current_task = (dague_dtd_task_t *)this_task;
    int current_dep, count, found_last_writer;
    dague_dtd_task_t *last_read = NULL, *out_task = NULL;
    dague_dtd_task_t *current_desc = NULL;
    int op_type_on_current_flow, desc_op_type, desc_flow_index, out_task_flow_index;
    dague_dtd_tile_t *tile;

    dep_t deps;
    dague_dep_data_description_t data;
    int rank_src = 0, rank_dst = 0, vpid_dst=0;

    for( current_dep = 0; current_dep < current_task->super.function->nb_flows; current_dep++ ) {
        count = 0; found_last_writer = 0;
        current_desc = current_task->desc[current_dep].task;
        op_type_on_current_flow = (current_task->desc[current_dep].op_type_parent & GET_OP_TYPE);
        tile = current_task->desc[current_dep].tile;

        /**
         * In case the same data is used for multiple flows, only the last reference
         * points to the potential successors. Every other use of the data will point
         * to ourself.
         */
        if(current_task == current_desc)
            continue;

        if( NULL == current_desc ) {
             if( INOUT == op_type_on_current_flow ||
                 OUTPUT == op_type_on_current_flow ||
                (current_task->dont_skip_releasing_data[current_dep])) {
#if defined (OVERLAP)
                if(!multithread_dag_build_1(current_task, current_dep)) { /* trying to release ownership */
#endif
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

        assert(current_desc != NULL);
        desc_op_type = (current_task->desc[current_dep].op_type & GET_OP_TYPE);
        desc_flow_index = current_task->desc[current_dep].flow_index;


        int tmp_desc_flow_index = 254;
        int8_t keep_fake_writer = 0;

        /* Create Fake output_task */
        dague_dtd_task_t *fake_writer = create_fake_writer_task( (dague_dtd_handle_t *)current_task->super.dague_handle , tile);
        out_task = fake_writer;
        out_task_flow_index = 0;

        dague_dtd_task_t *prev_task;
        while( NULL != current_desc ) {
            if( OUTPUT == desc_op_type || INOUT == desc_op_type ) {
                out_task = current_desc;
                out_task_flow_index = desc_flow_index;
                found_last_writer = 1;
                break; /* We have found our last_out_task, lets get out */
            }
            count++;

            assert(tmp_desc_flow_index!=255);
            assert(desc_flow_index!=255);
            tmp_desc_flow_index =  desc_flow_index;
            assert(tmp_desc_flow_index!=255);
            prev_task           =  last_read;
            last_read           =  current_desc;
            current_desc        =  current_desc->desc[tmp_desc_flow_index].task;
            desc_flow_index     =  last_read->desc[tmp_desc_flow_index].flow_index;
            desc_op_type        = (last_read->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);

            if( current_desc == NULL ) {
                if( !(keep_fake_writer = put_fake_writer_as_last_user(last_read, fake_writer, tmp_desc_flow_index)) ) {
                    current_desc    = last_read->desc[tmp_desc_flow_index].task;
                    desc_flow_index = last_read->desc[tmp_desc_flow_index].flow_index;
                    assert(desc_flow_index!=255);
                    desc_op_type    = (last_read->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);
                }
            }
        }

        if( !keep_fake_writer ) {
           fake_writer->super.function->release_task(eu, fake_writer);
        } else {
            dague_atomic_add_32b((int *)&(current_task->super.dague_handle->nb_local_tasks),1);
#if defined(DEBUG_HEAVY)
            dague_dtd_task_insert( (dague_dtd_handle_t *)current_task->super.dague_handle, fake_writer );
#endif
        }


#if 0
        int tmp_desc_flow_index;
        while( NULL != current_desc ) {
            if( OUTPUT == desc_op_type || INOUT == desc_op_type ) {
                out_task = current_desc;
                out_task_flow_index = desc_flow_index;
                found_last_writer = 1;
                break; /* We have found our last_out_task, lets get out */
            }
            count++;

            tmp_desc_flow_index =  desc_flow_index;
            desc_flow_index     =  current_desc->desc[desc_flow_index].flow_index;
            desc_op_type        = (current_desc->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);
            last_read           =  current_desc;
            current_desc        =  current_desc->desc[tmp_desc_flow_index].task;
        }
        if( !found_last_writer ) {
            int8_t keep_fake_writer = 0;
            /* Create Fake output_task */
            dague_dtd_task_t *fake_writer = create_fake_writer_task( (dague_dtd_handle_t *)current_task->super.dague_handle , tile);
            out_task = fake_writer;
            out_task_flow_index = 0;

            desc_flow_index = tmp_desc_flow_index;
            while( !(keep_fake_writer = put_fake_writer_as_last_user(last_read, fake_writer, desc_flow_index)) ) {
                tmp_desc_flow_index =  desc_flow_index;
                current_desc = last_read->desc[desc_flow_index].task;
                desc_flow_index = last_read->desc[desc_flow_index].flow_index;
                desc_op_type = (last_read->desc[tmp_desc_flow_index].op_type & GET_OP_TYPE);

                if( INOUT == desc_op_type || OUTPUT == desc_op_type ) {
                    out_task = current_desc;
                    out_task_flow_index = desc_flow_index;
                    break;
                }

                count++;
                last_read = current_desc;
            }

            if( !keep_fake_writer ) {
               fake_writer->super.function->release_task(eu, fake_writer);
            } else {
                dague_atomic_add_32b((int *)&(current_task->super.dague_handle->nb_local_tasks),1);
#if defined(DEBUG_HEAVY)
                dague_dtd_task_insert( (dague_dtd_handle_t *)current_task->super.dague_handle, fake_writer );
#endif
            }
        }
#endif

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

            tmp_desc->desc[tmp_desc_flow_index].task = out_task;
            tmp_desc->desc[tmp_desc_flow_index].flow_index = out_task_flow_index;
            tmp_desc->desc[tmp_desc_flow_index].op_type = out_task->desc[out_task_flow_index].op_type_parent;

            ontask( eu, (dague_execution_context_t *)tmp_desc, (dague_execution_context_t *)current_task,
                    &deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);
        }

        ontask( eu, (dague_execution_context_t *)out_task, (dague_execution_context_t *)current_task,
                    &deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);

        vpid_dst = (vpid_dst+1)%current_task->super.dague_handle->context->nb_vp;

    }
}
