#include <stdio.h>
#include "dague_config.h"
#include "dague.h"
#include "dague/data_distribution.h"
#include "dague/remote_dep.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

/* This function releases the ownership of data once the task is done with it
 * Arguments:   - the task who wants to release the data (dtd_task_t *)
                - the index of the flow for which this data was being used (int)
 * Returns:     - 0 if the task successfully released data / 
                  1 if the task got a descendant before releasing ownership (int)
 */
int
multithread_dag_build_1(const dague_execution_context_t *task, int flow_index)
{
    dtd_task_t *current_task = (dtd_task_t *) task;

    int i = dague_atomic_cas(&(current_task->desc[flow_index].tile->last_user.task), current_task, NULL);

    /* if we can not successfully set the last user of the tile to NULL then we
     * wait until we find a successor 
     */     
    if(i) {
        return 0; /* we are successful, we do not need to wait and there is no successor yet*/
    }else { /* we have a descendant but last time we checked we had none
             * so waiting for the descendant to show up in our list of descendants  
             */
        int unit_waited = 0;
        while(current_task->desc[flow_index].task == NULL) {
            unit_waited++;
            if (unit_waited % 1000 == 0) {
                unit_waited = 0;
                usleep(1);
            }
        }
        return 1;
    }
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
    dtd_task_t *current_task = (dtd_task_t*) this_task;
    dtd_task_t *current_desc_task, *tmp_task;
    dtd_task_t *out_task = NULL; /* last task that will be the descendant of every READ-ONLY task */ 
    /* both are initialized to high numbers to indicate garbage values */
    int flow_index_out_task = 99, op_type_out_task = 99; 
    dep_t* deps;

    dtd_successor_list_t *head_succ = NULL, *current_succ = NULL, *tmp_succ = NULL;

    uint32_t rank_src=0, rank_dst=0;
    int vpid_dst = 0, i;
    uint8_t tmp_flow_index, last_iterate_flag=0;
    uint8_t atomic_write_found;        

    /* Traversing through the successors, for each flow to build list of Read-Only tasks */ 
    for(i=0; i<current_task->total_flow; i++) {
        atomic_write_found = 0;
        head_succ = NULL;
        out_task = NULL;
        tmp_task = current_task;
        last_iterate_flag = 0;
        /* 
         * Not iterating if there's no descendant for this flow
           or the descendant is itself
         */
        if(current_task == current_task->desc[i].task) {
            continue;
        }
        if(NULL == current_task->desc[i].task) {
            /* if the operation type for this flow is not INPUT or ATOMIC_WRITE
             * we release the ownership as we found there's no successor for this flow 
             */
            if (INOUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) || 
                OUTPUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) ||
                (current_task->dont_skip_releasing_data[i])) { 
                if(!multithread_dag_build_1(this_task, i)) { /* trying to release ownership */
                    continue;
                } else {
                }
            } else  {
                continue;
            }
        }

        if ((current_task->desc[i].op_type & GET_OP_TYPE) == INOUT || 
            (current_task->desc[i].op_type & GET_OP_TYPE) == OUTPUT) {
            last_iterate_flag = 1;
        }
        
        tmp_flow_index  = i;
        current_desc_task = current_task->desc[i].task;

        /* checking if the first descendant is the ATOMIC_WRITE or not, otherwise ATOMIC_WRITE is treated as 
           normal INOUT and OUTPUT*/
        if((current_task->desc[i].op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
            atomic_write_found = 1;
        }
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
                        current_desc_task->super.function->name, current_desc_task->task_id, 
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

            if(NULL != current_desc_task->desc[dst_flow->flow_index].task &&
                current_desc_task->desc[dst_flow->flow_index].task != current_desc_task) {
                /* Check to stop building chain of ATOMIC_WRITE tasks when we find any task with 
                   other type of operation like INPUT, INOUT or OUTPUT */
                if(atomic_write_found && ((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) != ATOMIC_WRITE)) {
                    last_iterate_flag = 1;
                    op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                    flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index; 
                    out_task = current_desc_task->desc[dst_flow->flow_index].task;
                    dague_atomic_add_32b((int *) &(out_task->flow_count),-1);
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
                        dague_atomic_add_32b((int *) &(out_task->flow_count),-1);
                    }
                }
            }

            current_desc_task = tmp_task->desc[dst_flow->flow_index].task;
        }

        /* Activating all successors for each flow and setting the last OUT task as the descendant */
        current_succ = head_succ;        
        int task_is_ready;
        while(NULL != current_succ) {
            /* If there's a OUT task after at least one INPUT task we assign the OUT task as
             * the descendant for that flow for each of the other INPUT task(s) before it 
             */
            if(NULL != out_task) { 
                if(atomic_write_found) {
                    /* the op type is usually or'd with region info, in this case it does not
                       make a difference.
                    */
                    current_succ->task->desc[current_succ->flow_index].op_type_parent = ATOMIC_WRITE;
                } else {
                    current_succ->task->desc[current_succ->flow_index].op_type_parent = INPUT;
                }
                current_succ->task->desc[current_succ->flow_index].op_type = op_type_out_task;
                current_succ->task->desc[current_succ->flow_index].flow_index = flow_index_out_task;
                current_succ->task->desc[current_succ->flow_index].task = out_task;
                dague_atomic_add_32b((int *) &(out_task->flow_count),1);
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
        }        
    }
}
