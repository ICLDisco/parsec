#include <stdio.h>
#include "dague_config.h"
#include "dague.h"
#include "data_distribution.h"
#include "remote_dep.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

int
multithread_dag_build_1(const dague_execution_context_t *task, int flow_index)
{
    dtd_task_t *current_task = (dtd_task_t *) task;
    int i;

    /*printf("Current_task %p\t", current_task);
    if(current_task->desc[flow_index].tile->last_user.task != NULL){
        printf("Last user in tile : %p\n", current_task->desc[flow_index].tile->last_user.task);
    }else{
        printf("Last user in tile: NULL\n");
    
    }*/

    i = dague_atomic_cas(&current_task->desc[flow_index].tile->last_user.task, current_task, NULL);
        //printf("Successfully set to null for task with no descendant %d\n", i);
    /* if we can not successfully set the last user of the tile to NULL then we wait until we find a successor */
    if(i){
        return 0; /* we are successful, we do not need to wait and there is no successor yet*/
    }else {
        while(current_task->desc[flow_index].task == NULL){
        }
        return 1;
    }
}

void 
ordering_correctly_1(dague_execution_unit_t * eu,
             const dague_execution_context_t * this_task,
             uint32_t action_mask,
             dague_ontask_function_t * ontask,
             void *ontask_arg)
{
    dague_dtd_handle_t *dague_dtd_handle = (dague_dtd_handle_t*) this_task->dague_handle;
    dague_dep_data_description_t data;
    dtd_task_t *current_task = (dtd_task_t*) this_task;
    dtd_task_t *current_desc_task, *tmp_task;
    dtd_task_t *out_task = NULL; /* last task that will be the descendant of every READ-ONLY task */ 
    int flow_index_out_task = 99, op_type_out_task = 99;
    dep_t* deps;

    dtd_successor_list_t *head_succ = NULL, *current_succ = NULL, *tmp_succ = NULL;

    uint32_t rank_src=0, rank_dst=0;
    int __nb_elt = -1, vpid_dst = 0, i;
    uint8_t tmp_flow_index, last_iterate_flag=0;

    /* Traversing through the successors of each flow  to build list of Read-Only tasks */ 
    for(i=0; i<current_task->total_flow; i++) {
        head_succ = NULL;
        out_task = NULL;
        tmp_task = current_task;
        last_iterate_flag = 0;
        /** 
          * Not iterating if there's no descendant for this flow
         */
        if(NULL == current_task->desc[i].task) {
            //printf("No descendant for this flow %s id: %d and %d\n", current_task->super.function->name, current_task->task_id, i);
            if(INPUT != current_task->desc[i].op_type_parent){ /* task with input on a flow is special */
                if(!multithread_dag_build_1(this_task, i)){
                    continue;
                }
            } else  {
                continue;
            }
        }

        tmp_flow_index  = i;

        current_desc_task = current_task->desc[i].task;

        if (current_task->desc[i].op_type == INOUT || current_task->desc[i].op_type == OUTPUT) {
            last_iterate_flag = 1;
        }
    
        while(NULL != current_desc_task) {
            #if defined (SPIT_TRAVERSAL_INFO) 
            printf("Current successor: %s \t %d\n", current_desc_task->super.function->name, current_desc_task->task_id);
            printf("Total flow: %d  flow_satisfied: %d\n", current_desc_task->flow_count, current_desc_task->flow_satisfied); 
            #endif

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


            if (NULL == head_succ){
                head_succ = tmp_succ;
            } else {
                current_succ->next = tmp_succ;
            }
            current_succ = tmp_succ;
            

            /*ontask(eu, (dague_execution_context_t*)current_desc_task, (dague_execution_context_t*)current_task, deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg); */
            if(1 == last_iterate_flag ) {
                break;
            }
            tmp_task = current_desc_task;
            if ( current_desc_task->desc[dst_flow->flow_index].op_type == INOUT ||  current_desc_task->desc[dst_flow->flow_index].op_type == OUTPUT) {
                   last_iterate_flag = 1;
                   op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                   flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index; 
                   out_task = current_desc_task->desc[dst_flow->flow_index].task;
                   out_task->flow_count--; /* add atomic */
            }
            current_desc_task = tmp_task->desc[dst_flow->flow_index].task;
        }

        /* Activating all successors for each flow and setting the last OUT task as the descendant */
        current_succ = head_succ;        
        while(NULL != current_succ){
            ontask(eu, (dague_execution_context_t*)current_succ->task, (dague_execution_context_t*)current_task, 
                        current_succ->deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);
           
            if(NULL != out_task){ /* If there's a OUT task after at least on INPUT task we assign the OUT task as
                                     the descendant for that flow for each of the other INPUT task(s) before it */
                current_succ->task->desc[current_succ->flow_index].op_type_parent = INPUT;
                current_succ->task->desc[current_succ->flow_index].op_type = op_type_out_task;
                current_succ->task->desc[current_succ->flow_index].flow_index = flow_index_out_task;
                current_succ->task->desc[current_succ->flow_index].task = out_task;
                out_task->flow_count++; /* error can occor at this point */
                                        /* Lists errors:
                                            - Not sure whther we need to atomically increment it in multithreaded
                                              environment.
                                        */
            } else {
                if(INPUT == current_succ->task->desc[current_succ->flow_index].op_type_parent){
                    current_succ->task->desc[current_succ->flow_index].task = NULL;
                    if (current_succ->next == NULL) { /* treating the INPUT tasks specially */
                        multithread_dag_build_1((dague_execution_context_t *)current_succ->task, current_succ->flow_index);
                    }
                }
            }
            tmp_succ = current_succ;
            current_succ = current_succ->next;
            free((dague_flow_t *)(tmp_succ->deps->flow));
            free(tmp_succ->deps);
            free(tmp_succ);
        }        
    }
}
