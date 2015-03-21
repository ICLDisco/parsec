#include <stdio.h>
#include "dague_config.h"
#include "dague.h"
#include "data_distribution.h"
#include "remote_dep.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

/* returns 0 if the task successfully released the tile (data) */
int
multithread_dag_build_2(const dague_execution_context_t *task, int flow_index)
{

    dtd_task_t *current_task = (dtd_task_t *) task;
    int i;


    i = dague_atomic_cas(&current_task->desc[flow_index].task, current_task, NULL);
    
    printf("size of :%d \n",sizeof(&current_task->desc[flow_index].task));

    return 0;
}

/* Returns 0 if the task successfully released the tile (data) 
   waits otherwise for a descendant and returns 1 when one is found
*/
int
multithread_dag_build_1(const dague_execution_context_t *task, int flow_index)
{
    dtd_task_t *current_task = (dtd_task_t *) task;
    int i;

    //printf("Current_task in releasing a tile: %d\tfor flow index: %d\n", current_task->task_id, flow_index);
    /*if(current_task->desc[flow_index].tile->last_user.task != NULL){
        printf("Last user in tile : %p\n", current_task->desc[flow_index].tile->last_user.task);
    }else{
        printf("Last user in tile: NULL\n");
    
    }*/

    i = dague_atomic_cas(&(current_task->desc[flow_index].tile->last_user.task), current_task, NULL);
    //printf("Successfully set to null for task with no descendant %d\n", i);

    /* if we can not successfully set the last user of the tile to NULL then we wait until we find a successor */
    if(i){
        //printf("Successfully set to NULL\n");
        return 0; /* we are successful, we do not need to wait and there is no successor yet*/
    }else {
    /*
        while(1){
          cnt = 0;
          while(current_task->desc[flow_index].task == NULL && ++cnt < 50){
            mfence()
          }
          sched_yield();
          if(cnt<500) break;
        }
    */
        
        while(current_task->desc[flow_index].task == NULL){
        }
        //printf("Was releasing tile and then found a desc: %d\n", current_task->desc[flow_index].task->task_id);

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

#if defined(ATOMIC_WRITE_ENABLED)
    uint8_t atomic_write_found = 0;        
#endif

    /* Traversing through the successors of each flow  to build list of Read-Only tasks */ 
    for(i=0; i<current_task->total_flow; i++) {
#if defined(ATOMIC_WRITE_ENABLED)
        atomic_write_found = 0;
#endif
        head_succ = NULL;
        out_task = NULL;
        tmp_task = current_task;
        last_iterate_flag = 0;
        /** 
          * Not iterating if there's no descendant for this flow
         */
        if(NULL == current_task->desc[i].task) {
#if defined(ATOMIC_WRITE_ENABLED)
            if(INOUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) || OUTPUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) 
               || current_task->dont_skip_releasing_data[i]){ 
#else
            if((INPUT != (current_task->desc[i].op_type_parent) & GET_OP_TYPE) || current_task->dont_skip_releasing_data[i]){ /* task with input on a flow is special taking care
                                                                  at the end of this function */
#endif
                if(!multithread_dag_build_1(this_task, i)){
                    continue;
                }
            } else  {
                continue;
            }
        }

        tmp_flow_index  = i;

        current_desc_task = current_task->desc[i].task;

        if ((current_task->desc[i].op_type & GET_OP_TYPE) == INOUT || (current_task->desc[i].op_type & GET_OP_TYPE) == OUTPUT) {
            last_iterate_flag = 1;
        }
        
#if defined(ATOMIC_WRITE_ENABLED)
        /* checking if the first descendant is the ATOMIC_WRITE or not, otherwise ATOMIC_WRITE is treated as 
           normal INOUT and OUTPUT*/
        if((current_task->desc[i].op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
            atomic_write_found = 1;
        }
        if(!(current_desc_task->desc[current_task->desc[i].flow_index].task == NULL 
             || current_desc_task == current_desc_task->desc[current_task->desc[i].flow_index].task)){
            if(atomic_write_found
               &&  ((current_desc_task->desc[current_task->desc[i].flow_index].op_type & GET_OP_TYPE) != ATOMIC_WRITE)){
                last_iterate_flag = 1;
            }
        }
#endif
            
        while(NULL != current_desc_task) {
#if defined (SPIT_TRAVERSAL_INFO) 
            printf("Current successor: %s \t %d\nTotal flow: %d  flow_satisfied: %d\n", current_desc_task->super.function->name, current_desc_task->task_id, current_desc_task->flow_count, current_desc_task->flow_satisfied);
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

            if(NULL == head_succ){
                head_succ = tmp_succ;
            } else {
                current_succ->next = tmp_succ;
            }
            current_succ = tmp_succ;

            if(last_iterate_flag) {
                break;
            }

            tmp_task = current_desc_task;

            if(NULL != current_desc_task->desc[dst_flow->flow_index].task) {
#if defined (ATOMIC_WRITE_ENABLED) 
                /* Check to stop building chain of ATOMIC_WRITE tasks when we find any task with 
                   other type of operation like INPUT, INOUT or OUTPUT */
                if(atomic_write_found && ((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) != ATOMIC_WRITE)){
                    last_iterate_flag = 1;
                    op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                    flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index; 
                    out_task = current_desc_task->desc[dst_flow->flow_index].task;
                    out_task->flow_count--; /* The flow count needs to be adjusted as we play with it 
                                              to maintain correct order. 
                                              TODO: add atomic operation */

                }

                /* check to deal with ATOMIC_WRITE when building chain of INPUT tasks */
                if(!atomic_write_found && ((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) != INPUT)){
#else
                if((current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) == INOUT ||  (current_desc_task->desc[dst_flow->flow_index].op_type & GET_OP_TYPE) == OUTPUT) {
#endif
                    last_iterate_flag = 1;
                    /**
                    * Checking if the last task in the chain of INPUT task has any overlapping region or not.
                    * If yes we treat that as the task that needs to wait for the whole chain to finish.
                    * Otherwise we treat as another INPUT task in the chain 
                    */
                    if((current_desc_task->desc[dst_flow->flow_index].op_type_parent & GET_REGION_INFO) & (current_desc_task->desc[dst_flow->flow_index].op_type & GET_REGION_INFO)){
                        op_type_out_task = current_desc_task->desc[dst_flow->flow_index].op_type;
                        flow_index_out_task = current_desc_task->desc[dst_flow->flow_index].flow_index; 
                        out_task = current_desc_task->desc[dst_flow->flow_index].task;
                        out_task->flow_count--; /* The flow count needs to be adjusted as we play with it 
                                                  to maintain correct order. 
                                                  TODO: add atomic operation */
                    }
                }
            }

            current_desc_task = tmp_task->desc[dst_flow->flow_index].task;
        }

        /* Activating all successors for each flow and setting the last OUT task as the descendant */
        current_succ = head_succ;        
        int task_is_ready = 0;
        while(NULL != current_succ){
            task_is_ready = 0;
            task_is_ready = ontask(eu, (dague_execution_context_t*)current_succ->task, (dague_execution_context_t*)current_task, 
                        current_succ->deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);
           
            if(NULL != out_task){ /* If there's a OUT task after at least one INPUT task we assign the OUT task as
                                     the descendant for that flow for each of the other INPUT task(s) before it */
           
#if defined (ATOMIC_WRITE_ENABLED) 
                if(atomic_write_found){
                    current_succ->task->desc[current_succ->flow_index].op_type_parent = ATOMIC_WRITE;
                } else {
                    current_succ->task->desc[current_succ->flow_index].op_type_parent = INPUT;
                }
#else
                current_succ->task->desc[current_succ->flow_index].op_type_parent = INPUT;
#endif
                current_succ->task->desc[current_succ->flow_index].op_type = op_type_out_task;
                current_succ->task->desc[current_succ->flow_index].flow_index = flow_index_out_task;
                current_succ->task->desc[current_succ->flow_index].task = out_task;
                out_task->flow_count++; /* error can occur at this point */
                                        /* Lists errors:
                                            - Not sure whther we need to atomically increment it in multithreaded
                                              environment.
                                        */
            } else {
#if defined (ATOMIC_WRITE_ENABLED) 
                if(INPUT == (current_succ->task->desc[current_succ->flow_index].op_type_parent & GET_OP_TYPE) 
                   || ATOMIC_WRITE == (current_succ->task->desc[current_succ->flow_index].op_type_parent & GET_OP_TYPE)){

#else
                if(INPUT == (current_succ->task->desc[current_succ->flow_index].op_type_parent & GET_OP_TYPE)){
#endif
                    if (current_succ->next == NULL) { /* treating the INPUT tasks specially */
                        //multithread_dag_build_1((dague_execution_context_t *)current_succ->task, current_succ->flow_index);
                        current_succ->task->dont_skip_releasing_data[current_succ->flow_index] = 1;
                    } else {
                        current_succ->task->desc[current_succ->flow_index].task = NULL;
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
