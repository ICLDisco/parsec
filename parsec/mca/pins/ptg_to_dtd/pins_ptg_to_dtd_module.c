/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * This is a playground to transform Parameterized Task Graphs into insert_task type of
 * programming model. As the PTG engine of PaRSEC generate ready tasks we insert them into
 * a new DTD taskpool, and allow the new tasks to go through the system and eventually get
 * executed. Upon completion each of these tasks will trigger the completion of the
 * corresponding PTG task.
 *
 * The main complextity here is to synchronize the 2 parsec_taskpool_t, the one that the upper
 * level is manipulation (possible waiting on), and the one we create for internal purposes.
 */
#include "parsec/parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "pins_ptg_to_dtd.h"
#include "parsec/profiling.h"
#include "parsec/scheduling.h"
#include "parsec/utils/mca_param.h"
#include "parsec/devices/device.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#include <stdio.h>

extern parsec_mempool_t *parsec_dtd_tile_mempool;

/* Structure used to pack arguments of insert_task() */
typedef struct parsec_dtd_task_param_ptg_to_dtd_s parsec_dtd_task_param_ptg_to_dtd_t;
struct parsec_dtd_task_param_ptg_to_dtd_s {
    parsec_dtd_task_param_t  super;
    int             operation_type;
    int             tile_type_index;
};

/* list to push tasks in */
static parsec_list_t *dtd_global_deque = NULL;
/* for testing purpose of automatic insertion from Awesome PTG approach */
static parsec_dtd_taskpool_t *__dtd_taskpool = NULL;
/* We  use a global dc as PTG has varying behavior */
static parsec_data_collection_t *__dc;

/* Prototype of some of the static functions */
static void pins_init_ptg_to_dtd(parsec_context_t *master_context);
static void pins_fini_ptg_to_dtd(parsec_context_t *master_context);
static void pins_taskpool_init_ptg_to_dtd(struct parsec_taskpool_s *tp);
static void pins_taskpool_fini_ptg_to_dtd(struct parsec_taskpool_s *tp);
static int fake_hook_for_testing(parsec_execution_stream_t    *es,
                                 parsec_task_t *__this_task);

const parsec_pins_module_t parsec_pins_ptg_to_dtd_module = {
    &parsec_pins_ptg_to_dtd_component,
    {
        pins_init_ptg_to_dtd,
        pins_fini_ptg_to_dtd,
        pins_taskpool_init_ptg_to_dtd,
        pins_taskpool_fini_ptg_to_dtd,
        NULL,
        NULL
    }
};

/**
 * Make a copy of the original chores of the parsec_taskpool_t and replace them with a
 * set of temporary hooks allowing to transition all tasks as DTD tasks. Only make
 * a copy of the officially exposed nb_function hooks, protecting all those that are
 * hidden (such as the initialization functions).
 */
static void
copy_chores(parsec_taskpool_t *tp, parsec_dtd_taskpool_t *dtd_tp)
{
    int i, j;
    for( i = 0; NULL != tp->task_classes_array[i]; i++) {
        for( j = 0; NULL != tp->task_classes_array[i]->incarnations[j].hook; j++) {
            parsec_hook_t **hook_not_const = (parsec_hook_t **)&(tp->task_classes_array[i]->incarnations[j].hook);

            /* saving the CPU hook only */
            if (tp->task_classes_array[i]->incarnations[j].type == PARSEC_DEV_CPU) {
                dtd_tp->actual_hook[i].hook = tp->task_classes_array[i]->incarnations[j].hook;
            }
            /* copying the fake hook in all the hooks (CPU, GPU etc) */
            *hook_not_const = &fake_hook_for_testing;
        }
    }
}

static void pins_init_ptg_to_dtd(parsec_context_t *master_context)
{
    (void)master_context;
    __dc = malloc(sizeof(parsec_data_collection_t));
    parsec_data_collection_init( __dc, 0 , 0 );
}

static void pins_fini_ptg_to_dtd(parsec_context_t *master_context)
{
    parsec_data_collection_destroy( __dc );
    free(__dc);
    (void)master_context;
}

static int pins_taskpool_complete_callback(parsec_taskpool_t* ptg_tp, void* void_dtd_tp)
{
    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t*)void_dtd_tp;

    parsec_dtd_data_flush_all( (parsec_taskpool_t *)__dtd_taskpool, __dc );
    /* We can not wait until all tasks are done as:
     * 1. This callback is called from the complete hook of another task
     *    so we need to wait for at least until the counter reaches 2
     * 2. It might be the case that this last task from whose hook this callback
     *    has been called has "n" flows and this task writes last on all flows.
     *    Then all the special terminating tasks(copy_data_in_dist) will wait for
     *    the completion of this task.
     * There is no way to know for sure. We pass 10 as a arbitrary number.
     * If there is any task with more than 9 flows it will get stuck in the
     * following function call.
     */
    parsec_execute_and_come_back( dtd_tp->context, dtd_tp, 10);

    parsec_detach_all_dtd_taskpool_from_context( ptg_tp->context );

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * This PINS callback is triggered everytime the PaRSEC runtime is notified
 * about the eixtence of a new parsec_taskpool_t. For each taskpool not corresponding
 * to a DTD version, we create a DTD taskpool, a placeholder where all intermediary
 * tasks will belong to. As we need to be informed about the completion of the PTG
 * taskpool, we highjack the completion_callback and register our own function.
 */
static void pins_taskpool_init_ptg_to_dtd(parsec_taskpool_t *ptg_tp)
{
    /* We only convert PTG taskpools */
    if( PARSEC_TASKPOOL_TYPE_PTG != ptg_tp->taskpool_type )
        return;

    if( ptg_tp->destructor == (parsec_destruct_fn_t)parsec_dtd_taskpool_destruct ) {
        return;
    }
    if( __dtd_taskpool != NULL ) {
        parsec_taskpool_free((parsec_taskpool_t *)__dtd_taskpool);
    }

    parsec_dtd_data_collection_init( __dc );
    __dtd_taskpool = (parsec_dtd_taskpool_t *)parsec_dtd_taskpool_new( );
    dtd_global_deque = OBJ_NEW(parsec_list_t);
    copy_chores(ptg_tp, __dtd_taskpool);
    {
        parsec_event_cb_t lfct = NULL;
        void* ldata = NULL;
        parsec_taskpool_get_complete_callback(ptg_tp, &lfct, &ldata);
        if( NULL != lfct ) {
            parsec_taskpool_set_complete_callback((parsec_taskpool_t*)__dtd_taskpool, lfct, ldata);
        }
        parsec_taskpool_set_complete_callback((parsec_taskpool_t*)ptg_tp, pins_taskpool_complete_callback, __dtd_taskpool);
    }
    parsec_enqueue(ptg_tp->context, (parsec_taskpool_t*)__dtd_taskpool);
}

static void pins_taskpool_fini_ptg_to_dtd(parsec_taskpool_t *tp)
{
    if(tp->destructor == (parsec_destruct_fn_t)parsec_dtd_taskpool_destruct) {
        return;
    }

    parsec_dtd_data_collection_fini( __dc );
    OBJ_RELEASE(dtd_global_deque);
    dtd_global_deque = NULL;
}

/**
 * This function acts as the hook to connect the PaRSEC task with the actual task.
 * The function users passed while inserting task in PaRSEC is called in this procedure.
 * Called internally by the scheduler
 * Arguments:
 *   - the execution unit (parsec_execution_stream_t *)
 *   - the PaRSEC task (parsec_task_t *)
 */
static int
testing_hook_of_dtd_task(parsec_execution_stream_t *es,
                         parsec_dtd_task_t         *dtd_task)
{
    parsec_task_t *orig_task = dtd_task->orig_task;
    int rc = 0;

    PARSEC_TASK_PROF_TRACE(es->es_profile,
                          dtd_task->super.taskpool->profiling_array[2 * dtd_task->super.task_class->task_class_id],
                          &(dtd_task->super));

    /**
     * Check to see which interface, if it is the PTG inserting task in DTD then
     * this condition will be true
     */
    rc = ((parsec_dtd_task_class_t *)(dtd_task->super.task_class))->fpointer(es, orig_task);
    if(rc == PARSEC_HOOK_RETURN_DONE) {
        /* Completing the orig task */
        dtd_task->orig_task = NULL;
        __parsec_complete_execution( es, orig_task );
    }

    return rc;
}

/* chores and parsec_task_class_t structure initialization
 * This is for the ptg inserting task in dtd mode
 */
static const __parsec_chore_t dtd_chore_for_testing[] = {
    {.type      = PARSEC_DEV_CPU,
     .evaluate  = NULL,
     .hook      = (parsec_hook_t*)testing_hook_of_dtd_task },
    {.type      = PARSEC_DEV_NONE,
     .evaluate  = NULL,
     .hook      = NULL},             /* End marker */
};


/* Function to manage tiles once insert_task() is called, this functions is to
 * generate tasks from PTG and insert it using insert task interface.
 * This function checks if the tile structure(parsec_dtd_tile_t) is created for the data
 * already or not.
 * Arguments:   - parsec taskpool (parsec_dtd_taskpool_t *)
                - data descriptor (parsec_data_collection_t *)
                - key of this data (parsec_data_key_t)
 * Returns:     - tile, creates one if not already created, and returns that
                  tile, (parsec_dtd_tile_t *)
 */
static parsec_dtd_tile_t*
tile_manage_for_testing(parsec_data_t *data, parsec_data_key_t key, int arena_index)
{
    uint64_t data_ptr = (uint64_t)(uintptr_t)data;
    uint64_t combined_key = ( (data_ptr << 32) | ((uint32_t)key) );

    //uint64_t combined_key = ((((uint32_t)data)<<32) | ((uint32_t)key));

    parsec_dtd_tile_t *tile = parsec_dtd_tile_find(__dc, combined_key);
    if( NULL == tile ) {
        tile = (parsec_dtd_tile_t *) parsec_thread_mempool_allocate(parsec_dtd_tile_mempool->thread_mempools);
        tile->key                   = combined_key;
        tile->rank                  = 0;
        tile->flushed               = NOT_FLUSHED;
        tile->data_copy             = data->device_copies[0];
#if defined(PARSEC_HAVE_CUDA)
        tile->data_copy->readers    = 0;
#endif
        tile->arena_index           = arena_index;
        tile->dc                 = __dc;

        SET_LAST_ACCESSOR(tile);

        parsec_dtd_tile_retain(tile);
        parsec_dtd_tile_insert( combined_key,
                                tile, __dc );
    }
    return tile;
}

/* Prepare_input function */
static int
data_lookup_ptg_to_dtd_task(parsec_execution_stream_t *es,
                            parsec_task_t *this_task)
{
    (void)es;(void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

/*
 * INSERT Task Function.
 * Each time the user calls it a task is created with the respective parameters the user has passed.
 * For each task class a structure is created. (e.g. for Cholesky 4 function
 * structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are tracked from this function.
 */
static void
parsec_dtd_taskpool_insert_task_ptg_to_dtd( parsec_dtd_taskpool_t  *dtd_tp,
                              parsec_dtd_funcptr_t *fpointer, parsec_task_t *orig_task,
                              char *name_of_kernel, parsec_dtd_task_param_t *packed_parameters_head, int count_of_params )
{
    parsec_taskpool_t *parsec_tp = (parsec_taskpool_t *)dtd_tp;
    if( 0 == dtd_tp->enqueue_flag ) {
        parsec_enqueue( parsec_tp->context, parsec_tp );
    }

    parsec_dtd_task_param_t *current_paramm;
    int next_arg = -1, tile_op_type, flow_index = 0;
    void *tile;

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    uint64_t fkey = (uint64_t)(uintptr_t)fpointer + count_of_params;
    parsec_task_class_t *tc = (parsec_task_class_t *)parsec_dtd_find_task_class
                                                      (dtd_tp, fkey);

    if( NULL == tc ) {
        /* calculating the size of parameters for each task class*/
        long unsigned int size_of_params = 0;

        tc = parsec_dtd_create_task_class( dtd_tp, fpointer, name_of_kernel, count_of_params,
                                           size_of_params, count_of_params );

        __parsec_chore_t **incarnations = (__parsec_chore_t **)&(tc->incarnations);
        *incarnations                   = (__parsec_chore_t *)dtd_chore_for_testing;
        tc->prepare_input               = data_lookup_ptg_to_dtd_task;

#if defined(PARSEC_PROF_TRACE)
        parsec_dtd_add_profiling_info((parsec_taskpool_t *)dtd_tp, tc->task_class_id, name_of_kernel);
#endif /* defined(PARSEC_PROF_TRACE) */
    }

    parsec_dtd_task_t *this_task = parsec_dtd_create_and_initialize_task(dtd_tp, tc, 0/*setting rank as 0*/);
    this_task->orig_task = orig_task;

    /* Iterating through the parameters of the task */
    parsec_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param = NULL;
    void *value_block, *current_val;

    /* Getting the pointer to allocated memory by mempool */
    head_of_param_list = GET_HEAD_OF_PARAM_LIST(this_task);
    current_param      = head_of_param_list;
    value_block        = GET_VALUE_BLOCK(head_of_param_list, ((parsec_dtd_task_class_t*)tc)->count_of_params);
    current_val        = value_block;

    current_paramm = packed_parameters_head;

    int write_flow_count = 1;
    while(current_paramm != NULL) {
        tile         = current_paramm->pointer_to_tile;
        tile_op_type = ((parsec_dtd_task_param_ptg_to_dtd_t *)current_paramm)->operation_type;

        if( INOUT == (tile_op_type & GET_OP_TYPE) || OUTPUT == (tile_op_type & GET_OP_TYPE) ) {
            write_flow_count++;
        }

        parsec_dtd_set_params_of_task( this_task, tile, tile_op_type,
                                       &flow_index, &current_val,
                                       current_param, next_arg );

        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;

        current_paramm = current_paramm->next;
    }

    parsec_object_t *object;
    if( parsec_dtd_task_is_local(this_task) ) {
        /* retaining the local task as many write flows
         * as it has and one to indicate when we have
         * executed the task
         */
        object = (parsec_object_t *)this_task;
        (void)parsec_atomic_add_32b( &object->obj_reference_count, (write_flow_count) );

    }

    if( tmp_param != NULL )
        tmp_param->next = NULL;

    parsec_insert_dtd_task( (parsec_task_t *)this_task );
}

static int
fake_hook_for_testing(parsec_execution_stream_t *es,
                      parsec_task_t *this_task)
{
    static parsec_atomic_lock_t pins_ptg_to_dtd_atomic_lock = {PARSEC_ATOMIC_UNLOCKED};
    parsec_list_item_t* local_list = NULL;

    /* We will try to push our tasks in the same Global Deque.
     * Then we will try to take ownership of the Global Deque and try to pull the tasks from there and build a list.
     */
    /* push task in the global dequeue */
    parsec_list_push_back( dtd_global_deque, (parsec_list_item_t*)this_task );

    /* try to get ownership of global deque*/
    if( !parsec_atomic_trylock(&pins_ptg_to_dtd_atomic_lock) )
        return PARSEC_HOOK_RETURN_ASYNC;
  redo:
    /* Extract all the elements in the queue and then release the queue as fast as possible */
    local_list = parsec_list_unchain(dtd_global_deque);

    if( NULL == local_list ) {
        parsec_atomic_unlock(&pins_ptg_to_dtd_atomic_lock);
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    /* Successful in ataining the lock, now we will pop all the tasks out of the list and put it
     * in our local list.
     */
    for( this_task = (parsec_task_t*)local_list;
         NULL != this_task;
         this_task = (parsec_task_t*)local_list ) {

        int i, tmp_op_type;
        int count_of_params = 0;
        parsec_dtd_taskpool_t *dtd_tp= __dtd_taskpool;
        parsec_dtd_task_param_ptg_to_dtd_t *head_param = NULL, *current_param = NULL, *tmp_param = NULL;
        parsec_data_t *data;
        parsec_data_key_t key;

        local_list = parsec_list_item_ring_chop(local_list);

        for (i = 0; NULL != this_task->task_class->in[i]; i++) {

            tmp_op_type = this_task->task_class->in[i]->flow_flags;
            int op_type;
            parsec_dtd_tile_t *tile = NULL;

            if ((tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_RW) {
                op_type = INOUT;
            } else if( (tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_READ ) {
                op_type = INPUT;
            } else {
                continue;  /* next IN flow */
            }

            if( NULL != this_task->data[i].data_in ) {
                data = this_task->data[i].data_in->original;
                key = this_task->data[i].data_in->original->key;
                tile = tile_manage_for_testing(data, key, 0);
            }

            tmp_param = (parsec_dtd_task_param_ptg_to_dtd_t *) malloc(sizeof(parsec_dtd_task_param_ptg_to_dtd_t));
            tmp_param->super.pointer_to_tile = (void *)tile;
            tmp_param->operation_type = op_type;
            tmp_param->tile_type_index = 0;
            tmp_param->super.next = NULL;

            if(head_param == NULL) {
                head_param = tmp_param;
            } else {
                current_param->super.next = (parsec_dtd_task_param_t *)tmp_param;
            }
            count_of_params++;
            current_param = tmp_param;
        }

        for( i = 0; NULL != this_task->task_class->out[i]; i++) {
            int op_type;

            tmp_op_type = this_task->task_class->out[i]->flow_flags;
            parsec_dtd_tile_t *tile = NULL;
            if((tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_WRITE) {
                op_type = OUTPUT;
                if( NULL != this_task->data[i].data_out ) {
                    data = this_task->data[i].data_out->original;
                    key = this_task->data[i].data_out->original->key;
                    tile = tile_manage_for_testing(data, key, 0);
                }
            } else {
                continue;
            }

            tmp_param = (parsec_dtd_task_param_ptg_to_dtd_t *) malloc(sizeof(parsec_dtd_task_param_ptg_to_dtd_t));
            tmp_param->super.pointer_to_tile = (void *)tile;
            tmp_param->operation_type = op_type;
            tmp_param->tile_type_index = 0;
            tmp_param->super.next = NULL;

            if(head_param == NULL) {
                head_param = tmp_param;
            } else {
                current_param->super.next = (parsec_dtd_task_param_t *)tmp_param;
            }
            count_of_params++;
            current_param = tmp_param;
        }

        parsec_dtd_taskpool_insert_task_ptg_to_dtd( dtd_tp, __dtd_taskpool->actual_hook[this_task->task_class->task_class_id].hook,
                                       this_task, (char *)this_task->task_class->name, (parsec_dtd_task_param_t *)head_param, count_of_params );

        /* Cleaning the params */
        current_param = head_param;
        while( current_param != NULL ) {
            tmp_param = current_param;
            current_param = (parsec_dtd_task_param_ptg_to_dtd_t *)current_param->super.next;
            free(tmp_param);
        }
    }
    goto redo;
    (void)es;
}
