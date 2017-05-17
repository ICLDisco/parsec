/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * This is a playground to transform Parameterized Task Graphs into insert_task type of
 * programming model. As the PTG engine of PaRSEC generate ready tasks we insert them into
 * a new DTD handle, and allow the new tasks to go through the system and eventually get
 * executed. Upon completion each of these tasks will trigger the completion of the
 * corresponding PTG task.
 *
 * The main complextity here is to synchronize the 2 parsec_handle_t, the one that the upper
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
static parsec_dtd_handle_t *__dtd_handle = NULL;
/* We  use a global ddesc as PTG has varying behavior */
static parsec_ddesc_t *__ddesc;

/* Prototype of some of the static functions */
static void pins_init_ptg_to_dtd(parsec_context_t *master_context);
static void pins_fini_ptg_to_dtd(parsec_context_t *master_context);
static void pins_handle_init_ptg_to_dtd(struct parsec_handle_s *handle);
static void pins_handle_fini_ptg_to_dtd(struct parsec_handle_s *handle);
static int fake_hook_for_testing(parsec_execution_unit_t    *context,
                                 parsec_task_t *__this_task);

const parsec_pins_module_t parsec_pins_ptg_to_dtd_module = {
    &parsec_pins_ptg_to_dtd_component,
    {
        pins_init_ptg_to_dtd,
        pins_fini_ptg_to_dtd,
        pins_handle_init_ptg_to_dtd,
        pins_handle_fini_ptg_to_dtd,
        NULL,
        NULL
    }
};

/**
 * Make a copy of the original chores of the parsec_handle_t and replace them with a
 * set of temporary hooks allowing to transition all tasks as DTD tasks. Only make
 * a copy of the officially exposed nb_function hooks, protecting all those that are
 * hidden (such as the initialization functions).
 */
static void
copy_chores(parsec_handle_t *handle, parsec_dtd_handle_t *dtd_handle)
{
    int i, j;
    for( i = 0; i < (int)handle->nb_functions; i++) {
        for( j = 0; NULL != handle->functions_array[i]->incarnations[j].hook; j++) {
            parsec_hook_t **hook_not_const = (parsec_hook_t **)&(handle->functions_array[i]->incarnations[j].hook);

            /* saving the CPU hook only */
            if (handle->functions_array[i]->incarnations[j].type == PARSEC_DEV_CPU) {
                dtd_handle->actual_hook[i].hook = handle->functions_array[i]->incarnations[j].hook;
            }
            /* copying the fake hook in all the hooks (CPU, GPU etc) */
            *hook_not_const = &fake_hook_for_testing;
        }
    }
}

static void pins_init_ptg_to_dtd(parsec_context_t *master_context)
{
    (void)master_context;
    __ddesc = malloc(sizeof(parsec_ddesc_t));
    parsec_ddesc_init( __ddesc, 0 , 0 );
}

static void pins_fini_ptg_to_dtd(parsec_context_t *master_context)
{
    parsec_ddesc_destroy( __ddesc );
    free(__ddesc);
    (void)master_context;
}

static int pins_handle_complete_callback(parsec_handle_t* ptg_handle, void* void_dtd_handle)
{
    parsec_handle_t *dtd_handle = (parsec_handle_t*)void_dtd_handle;

    parsec_dtd_data_flush_all( (parsec_handle_t *)__dtd_handle, __ddesc );
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
    parsec_execute_and_come_back( dtd_handle->context, dtd_handle, 10);

    parsec_detach_all_dtd_handles_from_context( ptg_handle->context );

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * This PINS callback is triggered everytime the PaRSEC runtime is notified
 * about the eixtence of a new parsec_handle_t. For each handle not corresponding
 * to a DTD version, we create a DTD handle, a placeholder where all intermediary
 * tasks will belong to. As we need to be informed about the completion of the PTG
 * handle, we highjack the completion_callback and register our own function.
 */
static void pins_handle_init_ptg_to_dtd(parsec_handle_t *ptg_handle)
{
    if(ptg_handle->destructor == (parsec_destruct_fn_t)parsec_dtd_handle_destruct) {
        return;
    }
    if( __dtd_handle != NULL ) {
        parsec_handle_free((parsec_handle_t *)__dtd_handle);
    }

    parsec_dtd_ddesc_init( __ddesc );
    __dtd_handle = (parsec_dtd_handle_t *)parsec_dtd_handle_new( );
    dtd_global_deque = OBJ_NEW(parsec_list_t);
    copy_chores(ptg_handle, __dtd_handle);
    {
        parsec_event_cb_t lfct = NULL;
        void* ldata = NULL;
        parsec_get_complete_callback(ptg_handle, &lfct, &ldata);
        if( NULL != lfct ) {
            parsec_set_complete_callback((parsec_handle_t*)__dtd_handle, lfct, ldata);
        }
        parsec_set_complete_callback((parsec_handle_t*)ptg_handle, pins_handle_complete_callback, __dtd_handle);
    }
    parsec_enqueue(ptg_handle->context, (parsec_handle_t*)__dtd_handle);
}

static void pins_handle_fini_ptg_to_dtd(parsec_handle_t *handle)
{
    (void)handle;

    if(handle->destructor == (parsec_destruct_fn_t)parsec_dtd_handle_destruct) {
        return;
    }

    parsec_dtd_ddesc_fini( __ddesc );
    OBJ_RELEASE(dtd_global_deque);
    dtd_global_deque = NULL;
}

/**
 * This function acts as the hook to connect the PaRSEC task with the actual task.
 * The function users passed while inserting task in PaRSEC is called in this procedure.
 * Called internally by the scheduler
 * Arguments:
 *   - the execution unit (parsec_execution_unit_t *)
 *   - the PaRSEC task (parsec_task_t *)
 */
static int
testing_hook_of_dtd_task(parsec_execution_unit_t *context,
                         parsec_dtd_task_t       *dtd_task)
{
    parsec_task_t *orig_task = dtd_task->orig_task;
    int rc = 0;

    PARSEC_TASK_PROF_TRACE(context->eu_profile,
                          dtd_task->super.parsec_handle->profiling_array[2 * dtd_task->super.function->function_id],
                          &(dtd_task->super));

    /**
     * Check to see which interface, if it is the PTG inserting task in DTD then
     * this condition will be true
     */
    rc = ((parsec_dtd_function_t *)(dtd_task->super.function))->fpointer(context, orig_task);
    if(rc == PARSEC_HOOK_RETURN_DONE) {
        /* Completing the orig task */
        dtd_task->orig_task = NULL;
        __parsec_complete_execution( context, orig_task );
    }

    return rc;
}

/* chores and parsec_function_t structure initialization
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
 * Arguments:   - parsec handle (parsec_dtd_handle_t *)
                - data descriptor (parsec_ddesc_t *)
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

    parsec_dtd_tile_t *tile = parsec_dtd_tile_find(__ddesc, combined_key);
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
        tile->ddesc                 = __ddesc;

        SET_LAST_ACCESSOR(tile);

        parsec_dtd_tile_retain(tile);
        parsec_dtd_tile_insert( combined_key,
                                tile, __ddesc );
    }
    return tile;
}

/* Prepare_input function */
static int
data_lookup_ptg_to_dtd_task(parsec_execution_unit_t *context,
                            parsec_task_t *this_task)
{
    (void)context;(void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

/*
 * INSERT Task Function.
 * Each time the user calls it a task is created with the respective parameters the user has passed.
 * For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
 * structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are tracked from this function.
 */
static void
parsec_insert_task_ptg_to_dtd( parsec_dtd_handle_t  *parsec_dtd_handle,
                              parsec_dtd_funcptr_t *fpointer, parsec_task_t *orig_task,
                              char *name_of_kernel, parsec_dtd_task_param_t *packed_parameters_head, int count_of_params )
{
    parsec_handle_t *parsec_handle = (parsec_handle_t *)parsec_dtd_handle;
    if( 0 == parsec_dtd_handle->enqueue_flag ) {
        parsec_enqueue( parsec_handle->context, parsec_handle );
    }

    parsec_dtd_task_param_t *current_paramm;
    int next_arg = -1, tile_op_type, flow_index = 0;
    void *tile;

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    uint64_t fkey = (uint64_t)(uintptr_t)fpointer + count_of_params;
    parsec_function_t *function = (parsec_function_t *) parsec_dtd_function_find
                                                     (parsec_dtd_handle, fkey);

    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        long unsigned int size_of_params = 0;

        if (dump_function_info) {
            parsec_output(parsec_debug_output, "Function Created for task Class: %s\n Has %d parameters\n"
                   "Total Size: %lu\n", name_of_kernel, count_of_params, size_of_params);
        }

        function = parsec_dtd_create_function( parsec_dtd_handle, fpointer, name_of_kernel, count_of_params,
                                               size_of_params, count_of_params );

        __parsec_chore_t **incarnations = (__parsec_chore_t **)&(function->incarnations);
        *incarnations                  = (__parsec_chore_t *)dtd_chore_for_testing;
        function->prepare_input        = data_lookup_ptg_to_dtd_task;

#if defined(PARSEC_PROF_TRACE)
        parsec_dtd_add_profiling_info((parsec_handle_t *)parsec_dtd_handle, function->function_id, name_of_kernel);
#endif /* defined(PARSEC_PROF_TRACE) */
    }

    parsec_dtd_task_t *this_task = parsec_dtd_create_and_initialize_task(parsec_dtd_handle, function, 0/*setting rank as 0*/);
    this_task->orig_task = orig_task;

    /* Iterating through the parameters of the task */
    parsec_dtd_task_param_t *head_of_param_list, *current_param, *tmp_param = NULL;
    void *value_block, *current_val;

    /* Getting the pointer to allocated memory by mempool */
    head_of_param_list = GET_HEAD_OF_PARAM_LIST(this_task);
    current_param      = head_of_param_list;
    value_block        = GET_VALUE_BLOCK(head_of_param_list, ((parsec_dtd_function_t*)function)->count_of_params);
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

    parsec_insert_dtd_task( this_task );
}

static int
fake_hook_for_testing(parsec_execution_unit_t    *context,
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
        parsec_dtd_handle_t *dtd_handle = __dtd_handle;
        parsec_dtd_task_param_ptg_to_dtd_t *head_param = NULL, *current_param = NULL, *tmp_param = NULL;
        parsec_data_t *data;
        parsec_data_key_t key;

        local_list = parsec_list_item_ring_chop(local_list);

        for (i = 0; NULL != this_task->function->in[i]; i++) {

            tmp_op_type = this_task->function->in[i]->flow_flags;
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

        for( i = 0; NULL != this_task->function->out[i]; i++) {
            int op_type;

            tmp_op_type = this_task->function->out[i]->flow_flags;
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

        parsec_insert_task_ptg_to_dtd( dtd_handle, __dtd_handle->actual_hook[this_task->function->function_id].hook,
                                       this_task, (char *)this_task->function->name, (parsec_dtd_task_param_t *)head_param, count_of_params );

        /* Cleaning the params */
        current_param = head_param;
        while( current_param != NULL ) {
            tmp_param = current_param;
            current_param = (parsec_dtd_task_param_ptg_to_dtd_t *)current_param->super.next;
            free(tmp_param);
        }
    }
    (void)context;
    goto redo;
}
