/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/class/parsec_future.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"

static void parsec_datacopy_future_construct(parsec_base_future_t* future);

static void parsec_datacopy_future_cleanup_nested(parsec_base_future_t* future);
static void parsec_datacopy_future_destruct(parsec_base_future_t* future);

static void  parsec_datacopy_future_init(parsec_base_future_t* future,
                                        parsec_future_cb_fulfill cb,
                                        void * cb_fulfill_data_in,
                                        parsec_future_cb_match cb_match,
                                        void * cb_match_data_in,
                                        parsec_future_cb_cleanup cb_cleanup);
static void* parsec_datacopy_future_get_or_trigger_internal(parsec_base_future_t* future,
                                                            void* es,
                                                            void* task);
static void* parsec_datacopy_future_get_or_trigger(parsec_base_future_t* future,
                                                   parsec_future_cb_nested cb_setup_nested,
                                                   void* cb_data_in,
                                                   void* es,
                                                   void* task);
static void parsec_datacopy_future_set(parsec_base_future_t* future, void*data);


/*Function implementation */

/**
 *
 * Routine to initialize the future. This routine is not thread-safe.
 * The upper level must ensure data is only init once.
 *
 * @param[inout] future to be initialized.
 * @param[in] cb callback routine to fulfilled the future.
 * @param[in] cb_fulfill_data_in callback routine input data.
 * @param[in] cb_match callback routine to check if the tracked data
 * matches the requested data during get_or_trigger.
 * @param[in] cb_match_data_in input to be cb_match to compare the tracked data.
 * @param[in] cb_cleanup callback routine to clean up the future.
 */
static void parsec_datacopy_future_init(parsec_base_future_t* future,
                                        parsec_future_cb_fulfill cb,
                                        void * cb_fulfill_data_in,
                                        parsec_future_cb_match cb_match,
                                        void * cb_match_data_in,
                                        parsec_future_cb_cleanup cb_cleanup)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    d_fut->super.status = PARSEC_DATA_FUTURE_STATUS_INIT;
    d_fut->super.cb_fulfill = cb;
    d_fut->nested_enable  = 1;
    d_fut->cb_fulfill_data_in = cb_fulfill_data_in;
    d_fut->cb_match = cb_match;
    d_fut->cb_match_data_in = cb_match_data_in;
    d_fut->cb_cleanup = cb_cleanup;
    d_fut->nested_futures = NULL;
}

/**
 *
 * Internal routine to get the data tracked by the future or trigger its
 * fulfillment.
 *
 * @param[in] future to be trigger.
 * @param[in] es parsec_execution_stream_t.
 * @param[in] tp parsec_taskpool_t.
 * @param[in] task current parsec_task_t.
 * @return data tracked by the future; NULL in case the future is not fulfilled yet.
 */
static void* parsec_datacopy_future_get_or_trigger_internal(parsec_base_future_t* future,
                                                            void* es,
                                                            void* task)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    if( !(d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED) ){
        /* Try to trigger */
        parsec_atomic_lock(&d_fut->super.future_lock);
        if( !(d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_TRIGGERED) ){
            d_fut->super.status |= PARSEC_DATA_FUTURE_STATUS_TRIGGERED;
            d_fut->super.cb_fulfill(future, &d_fut->cb_fulfill_data_in, es, task);
        }
        parsec_atomic_unlock(&d_fut->super.future_lock);

        if( ! (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED) ){
            return NULL;
        }
    }

    return d_fut->super.tracked_data;
}

/**
 *
 * Routine to get the data tracked by the future or trigger its
 * fulfillment.
 * In case cb_setup_nested and cb_data_in are given and the data tracked by the
 * future doesn't match the requested specifications (cb_data_in) a new nested
 * future will be set up and trigger.
 *
 * @param[in] future to be triggered.
 * @param[in] cb_setup_nested callback routine to set up a nested future or NULL.
 * @param[in] cb_data_in requested specifications or NULL.
 * @param[in] es parsec_execution_stream_t.
 * @param[in] tp parsec_taskpool_t.
 * @param[in] task current parsec_task_t.
 * @return data tracked by the future matching the specification; NULL in
 * case the future is not fulfilled yet.
 */
static void* parsec_datacopy_future_get_or_trigger(parsec_base_future_t* future,
                                                   parsec_future_cb_nested cb_setup_nested,
                                                   void* cb_data_in,
                                                   void* es,
                                                   void* task)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    void *data = NULL;
    if( NULL == cb_data_in ){ /* No request for checking if target data matches */
        return parsec_datacopy_future_get_or_trigger_internal(future, es, task);
    }

    /* Only complete the base future if it is a match */
    /* Check if target data matches the given specs */
    if( d_fut->cb_match(future, d_fut->cb_match_data_in, cb_data_in)){
        /* future data matches requested version */
        return parsec_datacopy_future_get_or_trigger_internal(future, es, task);
    }

    assert(d_fut->nested_enable); /*second level future is able to create more nested versions */

    /* lock to check if nested future data matches requested specs */
    parsec_atomic_lock(&d_fut->super.future_lock);
    if( NULL == d_fut->nested_futures ){
        d_fut->nested_futures = PARSEC_OBJ_NEW(parsec_list_t);
    }

    parsec_list_item_t *item;
    for(item = PARSEC_LIST_ITERATOR_FIRST(d_fut->nested_futures);
        item != PARSEC_LIST_ITERATOR_END(d_fut->nested_futures);
        item = PARSEC_LIST_ITERATOR_NEXT(item) ) {
        data = parsec_datacopy_future_get_or_trigger_internal(
                ((parsec_base_future_t*)item), es, task);
        if( ( NULL == data )/* future being generating */
           || ( d_fut->cb_match(future,
                   ((parsec_datacopy_future_t*)item)->cb_match_data_in, cb_data_in) )/* data matches requested specs */
           )
        {
            parsec_atomic_unlock(&d_fut->super.future_lock);
            return data;
        }
    }

    /* create new nested future relying on the callback to set it up */
    assert(cb_setup_nested != NULL);
    parsec_datacopy_future_t* new_nested_fut;
    cb_setup_nested(((parsec_base_future_t**)&new_nested_fut), d_fut, cb_data_in);
    PARSEC_OBJ_CONSTRUCT(&new_nested_fut->super.item, parsec_list_item_t);
    new_nested_fut->nested_enable = 0;
    parsec_list_nolock_push_back(d_fut->nested_futures, (parsec_list_item_t*) &new_nested_fut->super.item);

    /* unlock; this prevents no other thread to check the nested list of until the
     * new nested future is completely set up */
    parsec_atomic_unlock(&d_fut->super.future_lock);

    return parsec_datacopy_future_get_or_trigger_internal(
            ((parsec_base_future_t*)new_nested_fut), es, task);
}

/**
 * Function to set up the target data of the future. This routine is not thread-safe.
 *
 * The upper level should only invoke this routine during the fulfill callback (a
 * datacopy_future will be triggered only once, thus, data will only be set once) or
 * otherwise, the upper level must ensure data is only set up once.
 *
 * E.g. when using datacopy futures to reshape data, the future can be set
 * after init by the thread that is creating it; or otherwise on the callback that
 * performs the reshaping when it is triggered.
 *
 * @param[in] future.
 * @param[in] data tracked by the future that fulfills it.
 */
static void parsec_datacopy_future_set(parsec_base_future_t* future, void*data)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    assert( !(d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED) );
    d_fut->super.tracked_data = data;
    d_fut->super.status |= PARSEC_DATA_FUTURE_STATUS_COMPLETED;
}

/**
 * Routine to fill up base future specifications. This routine should not be called
 * on a datacopy future.
 * @param[in] future.
 * @return 0
 */
static int parsec_datacopy_future_is_ready(parsec_base_future_t* future)
{
    (void)future;
    return 0;
}

/**
 * Routine to fill up base future specifications. This routine should not be called
 * on a datacopy future.
 * @param[in] future.
 * @return NULL
 */
static void* parsec_datacopy_future_get(parsec_base_future_t* future)
{
    (void)future;
    return NULL;
}


/* Set up datacopy_future routines.
 */
static parsec_future_fn_t parsec_datacopy_future_functions = {
    .is_ready         = parsec_datacopy_future_is_ready,
    .get              = parsec_datacopy_future_get,
    .get_or_trigger   = (parsec_future_get_or_trigger_t)parsec_datacopy_future_get_or_trigger,
    .set              = parsec_datacopy_future_set,
    .future_init      = (parsec_future_init_t)parsec_datacopy_future_init
};

/**
 *
 * Routine to construct a datacopy_future.
 *
 * @param[in] future to be constructed.
 */
static void parsec_datacopy_future_construct(parsec_base_future_t* future)
{
    parsec_atomic_lock_t temp = PARSEC_ATOMIC_UNLOCKED;
    future->future_lock = temp;
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    d_fut->super.future_class = &parsec_datacopy_future_functions;
    d_fut->super.status = 0;
    d_fut->super.cb_fulfill = NULL;
    d_fut->super.tracked_data = NULL;
    d_fut->cb_fulfill_data_in = NULL;
    d_fut->cb_match = NULL;
    d_fut->cb_cleanup = NULL;
}

/**
 *
 * Routine to destruct the nested futures of a datacopy_future.
 *
 * @param[in] future to be cleanup.
 */
static void parsec_datacopy_future_cleanup_nested(parsec_base_future_t* future)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;

    PARSEC_DEBUG_VERBOSE(14, parsec_debug_output,
                         "RESHAPE_PROMISE CLEANUP_NESTED fut %p (copy %p) %s status: %s & %s",
                         d_fut, d_fut->super.tracked_data,
                         (d_fut->nested_enable == 1)? "root": "leave",
                         (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_TRIGGERED)? "TRIGGERED": "NOTTRIGGERED",
                         (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED)? "COMPLETED": "NOTCOMPLETED");

    /* Base future doesn't need to be completed, it can be used as an intermediate to create nested futures.
     * However, nested futures have to be completed.
     */
    if(!d_fut->nested_enable)
            assert( (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED) );

    if(d_fut->nested_futures != NULL){
        parsec_datacopy_future_t* nested_future;
        while(parsec_list_nolock_is_empty(d_fut->nested_futures) == 0){
            nested_future = (parsec_datacopy_future_t*)parsec_list_nolock_pop_front(d_fut->nested_futures);
            PARSEC_DEBUG_VERBOSE(14, parsec_debug_output,
                                 "RESHAPE_PROMISE CLEANUP_NESTED fut %p (copy %p) have nested %p",
                                 d_fut, d_fut->super.tracked_data, nested_future);
            /* manually run destructor - workaround: future -> base_future -> list item -> object, fix? */
            parsec_datacopy_future_destruct((parsec_base_future_t*)nested_future);
            PARSEC_OBJ_RELEASE(nested_future);
        }
        PARSEC_OBJ_RELEASE(d_fut->nested_futures);
    }
}

/**
 *
 * Routine to destruct a datacopy_future.
 * The routine will clean up the nested futures, if there are any, and invoke
 * the user callback cleanup.
 *
 * @param[in] future to be destructed.
 */
static void parsec_datacopy_future_destruct(parsec_base_future_t* future)
{
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;

    PARSEC_DEBUG_VERBOSE(14, parsec_debug_output,
                         "RESHAPE_PROMISE DESTROY fut %p (copy %p) %s status: %s & %s",
                         d_fut, d_fut->super.tracked_data,
                         (d_fut->nested_enable == 1)? "root": "leave",
                         (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_TRIGGERED)? "TRIGGERED": "NOTTRIGGERED",
                         (d_fut->super.status & PARSEC_DATA_FUTURE_STATUS_COMPLETED)? "COMPLETED": "NOTCOMPLETED");

    /* Clean up nested futures*/
    parsec_datacopy_future_cleanup_nested(future);

    /* User clean up routine */
    if(d_fut->cb_cleanup != NULL){
        d_fut->cb_cleanup(future);
    }
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_datacopy_future_t, parsec_base_future_t,
                   parsec_datacopy_future_construct, parsec_datacopy_future_destruct);
