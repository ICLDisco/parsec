/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/class/parsec_future.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"

static void parsec_base_future_construct(parsec_base_future_t* future);
static void parsec_countable_future_construct(parsec_base_future_t* future);

static int   parsec_base_future_is_ready(parsec_base_future_t* future);
static void  parsec_base_future_set(parsec_base_future_t* future, void* data);
static void* parsec_base_future_get(parsec_base_future_t* future);
static void  parsec_base_future_init(parsec_base_future_t* future, parsec_future_cb_fulfill cb);

static void  parsec_countable_future_set(parsec_base_future_t* future, void* data);
static void  parsec_countable_future_init(parsec_base_future_t* future, parsec_future_cb_fulfill cb, int count, ...);

/*Function implementation */

static int parsec_base_future_is_ready(parsec_base_future_t* future)
{
    return (future->status & PARSEC_DATA_FUTURE_STATUS_COMPLETED);
}

static void parsec_base_future_set(parsec_base_future_t* future, void* data)
{
    if(parsec_atomic_cas_ptr(&(future->tracked_data), NULL, data)) {
        parsec_atomic_wmb();
        /* increment flag to indicate data is ready */
        future->status |= PARSEC_DATA_FUTURE_STATUS_COMPLETED;
        if(future->cb_fulfill != NULL){
            future->cb_fulfill(future);
        }
#if defined(PARSEC_DEBUG_NOISIER)
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
            "Set base future %p to ready state %d, with data %p", future, future->status, data);
#endif
    }else{
        parsec_warning("Trying to set a base future that is already in a ready state");
    }
}

static void* parsec_base_future_get(parsec_base_future_t* future)
{
    /* 
     * blocking get
     * TODO: Don't do busy wait
     * */
    while(1){
        if(parsec_base_future_is_ready(future)){
            parsec_atomic_rmb();
            return future->tracked_data;
        }
    }
    return NULL;
}

static void parsec_base_future_init(parsec_base_future_t* future, parsec_future_cb_fulfill cb)
{
    future->status = PARSEC_DATA_FUTURE_STATUS_INIT;
    future->cb_fulfill = cb;
}

/*
 * Countable futures
 */

static void parsec_countable_future_set(parsec_base_future_t* future, void* data)
{
    (void) data; /* not used since can't guarantee order between the sets */
    parsec_countable_future_t* c_fut = (parsec_countable_future_t*)future; 
    if(0 == parsec_atomic_fetch_dec_int32(&(c_fut->count))-1){
        c_fut->super.status |= PARSEC_DATA_FUTURE_STATUS_COMPLETED;
        if( NULL != future->cb_fulfill ){
            future->cb_fulfill(future);
        }
#if defined(PARSEC_DEBUG_NOISIER)
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                "Set countable future %p to ready state", future);
#endif
    } else {
        if(parsec_base_future_is_ready(future)) {
            parsec_warning("Trying to set a countable future that is already in a ready state");
        }
    }
}

static void parsec_countable_future_init(parsec_base_future_t* future, parsec_future_cb_fulfill cb, int count, ...)
{
    parsec_countable_future_t* c_fut = (parsec_countable_future_t*)future;
    c_fut->super.status = PARSEC_DATA_FUTURE_STATUS_INIT;
    c_fut->super.cb_fulfill = cb;
    c_fut->count = count;
}

/*Global function structs for base and countable future, assigned to pointer in construct function */
static parsec_future_fn_t parsec_base_future_functions = {
    .is_ready         = parsec_base_future_is_ready,
    .set              = parsec_base_future_set,
    .get              = parsec_base_future_get,
    .future_init      = (parsec_future_init_t)parsec_base_future_init
};

static parsec_future_fn_t parsec_countable_future_functions = {
    .is_ready         = parsec_base_future_is_ready,
    .set              = parsec_countable_future_set,
    .get              = parsec_base_future_get,
    .future_init      = (parsec_future_init_t)parsec_countable_future_init
};

static void parsec_base_future_construct(parsec_base_future_t* future)
{
    //initialize the elements in the struct
    future->future_class = &parsec_base_future_functions;
    future->status = 0;
    future->cb_fulfill = NULL;
    future->tracked_data = NULL;
    parsec_atomic_lock_t temp = PARSEC_ATOMIC_UNLOCKED;
    future->future_lock = temp;
}

static void parsec_countable_future_construct(parsec_base_future_t* future)
{
    parsec_atomic_lock_t temp = PARSEC_ATOMIC_UNLOCKED;
    future->future_lock = temp;
    parsec_countable_future_t* c_fut = (parsec_countable_future_t*)future;
    c_fut->super.future_class = &parsec_countable_future_functions;
    c_fut->super.status = 0;
    c_fut->super.cb_fulfill = NULL;
    c_fut->super.tracked_data = NULL;
    c_fut->count = 1;
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_base_future_t, parsec_list_item_t,/*parsec_object_t,*/
                   parsec_base_future_construct, NULL);

PARSEC_OBJ_CLASS_INSTANCE(parsec_countable_future_t, parsec_base_future_t,
                   parsec_countable_future_construct, NULL);

