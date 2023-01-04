/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <assert.h>
#include "parsec/parsec_mpi_funnelled.h"

parsec_comm_engine_t parsec_ce;
static parsec_list_t *parsec_ce_callback_list = NULL;
static int32_t parsec_ce_callback_nextid = 1;

typedef struct {
    parsec_list_item_t super;
    int callback_id;
    parsec_comm_engine_updown_fn_t *up;
    void *up_data;
    parsec_comm_engine_updown_fn_t *down;
    void *down_data;
} parsec_comm_engine_callback_item_t;

/* This function will be called by the runtime */
parsec_comm_engine_t *
parsec_comm_engine_init(parsec_context_t *parsec_context)
{
    /* call the selected module init */
    parsec_comm_engine_t *ce = mpi_funnelled_init(parsec_context);

    assert(ce->capabilites.sided > 0 && ce->capabilites.sided < 3);

    if(NULL == parsec_ce_callback_list)
        parsec_ce_callback_list = PARSEC_OBJ_NEW(parsec_list_t);

    for(parsec_list_item_t *item = PARSEC_LIST_ITERATOR_FIRST(parsec_ce_callback_list);
        item != PARSEC_LIST_ITERATOR_END(parsec_ce_callback_list);
        item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        parsec_comm_engine_callback_item_t *ce_item = (parsec_comm_engine_callback_item_t*)item;
        if(NULL != ce_item->up) {
            ce_item->up(ce, ce_item->up_data);
        }
    }

    return ce;
}

int
parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine)
{
    if(NULL != parsec_ce_callback_list) {
        for(parsec_list_item_t *item = PARSEC_LIST_ITERATOR_FIRST(parsec_ce_callback_list);
            item != PARSEC_LIST_ITERATOR_END(parsec_ce_callback_list);
            item = PARSEC_LIST_ITERATOR_NEXT(item)) {
            parsec_comm_engine_callback_item_t *ce_item = (parsec_comm_engine_callback_item_t*)item;
            if(NULL != ce_item->down) {
                ce_item->down(comm_engine, ce_item->down_data);
            }
        }
    }

    /* call the selected module fini */
    return mpi_funnelled_fini(comm_engine);
}

int32_t parsec_comm_engine_register_callback(parsec_comm_engine_updown_fn_t *up,
                                             void *up_data,
                                             parsec_comm_engine_updown_fn_t *down,
                                             void *down_data)
{
    parsec_comm_engine_callback_item_t *new_item = malloc(sizeof(parsec_comm_engine_callback_item_t));
    PARSEC_OBJ_CONSTRUCT(new_item, parsec_list_item_t);
    PARSEC_LIST_ITEM_SINGLETON(new_item);
    new_item->callback_id = parsec_atomic_fetch_inc_int32(&parsec_ce_callback_nextid);
    new_item->up = up;
    new_item->up_data = up_data;
    new_item->down = down;
    new_item->down_data = down_data;

    if(NULL == parsec_ce_callback_list) {
        parsec_ce_callback_list = PARSEC_OBJ_NEW(parsec_list_t);
    }

    parsec_list_append(parsec_ce_callback_list, &new_item->super);

    if(NULL != parsec_ce.tag_register && NULL != up) {
        up(&parsec_ce, up_data);
    }

    return new_item->callback_id;
}

int parsec_comm_engine_unregister_callback(int32_t callback_id)
{
    parsec_list_item_t *item = NULL;
    parsec_comm_engine_callback_item_t *ce_item = NULL;

    if(NULL == parsec_ce_callback_list)
        return PARSEC_ERR_NOT_FOUND;
    parsec_list_lock(parsec_ce_callback_list);
    for(item = PARSEC_LIST_ITERATOR_FIRST(parsec_ce_callback_list);
        item != PARSEC_LIST_ITERATOR_END(parsec_ce_callback_list);
        item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        ce_item = (parsec_comm_engine_callback_item_t*)item;
        if(callback_id == ce_item->callback_id) {
            parsec_list_nolock_remove(parsec_ce_callback_list, item);
            break;
        }
    }
    parsec_list_unlock(parsec_ce_callback_list);
    if(item == PARSEC_LIST_ITERATOR_END(parsec_ce_callback_list))
        return PARSEC_ERR_NOT_FOUND;
    if(parsec_list_is_empty(parsec_ce_callback_list)) {
        PARSEC_OBJ_RELEASE(parsec_ce_callback_list);
        parsec_ce_callback_list = NULL;
    }
    if(NULL != ce_item->down && NULL != parsec_ce.tag_unregister) {
        ce_item->down(&parsec_ce, ce_item->down_data);
    }
    return PARSEC_SUCCESS;
}
