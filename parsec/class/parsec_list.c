/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <parsec_config.h>
#include "parsec/class/list.h"

/**
 * The list_item object instance.
 */
static inline void
parsec_list_item_construct( parsec_list_item_t* item )
{
    item->list_prev = item;
    item->list_next = item;
    item->aba_key = 0;
#if defined(PARSEC_DEBUG_PARANOID)
    item->refcount = 0;
    item->belong_to = (void*)0xdeadbeef;
#endif
}

OBJ_CLASS_INSTANCE(parsec_list_item_t, parsec_object_t,
                   parsec_list_item_construct, NULL);

/**
 * And now the list instance.
 */

static inline void
parsec_list_construct( parsec_list_t* list )
{
    parsec_list_item_construct(&list->ghost_element);
    PARSEC_ITEM_ATTACH(list, &list->ghost_element);
    list->ghost_element.list_next = &list->ghost_element;
    list->ghost_element.list_prev = &list->ghost_element;
    parsec_atomic_unlock(&list->atomic_lock);
}

static inline void
parsec_list_destruct( parsec_list_t* list )
{
    assert(parsec_list_is_empty(list)); (void)list;
}

OBJ_CLASS_INSTANCE(parsec_list_t, parsec_object_t,
                   parsec_list_construct, parsec_list_destruct);


