/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_obj_freelist.h"

static void *parsec_obj_freelist_alloc(parsec_obj_allocator_t *allocator, parsec_class_t *cls)
{
    parsec_obj_freelist_t *freelist = (parsec_obj_freelist_t *)allocator;
    parsec_list_item_t *item;

    assert(NULL != freelist->lifo);  /* the free-list was destructed */
    assert(cls->cls_sizeof <= freelist->elt_size); (void)cls;
    item = parsec_lifo_pop(freelist->lifo);
    if( NULL == item ) {
        item = parsec_lifo_item_alloc(freelist->lifo, freelist->elt_size);
        parsec_atomic_fetch_inc_int32(&freelist->nb_elt);
    }
    return item;
}

static void parsec_obj_freelist_release(parsec_obj_allocator_t *allocator, parsec_object_t *object)
{
    parsec_obj_freelist_t *freelist = (parsec_obj_freelist_t *)allocator;
    /* the free-list must outlive the objects allocated from it */
    assert(NULL != freelist->lifo);
    /* only objects created from this free-list can be returned to it */
    assert(object->obj_class->cls_sizeof <= freelist->elt_size);
    parsec_obj_destruct(object);
    parsec_lifo_push(freelist->lifo, parsec_list_item_from_storage(object));
}

int parsec_obj_freelist_construct(parsec_obj_freelist_t *freelist, size_t elt_size)
{
    freelist->super.obj_alloc   = parsec_obj_freelist_alloc;
    freelist->super.obj_release = parsec_obj_freelist_release;
    freelist->lifo     = PARSEC_OBJ_NEW(parsec_lifo_t);
    freelist->elt_size = elt_size < sizeof(parsec_list_item_t) ? sizeof(parsec_list_item_t) : elt_size;
    freelist->nb_elt   = 0;
    return (NULL == freelist->lifo) ? PARSEC_ERR_OUT_OF_RESOURCE : PARSEC_SUCCESS;
}

int32_t parsec_obj_freelist_destruct(parsec_obj_freelist_t *freelist)
{
    parsec_list_item_t *item;
    int32_t nb_elt = freelist->nb_elt;

    while( NULL != (item = parsec_lifo_pop(freelist->lifo)) ) {
        parsec_lifo_item_storage_free(item);  /* raw storage, no longer an object */
        nb_elt--;
    }
    PARSEC_OBJ_RELEASE(freelist->lifo);
    freelist->nb_elt = 0;
    return nb_elt;
}
