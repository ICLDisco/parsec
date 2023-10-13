/**
 * Copyright (c) 2020-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/class/info.h"
#include "parsec/sys/atomic.h"

#include <string.h>

/* To create object of class parsec_info_t that inherits parsec_list_t
 * class
 */
static void parsec_info_constructor(parsec_object_t *obj)
{
    parsec_info_t *nfo = (parsec_info_t*)obj;
    nfo->max_id = -1;
    PARSEC_OBJ_CONSTRUCT(&nfo->ioa_list, parsec_list_t);
}

static void parsec_info_destructor(parsec_object_t *obj)
{
    parsec_info_t *nfo = (parsec_info_t*)obj;
    parsec_list_item_t *item, *next;
    for(item = PARSEC_LIST_ITERATOR_FIRST(&nfo->info_list);
        item != PARSEC_LIST_ITERATOR_END(&nfo->info_list);
        item = next) {
        next = PARSEC_LIST_ITERATOR_NEXT(item);
        parsec_info_entry_t *ie = (parsec_info_entry_t*)item;
        parsec_info_unregister(nfo, ie->iid, NULL);
    }
    PARSEC_OBJ_DESTRUCT(&nfo->ioa_list);
    /* nfo->info_list is the parent and will be destructed at exit */
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_info_t, parsec_list_t, parsec_info_constructor, parsec_info_destructor);

parsec_info_id_t parsec_info_register(parsec_info_t *nfo, const char *name,
                                      parsec_info_destructor_t destructor, void *des_data,
                                      parsec_info_constructor_t constructor, void *cons_data,
                                      void *cb_data)
{
    parsec_list_item_t *item, *next_item;
    parsec_info_entry_t *ie, *nie;
    parsec_info_id_t ret = 0;

    /* Assume that the info does not exist in the list yet,
     * and preemptively allocate the structure to hold it.
     * We do this now to avoid doing a malloc while holding
     * the lock on the list. Registering the same info twice
     * is an error, so useless malloc should not happen anyway. */
    nie = malloc(sizeof(parsec_info_entry_t));
    PARSEC_OBJ_CONSTRUCT(nie, parsec_list_item_t);
    nie->info = nfo;
    nie->name = strdup(name);
    nie->destructor = destructor;
    nie->des_data = des_data;
    nie->constructor = constructor;
    nie->cons_data = cons_data;
    nie->cb_data = cb_data;

    parsec_list_lock(&nfo->info_list);
    /* By default, we assume we're going to append the
     * new element at the end of the list */
    next_item = PARSEC_LIST_ITERATOR_END(&nfo->info_list);
    /* we iterate over the list, and as long as we have not found
     * our position (next_item == END), we check that the iid
     * of each item is a contiguous index: 0, 1, 2, ...
     * If we find a hole for the first time
     *    (next_item == END && ret != item->iid)
     * then, we take this spot: we stop incrementing ret at each
     * step, and save next_item to be the current item (which will
     * thus become the successor of the item we will insert).
     * This keeps the list monotonously increasing, and the index
     * space as dense as possible.
     * We need to continue iterating over the list to check that the
     * info has not been previously registered, but we don't change
     * ret or next_item anymore. */
    for(item = PARSEC_LIST_ITERATOR_FIRST(&nfo->info_list);
        item != PARSEC_LIST_ITERATOR_END(&nfo->info_list);
        item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        ie = (parsec_info_entry_t*)item;
        if( PARSEC_LIST_ITERATOR_END(&nfo->info_list) == next_item ) {
            if( ie->iid == ret ) {
                ret++;
            } else {
                next_item = PARSEC_LIST_ITERATOR_NEXT(item);
            }
        }
        if( 0 == strcmp(ie->name, name) ) {
            parsec_list_unlock(&nfo->info_list);
            free(nie); /* This should not happen often */
            return PARSEC_INFO_ID_UNDEFINED;
        }
    }
    nie->iid = ret;
    parsec_list_nolock_add_before(&nfo->info_list, next_item, &nie->list_item);
    if(ret > nfo->max_id)
        nfo->max_id = ret;
    parsec_list_unlock(&nfo->info_list);
    return ret;
}

parsec_info_id_t parsec_info_unregister(parsec_info_t *nfo, parsec_info_id_t iid,
                                        void **pcb_data)
{
    parsec_list_item_t *item, *next, *item2;
    parsec_info_entry_t *ie, *found = NULL;
    parsec_info_object_array_t *ioa;
    int max_id = -1;

    parsec_list_lock(&nfo->info_list);
    for(item = PARSEC_LIST_ITERATOR_FIRST(&nfo->info_list);
        item != PARSEC_LIST_ITERATOR_END(&nfo->info_list);
        item = next) {
        next = PARSEC_LIST_ITERATOR_NEXT(item);
        ie = (parsec_info_entry_t*)item;
        if( ie->iid == iid ) {
            if(NULL != ie->destructor) {
                parsec_list_lock(&nfo->ioa_list);
                for(item2 = PARSEC_LIST_ITERATOR_FIRST(&nfo->ioa_list);
                    item2 != PARSEC_LIST_ITERATOR_END(&nfo->ioa_list);
                    item2 = PARSEC_LIST_ITERATOR_NEXT(item2)) {
                    ioa = (parsec_info_object_array_t*)item2;
                    if(iid < ioa->known_infos && NULL != ioa->info_objects[iid]) {
                        ie->destructor(ioa->info_objects[iid], ie->des_data);
                        ioa->info_objects[iid] = NULL;
                    }
                }
                parsec_list_unlock(&nfo->ioa_list);
            }
            parsec_list_nolock_remove(&nfo->info_list, item);
            assert(NULL == found);
            found = ie;
            if( iid != nfo->max_id ) /* We don't care to find the next max */
                break;
        } else {
            if(ie->iid > max_id)
                max_id = ie->iid;
        }
    }
    if(iid == nfo->max_id)
        nfo->max_id = max_id;
    parsec_list_unlock(&nfo->info_list);

    if( NULL == found )
        return PARSEC_INFO_ID_UNDEFINED;

    if(NULL != pcb_data)
        *pcb_data = found->cb_data;
    free(found->name);
    free(found);
    return iid;
}

parsec_info_id_t parsec_info_lookup(parsec_info_t *nfo, const char *name, void **pcb_data)
{
    parsec_list_item_t *item;
    parsec_info_entry_t *ie;
    int ret = PARSEC_INFO_ID_UNDEFINED;

    parsec_list_lock(&nfo->info_list);
    for(item = PARSEC_LIST_ITERATOR_FIRST(&nfo->info_list);
        item != PARSEC_LIST_ITERATOR_END(&nfo->info_list);
        item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        ie = (parsec_info_entry_t*)item;
        if( !strcmp(ie->name, name) ) {
            ret = ie->iid;
            if(NULL != pcb_data)
                *pcb_data = ie->cb_data;
            break;
        }
    }
    parsec_list_unlock(&nfo->info_list);
    return ret;
}

static parsec_info_entry_t *parsec_info_lookup_by_iid(parsec_info_t *nfo, int iid)
{
    parsec_list_item_t *item;
    parsec_info_entry_t *ie;
    parsec_info_entry_t *ret = NULL;

    parsec_list_lock(&nfo->info_list);
    for(item = PARSEC_LIST_ITERATOR_FIRST(&nfo->info_list);
        item != PARSEC_LIST_ITERATOR_END(&nfo->info_list);
        item = PARSEC_LIST_ITERATOR_NEXT(item)) {
        ie = (parsec_info_entry_t*)item;
        if( ie->iid == iid ) {
            ret = ie;
            break;
        }
    }
    parsec_list_unlock(&nfo->info_list);
    return ret;
}

static void parsec_info_object_array_constructor(parsec_object_t *obj)
{
    parsec_info_object_array_t *oa = (parsec_info_object_array_t*)obj;
    oa->known_infos = -1;
    oa->info_objects = NULL;
    oa->infos = NULL;
    parsec_atomic_rwlock_init(&oa->rw_lock);
    PARSEC_LIST_ITEM_SINGLETON(&oa->list_item);
}

/* The constructor cannot set the info, as it does not take additional
 * parameters. Thus, it is needed to call init after constructing the
 * info_object_array. */
void parsec_info_object_array_init(parsec_info_object_array_t *oa, parsec_info_t *nfo, void *cons_obj)
{
    oa->known_infos = nfo->max_id+1;
    parsec_list_push_front(&nfo->ioa_list, &oa->list_item);
    if(oa->known_infos == 0)
        oa->info_objects = NULL;
    else
        oa->info_objects = calloc(sizeof(void*), oa->known_infos);
    oa->infos = nfo;
    oa->cons_obj = cons_obj;
}

static void parsec_info_object_array_destructor(parsec_object_t *obj)
{
    parsec_list_item_t *next, *item;
    parsec_info_object_array_t *oa = (parsec_info_object_array_t*)obj;
    /* If we are a singleton, we have already been removed from the list */
    if(oa->list_item.list_next != &oa->list_item) {
        parsec_list_lock(&oa->infos->ioa_list);
        for (item = PARSEC_LIST_ITERATOR_FIRST(&oa->infos->ioa_list);
             item != PARSEC_LIST_ITERATOR_END(&oa->infos->ioa_list);
             item = next) {
            next = PARSEC_LIST_ITERATOR_NEXT(item);
            if (item == &oa->list_item) {
                parsec_list_nolock_remove(&oa->infos->ioa_list, item);
                break;
            }
        }
        parsec_list_unlock(&oa->infos->ioa_list);
    }
    if(NULL != oa->info_objects)
        free(oa->info_objects);
    oa->info_objects = NULL;
    oa->infos = NULL;
    oa->known_infos = -1;
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_info_object_array_t, parsec_list_item_t,
                          parsec_info_object_array_constructor,
                          parsec_info_object_array_destructor);

static void parsec_ioa_resize_and_rdlock(parsec_info_object_array_t *oa, parsec_info_id_t iid)
{
    parsec_atomic_rwlock_rdlock(&oa->rw_lock);
    if(iid >= oa->known_infos) {
        int ns;
        parsec_atomic_rwlock_rdunlock(&oa->rw_lock);
        parsec_atomic_rwlock_wrlock(&oa->rw_lock);
        if(iid >= oa->known_infos) {
            assert(oa->infos->max_id >= iid);
            ns = oa->infos->max_id + 1;
            if(oa->known_infos > 0) {
                oa->info_objects = realloc(oa->info_objects, sizeof(void *) * ns);
                memset(&oa->info_objects[oa->known_infos - 1], 0, ns - oa->known_infos);
            } else {
                oa->info_objects = calloc(sizeof(void*), ns);
            }
            oa->known_infos = ns;
        }
        parsec_atomic_rwlock_wrunlock(&oa->rw_lock);
        parsec_atomic_rwlock_rdlock(&oa->rw_lock);
    }
}

void *parsec_info_set(parsec_info_object_array_t *oa, parsec_info_id_t iid, void *info)
{
    void *ret;
    parsec_ioa_resize_and_rdlock(oa, iid);
    ret = oa->info_objects[iid];
    oa->info_objects[iid] = info;
    parsec_atomic_rwlock_rdunlock(&oa->rw_lock);
    return ret;
}

void *parsec_info_test_and_set(parsec_info_object_array_t *oa, parsec_info_id_t iid, void *info, void *old)
{
    void *ret;
    parsec_ioa_resize_and_rdlock(oa, iid);
    if( parsec_atomic_cas_ptr(&oa->info_objects[iid], old, info) ) {
        parsec_atomic_rwlock_rdunlock(&oa->rw_lock);
        return info;
    }
    ret = oa->info_objects[iid];
    parsec_atomic_rwlock_rdunlock(&oa->rw_lock);
    return ret;
}

void *parsec_info_get(parsec_info_object_array_t *oa, parsec_info_id_t iid)
{
    void *ret, *nio;
    parsec_info_entry_t *ie;

    parsec_ioa_resize_and_rdlock(oa, iid);
    ret = oa->info_objects[iid];
    parsec_atomic_rwlock_rdunlock(&oa->rw_lock);
    if(NULL != ret)
        return ret;
    ie = parsec_info_lookup_by_iid(oa->infos, iid);
    if(NULL == ie->constructor)
        return ret;
    nio = ie->constructor(oa->cons_obj, ie->cons_data);
    ret = parsec_info_test_and_set(oa, iid, nio, NULL);
    if(ret != nio && NULL != ie->destructor) {
        ie->destructor(nio, ie->des_data);
    }
    return ret;
}
