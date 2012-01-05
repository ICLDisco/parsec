#ifndef _priority_sorted_queue_h
#define _priority_sorted_queue_h

#include "dague_config.h"
#include "lifo.h"
#include "dague.h"

/**
 * Elements in these queues are only dague_context_t types.
 *  They are inserted sorted by decreasing ->priority
 *  And popped from the beginning of the queue (hence smallest priority first)
 */

typedef struct dague_priority_sorted_list {
    volatile uint32_t  queue_lock;
    dague_list_item_t  queue_ghost;
    uint32_t nb_elt;
} dague_priority_sorted_list_t;

static inline void dague_priority_sorted_list_construct( dague_priority_sorted_list_t *l )
{
    l->queue_lock = 0;
    l->nb_elt = 0;
    dague_list_item_construct( &l->queue_ghost );
}

static inline void dague_priority_sorted_list_destruct( dague_priority_sorted_list_t *l )
{
    (void)l;
}

static inline dague_list_item_t *dague_sort_list_by_priority_nolock( dague_list_item_t *m )
{
    dague_list_item_t *pivot = m, *e, *ne;
    dague_list_item_t *smaller = NULL;
    dague_list_item_t *bigger = NULL;

    if( NULL == pivot )
        return NULL;

    e = (dague_list_item_t *)pivot->list_next;
    while( NULL != e ) {
        ne = (dague_list_item_t *)e->list_next;
        if(((dague_execution_context_t*)e)->priority <= ((dague_execution_context_t*)pivot)->priority) {
            e->list_next = smaller;
            smaller = e;
        } else {
            e->list_next = bigger;
            bigger = e;
        }
        e = ne;
    }
    smaller = dague_sort_list_by_priority_nolock(smaller);
    bigger  = dague_sort_list_by_priority_nolock(bigger);
    pivot->list_next = bigger;
    if( NULL == smaller )
        return pivot;
    for(e = smaller; NULL != e->list_next; e = (dague_list_item_t *)e->list_next) /* nothing */ ;
    e->list_next = pivot;
    return smaller;
}

/**
 * m is a circular list of elements
 */
static inline void dague_priority_sorted_list_merge( dague_priority_sorted_list_t *l,
                                                     dague_list_item_t *m )
{
    dague_list_item_t *e, *ne, *i;
    uint32_t nb = 0;

    m->list_prev->list_next = NULL;
    m = dague_sort_list_by_priority_nolock( m );

    dague_atomic_lock( &l->queue_lock );
    i = (dague_list_item_t *)l->queue_ghost.list_next;
    e = m;

    do {
        ne = (dague_list_item_t *)e->list_next;

        while( i != &l->queue_ghost && ((dague_execution_context_t*)i)->priority < ((dague_execution_context_t*)e)->priority ) {
            i = (dague_list_item_t *)i->list_next;
        }

        e->list_next = i;
        e->list_prev = (dague_list_item_t *)i->list_prev;
        i->list_prev->list_next = e;
        i->list_prev = e;
        nb++;

        e = ne;
    } while( e != NULL );
    l->nb_elt += nb;
    dague_atomic_unlock( &l->queue_lock );
}

static inline dague_list_item_t *dague_priority_sorted_list_pop_front( dague_priority_sorted_list_t *l )
{
    volatile dague_list_item_t *e;

    dague_atomic_lock( &l->queue_lock );

    e = l->queue_ghost.list_next;
    if( e == &l->queue_ghost )
        e = NULL;
    else {
        e->list_next->list_prev = &l->queue_ghost;
        l->queue_ghost.list_next = e->list_next;
        e->list_next = e;
        e->list_prev = e;
    }
    l->nb_elt--;
    dague_atomic_unlock( &l->queue_lock );
    return (dague_list_item_t *)e;
}

static inline int dague_priority_sorted_list_empty( dague_priority_sorted_list_t *l )
{
    int r;
    dague_atomic_lock( &l->queue_lock );
    r = (l->queue_ghost.list_next == &l->queue_ghost);
    dague_atomic_unlock( &l->queue_lock );

    return r;
}

#endif
