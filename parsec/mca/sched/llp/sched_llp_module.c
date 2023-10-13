/**
 * Copyright (c) 2021-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/class/lifo.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/llp/sched_llp.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

/* Whether we check for the provided ring to be sorted and handle unsorted
 * rings properly. Otherwise, we expect decending ordering and insert the
 * ring after the position where the first task fits in. */
#define CHECK_RING_SORTED 0

/**
 * Module functions
 */
static int sched_llp_install(parsec_context_t* master);
static int sched_llp_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_llp_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static void sched_llp_remove(parsec_context_t* master);
static int flow_llp_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);


/**
 * @brief Merge a sorted ring of elements into a LIFO
 *
 * @details Take a ring of elements, and push all the elements of items in
 *          front of the LIFO, honoring the priority at offset.
 *          If no other thread may push elements into the LIFO concurrently,
 *          single_writer == true may be provided to allow for some
 *          optimizations. The resulting LIFO is only guaranteed to be
 *          sorted if the ring was properly sorted.
 *
 * @param[inout] lifo the LIFO into which to push the elements
 * @param[inout] items the elements ring to push in front
 * @param distance minimum distance after which to push the first element
 *                 (if sufficient elements are in the LIFO)
 * @param offset the offset at which to find the priority field
 * @param single_writer whether this is the only thread adding elements to the LIFO
 *
 * @remark this function is thread safe
 */
static
void lifo_chain_sorted( parsec_lifo_t* lifo,
                        parsec_list_item_t* items,
                        int distance,
                        size_t offset,
                        bool single_writer);


const parsec_sched_module_t parsec_sched_llp_module = {
    &parsec_sched_llp_component,
    {
        sched_llp_install,
        flow_llp_init,
        sched_llp_schedule,
        sched_llp_select,
        NULL,
        sched_llp_remove
    }
};

/**
 * @brief define a Lifo with local counter
 *
 * @details the lifo is augmented with a local counter.
 *   The counter is in fact completely independent from
 *   the lifo itself: the lifo belongs to a thread, and
 *   the counter belongs to the same thread. Using atomic
 *   operations, the lifo may be modified by other threads,
 *   but the counter may not. Each thread counts on its
 *   counter only (not using atomic operations), but they
 *   count all lifo modifications (including modifications
 *   of another lifo
 *
 *   So, the sum counters should represent a good approximation
 *   of the number of tasks in the set of lifos, but each counter
 *   represents how many insert/remove a given thread did, not
 *   how many items are in the corresponding lifo
 */
typedef struct {
    parsec_object_t    super;
    parsec_lifo_t   lifo;
#if defined(PARSEC_PAPI_SDE)
    int           local_counter;
#endif
} parsec_lifo_with_prio_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_lifo_with_prio_t);

static inline void parsec_list_with_prio_construct( parsec_lifo_with_prio_t* list )
{
    PARSEC_OBJ_CONSTRUCT(&list->lifo, parsec_lifo_t);
#if defined(PARSEC_PAPI_SDE)
    list->local_counter = 0;
#endif
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_lifo_with_prio_t, parsec_object_t,
                   parsec_list_with_prio_construct, NULL);

#if defined(PARSEC_PAPI_SDE)
static long long int parsec_lifo_with_local_counter_length( parsec_vp_t *vp )
{
    int t;
    long long int sum = 0;
    parsec_execution_stream_t *es;
    parsec_lifo_with_prio_t *sched_obj;

    for(t = 0; t < vp->nb_cores; t++) {
        es = vp->execution_streams[t];
        sched_obj = (parsec_lifo_with_prio_t*)es->scheduler_object;
        sum += sched_obj->local_counter;
    }
    return sum;
}
#endif

/**
 * @brief
 *   Installs the scheduler on a parsec context
 *
 * @details
 *   This function has nothing to do, as all operations are done in
 *   init.
 *
 *  @param[INOUT] master the parsec_context_t on which this scheduler should be installed
 *  @return PARSEC_SUCCESS iff this scheduler has been installed
 */
static int sched_llp_install( parsec_context_t *master )
{
    (void)master;
    return PARSEC_SUCCESS;
}

/**
 * @brief
 *    Initialize the scheduler on the calling execution stream
 *
 * @details
 *    Creates a LIFO per execution stream, store it into es->scheduling_object, and
 *    synchronize with the other execution streams using the barrier
 *
 *  @param[INOUT] es      the calling execution stream
 *  @param[INOUT] barrier the barrier used to synchronize all the es
 *  @return PARSEC_SUCCESS in case of success, a negative number otherwise
 */
static int flow_llp_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    /* Every flow creates its own local object */
    parsec_lifo_with_prio_t *lifo = PARSEC_OBJ_NEW(parsec_lifo_with_prio_t);
    es->scheduler_object = lifo;

    /* All local allocations are now completed. Synchronize with the other threads
	 * before they start stealing from each other. */
    parsec_barrier_wait(barrier);

#if defined(PARSEC_PAPI_SDE)
    if( 0 == es->th_id ) {
        char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
        snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=LLP", es->virtual_process->vp_id);
        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_lifo_with_local_counter_length, es->virtual_process);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LLP", PAPI_SDE_SUM);
    }
#endif
    return PARSEC_SUCCESS;
}

/**
 * @brief
 *   Selects a task to run
 *
 * @details
 *   Take the head of the calling execution stream LIFO as the selected task;
 *   if that LIFO is empty, iterate over all other execution streams LIFOs,
 *   using the eu_id as an index (modulo the number of execution streams in this
 *   virtual process).
 *
 *   @param[INOUT] es     the calling execution stream
 *   @param[OUT] distance the distance of the selected task. We return here
 *                        how many LIFOs that are empty were tried
 *   @return the selected task
 */
static parsec_task_t* sched_llp_select(parsec_execution_stream_t *es,
                                       int32_t* distance)
{
    parsec_task_t *task = NULL;
    parsec_lifo_with_prio_t *es_sched_obj = (parsec_lifo_with_prio_t*)es->scheduler_object;
    int i;

    task = (parsec_task_t*)parsec_lifo_pop(&es_sched_obj->lifo);

    if (NULL == task) {
        for(i = (es->th_id + 1) % es->virtual_process->nb_cores;
            i != es->th_id;
            i = (i+1) % es->virtual_process->nb_cores) {
            parsec_lifo_with_prio_t *sched_obj;
            sched_obj = (parsec_lifo_with_prio_t*)es->virtual_process->execution_streams[i]->scheduler_object;
            task = (parsec_task_t*)parsec_lifo_pop(&sched_obj->lifo);
            if( NULL != task ) {
                *distance = (i - es->th_id + es->virtual_process->nb_cores) % es->virtual_process->nb_cores;
#if defined(PARSEC_PAPI_SDE)
                es_sched_obj->local_counter--;
#endif
                break;
            }
        }
    } else {
#if defined(PARSEC_PAPI_SDE)
        es_sched_obj->local_counter--;
#endif
        *distance = 0;
    }

    return task;
}

/**
 * @brief
 *  Schedule a set of ready tasks on the calling execution stream
 *
 * @details
 *  Chain the set of tasks into the local LIFO of the calling es.
 *
 *   @param[INOUT] es          the calling execution stream
 *   @param[INOUT] new_context the ring of ready tasks to schedule
 *   @param[IN] distance       the distance hint
 *   @return PARSEC_SUCCESS in case of success, a negative number
 *                          otherwise.
 */
static int sched_llp_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_lifo_with_prio_t *es_sched_obj = (parsec_lifo_with_prio_t*)es->scheduler_object;
#if defined(PARSEC_PAPI_SDE)
    int len = 0;
    _LIST_ITEM_ITERATOR(new_context, &new_context->super, item, {len++; });
    es_sched_obj->local_counter+=len;
#endif

    lifo_chain_sorted(&es_sched_obj->lifo, &new_context->super, distance,
                      parsec_execution_context_priority_comparator,
                      /* the comm thread might write into thread 0' s queue */
                      (es->th_id != 0));

    return PARSEC_SUCCESS;
}

/**
 * @brief
 *  Removes the scheduler from the parsec_context_t
 *
 * @details
 *  Release the LIFO for each execution stream
 *
 *  @param[INOUT] master the parsec_context_t from which the scheduler should
 *                       be removed
 */
static void sched_llp_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_lifo_with_prio_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            if (es != NULL) {
                sched_obj = (parsec_lifo_with_prio_t*)es->scheduler_object;
                PARSEC_OBJ_RELEASE(sched_obj);
                es->scheduler_object = NULL;
            }
            PARSEC_PAPI_SDE_UNREGISTER_COUNTER("PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=LLP", vp->vp_id);
        }
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("PARSEC::SCHEDULER::PENDING_TASKS::SCHED=LLP");
}


static
parsec_list_item_t* lifo_merge_ring(parsec_list_item_t *next_in_lifo,
                                    parsec_list_item_t *ring,
                                    size_t offset,
                                    int distance,
                                    bool sorted)
{

    parsec_list_item_t *next;
    parsec_list_item_t *list;

    list = next = next_in_lifo;

    parsec_list_item_t* prev = NULL;
    int d = 0;
    do {
        /* Find the right place for the first element in the ring
         * NOTE: we put new elements *before* old elements with same priority */
        while (next != NULL && !(d < distance || A_HIGHER_PRIORITY_THAN_B(next, ring, offset)))
        {
            prev = next;
            next = (parsec_list_item_t*)next->list_next;
            ++d;
        }

        /* check if we can put all elements in place at once */
        if (sorted && (next == NULL || !A_HIGHER_PRIORITY_THAN_B(next, ring->list_prev, offset))) {
            if (prev != NULL) {
                prev->list_next = ring;
            } else {
                list = ring;
            }
            ring->list_prev->list_next = next;
            ring->list_prev = NULL;
            break;
        } else {
            /* insert a single element */
            parsec_list_item_t* item = ring;
            ring = parsec_list_item_ring_chop(ring);
            if (NULL != prev) {
                item->list_next = prev->list_next;
                prev->list_next = item;
            } else {
                item->list_next = next;
                next = list = item;
            }
        }

    } while (NULL != ring);

    return list;
}

#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)

void lifo_chain_sorted( parsec_lifo_t* lifo,
                        parsec_list_item_t* ring,
                        int distance,
                        size_t offset,
                        bool single_writer)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, ring);

    /* first: mark the lifo as empty */
    parsec_list_item_t *next;
    parsec_list_item_t *list;

    bool sorted = true;

#if CHECK_RING_SORTED
    int last_prio = COMPARISON_VAL(ring, offset);
    next = ring;
    do {
        next = PARSEC_LIST_ITEM_NEXT(next);
        if (last_prio < COMPARISON_VAL(next, offset)) {
            sorted = false;
            break;
        }
    } while (next != ring->list_prev);
#endif // CHECK_RING_SORTED

repeat:

    /* first try to push the whole ring or detach existing elements */
    do {
        parsec_counted_pointer_t old_head;
        old_head.data.guard.counter = lifo->lifo_head.data.guard.counter;
        next = old_head.data.item = lifo->lifo_head.data.item;
        if (sorted && distance == 0 &&
            (next == NULL ||
             !A_HIGHER_PRIORITY_THAN_B(next, ring->list_prev, offset))) {
            /* try to push to front of lifo */
            ring->list_prev->list_next = next;
            parsec_atomic_wmb();
            if (parsec_update_counted_pointer(&lifo->lifo_head, old_head, ring)) {
                return;
            }
            /* restore the ring */
            ring->list_prev->list_next = ring;
        } else if (parsec_update_counted_pointer(&lifo->lifo_head, old_head, NULL)) {
            /* detached all elements from the lifo */
            break;
        }
        /* DO some kind of pause to release the bus */
    } while (1);

    list = next;

    /* merge the ring into the lifo */

    list = lifo_merge_ring(list, ring, offset, distance, sorted);

    /* push the merged list back into place */

    if (single_writer) {
        /* the caller guaranteed that no other thread has put elements into
         * the LIFO, so just reset the head pointer */
        lifo->lifo_head.data.guard.counter++;
        parsec_atomic_wmb();
        lifo->lifo_head.data.item = list;
    } else {

        parsec_list_item_t *new_ring = NULL;

        /* pop out any items that might have been added in between */
        do {
            parsec_counted_pointer_t old_head;
            old_head.data.guard.counter = lifo->lifo_head.data.guard.counter;
            parsec_atomic_rmb ();
            next = old_head.data.item = lifo->lifo_head.data.item;
            if (parsec_update_counted_pointer(&lifo->lifo_head, old_head, list)) {
                if (next == NULL) {
                    /* successfully reattached the chain and no new elements
                     * were added in between */
                    break;
                }
                /* form a ring of popped items */
                parsec_list_item_t *last = next;
				parsec_list_item_t *prev_last = last;
                while (last != NULL) {
					prev_last = last;
                    last = PARSEC_LIST_ITEM_NEXT(last);
                }
                new_ring = parsec_list_item_ring(next, prev_last);
                break;
            }
        } while (1);

        if (NULL != new_ring) {
            ring = new_ring;
            goto repeat;
        }
    }
}

#elif defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR)


void lifo_chain_sorted( parsec_lifo_t* lifo,
                               parsec_list_item_t* ring,
                               int distance,
                               size_t offset,
                               bool single_writer)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif

    /* first: mark the lifo as empty */
    parsec_list_item_t *next;
    parsec_list_item_t *list;

    bool sorted = true;
#if CHECK_RING_SORTED
    int last_prio = COMPARISON_VAL(ring, offset);
    next = ring;
    do {
        PARSEC_ITEM_ATTACH(lifo, next);
        next = PARSEC_LIST_ITEM_NEXT(next);
        if (last_prio < COMPARISON_VAL(next, offset)) {
            sorted = false;
            break;
        }
    } while (next != ring->list_prev);
#endif // CHECK_RING_SORTED

repeat:

    /* first try to detach existing elements */

    do {
        next = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&lifo->lifo_head.data.item);
        if (sorted && distance == 0 &&
            (next == NULL ||
             !A_HIGHER_PRIORITY_THAN_B(next, ring->list_prev, offset))) {
            /* try to push to front of lifo */
            ring->list_prev->list_next = next;
            parsec_atomic_wmb();
            if (parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)ring)) {
                return;
            }
            /* restore the ring */
            ring->list_prev->list_next = ring;
        } else if (parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, NULL)) {
            /* detached all elements from the lifo */
            break;
        }
        /* DO some kind of pause to release the bus */
    } while (1);

    list = next;

    /* merge the ring into the lifo */

    list = lifo_merge_ring(list, ring, offset, distance, sorted);

    /* push the merged list back into place */

    if (single_writer) {
        /* the caller guaranteed that no other thread has put elements into
         * the LIFO, so just reset the head pointer */
        parsec_atomic_wmb();
        lifo->lifo_head.data.item = list;
    } else {

        parsec_list_item_t *new_ring = NULL;

        parsec_atomic_wmb();
        /* pop out any items that might have been added in between */
        do {
            next = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&lifo->lifo_head.data.item);
            if (parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)list)) {
                if (next == NULL) {
                    /* successfully reattached the chain */
                    break;
                }

                /* form a ring of popped items to repeat the process */
                parsec_list_item_t *last = next;
                while (last != NULL) {
                    last = PARSEC_LIST_ITEM_NEXT(last);
                }
                new_ring = parsec_list_item_ring(next, last);
                break;
            }
        } while (1);

        if (NULL != new_ring) {
            ring = new_ring;
            goto repeat;
        }
    }
}


#elif defined(PARSEC_USE_64BIT_LOCKFREE_LIST)

#error Lock-free 64bit is not supported!

#else /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128) || defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR) || defined(PARSEC_USE_64BIT_LOCKFREE_LIST) */


void lifo_chain_sorted( parsec_lifo_t* lifo,
                        parsec_list_item_t* ring,
                        int distance,
                        size_t offset,
                        bool single_writer)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif

    /* first: mark the lifo as empty */
    parsec_list_item_t *next;
    parsec_list_item_t *list;

    bool sorted = true;
#if CHECK_RING_SORTED
    int last_prio = COMPARISON_VAL(ring, offset);
    next = ring;
    do {
        PARSEC_ITEM_ATTACH(lifo, next);
        next = PARSEC_LIST_ITEM_NEXT(next);
        if (last_prio < COMPARISON_VAL(next, offset)) {
            sorted = false;
            break;
        }
    } while (next != ring->list_prev);
#endif // CHECK_RING_SORTED

    /* first try to detach existing elements */
    parsec_atomic_lock(&lifo->lifo_head.data.guard.lock);

repeat:

    next = lifo->lifo_head.data.item;
    if (sorted && distance == 0 &&
        (next == NULL ||
          !A_HIGHER_PRIORITY_THAN_B(next, ring->list_prev, offset))) {
        /* push to front of lifo and be done */
        ring->list_prev->list_next = next;
        lifo->lifo_head.data.item = ring;
        parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
        return;
    }
    /* detach all elements from the lifo */
    lifo->lifo_head.data.item = NULL;
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);

    list = next;

    /* merge the ring into the lifo */

    list = lifo_merge_ring(list, ring, offset, distance, sorted);

    /* push the merged list back into place */

    if (single_writer) {
        /* the caller guaranteed that no other thread has put elements into
         * the LIFO, so just reset the head pointer */
        parsec_atomic_wmb();
        lifo->lifo_head.data.item = list;
    } else {

        parsec_list_item_t *new_ring = NULL;

        parsec_atomic_lock(&lifo->lifo_head.data.guard.lock);
        next = lifo->lifo_head.data.item;
        lifo->lifo_head.data.item = list;

        /* check if there are new items */
        if (next != NULL) {
            parsec_list_item_t *last = next;
            while (last != NULL) {
                last = PARSEC_LIST_ITEM_NEXT(last);
            }
            new_ring = parsec_list_item_ring(next, last);
        }

        if (NULL != new_ring) {
            ring = new_ring;
            /* keep the lock */
            goto repeat;
        }

        parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
    }
}


#endif
