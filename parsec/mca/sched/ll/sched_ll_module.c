/**
 * Copyright (c) 2017-2018 The University of Tennessee and The University
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
#include "parsec/mca/sched/ll/sched_ll.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

#if defined(PARSEC_PROF_TRACE)
#define TAKE_TIME(ES_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((ES_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(ES_PROFILE, KEY, ID) do {} while(0)
#endif

/**
 * Module functions
 */
static int sched_ll_install(parsec_context_t* master);
static int sched_ll_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_ll_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static void sched_ll_remove(parsec_context_t* master);
static int flow_ll_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static int sched_ll_warning_issued = 0;

const parsec_sched_module_t parsec_sched_ll_module = {
    &parsec_sched_ll_component,
    {
        sched_ll_install,
        flow_ll_init,
        sched_ll_schedule,
        sched_ll_select,
        NULL,
        sched_ll_remove
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
    parsec_lifo_t lifo;
#if defined(PARSEC_PAPI_SDE)
    int           local_counter;
#endif
} parsec_lifo_with_local_counter_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_lifo_with_local_counter_t);

static inline void parsec_lifo_with_local_counter_construct( parsec_lifo_with_local_counter_t* lifo )
{
#if defined(PARSEC_PAPI_SDE)
    lifo->local_counter = 0;
#else
    (void)lifo;
#endif
}

OBJ_CLASS_INSTANCE(parsec_lifo_with_local_counter_t, parsec_lifo_t,
                   parsec_lifo_with_local_counter_construct, NULL);

#if defined(PARSEC_PAPI_SDE)
static long long int parsec_lifo_with_local_counter_length( parsec_vp_t *vp )
{
    int t;
    long long int sum = 0;
    parsec_execution_stream_t *es;
    parsec_lifo_with_local_counter_t *sched_obj;

    for(t = 0; t < vp->nb_cores; t++) {
        es = vp->execution_streams[t];
        sched_obj = (parsec_lifo_with_local_counter_t*)es->scheduler_object;
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
static int sched_ll_install( parsec_context_t *master )
{
    sched_ll_warning_issued = 0;
    (void)master;
    return 0;
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
static int flow_ll_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    /* Every flow creates its own local object */
    es->scheduler_object = OBJ_NEW(parsec_lifo_with_local_counter_t);

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

#if defined(PARSEC_PAPI_SDE)
    if( 0 == es->th_id ) {
        char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
        snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=LL", es->virtual_process->vp_id);
        papi_sde_register_fp_counter(parsec_papi_sde_handle, event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_lifo_with_local_counter_length, es->virtual_process);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=LL", PAPI_SDE_SUM);
    }
#endif

    return 0;
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
static parsec_task_t* sched_ll_select(parsec_execution_stream_t *es,
                                      int32_t* distance)
{
    parsec_task_t *task = NULL;
    parsec_lifo_with_local_counter_t *es_sched_obj = (parsec_lifo_with_local_counter_t*)es->scheduler_object;
    parsec_lifo_with_local_counter_t *sched_obj;
    int i, d = 0;

    task = (parsec_task_t*)parsec_lifo_pop(&es_sched_obj->lifo);
    if( NULL == task ) {
        for(i = (es->th_id + 1) % es->virtual_process->nb_cores;
            i != es->th_id;
            i = (i+1) % es->virtual_process->nb_cores) {
            d++;
            sched_obj = (parsec_lifo_with_local_counter_t*)es->virtual_process->execution_streams[i]->scheduler_object;
            task = (parsec_task_t*)parsec_lifo_pop(&sched_obj->lifo);
            if( NULL != task ) {
                *distance = d;
#if defined(PARSEC_PAPI_SDE)
                es_sched_obj->local_counter--;
#endif
                return task;
            }
        }
        return NULL;
    } else {
#if defined(PARSEC_PAPI_SDE)
        es_sched_obj->local_counter--;
#endif
        *distance = 0;
        return task;
    }
}

/**
 * @brief
 *  Schedule a set of ready tasks on the calling execution stream
 *
 * @details
 *  Chain the set of tasks into the local LIFO of the calling es.
 *  Distance hint is ignored at this time.
 *
 *   @param[INOUT] es          the calling execution stream
 *   @param[INOUT] new_context the ring of ready tasks to schedule
 *   @param[IN] distance       the distance hint
 *   @return PARSEC_SUCCESS in case of success, a negative number 
 *                          otherwise.
 */
static int sched_ll_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_lifo_with_local_counter_t *es_sched_obj = (parsec_lifo_with_local_counter_t*)es->scheduler_object;
    parsec_lifo_with_local_counter_t *sched_obj;
#if defined(PARSEC_PAPI_SDE)
    int len = 0;
    _LIST_ITEM_ITERATOR(new_context, &new_context->super, item, {len++; });
    es_sched_obj->local_counter+=len;
#endif
    if( distance > 0 ) {
        parsec_vp_t *vp = es->virtual_process;
        int target;
        if( (vp->nb_cores == 1) && (sched_ll_warning_issued == 0) ) {
            parsec_warning("Local LIFO scheduler is unable to implement active wait with a single thread.\n"
                           "This run is at risk of live-lock\n");
            sched_ll_warning_issued = 1;
        }
        target = (es->th_id + distance) % vp->nb_cores;
        if( target == es->th_id )
            target = (es->th_id + 1) % vp->nb_cores;
        sched_obj = (parsec_lifo_with_local_counter_t*)vp->execution_streams[target]->scheduler_object;
        parsec_lifo_chain(&sched_obj->lifo, (parsec_list_item_t*)new_context);
    } else {
        parsec_lifo_chain(&es_sched_obj->lifo, (parsec_list_item_t*)new_context);
    }
    return 0;
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
static void sched_ll_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_lifo_with_local_counter_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            if (es != NULL) {
                sched_obj = (parsec_lifo_with_local_counter_t*)es->scheduler_object;
                OBJ_RELEASE(sched_obj);
                es->scheduler_object = NULL;
            }
            PARSEC_PAPI_SDE_UNREGISTER_COUNTER("PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=LL", vp->vp_id);
        }
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("PARSEC::SCHEDULER::PENDING_TASKS::SCHED=LL");
}
