/**
 * Copyright (c) 2022      The University of Tennessee and The University
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
#include "parsec/mca/sched/pcb/sched_pcb.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

/**
 * Module functions
 */
static int sched_pcb_install(parsec_context_t* master);
static int sched_pcb_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_pcb_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static void sched_pcb_remove(parsec_context_t* master);
static int sched_pcb_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static int sched_pcb_warning_issued = 0;

const parsec_sched_module_t parsec_sched_pcb_module = {
    &parsec_sched_pcb_component,
    {
        sched_pcb_install,
        sched_pcb_init,
        sched_pcb_schedule,
        sched_pcb_select,
        NULL,
        sched_pcb_remove
    }
};

/**
 * @brief scheduler structure: each subset of threads holds a single (locked) list sorted by priority
 *        This structure is locked for any access. Any threads can push in this structure,
 *        only threads that belong to the subset can pop from it.
 */
typedef struct sched_pcb_scheduler_object_s {
    parsec_list_t  group_tasks;    /**< List of tasks bound to the group */
    parsec_list_t *shared_tasks;   /**< List of tasks shared between all processes */
    int            group_id;       /**< Group identifier for this group. NB: group identifiers start at 1!! -- 0 identifies the group of shared tasks */
    int            nb_groups;      /**< Number of groups found for this process */
    struct sched_pcb_scheduler_object_s *groups[1]; /**< nb_groups long array of scheduler objects to quickly find any other list from any scheduler object */
} sched_pcb_scheduler_object_t;
#define SCHED_PCB_SO(es) ((sched_pcb_scheduler_object_t*) (es)->scheduler_object)

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
static int sched_pcb_install( parsec_context_t *master )
{
    sched_pcb_warning_issued = 0;
    (void)master;
    return PARSEC_SUCCESS;
}

#if defined(PARSEC_PAPI_SDE)
/**
 * @brief counts the number of items in the group tasks
 * 
 * @details only used if PAPI_SDE is enabled, this will be called by PAPI
 *   in case the corresponding counter is considered.
 * 
 */
static long long int sched_pcb_group_tasks_length( void *_es )
{
    parsec_execution_stream_t *es = (parsec_execution_stream_t *)_es;
    sched_pcb_scheduler_object_t *so = SCHED_PCB_SO(es);
    long long int len = 0;
    PARSEC_LIST_ITERATOR(&so->group_tasks, item, {len++;});
    return len;
}

/**
 * @brief counts the number of items in the shared tasks
 * 
 * @details only used if PAPI_SDE is enabled, this will be called by PAPI
 *   in case the corresponding counter is considered.
 * 
 */
static long long int sched_pcb_shared_tasks_length( void *_es )
{
    parsec_execution_stream_t *es = (parsec_execution_stream_t *)_es;
    sched_pcb_scheduler_object_t *so = SCHED_PCB_SO(es);
    long long int len = 0;
    PARSEC_LIST_ITERATOR(so->shared_tasks, item, {len++;});
    return len;
}
#endif

// Make sure the group requested by the priority fits in the existing set of groups
static int sched_pcb_group(int p, sched_pcb_scheduler_object_t *so) {
    int group = (p & sched_pcb_group_mask);
    if(-1 == p) 
        return 0;
    if(0 == group)
        return 0;
    group = group >> sched_pcb_group_shift;
    group = ((group - 1) % so->nb_groups) + 1;
    return group;    
}

/**
 * @brief
 *    Initialize the scheduler on the calling execution stream
 *
 * @details
 *    Creates a list per group if this es is the master of the group, store it into es->scheduling_object, and
 *    synchronize with the other execution streams using the barrier
 *
 *  @param[INOUT] es      the calling execution stream
 *  @param[INOUT] barrier the barrier used to synchronize all the es
 *  @return PARSEC_SUCCESS in case of success, a negative number otherwise
 */
static int sched_pcb_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    sched_pcb_scheduler_object_t *so;
    // If there is no HWLOC, we always have 1 group per thread
    int master_id = es->th_id;
    int nb_groups = es->virtual_process->nb_cores;

#if defined(PARSEC_HAVE_HWLOC)
    master_id = parsec_hwloc_master_id(sched_pcb_sharing_level, es->th_id);

    nb_groups = 0;
    for(int t = 0; t < es->virtual_process->nb_cores; t++) {
        if(t == parsec_hwloc_master_id(sched_pcb_sharing_level, t))
            nb_groups++;
    }
#endif

    if(master_id == es->th_id) {
        so = (sched_pcb_scheduler_object_t*)malloc(sizeof(sched_pcb_scheduler_object_t) + (nb_groups-1)*sizeof(sched_pcb_scheduler_object_t*));
        so->group_id = -1;
        so->nb_groups = nb_groups;
        PARSEC_OBJ_CONSTRUCT(&so->group_tasks, parsec_list_t);
        es->scheduler_object = so;

        if(0 == es->th_id) {
            so->shared_tasks = PARSEC_OBJ_NEW(parsec_list_t);
        }

        parsec_barrier_wait(barrier);

        if(0 != es->th_id) {
            so->shared_tasks = SCHED_PCB_SO(es->virtual_process->execution_streams[0])->shared_tasks;
        }

#if defined(PARSEC_PAPI_SDE)
        char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
        snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=PCB", es->th_id);
        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_int, (papi_sde_fptr_t)sched_pcb_group_tasks_length, es);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=PCB", PAPI_SDE_SUM);

        if(0 == es->th_id) {
            snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "SCHEDULER::PENDING_TASKS::QUEUE=SHARED::SCHED=PCB", es->th_id);
            parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_int, (papi_sde_fptr_t)sched_pcb_shared_tasks_length, es);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=PCB", PAPI_SDE_SUM);
        }
#endif
    } else {
        assert(0 != es->th_id);
        parsec_barrier_wait(barrier);
        es->scheduler_object = es->virtual_process->execution_streams[master_id]->scheduler_object;
    }

    // Thread 0 needs to wait that all others have set their scheduler object
    parsec_barrier_wait(barrier);

    if(0 == es->th_id) {
        // Core 0 names all the groups in sequence
        int group_id = 1; // NB: group IDs start at 1, because 0 is reserved to designate the shared list of tasks
        sched_pcb_scheduler_object_t *tso;
        for(int t = 0; t < es->virtual_process->nb_cores; t++) {
            tso = SCHED_PCB_SO(es->virtual_process->execution_streams[t]);
            if(tso->group_id == -1) {
                so->groups[group_id-1] = tso;
                tso->group_id = group_id++;
            }
        }
    }

    // Make sure that group assignments and the array of groups are visible by all before copying group 1 array
    parsec_barrier_wait(barrier);

    if(so->group_id > 1 && es->th_id == master_id) {
        sched_pcb_scheduler_object_t *so0 = SCHED_PCB_SO(es->virtual_process->execution_streams[0]);
        memcpy(so->groups, so0->groups, sizeof(sched_pcb_scheduler_object_t*)*nb_groups);
    }

    // and make sure that the groups array is visible by any other thread in this group
    parsec_barrier_wait(barrier);

    return PARSEC_SUCCESS;
}

/**
 * @brief
 *   Selects a task to run
 *
 * @details
 *   Take the highest priority task between the head of the calling execution stream's group list
 *   and the shared list of tasks; do nothing if both are empty.
 *
 *   @param[INOUT] es     the calling execution stream
 *   @param[OUT] distance the distance of the selected task. This scheduler
 *     always returns 0
 *   @return the selected task
 */
static parsec_task_t* sched_pcb_select(parsec_execution_stream_t *es,
                                      int32_t* distance)
{
    parsec_task_t *group_task = NULL, *shared_task = NULL, *pop_task = NULL, *candidate_task = NULL;
    parsec_list_t *candidate_list = NULL;
    sched_pcb_scheduler_object_t *so = SCHED_PCB_SO(es);
    int group_task_priority, shared_task_priority;

    *distance = 0;

    for(;;) {
        // Peak at the head of both lists to find which task has the highest priority
        parsec_list_lock(&so->group_tasks);
        group_task = (parsec_task_t*)PARSEC_LIST_ITERATOR_FIRST(&so->group_tasks);
        if((parsec_list_item_t*)group_task != PARSEC_LIST_ITERATOR_END(&so->group_tasks)) {
            group_task_priority = group_task->priority;
        } else {
            group_task = NULL;
        }
        parsec_list_unlock(&so->group_tasks);
        parsec_list_lock(so->shared_tasks);
        shared_task = (parsec_task_t*)PARSEC_LIST_ITERATOR_FIRST(so->shared_tasks);
        if((parsec_list_item_t*)shared_task != PARSEC_LIST_ITERATOR_END(so->shared_tasks)) {
            shared_task_priority = shared_task->priority;
        } else {
            shared_task = NULL;
        }
        parsec_list_unlock(so->shared_tasks);

        // If one of the lists is empty, return the head of the other without caring for priority
        if( NULL == shared_task ) {
            pop_task = (parsec_task_t*)parsec_list_pop_front(&so->group_tasks);
            if(NULL != pop_task) {
                PARSEC_LIST_ITEM_SINGLETON(pop_task);
            }
            return pop_task;
        }
        if( NULL == group_task ) {
            pop_task = (parsec_task_t*)parsec_list_pop_front(so->shared_tasks);
            if(NULL != pop_task) {
                PARSEC_LIST_ITEM_SINGLETON(pop_task);
            }
            return pop_task;
        }

        // Determine which list has the highest priority task
        if( shared_task_priority > group_task_priority ) {
            candidate_task = shared_task;
            candidate_list = so->shared_tasks;
        } else {
            candidate_task = group_task;
            candidate_list = &so->group_tasks;
        }

        // Try to pop it... But if the list head has changed, do not pop a random task, push
        // it back, and start again.
        pop_task = (parsec_task_t*)parsec_list_pop_front(candidate_list);
        PARSEC_LIST_ITEM_SINGLETON(pop_task);
        if(pop_task == candidate_task) {
            return pop_task;
        }
        parsec_list_push_sorted(candidate_list, &pop_task->super, parsec_execution_context_priority_comparator);
    }
}

/**
 * @brief
 *  Schedule a set of ready tasks on the calling execution stream
 *
 * @details
 *  Split the chain of tasks based on their priority, and the
 *  group this priority points to.
 *
 *   @param[INOUT] es          the calling execution stream
 *   @param[INOUT] new_context the ring of ready tasks to schedule
 *   @param[IN] distance       the distance hint is ignored...
 *   @return PARSEC_SUCCESS in case of success, a negative number
 *                          otherwise.
 */
static int sched_pcb_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    sched_pcb_scheduler_object_t *so;
    parsec_list_t *target;
    parsec_task_t *grp_tasks, *other_tasks;
    int group;

    (void)distance;

    so = SCHED_PCB_SO(es);

    other_tasks = new_context;
    while(other_tasks != NULL) {
        grp_tasks = other_tasks;
        other_tasks = (parsec_task_t*)parsec_list_item_ring_chop(&grp_tasks->super);
        PARSEC_LIST_ITEM_SINGLETON(&grp_tasks->super);

        group = sched_pcb_group(grp_tasks->priority, so);

        // Chain all other tasks belonging to the same group in ctx
        while(NULL != other_tasks && sched_pcb_group(other_tasks->priority, so) == group) {
            parsec_task_t *tmp = other_tasks;
            other_tasks = (parsec_task_t*)parsec_list_item_ring_chop(&tmp->super);
            PARSEC_LIST_ITEM_SINGLETON(&tmp->super);
            parsec_list_item_ring_push(&grp_tasks->super, &tmp->super);
        }

        // We found at least one task left in other_tasks that belongs to a different group,
        // we continue iterating on other_tasks, but now we have a sentinel to stop the iteration
        if(NULL != other_tasks) {
            parsec_task_t *t = (parsec_task_t*)PARSEC_LIST_ITEM_NEXT(&other_tasks->super);
            do {
                if(group == sched_pcb_group(t->priority, so)) {
                    parsec_task_t *tmp = t;
                    t = (parsec_task_t*)parsec_list_item_ring_chop(&tmp->super);
                    PARSEC_LIST_ITEM_SINGLETON(&tmp->super);
                    assert(NULL != t /* other_task should at least belong to this ring */); 

                    parsec_list_item_ring_push(&grp_tasks->super, &tmp->super);
                } else {
                    t = (parsec_task_t*)PARSEC_LIST_ITEM_NEXT(&t->super);
                }
            } while(t != other_tasks);
        }

        // Chain the tasks belong to group into the appropriate target list
        if(group > 0)
            target = &so->groups[group-1]->group_tasks;
        else
            target = so->shared_tasks;
        parsec_list_chain_sorted(target, &grp_tasks->super, parsec_execution_context_priority_comparator);
    }

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
static void sched_pcb_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    sched_pcb_scheduler_object_t *so;
    parsec_vp_t *vp;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            int master_id = t;
#if defined(PARSEC_HAVE_HWLOC)
            master_id = parsec_hwloc_master_id(sched_pcb_sharing_level, t);
#endif
            es = vp->execution_streams[t];
            if(t == master_id) {
                so = SCHED_PCB_SO(es);
                assert(so->group_tasks.super.obj_reference_count == 1);
                PARSEC_OBJ_DESTRUCT(&so->group_tasks);

                if(0 == t) {
                    PARSEC_OBJ_RELEASE(so->shared_tasks);
                    assert(NULL == so->shared_tasks);
                }

                free(so);
                PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=PCB", t);
            }

            es->scheduler_object = NULL;
        }
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=PCB");
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS");
}
