/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/profiling.h"
#include "datarepo.h"
#include "parsec/execution_stream.h"
#include "parsec/vpmap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/os-spec-timing.h"
#include "parsec/remote_dep.h"
#include "parsec/scheduling.h"
#include "parsec/papi_sde.h"

#include "parsec/debug_marks.h"
#include "parsec/ayudame.h"
#include "parsec/constants.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/class/list.h"
#include "parsec/utils/debug.h"
#include "parsec/dictionary.h"
#include "parsec/utils/backoff.h"

#include <signal.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif /* defined(PARSEC_HAVE_STRING_H) */
#include <sched.h>
#include <sys/types.h>
#if defined(PARSEC_HAVE_ERRNO_H)
#include <errno.h>
#endif  /* defined(PARSEC_HAVE_ERRNO_H) */
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* defined(PARSEC_HAVE_UNISTD_H) */
#if defined(PARSEC_PROF_RUSAGE_EU) && defined(PARSEC_HAVE_GETRUSAGE) && defined(PARSEC_HAVE_RUSAGE_THREAD) && !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>

static void parsec_rusage_per_es(parsec_execution_stream_t* es, bool print)
{
    struct rusage current;
    getrusage(RUSAGE_THREAD, &current);
    if( print ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - es->_es_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - es->_es_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - es->_es_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - es->_es_rusage.ru_stime.tv_usec) / 1000000.0);

        parsec_inform(
                      "Resource Usage Exec. Unit VP: %i Thread: %i (Core %i, socket %i)\n"
                      "-------------------------------------------------------------\n"
                      "User Time   (secs)          : %10.3f\n"
                      "System Time (secs)          : %10.3f\n"
                      "Total Time  (secs)          : %10.3f\n"
                      "Minor Page Faults           : %10ld\n"
                      "Major Page Faults           : %10ld\n"
                      "Swap Count                  : %10ld\n"
                      "Voluntary Context Switches  : %10ld\n"
                      "Involuntary Context Switches: %10ld\n"
                      "Block Input Operations      : %10ld\n"
                      "Block Output Operations     : %10ld\n"
                      "Maximum Resident Memory     : %10ld\n"
                      "=============================================================\n"
                      , es->virtual_process->vp_id, es->th_id, es->core_id, es->socket_id,
                      usr, sys, usr + sys,
                      (current.ru_minflt  - es->_es_rusage.ru_minflt), (current.ru_majflt  - es->_es_rusage.ru_majflt),
                      (current.ru_nswap   - es->_es_rusage.ru_nswap) , (current.ru_nvcsw   - es->_es_rusage.ru_nvcsw),
                      (current.ru_inblock - es->_es_rusage.ru_inblock), (current.ru_oublock - es->_es_rusage.ru_oublock),
                      current.ru_maxrss);

    }
    es->_es_rusage = current;
    return;
}
#define parsec_rusage_per_es(eu, b) do { if(parsec_want_rusage > 1) parsec_rusage_per_es(eu, b); } while(0)
#else
#define parsec_rusage_per_es(eu, b) do {} while(0)
#endif /* defined(PARSEC_HAVE_GETRUSAGE) defined(PARSEC_PROF_RUSAGE_EU) */

#if 0
/*
 * Disabled by now.
 */
int __parsec_context_wait_task( parsec_execution_stream_t* es,
                                parsec_task_t* task )
{
    (void)es;
    switch(task->status) {
    case PARSEC_TASK_STATUS_NONE:
#if defined(PARSEC_DEBUG)
        char tmp[MAX_TASK_STRLEN];
        parsec_degug_verbose(5, parsec_debug_output, "thread %d of VP %d Execute %s\n", es->th_id, es->virtual_process->vp_id,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task));
#endif
        return -1;

    case PARSEC_TASK_STATUS_PREPARE_INPUT:
        task->status = PARSEC_TASK_STATUS_EVAL;
        break;
    case PARSEC_TASK_STATUS_EVAL:
        task->status = PARSEC_TASK_STATUS_HOOK;
        break;
    case PARSEC_TASK_STATUS_HOOK:
        task->status = PARSEC_TASK_STATUS_PREPARE_OUTPUT;
        break;
    case PARSEC_TASK_STATUS_PREPARE_OUTPUT:
        task->status = PARSEC_TASK_STATUS_COMPLETE;
        break;
    case PARSEC_TASK_STATUS_COMPLETE:
        break;
    }
    return -1;
}
#endif

int __parsec_execute( parsec_execution_stream_t* es,
                      parsec_task_t* task )
{
    const parsec_task_class_t* tc = task->task_class;
    parsec_evaluate_function_t* eval;
    int rc;
#if defined(PARSEC_DEBUG)
    char tmp[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task);
#endif
    PARSEC_AYU_TASK_RUN(es->th_id, task);
    unsigned int chore_id;

    /* Find first bit in chore_mask that is not 0 */
    for(chore_id = 0; NULL != tc->incarnations[chore_id].hook; chore_id++)
        if( 0 != (task->chore_mask & (1<<chore_id)) )
            break;

    if (chore_id == sizeof(task->chore_mask)*8) {
#if !defined(PARSEC_DEBUG)
        char tmp[MAX_TASK_STRLEN];
        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task);
#endif
        parsec_warning("Task %s ran out of valid incarnations. Consider it complete",
                       tmp);
        return PARSEC_HOOK_RETURN_ERROR;
    }

    PARSEC_PINS(es, EXEC_BEGIN, task);
    /* Try all the incarnations until one agree to execute. */
    do {
        if( NULL != (eval = tc->incarnations[chore_id].evaluate) ) {
            rc = eval(task);
            if( PARSEC_HOOK_RETURN_DONE != rc ) {
                if( PARSEC_HOOK_RETURN_NEXT != rc ) {
#if defined(PARSEC_DEBUG)
                    parsec_debug_verbose(5, parsec_debug_output, "Thread %d of VP %d Failed to evaluate %s[%d] chore %d",
                                         es->th_id, es->virtual_process->vp_id,
                                         tmp, tc->incarnations[chore_id].type,
                                         chore_id);
#endif
                    break;
                }
                goto next_chore;
            }
        }

#if defined(PARSEC_DEBUG)
        parsec_debug_verbose(5, parsec_debug_output, "Thread %d of VP %d Execute %s[%d] chore %d",
                             es->th_id, es->virtual_process->vp_id,
                             tmp, tc->incarnations[chore_id].type,
                             chore_id);
#endif
        parsec_hook_t *hook = tc->incarnations[chore_id].hook;

        rc = hook( es, task );
#if defined(PARSEC_PROF_TRACE)
        task->prof_info.task_return_code = rc;
        PARSEC_PINS(es, EXEC_END, task);
#endif
        if( PARSEC_HOOK_RETURN_NEXT != rc ) {
            if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
                /* Let's assume everything goes just fine */
                task->status = PARSEC_TASK_STATUS_COMPLETE;
            }
            return rc;
        }
    next_chore:
        /* Mark this chore as tested */
        task->chore_mask &= ~( 1<<chore_id );
        /* Find next chore to try */
        for(chore_id = chore_id+1; NULL != tc->incarnations[chore_id].hook; chore_id++)
            if( 0 != (task->chore_mask & (1<<chore_id)) )
                break;
    } while(NULL != tc->incarnations[chore_id].hook);
    assert(task->status == PARSEC_TASK_STATUS_HOOK);
    /* We're out of luck, no more chores */
    PARSEC_PINS(es, EXEC_END, task);
    return PARSEC_HOOK_RETURN_ERROR;
}

/* Update the number of runtime associated activities. When this counter
 * reaches zero the taskpool is considered as completed, and all
 * resources will be marked for release. It should be noted that for as long
 * as the DSL might add additional tasks into the taskpool, it should hold
 * one reference to the runtime activities, preventing the runtime from
 * completing the taskpool too early.
 */
int parsec_taskpool_update_runtime_nbtask(parsec_taskpool_t *tp, int32_t nb_tasks)
{
    tp->tdm.module->taskpool_addto_runtime_actions(tp, nb_tasks);
    return 0;
}

static inline int all_tasks_done(parsec_context_t* context)
{
    return (context->active_taskpools == 0);
}

void parsec_taskpool_termination_detected(parsec_taskpool_t *tp)
{
    if( NULL != tp->on_complete ) {
        (void)tp->on_complete( tp, tp->on_complete_data );
    }
    (void)parsec_atomic_fetch_dec_int32( &(tp->context->active_taskpools) );
    PARSEC_PINS_TASKPOOL_FINI(tp);
}

parsec_sched_module_t *parsec_current_scheduler           = NULL;
static parsec_sched_base_component_t *scheduler_component = NULL;

void parsec_remove_scheduler( parsec_context_t *parsec )
{
    if( NULL != parsec_current_scheduler ) {
        parsec_current_scheduler->module.remove( parsec );
        assert( NULL != scheduler_component );
        mca_component_close( (mca_base_component_t*)scheduler_component );
        parsec_current_scheduler = NULL;
        scheduler_component = NULL;
    }
}

int parsec_set_scheduler( parsec_context_t *parsec )
{
    mca_base_component_t **scheds;
    mca_base_module_t    *new_scheduler = NULL;
    mca_base_component_t *new_component = NULL;

    assert(NULL == parsec_current_scheduler);
    scheds = mca_components_open_bytype( "sched" );
    mca_components_query(scheds,
                         &new_scheduler,
                         &new_component);
    mca_components_close(scheds);

    if( NULL == new_scheduler ) {
        return PARSEC_ERROR;
    }

    parsec_remove_scheduler( parsec );
    parsec_current_scheduler   = (parsec_sched_module_t*)new_scheduler;
    scheduler_component = (parsec_sched_base_component_t*)new_component;

    parsec_debug_verbose(4, parsec_debug_output, " Installing scheduler %s", parsec_current_scheduler->component->base_version.mca_component_name);
    PROFILING_SAVE_sINFO("sched", (char *)parsec_current_scheduler->component->base_version.mca_component_name);

    parsec_current_scheduler->module.install( parsec );
    return PARSEC_SUCCESS;
}

/*
 * Dispatch a ring of tasks to the requested execution stream, using the provided
 * distance. This function provides little benefit by itself, but it allows to
 * have a common place where tasks can be seen before being delivered to the
 * scheduler.
 *
 * In general, this is where we end up after the release_dep_fct is called and
 * generates a readylist.
 */
inline int
__parsec_schedule(parsec_execution_stream_t* es,
                  parsec_task_t* tasks_ring,
                  int32_t distance)
{
    int ret;

#if defined(PARSEC_DEBUG_PARANOID) || defined(PARSEC_DEBUG_NOISIER)
    {
        parsec_task_t* task = tasks_ring;
        char task_string[MAX_TASK_STRLEN];

        do {
            (void)parsec_task_snprintf(task_string, MAX_TASK_STRLEN, task);
#if defined(PARSEC_DEBUG_PARANOID)
            const struct parsec_flow_s* flow;
            for( int i = 0; NULL != (flow = task->task_class->in[i]); i++ ) {
                if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) continue;
                if( NULL != task->data[flow->flow_index].source_repo_entry ) {
                    if( NULL == task->data[flow->flow_index].data_in ) {
                        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Task %s has flow %s data_repo != NULL but a data == NULL (%s:%d)",
                                             task_string,
                                             flow->name, __FILE__, __LINE__);
                    }
                }
            }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "thread %d of VP %d Schedules %s (distance %d)",
                                 es->th_id, es->virtual_process->vp_id,
                                 task_string, distance );
            task = (parsec_task_t*)task->super.list_next;
        } while ( task != tasks_ring );
    }
#endif  /* defined(PARSEC_DEBUG_PARANOID) || defined(PARSEC_DEBUG_NOISIER) */

#if defined(PARSEC_PAPI_SDE)
    {
        int len = 0;
        parsec_task_t *task = tasks_ring;
        _LIST_ITEM_ITERATOR(task, &task->super, item, {len++; });
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_TASKS_ENABLED, len);
    }
#endif  /* defined(PARSEC_PAPI_SDE) */

    ret = parsec_current_scheduler->module.schedule(es, tasks_ring, distance);

    return ret;
}

/*
 * Schedule an array of rings of tasks with one entry per virtual process.
 * If an execution stream is provided, this function will save the highest
 * priority task (assuming the ring is ordered or the first task in the ring
 * otherwise) on the current execution stream virtual process as the next
 * task to be executed on the provided execution stream. Everything else gets
 * pushed into the execution stream 0 of the corresponding virtual process.
 * If the provided execution stream is NULL, all tasks are delivered to their
 * respective vp.
 *
 * Beware, as the manipulation of next_task is not protected, an exeuction
 * stream should never be used concurrently in two call to this function (or
 * a thread should never `borrow` an execution stream for this call).
 */
int __parsec_schedule_vp(parsec_execution_stream_t* es,
                         parsec_task_t** task_rings,
                         int32_t distance)
{
    parsec_execution_stream_t* target_es;
    const parsec_vp_t** vps = (const parsec_vp_t**)es->virtual_process->parsec_context->virtual_processes;
    int ret = 0;

#if  defined(PARSEC_DEBUG_PARANOID)
    /* As the setting of the next_task is not protected no thread should call
     * this function with a stream other than its own. */
    assert( (NULL == es) || (parsec_my_execution_stream() == es) );
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

    if( NULL == es || !parsec_runtime_keep_highest_priority_task ) {
        for(int vp = 0; vp < es->virtual_process->parsec_context->nb_vp; vp++ ) {
            parsec_task_t* ring = task_rings[vp];
            if( NULL == ring ) continue;

            target_es = vps[vp]->execution_streams[0];

            ret = __parsec_schedule(target_es, ring, distance);
            if( 0 != ret )
                return ret;

            task_rings[vp] = NULL;  /* remove the tasks already scheduled */
        }
        return ret;
    }
    for(int vp = 0; vp < es->virtual_process->parsec_context->nb_vp; vp++ ) {
        parsec_task_t* ring = task_rings[vp];
        if( NULL == ring ) continue;

        target_es = vps[vp]->execution_streams[0];

        if( vp == es->virtual_process->vp_id ) {
            if( NULL == es->next_task ) {
                es->next_task = ring;
                ring = (parsec_task_t*)parsec_list_item_ring_chop(&ring->super);
                if( NULL == ring ) {
                    task_rings[vp] = NULL;  /* remove the tasks already scheduled */
                    continue;
                }
            }
            target_es = es;
        }
        ret = __parsec_schedule(target_es, ring, distance);
        if( 0 != ret )
            return ret;

        task_rings[vp] = NULL;  /* remove the tasks already scheduled */
    }
    return ret;
}

/**
 * @brief Reschedule a task on the most appropriate resource.
 *
 * @details The function reschedules a task, by trying to locate it as closer
 *          as possible to the current execution unit. If not available
 *          execution unit was found, the task is rescheduled on the same
 *          execution unit. To find the most appropriate execution unit
 *          we start from the next execution unit after the current one, and
 *          iterate over all existing execution units (in the current VP,
 *          then on the next VP and so on).
 *
 * @param [IN] es, the start execution_stream (normall it is the current one).
 * @param [IN] task, the task to be rescheduled.
 *
 * @return parsec scheduling return code
 */
int __parsec_reschedule(parsec_execution_stream_t* es, parsec_task_t* task)
{
    parsec_context_t* context = es->virtual_process->parsec_context;
    parsec_vp_t* vp_context = es->virtual_process;

    int vp, start_vp = vp_context->vp_id, next_vp;
    int start_eu = (es->th_id + 1) % context->virtual_processes[start_vp]->nb_cores;

    for( vp = start_vp, next_vp = (start_vp + 1) % context->nb_vp;
         next_vp != vp_context->vp_id;
         ++vp) {
        if( 1 != context->virtual_processes[vp]->nb_cores ) {
            return __parsec_schedule(context->virtual_processes[vp]->execution_streams[start_eu], task, 0);
        }
        else if ( context->virtual_processes[vp]->vp_id != vp_context->vp_id ) {
            /* VP contains only one EU, and it's not my VP, so not me */
            return __parsec_schedule(context->virtual_processes[vp]->execution_streams[0], task, 0);
        }
        start_eu = 0;  /* with the exception of the first es, we always iterate from 0 */
    }
    /* no luck so far, let's reschedule the task on the same execution unit */
    return __parsec_schedule(es, task, 0);
}

int __parsec_complete_execution( parsec_execution_stream_t *es,
                                 parsec_task_t *task )
{
    int rc = 0;

    /* complete execution PINS event includes the preparation of the
     * output and the and the call to complete_execution.
     */
    PARSEC_PINS(es, COMPLETE_EXEC_BEGIN, task);

    if( NULL != task->task_class->prepare_output ) {
        task->task_class->prepare_output( es, task );
    }
    if( NULL != task->task_class->complete_execution )
        rc = task->task_class->complete_execution( es, task );

    PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_TASKS_RETIRED, 1);
    PARSEC_PINS(es, COMPLETE_EXEC_END, task);
    PARSEC_AYU_TASK_COMPLETE(task);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( es->th_id, es->virtual_process->vp_id, task );

    /* Release the execution context */
    (void)task->task_class->release_task( es, task );
    
    return rc;
}

int __parsec_task_progress( parsec_execution_stream_t* es,
                            parsec_task_t* task,
                            int distance)
{
    int rc = PARSEC_HOOK_RETURN_DONE;

    PARSEC_PINS(es, SELECT_END, task);

    if(task->status <= PARSEC_TASK_STATUS_PREPARE_INPUT) {
        PARSEC_PINS(es, PREPARE_INPUT_BEGIN, task);
        rc = task->task_class->prepare_input(es, task);
        PARSEC_PINS(es, PREPARE_INPUT_END, task);
    }
    switch(rc) {
    case PARSEC_HOOK_RETURN_DONE: {
        if(task->status <= PARSEC_TASK_STATUS_HOOK) {
            rc = __parsec_execute( es, task );
        }
        /* We're good to go ... */
        switch(rc) {
        case PARSEC_HOOK_RETURN_DONE:    /* This execution succeeded */
            task->status = PARSEC_TASK_STATUS_COMPLETE;
            __parsec_complete_execution( es, task );
            break;
        case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
            task->status = PARSEC_TASK_STATUS_HOOK;
            if(0 == task->priority) {
                SET_LOWEST_PRIORITY(task, parsec_execution_context_priority_comparator);
            } else
                task->priority /= 10;  /* demote the task */
            PARSEC_LIST_ITEM_SINGLETON(task);
            __parsec_schedule(es, task, distance + 1);
            break;
        case PARSEC_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                          * even try to change it's state, the completion
                                          * will be triggered asynchronously. */
            break;
        case PARSEC_HOOK_RETURN_NEXT:    /* Try next variant [if any] */
        case PARSEC_HOOK_RETURN_DISABLE: /* Disable the device, something went wrong */
        case PARSEC_HOOK_RETURN_ERROR:   /* Some other major error happened */
            assert( 0 ); /* Internal error: invalid return value */
        }
        break;
    }
    case PARSEC_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                      * even try to change it's state, the completion
                                      * will be triggered asynchronously. */
        break;
    case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
        if(0 == task->priority) {
            SET_LOWEST_PRIORITY(task, parsec_execution_context_priority_comparator);
        } else
            task->priority /= 10;  /* demote the task */
        PARSEC_LIST_ITEM_SINGLETON(task);
        __parsec_schedule(es, task, distance + 1);
        break;
    default:
        assert( 0 ); /* Internal error: invalid return value for data_lookup function */
    }

    // subsequent select begins
    PARSEC_PINS(es, SELECT_BEGIN, NULL);
    return rc;
}

int __parsec_context_wait( parsec_execution_stream_t* es )
{
    uint64_t misses_in_a_row;
    parsec_context_t* parsec_context = es->virtual_process->parsec_context;
    int32_t my_barrier_counter = parsec_context->__parsec_internal_finalization_counter;
    parsec_task_t* task;
    int nbiterations = 0, distance, rc;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    if( !PARSEC_THREAD_IS_MASTER(es) ) {
        /* Wait until all threads are done binding themselves
         * (see parsec_init) */
        parsec_barrier_wait( &(parsec_context->barrier) );
        my_barrier_counter = 1;
    } else {
#if defined(DISTRIBUTED)
        if( (1 == parsec_communication_engine_up) &&
            (es->virtual_process[0].parsec_context->nb_nodes == 1) ) {
            /* If there is a single process run and the main thread is in charge of
             * progressing the communications we need to make sure the comm engine
             * is ready for primetime. */
            remote_dep_mpi_on(parsec_context);
        }
#endif /* defined(DISTRIBUTED) */
        /* The master thread might not have to trigger the barrier if the other
         * threads have been activated by a previous start.
         */
        if( PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE & parsec_context->flags ) {
            goto skip_first_barrier;
        }
        parsec_context->flags |= PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE;
    }

    parsec_rusage_per_es(es, false);

    /* first select begin, right before the wait_for_the... goto label */
    PARSEC_PINS(es, SELECT_BEGIN, NULL);

    /* The main loop where all the threads will spend their time */
  wait_for_the_next_round:
    /* Wait until all threads are here and the main thread signal the beginning of the work */
    parsec_barrier_wait( &(parsec_context->barrier) );

    if( parsec_context->__parsec_internal_finalization_in_progress ) {
        my_barrier_counter++;
        for(; my_barrier_counter <= parsec_context->__parsec_internal_finalization_counter; my_barrier_counter++ ) {
            parsec_barrier_wait( &(parsec_context->barrier) );
        }
        goto finalize_progress;
    }

    if( NULL == parsec_current_scheduler ) {
        parsec_fatal("Main thread entered parsec_context_wait, while a scheduler is not selected yet!");
        return -1;
    }
  skip_first_barrier:
    while( !all_tasks_done(parsec_context) ) {

        if(PARSEC_THREAD_IS_MASTER(es)) {
            /* Here we detach all dtd taskpools registered with us */
            parsec_detach_all_dtd_taskpool_from_context(parsec_context);
        }

#if defined(DISTRIBUTED)
        if( (1 == parsec_communication_engine_up) &&
            (es->virtual_process[0].parsec_context->nb_nodes == 1) &&
            PARSEC_THREAD_IS_MASTER(es) ) {
            /* check for remote deps completion */
            while(parsec_remote_dep_progress(es) > 0)  {
                misses_in_a_row = 0;
            }
        }
#endif /* defined(DISTRIBUTED) */

        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = parsec_exponential_backoff(es, misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }
        misses_in_a_row++;  /* assume we fail to extract a task */

        if( NULL == (task = es->next_task) ) {
            task = parsec_current_scheduler->module.select(es, &distance);
        } else {
            es->next_task = NULL;
            distance = 1;
        }

        if( task != NULL ) {
            misses_in_a_row = 0;  /* reset the misses counter */

            rc = __parsec_task_progress(es, task, distance);
            (void)rc;  /* for now ignore the return value */

            nbiterations++;
        }
    }

    parsec_rusage_per_es(es, true);

    /* We're all done ? */
    parsec_barrier_wait( &(parsec_context->barrier) );

#if defined(PARSEC_SIM)
    if( PARSEC_THREAD_IS_MASTER(es) ) {
        parsec_vp_t *vp;
        int32_t my_vpid, my_idx;
        int largest_date = 0;
        for(my_vpid = 0; my_vpid < parsec_context->nb_vp; my_vpid++) {
            vp = parsec_context->virtual_processes[my_vpid];
            for(my_idx = 0; my_idx < vp->nb_cores; my_idx++) {
                if( vp->execution_streams[my_idx]->largest_simulation_date > largest_date )
                    largest_date = vp->execution_streams[my_idx]->largest_simulation_date;
            }
        }
        parsec_context->largest_simulation_date = largest_date;
    }
    parsec_barrier_wait( &(parsec_context->barrier) );
    es->largest_simulation_date = 0;
#endif

    if( !PARSEC_THREAD_IS_MASTER(es) ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
    // final select end - can we mark this as special somehow?
    // actually, it will already be obviously special, since it will be the only select
    // that has no context
    PARSEC_PINS(es, SELECT_END, NULL);

    if( parsec_context->__parsec_internal_finalization_in_progress ) {
        PARSEC_PINS_THREAD_FINI(es);
    }

    return nbiterations;
}

int parsec_context_add_taskpool( parsec_context_t* context, parsec_taskpool_t* tp )
{
    if( NULL == parsec_current_scheduler) {
        parsec_set_scheduler( context );
    }

    tp->context = context;  /* save the context */

    PARSEC_PINS_TASKPOOL_INIT(tp);  /* PINS taskpool initialization */

    /* If the DSL did not install a termination detection module,
     * assume that the old behavior (local detection when local 
     * number of tasks is 0) is expected: install the local termination
     * detection module, and declare the taskpool as ready */
    if( tp->tdm.module == NULL ) {
        parsec_termdet_open_module(tp, "local");
        assert( NULL != tp->tdm.module );
        tp->tdm.module->monitor_taskpool(tp, parsec_taskpool_termination_detected);
        tp->tdm.module->taskpool_ready(tp);
    }
    
    /* Update the number of pending taskpools */
    (void)parsec_atomic_fetch_inc_int32( &context->active_taskpools );

    /* If necessary trigger the on_enqueue callback */
    if( NULL != tp->on_enqueue ) {
        tp->on_enqueue(tp, tp->on_enqueue_data);
    }

#if defined(PARSEC_PROF_TRACE)
    if( parsec_profile_enabled )
        parsec_profiling_add_taskpool_properties(tp);
#endif

    if( NULL != tp->startup_hook ) {
        parsec_task_t **startup_list;
        int vpid;

        /* These pointers need to be initialized to NULL */
        startup_list = (parsec_task_t**)alloca( context->nb_vp * sizeof(parsec_task_t*) );
        for(vpid = 0; vpid < context->nb_vp; startup_list[vpid++] = NULL);

        tp->startup_hook(context, tp, startup_list);

        __parsec_schedule_vp(parsec_my_execution_stream(),
                             startup_list, 0);
    }

    return PARSEC_SUCCESS;
}

static inline int
__parsec_context_cas_or_flag(parsec_context_t* context,
                            uint32_t flags)
{
    uint32_t current_flags = context->flags;
    /* if the flags are already set don't reset them */
    if( flags == (current_flags & flags) ) return 0;
    return parsec_atomic_cas_int32(&context->flags,
                                   current_flags,
                                   current_flags | flags);
}

/*
 * If there are enqueued taskpools waiting to be executed launch the other threads
 * and then return. Mark the internal structures in such a way that we can't
 * start the context mutiple times without completions.
 *
 * @returns: 0 if the other threads in this context have been started, 1 if the
 * context was already active, 2 if there was nothing to do and no threads have
 * been activated.
 */
int parsec_context_start( parsec_context_t* context )
{
    /* Context already active */
    if( PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags )
        return 1;
    /* Start up the context */
    if( __parsec_context_cas_or_flag(context, PARSEC_CONTEXT_FLAG_COMM_ACTIVE) ) {
        (void)parsec_remote_dep_on(context);
        /* Mark the context so that we will skip the initial barrier during the _wait */
        context->flags |= PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE;
        /* Wake up the other threads */
        parsec_barrier_wait( &(context->barrier) );
        /* we keep one extra reference on the context to make sure we only match this with an
         * explicit call to parsec_context_wait.
         */
        (void)parsec_atomic_fetch_inc_int32( &context->active_taskpools );
        return 0;
    }
    return 1;  /* Someone else start it up */
}

int parsec_context_test( parsec_context_t* context )
{
    return all_tasks_done(context);
}

int parsec_context_wait( parsec_context_t* context )
{
    int ret = 0;

    if( !(PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags) ) {
        parsec_warning("parsec_context_wait detected on a non started context\n");
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if( __parsec_context_cas_or_flag(context,
                                    PARSEC_CONTEXT_FLAG_COMM_ACTIVE) ) {
        (void)parsec_remote_dep_on(context);
    }
    /* Remove the additional active_taskpool to signal the runtime that we
     * are ready to complete a scheduling epoch.
     */
    int active = parsec_atomic_fetch_dec_int32( &context->active_taskpools ) - 1;
    if( active < 0 ) {
        parsec_warning("parsec_context_wait detected on a non-started context\n");
        /* put the context back on it's original state */
        (void)parsec_atomic_fetch_inc_int32( &context->active_taskpools );
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    ret = __parsec_context_wait( context->virtual_processes[0]->execution_streams[0] );

    context->__parsec_internal_finalization_counter++;
    (void)parsec_remote_dep_off(context);
    assert(context->flags & PARSEC_CONTEXT_FLAG_COMM_ACTIVE);
    assert(context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    context->flags ^= (PARSEC_CONTEXT_FLAG_COMM_ACTIVE | PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    return (ret >= 0)? PARSEC_SUCCESS: ret;
}
