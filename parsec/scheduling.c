/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
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
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/class/list.h"
#include "parsec/utils/debug.h"

#include <signal.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif /* defined(PARSEC_HAVE_STRING_H) */
#include <sched.h>
#include <sys/types.h>
#if defined(PARSEC_HAVE_ERRNO_H)
#include <errno.h>
#endif  /* defined(PARSEC_HAVE_ERRNO_H) */
#if defined(PARSEC_HAVE_SCHED_SETAFFINITY)
#include <linux/unistd.h>
#endif  /* defined(PARSEC_HAVE_SCHED_SETAFFINITY) */
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_SCHEDULING_EVENTS)
#define TAKE_TIME(ES_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((ES_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(ES_PROFILE, KEY, ID) do {} while(0)
#endif

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
    AYU_TASK_RUN(es->th_id, task);

    if (NULL == tc->incarnations[task->chore_id].hook) {
#if !defined(PARSEC_DEBUG)
        char tmp[MAX_TASK_STRLEN];
        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task);
#endif
        parsec_warning("Task %s[%d] run out of valid incarnations. Consider it complete",
                       tmp, tc->incarnations[task->chore_id].type);
        return PARSEC_HOOK_RETURN_ERROR;
    }

    PINS(es, EXEC_BEGIN, task);
    /* Try all the incarnations until one agree to execute. */
    do {
        if( NULL != (eval = tc->incarnations[task->chore_id].evaluate) ) {
            rc = eval(task);
            if( PARSEC_HOOK_RETURN_DONE != rc ) {
                if( PARSEC_HOOK_RETURN_NEXT != rc ) {
#if defined(PARSEC_DEBUG)
                    parsec_debug_verbose(5, parsec_debug_output, "Thread %d of VP %d Failed to evaluate %s[%d] chore %d",
                                         es->th_id, es->virtual_process->vp_id,
                                         tmp, tc->incarnations[task->chore_id].type,
                                         task->chore_id);
#endif
                    break;
                }
                goto next_chore;
            }
        }

#if defined(PARSEC_DEBUG)
        parsec_debug_verbose(5, parsec_debug_output, "Thread %d of VP %d Execute %s[%d] chore %d",
                             es->th_id, es->virtual_process->vp_id,
                             tmp, tc->incarnations[task->chore_id].type,
                             task->chore_id);
#endif
        parsec_hook_t *hook = tc->incarnations[task->chore_id].hook;

        rc = hook( es, task );
        if( PARSEC_HOOK_RETURN_NEXT != rc ) {
            PINS(es, EXEC_END, task);
            if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
                /* Let's assume everything goes just fine */
                task->status = PARSEC_TASK_STATUS_COMPLETE;
            }
            return rc;
        }
    next_chore:
        task->chore_id++;

    } while(NULL != tc->incarnations[task->chore_id].hook);
    assert(task->status == PARSEC_TASK_STATUS_HOOK);
    /* We're out of luck, no more chores */
    PINS(es, EXEC_END, task);
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
    int remaining;

    assert( tp->nb_pending_actions != 0 );
    remaining = tp->update_nb_runtime_task( tp, nb_tasks );
    assert( 0 <= remaining );
    return parsec_check_complete_cb(tp, tp->context, remaining);
}

static inline int all_tasks_done(parsec_context_t* context)
{
    return (context->active_taskpools == 0);
}

int parsec_check_complete_cb(parsec_taskpool_t *tp, parsec_context_t *context, int remaining)
{
    if( 0 == remaining ) {
        /* A parsec taskpool has been completed. Call the attached callback if
         * necessary, then update the main engine.
         */
        if( NULL != tp->on_complete ) {
            (void)tp->on_complete( tp, tp->on_complete_data );
        }
        (void)parsec_atomic_fetch_dec_int32( &context->active_taskpools );
        PINS_TASKPOOL_FINI(tp);
        return 1;
    }
    return 0;
}

parsec_sched_module_t *current_scheduler                  = NULL;
static parsec_sched_base_component_t *scheduler_component = NULL;

void parsec_remove_scheduler( parsec_context_t *parsec )
{
    if( NULL != current_scheduler ) {
        current_scheduler->module.remove( parsec );
        assert( NULL != scheduler_component );
        mca_component_close( (mca_base_component_t*)scheduler_component );
        current_scheduler = NULL;
        scheduler_component = NULL;
    }
}

int parsec_set_scheduler( parsec_context_t *parsec )
{
    mca_base_component_t **scheds;
    mca_base_module_t    *new_scheduler = NULL;
    mca_base_component_t *new_component = NULL;

    assert(NULL == current_scheduler);
    scheds = mca_components_open_bytype( "sched" );
    mca_components_query(scheds,
                         &new_scheduler,
                         &new_component);
    mca_components_close(scheds);

    if( NULL == new_scheduler ) {
        return 0;
    }

    parsec_remove_scheduler( parsec );
    current_scheduler   = (parsec_sched_module_t*)new_scheduler;
    scheduler_component = (parsec_sched_base_component_t*)new_component;

    parsec_debug_verbose(4, parsec_debug_output, " Installing scheduler %s", current_scheduler->component->base_version.mca_component_name);
    PROFILING_SAVE_sINFO("sched", (char *)current_scheduler->component->base_version.mca_component_name);

    current_scheduler->module.install( parsec );
    return 1;
}

/*
 * This is where we end up after the release_dep_fct is called and generates a
 * readylist. the new_context IS the readylist.
 */
int __parsec_schedule(parsec_execution_stream_t* es,
                      parsec_task_t* tasks_ring,
                      int32_t distance)
{
    int ret;
    int len;
    parsec_task_t *task = tasks_ring;

#if defined(PARSEC_DEBUG_PARANOID) || defined(PARSEC_DEBUG_NOISIER)
    {
        parsec_task_t* task = tasks_ring;
        char task_string[MAX_TASK_STRLEN];

        do {
            (void)parsec_task_snprintf(task_string, MAX_TASK_STRLEN, task);
#if defined(PARSEC_DEBUG_PARANOID)
            const struct parsec_flow_s* flow;
            for( int i = 0; NULL != (flow = task->task_class->in[i]); i++ ) {
                if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) continue;
                if( NULL != task->data[flow->flow_index].data_repo ) {
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

    len = 0;
    _LIST_ITEM_ITERATOR(task, &task->super, item, {len++; });
    PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_TASKS_ENABLED, len);
    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME(es->es_profile, schedule_push_begin, 0);
     */
    ret = current_scheduler->module.schedule(es, tasks_ring, distance);
    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME( es->es_profile, schedule_push_end, 0);
     */

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

#ifdef  PARSEC_HAVE_SCHED_SETAFFINITY
#define gettid() syscall(__NR_gettid)
#endif /* PARSEC_HAVE_SCHED_SETAFFINITY */

#define TIME_STEP 5410
#define MIN(x, y) ( (x)<(y)?(x):(y) )
static inline unsigned long exponential_backoff(uint64_t k)
{
    unsigned int n = MIN( 64, k );
    unsigned int r = (unsigned int) ((double)n * ((double)rand()/(double)RAND_MAX));
    return r * TIME_STEP;
}

int __parsec_complete_execution( parsec_execution_stream_t *es,
                                 parsec_task_t *task )
{
    parsec_taskpool_t *tp = task->taskpool;
    int rc = 0;

    /* complete execution PINS event includes the preparation of the
     * output and the and the call to complete_execution.
     */
    PINS(es, COMPLETE_EXEC_BEGIN, task);

    if( NULL != task->task_class->prepare_output ) {
        task->task_class->prepare_output( es, task );
    }
    if( NULL != task->task_class->complete_execution )
        rc = task->task_class->complete_execution( es, task );

    PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_TASKS_RETIRED, 1);
    PINS(es, COMPLETE_EXEC_END, task);
    AYU_TASK_COMPLETE(task);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( es->th_id, es->virtual_process->vp_id, task );

    /* Release the execution context */
    task->task_class->release_task( es, task );

    /* Check to see if the DSL has marked the taskpool as completed */
    if( 0 == tp->nb_tasks ) {
        /* The taskpool has been marked as complete. Unfortunately, it is possible
         * that multiple threads are completing tasks associated with this taskpool
         * simultaneously and we need to release the runtime action associated with
         * this taskpool tasks once. We need to protect this action by atomically
         * setting the number of tasks to a non-zero value.
         */
        if( parsec_atomic_cas_int32(&tp->nb_tasks, 0, PARSEC_RUNTIME_RESERVED_NB_TASKS) )
            parsec_taskpool_update_runtime_nbtask(tp, -1);
    }

    return rc;
}

int __parsec_task_progress( parsec_execution_stream_t* es,
                            parsec_task_t* task,
                            int distance)
{
    int rc = PARSEC_HOOK_RETURN_DONE;

    PINS(es, SELECT_END, task);

    if(task->status <= PARSEC_TASK_STATUS_PREPARE_INPUT) {
        PINS(es, PREPARE_INPUT_BEGIN, task);
        rc = task->task_class->prepare_input(es, task);
        PINS(es, PREPARE_INPUT_END, task);
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
    PINS(es, SELECT_BEGIN, NULL);
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
    PINS(es, SELECT_BEGIN, NULL);

    /* The main loop where all the threads will spend their time */
  wait_for_the_next_round:
    /* Wait until all threads are here and the main thread signal the begining of the work */
    parsec_barrier_wait( &(parsec_context->barrier) );

    if( parsec_context->__parsec_internal_finalization_in_progress ) {
        my_barrier_counter++;
        for(; my_barrier_counter <= parsec_context->__parsec_internal_finalization_counter; my_barrier_counter++ ) {
            parsec_barrier_wait( &(parsec_context->barrier) );
        }
        goto finalize_progress;
    }

    if( NULL == current_scheduler ) {
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
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }
        misses_in_a_row++;  /* assume we fail to extract a task */

        task = current_scheduler->module.select(es, &distance);

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
    PINS(es, SELECT_END, NULL);

    if( parsec_context->__parsec_internal_finalization_in_progress ) {
        PINS_THREAD_FINI(es);
    }
    
    return nbiterations;
}

/*  *********** COMPOSITION OF PARSEC_TASKPOOLS ***************  */
typedef struct parsec_compound_state_t {
    parsec_context_t* ctx;
    int nb_taskpools;
    int completed_taskpools;
    parsec_taskpool_t* taskpools_array[1];
} parsec_compound_state_t;

static void parsec_compound_destructor( parsec_taskpool_t* compound_tp )
{
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound_tp->task_classes_array;
    PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool destructor %p",
                         compound_tp);
    free(compound_state);
    free(compound_tp);
}

static int parsec_composed_cb( parsec_taskpool_t* o, void* cbdata )
{
    parsec_taskpool_t* compound = (parsec_taskpool_t*)cbdata;
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound->task_classes_array;
    int completed_taskpools = compound_state->completed_taskpools++;
    assert( o == compound_state->taskpools_array[completed_taskpools] ); (void)o;
    if( --compound->nb_pending_actions ) {
        assert( NULL != compound_state->taskpools_array[completed_taskpools+1] );
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p enable taskpool %p",
                             compound, compound_state->taskpools_array[completed_taskpools+1]);
        parsec_context_add_taskpool(compound_state->ctx,
                                    compound_state->taskpools_array[completed_taskpools+1]);
    } else {
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool completed %p",
                             compound);
        parsec_check_complete_cb(compound, compound_state->ctx, 0 /* no tp left on this compound */);
    }
    return 0;
}

static void parsec_compound_startup( parsec_context_t *context,
                                     parsec_taskpool_t *compound_tp,
                                     parsec_task_t** startup_list )
{
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound_tp->task_classes_array;

    compound_state->ctx = context;
    compound_tp->nb_pending_actions = compound_state->nb_taskpools;
    PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p starting with %d taskpools",
                         compound_tp, compound_tp->nb_pending_actions);
    for( int i = 0; i < compound_state->nb_taskpools; i++ ) {
        parsec_taskpool_t* o = compound_state->taskpools_array[i];
        assert( NULL != o );
        o->on_complete      = parsec_composed_cb;
        o->on_complete_data = compound_tp;
    }
    parsec_context_add_taskpool(compound_state->ctx, compound_state->taskpools_array[0]);
    (void)startup_list;
}

parsec_taskpool_t* parsec_compose( parsec_taskpool_t* start,
                                   parsec_taskpool_t* next )
{
    parsec_taskpool_t* compound = NULL;
    parsec_compound_state_t* compound_state = NULL;

    if( PARSEC_TASKPOOL_TYPE_COMPOUND == start->taskpool_type ) {  /* start is already a compound taskpool */
        compound = start;
        compound_state = (parsec_compound_state_t*)compound->task_classes_array;
        compound_state->taskpools_array[compound_state->nb_taskpools++] = next;
        /* must always be NULL terminated */
        compound_state->taskpools_array[compound_state->nb_taskpools]   = NULL;
        /* make room for NULL terminating, if necessary */
        if( 0 == (compound_state->nb_taskpools % 16) ) {
            compound_state = realloc(compound_state,
                                     (sizeof(parsec_compound_state_t) +
                                      (compound_state->nb_taskpools + 16) * sizeof(void*)));
            compound->task_classes_array = (void*)compound_state;
        }
        compound_state->taskpools_array[compound_state->nb_taskpools] = NULL;
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p add %d taskpool %p",
                             compound, compound_state->nb_taskpools, next );
    } else {
        compound = calloc(1, sizeof(parsec_taskpool_t));
        compound->taskpool_type      = PARSEC_TASKPOOL_TYPE_COMPOUND;
        compound->task_classes_array = malloc(sizeof(parsec_compound_state_t) + 16 * sizeof(void*));

        compound_state = (parsec_compound_state_t*)compound->task_classes_array;
        compound_state->taskpools_array[0] = start;
        compound_state->taskpools_array[1] = next;
        compound_state->taskpools_array[2] = NULL;
        compound_state->completed_taskpools = 0;
        compound_state->nb_taskpools = 2;
        compound->startup_hook = parsec_compound_startup;
        compound->destructor = parsec_compound_destructor;
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p started with %p and %p taskpools",
                             compound, start, next );
    }
    return compound;
}
/* END: Composition */

int parsec_context_add_taskpool( parsec_context_t* context, parsec_taskpool_t* tp )
{
    if( NULL == current_scheduler) {
        parsec_set_scheduler( context );
    }

    tp->context = context;  /* save the context */

    PINS_TASKPOOL_INIT(tp);  /* PINS taskpool initialization */

    /* Update the number of pending taskpools */
    (void)parsec_atomic_fetch_inc_int32( &context->active_taskpools );

    /* If necessary trigger the on_enqueue callback */
    if( NULL != tp->on_enqueue ) {
        tp->on_enqueue(tp, tp->on_enqueue_data);
    }

    if( NULL != tp->startup_hook ) {
        parsec_task_t **startup_list;
        int vpid;

        /* These pointers need to be initialized to NULL */
        startup_list = (parsec_task_t**)alloca( context->nb_vp * sizeof(parsec_task_t*) );
        for(vpid = 0; vpid < context->nb_vp; startup_list[vpid++] = NULL);

        tp->startup_hook(context, tp, startup_list);

        for(vpid = 0; vpid < context->nb_vp; vpid++) {
            if( NULL == startup_list[vpid] )
                continue;

            /* The tasks are ordered by priority, so just make them available */
            __parsec_schedule(context->virtual_processes[vpid]->execution_streams[0],
                              startup_list[vpid], 0);
        }
    } else {
        parsec_check_complete_cb(tp, context, tp->nb_pending_actions);
    }

    return 0;
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
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads have
 * been activated.
 */
int parsec_context_start( parsec_context_t* context )
{
    /* Context already active */
    if( PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags )
        return -1;
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
    return -1;  /* Someone else start it up */
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
        return -1;
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
        return -1;
    }

    ret = __parsec_context_wait( context->virtual_processes[0]->execution_streams[0] );

    context->__parsec_internal_finalization_counter++;
    (void)parsec_remote_dep_off(context);
    assert(context->flags & PARSEC_CONTEXT_FLAG_COMM_ACTIVE);
    assert(context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    context->flags ^= (PARSEC_CONTEXT_FLAG_COMM_ACTIVE | PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    return ret;
}
