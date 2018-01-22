/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
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

#include "parsec/debug_marks.h"
#include "parsec/ayudame.h"
#include "parsec/constants.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/class/list.h"

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

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
#define PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER 65536
typedef struct {
    int      thread_id;
    int      vp_id;
    int32_t  priority;
    uint32_t step;
} sched_priority_trace_t;
static sched_priority_trace_t sched_priority_trace[PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER];
static uint32_t sched_priority_trace_counter;
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

/* Increases the number of runtime associated activities (decreases if
 *   nb_tasks is negative). When this counter reaches zero the taskpool
 *   is considered as completed, and all resources will be marked for
 *   release.
 */
int parsec_taskpool_update_runtime_nbtask(parsec_taskpool_t *tp, int32_t nb_tasks)
{
    int remaining;

    assert( tp->nb_pending_actions != 0 );
    remaining = tp->update_nb_runtime_task( tp, nb_tasks );
    assert( 0<= remaining );
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
        (void)parsec_atomic_dec_32b( &(context->active_taskpools) );
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
                      parsec_task_t* new_context,
                      int32_t distance)
{
    int ret;

#if defined(PARSEC_DEBUG_PARANOID) && defined(PARSEC_DEBUG_NOISIER)
    {
        parsec_task_t* context = new_context;
        const struct parsec_flow_s* flow;
        int set_parameters, i;
        char tmp[MAX_TASK_STRLEN];

        do {
            for( i = set_parameters = 0; NULL != (flow = context->task_class->in[i]); i++ ) {
                if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) continue;
                if( NULL != context->data[flow->flow_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[flow->flow_index].data_in ) {
                        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Task %s has flow %s data_repo != NULL but a data == NULL (%s:%d)",
                                parsec_task_snprintf(tmp, MAX_TASK_STRLEN, context),
                                flow->name, __FILE__, __LINE__);
                    }
                }
            }
            /*if( set_parameters > 1 ) {
                parsec_fatal( "Task %s has more than one input flow set (impossible)!! (%s:%d)",
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, context), __FILE__, __LINE__);
            }*/ /* Change it as soon as dtd has a running version */
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "thread %d of VP %d Schedules %s (distance %d)",
                    es->th_id, es->virtual_process->vp_id,
                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, context), distance );
            context = (parsec_task_t*)context->super.list_next;
        } while ( context != new_context );
    }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME(es->es_profile, schedule_push_begin, 0);
     */
    ret = current_scheduler->module.schedule(es, new_context, distance);
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
        if( parsec_atomic_cas_32b((uint32_t*)&tp->nb_tasks, 0, PARSEC_RUNTIME_RESERVED_NB_TASKS) )
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

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
    {
        uint32_t my_idx = parsec_atomic_inc_32b(&sched_priority_trace_counter);
        if(my_idx < PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
            sched_priority_trace[my_idx].step      = es->sched_nb_tasks_done++;
            sched_priority_trace[my_idx].thread_id = es->th_id;
            sched_priority_trace[my_idx].vp_id     = es->virtual_process->vp_id;
            sched_priority_trace[my_idx].priority  = task->priority;
        }
    }
#endif

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
    int rc, nbiterations = 0, distance;
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

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
    parsec_inform("#Scheduling: th <%3d/%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu",
            es->th_id, es->virtual_process->vp_id, nbiterations, (long long unsigned int)found_local,
            (long long unsigned int)found_remote,
            (long long unsigned int)found_victim,
            (long long unsigned int)miss_local,
            (long long unsigned int)miss_victim );

    if( PARSEC_THREAD_IS_MASTER(es) ) {
        char  priority_trace_fname[64];
        FILE *priority_trace = NULL;
        sprintf(priority_trace_fname, "priority_trace-%d.dat", es->virtual_process->parsec_context->my_rank);
        priority_trace = fopen(priority_trace_fname, "w");
        if( NULL != priority_trace ) {
            uint32_t my_idx;
            fprintf(priority_trace,
                    "#Step\tPriority\tThread\tVP\n"
                    "#Tasks are ordered in execution order\n");
            for(my_idx = 0; my_idx < MIN(sched_priority_trace_counter, PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER); my_idx++) {
                fprintf(priority_trace, "%d\t%d\t%d\t%d\n",
                        sched_priority_trace[my_idx].step, sched_priority_trace[my_idx].priority,
                        sched_priority_trace[my_idx].thread_id, sched_priority_trace[my_idx].vp_id);
            }
            fclose(priority_trace);
        }
    }
#endif  /* PARSEC_REPORT_STATISTICS */

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

static int parsec_composed_cb( parsec_taskpool_t* o, void* cbdata )
{
    parsec_taskpool_t* compound = (parsec_taskpool_t*)cbdata;
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound->task_classes_array;
    int completed_taskpools = compound_state->completed_taskpools++;
    assert( o == compound_state->taskpools_array[completed_taskpools] ); (void)o;
    if( compound->nb_pending_actions-- ) {
        assert( NULL != compound_state->taskpools_array[completed_taskpools+1] );
        parsec_enqueue(compound_state->ctx,
                      compound_state->taskpools_array[completed_taskpools+1]);
    }
    return 0;
}

static void parsec_compound_startup( parsec_context_t *context,
                                    parsec_taskpool_t *compound_tp,
                                    parsec_task_t** startup_list)
{
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound_tp->task_classes_array;
    parsec_taskpool_t* first = compound_state->taskpools_array[0];
    int i;

    assert( NULL != first );
    first->startup_hook(context, first, startup_list);
    compound_state->ctx = context;
    compound_tp->nb_pending_actions = compound_state->nb_taskpools;
    for( i = 0; i < compound_state->nb_taskpools; i++ ) {
        parsec_taskpool_t* o = compound_state->taskpools_array[i];
        assert( NULL != o );
        o->on_complete      = parsec_composed_cb;
        o->on_complete_data = compound_tp;
    }
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
            compound_state = realloc(compound_state, sizeof(parsec_compound_state_t) +
                            (1 + compound_state->nb_taskpools / 16) * 16 * sizeof(void*));
            compound->task_classes_array = (void*)compound_state;
        }
        compound_state->taskpools_array[compound_state->nb_taskpools] = NULL;
    } else {
        compound = calloc(1, sizeof(parsec_taskpool_t));
        compound->taskpool_type      = PARSEC_TASKPOOL_TYPE_COMPOUND;
        compound->task_classes_array = malloc(sizeof(parsec_compound_state_t) + 16 * sizeof(void*));

        compound_state = (parsec_compound_state_t*)compound->task_classes_array;
        compound_state->taskpools_array[0] = start;
        compound_state->taskpools_array[1] = next;
        compound_state->taskpools_array[2] = NULL;
        compound_state->completed_taskpools = 0;
        compound->startup_hook = parsec_compound_startup;
    }
    return compound;
}
/* END: Composition */

int parsec_enqueue( parsec_context_t* context, parsec_taskpool_t* tp )
{
    if( NULL == current_scheduler) {
        parsec_set_scheduler( context );
    }

    tp->context = context;  /* save the context */

    PINS_TASKPOOL_INIT(tp);  /* PINS taskpool initialization */

    /* Update the number of pending taskpools */
    (void)parsec_atomic_inc_32b( &(context->active_taskpools) );

    /* If necessary trigger the on_enqueue callback */
    if( NULL != tp->on_enqueue ) {
        tp->on_enqueue(tp, tp->on_enqueue_data);
    }

    if( NULL != tp->startup_hook ) {
        parsec_task_t **startup_list;
        int p;
        /* These pointers need to be initialized to NULL; doing it with calloc */
        startup_list = (parsec_task_t**)calloc( vpmap_get_nb_vp(), sizeof(parsec_task_t*) );
        if( NULL == startup_list ) {  /* bad bad */
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        tp->startup_hook(context, tp, startup_list);
        for(p = 0; p < vpmap_get_nb_vp(); p++) {
            if( NULL != startup_list[p] ) {
                parsec_list_t temp;

                OBJ_CONSTRUCT( &temp, parsec_list_t );
                /* Order the tasks by priority */
                parsec_list_chain_sorted(&temp, (parsec_list_item_t*)startup_list[p],
                                        parsec_execution_context_priority_comparator);
                startup_list[p] = (parsec_task_t*)parsec_list_nolock_unchain(&temp);
                OBJ_DESTRUCT(&temp);
                /* We should add these tasks on the system queue when there is one */
                __parsec_schedule(context->virtual_processes[p]->execution_streams[0],
                                  startup_list[p], 0);
            }
        }
        free(startup_list);
    } else {
        parsec_check_complete_cb(tp, context, tp->nb_pending_actions);
    }
#if defined(PARSEC_SCHED_REPORT_STATISTICS)
    sched_priority_trace_counter = 0;
#endif

    return 0;
}

static inline int
__parsec_context_cas_or_flag(parsec_context_t* context,
                            uint32_t flags)
{
    uint32_t current_flags = context->flags;
    /* if the flags are already set don't reset them */
    if( flags == (current_flags & flags) ) return 0;
    return parsec_atomic_cas_32b(&context->flags,
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
        (void)parsec_atomic_inc_32b( (uint32_t*)&(context->active_taskpools) );
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
    int active = parsec_atomic_dec_32b( &(context->active_taskpools) );
    if( active < 0 ) {
        parsec_warning("parsec_context_wait detected on a non-started context\n");
        /* put the context back on it's original state */
        (void)parsec_atomic_inc_32b( &(context->active_taskpools) );
        return -1;
    }

    /* Here we wait on all dtd taskpools registered with us */
    parsec_detach_all_dtd_taskpool_from_context( context );

    ret = __parsec_context_wait( context->virtual_processes[0]->execution_streams[0] );

    context->__parsec_internal_finalization_counter++;
    (void)parsec_remote_dep_off(context);
    assert(context->flags & PARSEC_CONTEXT_FLAG_COMM_ACTIVE);
    assert(context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    context->flags ^= (PARSEC_CONTEXT_FLAG_COMM_ACTIVE | PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    return ret;
}
