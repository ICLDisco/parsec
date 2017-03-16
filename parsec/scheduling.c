/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec_internal.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/profiling.h"
#include "datarepo.h"
#include "parsec/execution_unit.h"
#include "parsec/vpmap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/os-spec-timing.h"
#include "parsec/remote_dep.h"
#include "parsec/scheduling.h"

#include "parsec/debug_marks.h"
#include "parsec/ayudame.h"
#include "parsec/constants.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

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
#define TAKE_TIME(EU_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
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

static void parsec_rusage_per_eu(parsec_execution_unit_t* eu, bool print) {
    struct rusage current;
    getrusage(RUSAGE_THREAD, &current);
    if( print ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - eu->_eu_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - eu->_eu_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - eu->_eu_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - eu->_eu_rusage.ru_stime.tv_usec) / 1000000.0);

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
                , eu->virtual_process->vp_id, eu->th_id, eu->core_id, eu->socket_id,
                usr, sys, usr + sys,
                (current.ru_minflt  - eu->_eu_rusage.ru_minflt), (current.ru_majflt  - eu->_eu_rusage.ru_majflt),
                (current.ru_nswap   - eu->_eu_rusage.ru_nswap) , (current.ru_nvcsw   - eu->_eu_rusage.ru_nvcsw),
                (current.ru_inblock - eu->_eu_rusage.ru_inblock), (current.ru_oublock - eu->_eu_rusage.ru_oublock),
                current.ru_maxrss);

    }
    eu->_eu_rusage = current;
    return;
}
#define parsec_rusage_per_eu(eu, b) do { if(parsec_want_rusage > 1) parsec_rusage_per_eu(eu, b); } while(0)
#else
#define parsec_rusage_per_eu(eu, b) do {} while(0)
#endif /* defined(PARSEC_HAVE_GETRUSAGE) defined(PARSEC_PROF_RUSAGE_EU) */

#if 0
/*
 * Disabled by now.
 */
int __parsec_context_wait_task( parsec_execution_unit_t* eu_context,
                           parsec_execution_context_t* task )
{
    (void)eu_context;
    switch(task->status) {
        case PARSEC_TASK_STATUS_NONE:
#if defined(PARSEC_DEBUG)
            char tmp[MAX_TASK_STRLEN];
            parsec_degug_verbose(5, parsec_debug_output, "thread %d of VP %d Execute %s\n", eu_context->th_id, eu_context->virtual_process->vp_id,
                   parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, task));
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

int __parsec_execute( parsec_execution_unit_t* eu_context,
                     parsec_execution_context_t* exec_context )
{
    const parsec_function_t* function = exec_context->function;
    int rc;
#if defined(PARSEC_DEBUG)
    char tmp[MAX_TASK_STRLEN];
    parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif
    AYU_TASK_RUN(eu_context->th_id, exec_context);

    if (NULL == function->incarnations[exec_context->chore_id].hook) {
#if !defined(PARSEC_DEBUG)
        char tmp[MAX_TASK_STRLEN];
        parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif
        parsec_warning("Task %s[%d] run out of valid incarnations. Consider it complete",
                       tmp, function->incarnations[exec_context->chore_id].type);
        return PARSEC_HOOK_RETURN_ERROR;
    }

    PINS(eu_context, EXEC_BEGIN, exec_context);
    /* Try all the incarnations until one agree to execute. */
    do {
#if defined(PARSEC_DEBUG)
        parsec_debug_verbose(5, parsec_debug_output, "Thread %d of VP %d Execute %s[%d] chore %d",
                            eu_context->th_id, eu_context->virtual_process->vp_id,
                            tmp, function->incarnations[exec_context->chore_id].type,
                            exec_context->chore_id);
#endif
        parsec_hook_t *hook = function->incarnations[exec_context->chore_id].hook;

        rc = hook( eu_context, exec_context );
        if( PARSEC_HOOK_RETURN_NEXT != rc ) {
            PINS(eu_context, EXEC_END, exec_context);
            if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
                /* Let's assume everything goes just fine */
                exec_context->status = PARSEC_TASK_STATUS_COMPLETE;
            }
            return rc;
        }
        exec_context->chore_id++;

    } while(NULL != function->incarnations[exec_context->chore_id].hook);
    assert(exec_context->status == PARSEC_TASK_STATUS_HOOK);
    /* We're out of luck, no more chores */
    PINS(eu_context, EXEC_END, exec_context);
    return PARSEC_HOOK_RETURN_ERROR;
}

/* Increases the number of runtime associated activities (decreases if
 *   nb_tasks is negative). When this counter reaches zero the handle is
 *   considered as completed, and all resources will be marked for
 *   release.
 */
int parsec_handle_update_runtime_nbtask(parsec_handle_t *handle, int32_t nb_tasks)
{
    int remaining;

    assert( handle->nb_pending_actions != 0 );
    remaining = handle->update_nb_runtime_task( handle, nb_tasks );
    assert( 0<= remaining );
    return parsec_check_complete_cb(handle, handle->context, remaining);
}

static inline int all_tasks_done(parsec_context_t* context)
{
    return (context->active_objects == 0);
}

int parsec_check_complete_cb(parsec_handle_t *parsec_handle, parsec_context_t *context, int remaining)
{
   if( 0 == remaining ) {
        /* A parsec object has been completed. Call the attached callback if
         * necessary, then update the main engine.
         */
        if( NULL != parsec_handle->on_complete ) {
            (void)parsec_handle->on_complete( parsec_handle, parsec_handle->on_complete_data );
        }
        (void)parsec_atomic_dec_32b( &(context->active_objects) );
        PINS_HANDLE_FINI(parsec_handle);
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
int __parsec_schedule( parsec_execution_unit_t* eu_context,
                      parsec_execution_context_t* new_context,
                      int32_t distance)
{
    int ret;

#if defined(PARSEC_DEBUG_PARANOID) && defined(PARSEC_DEBUG_NOISIER)
    {
        parsec_execution_context_t* context = new_context;
        const struct parsec_flow_s* flow;
        int set_parameters, i;
        char tmp[MAX_TASK_STRLEN];

        do {
            for( i = set_parameters = 0; NULL != (flow = context->function->in[i]); i++ ) {
                if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) continue;
                if( NULL != context->data[flow->flow_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[flow->flow_index].data_in ) {
                        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Task %s has flow %s data_repo != NULL but a data == NULL (%s:%d)",
                                parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context),
                                flow->name, __FILE__, __LINE__);
                    }
                }
            }
            /*if( set_parameters > 1 ) {
                parsec_fatal( "Task %s has more than one input flow set (impossible)!! (%s:%d)",
                        parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context), __FILE__, __LINE__);
            }*/ /* Change it as soon as dtd has a running version */
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,  "thread %d of VP %d Schedules %s (distance %d)",
                    eu_context->th_id, eu_context->virtual_process->vp_id,
                    parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context), distance );
            context = (parsec_execution_context_t*)context->super.list_item.list_next;
        } while ( context != new_context );
    }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);
     */
    ret = current_scheduler->module.schedule(eu_context, new_context, distance);
    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME( eu_context->eu_profile, schedule_push_end, 0);
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
 * @param [IN] eu_context, the start execution_unit (normall it is the current one).
 * @param [IN] task, the task to be rescheduled.
 *
 * @return parsec scheduling return code
 */
int __parsec_reschedule(parsec_execution_unit_t* eu_context, parsec_execution_context_t* task)
{
    parsec_context_t* context = eu_context->virtual_process->parsec_context;
    parsec_vp_t* vp_context = eu_context->virtual_process;

    int vp, start_vp = vp_context->vp_id, next_vp;
    int eu, start_eu = (eu_context->th_id + 1) % context->virtual_processes[start_vp]->nb_cores, next_eu;

    for( vp = start_vp, next_vp = (start_vp + 1) % context->nb_vp;
         next_vp != vp_context->vp_id;
         ++vp) {
        if( 1 != context->virtual_processes[vp]->nb_cores ) {
            for( eu = start_eu, next_eu = (start_eu + 1) % context->virtual_processes[vp]->nb_cores;
                 next_eu != start_eu;
                 ++eu ) {
                return __parsec_schedule(context->virtual_processes[vp]->execution_units[eu], task, 0);
            }
        }
        else if ( context->virtual_processes[vp]->vp_id != vp_context->vp_id ) {
            /* VP contains only one EU, and it's not my VP, so not me */
            return __parsec_schedule(context->virtual_processes[vp]->execution_units[0], task, 0);
        }
        start_eu = 0;  /* with the exception of the first eu_context, we always iterate from 0 */
    }
    /* no luck so far, let's reschedule the task on the same execution unit */
    return __parsec_schedule(eu_context, task, 0);
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

int __parsec_complete_execution( parsec_execution_unit_t *eu_context,
                                parsec_execution_context_t *exec_context )
{
    parsec_handle_t *handle = exec_context->parsec_handle;
    int rc = 0;

    /* complete execution PINS event includes the preparation of the
     * output and the and the call to complete_execution.
     */
    PINS(eu_context, COMPLETE_EXEC_BEGIN, exec_context);

    if( NULL != exec_context->function->prepare_output ) {
        exec_context->function->prepare_output( eu_context, exec_context );
    }
    if( NULL != exec_context->function->complete_execution )
        rc = exec_context->function->complete_execution( eu_context, exec_context );

    PINS(eu_context, COMPLETE_EXEC_END, exec_context);
    AYU_TASK_COMPLETE(exec_context);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( eu_context->th_id, eu_context->virtual_process->vp_id, exec_context );

    /* Release the execution context */
    exec_context->function->release_task( eu_context, exec_context );

    /* Check to see if the DSL has marked the handle as completed */
    if( 0 == handle->nb_tasks ) {
        /* The handle has been marked as complete. Unfortunately, it is possible
         * that multiple threads are completing tasks associated with this handle
         * simultaneously and we need to release the runtime action associated with
         * this handle tasks once. We need to protect this action by atomically
         * setting the number of tasks to a non-zero value.
         */
        if( parsec_atomic_cas_32b((uint32_t*)&handle->nb_tasks, 0, PARSEC_RUNTIME_RESERVED_NB_TASKS) )
            parsec_handle_update_runtime_nbtask(handle, -1);
    }

    return rc;
}

int __parsec_context_wait( parsec_execution_unit_t* eu_context )
{
    uint64_t misses_in_a_row;
    parsec_context_t* parsec_context = eu_context->virtual_process->parsec_context;
    int32_t my_barrier_counter = parsec_context->__parsec_internal_finalization_counter;
    parsec_execution_context_t* exec_context;
    int rc, nbiterations = 0, distance;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    if( !PARSEC_THREAD_IS_MASTER(eu_context) ) {
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

    parsec_rusage_per_eu(eu_context, false);

    /* first select begin, right before the wait_for_the... goto label */
    PINS(eu_context, SELECT_BEGIN, NULL);

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
            (eu_context->virtual_process[0].parsec_context->nb_nodes == 1) &&
            PARSEC_THREAD_IS_MASTER(eu_context) ) {
            /* check for remote deps completion */
            while(parsec_remote_dep_progress(eu_context) > 0)  {
                misses_in_a_row = 0;
            }
        }
#endif /* defined(DISTRIBUTED) */

        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }

        exec_context = current_scheduler->module.select(eu_context, &distance);

        if( exec_context != NULL ) {
            PINS(eu_context, SELECT_END, exec_context);
            misses_in_a_row = 0;

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
            {
                uint32_t my_idx = parsec_atomic_inc_32b(&sched_priority_trace_counter);
                if(my_idx < PARSEC_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
                    sched_priority_trace[my_idx].step = eu_context->sched_nb_tasks_done++;
                    sched_priority_trace[my_idx].thread_id = eu_context->th_id;
                    sched_priority_trace[my_idx].vp_id     = eu_context->virtual_process->vp_id;
                    sched_priority_trace[my_idx].priority  = exec_context->priority;
                }
            }
#endif

            rc = PARSEC_HOOK_RETURN_DONE;
            if(exec_context->status <= PARSEC_TASK_STATUS_PREPARE_INPUT) {
                PINS(eu_context, PREPARE_INPUT_BEGIN, exec_context);
                rc = exec_context->function->prepare_input(eu_context, exec_context);
                PINS(eu_context, PREPARE_INPUT_END, exec_context);
            }
            switch(rc) {
            case PARSEC_HOOK_RETURN_DONE: {
                if(exec_context->status <= PARSEC_TASK_STATUS_HOOK) {
                    rc = __parsec_execute( eu_context, exec_context );
                }
                /* We're good to go ... */
                switch(rc) {
                case PARSEC_HOOK_RETURN_DONE:    /* This execution succeeded */
                    exec_context->status = PARSEC_TASK_STATUS_COMPLETE;
                    __parsec_complete_execution( eu_context, exec_context );
                    break;
                case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
                    exec_context->status = PARSEC_TASK_STATUS_HOOK;
                    if(0 == exec_context->priority) {
                        SET_LOWEST_PRIORITY(exec_context, parsec_execution_context_priority_comparator);
                    } else
                        exec_context->priority /= 10;  /* demote the task */
                    PARSEC_LIST_ITEM_SINGLETON(exec_context);
                    __parsec_schedule(eu_context, exec_context, distance + 1);
                    exec_context = NULL;
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
                nbiterations++;
                break;
            }
            case PARSEC_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                             * even try to change it's state, the completion
                                             * will be triggered asynchronously. */
                break;
            case PARSEC_HOOK_RETURN_AGAIN:   /* Reschedule later */
                if(0 == exec_context->priority) {
                    SET_LOWEST_PRIORITY(exec_context, parsec_execution_context_priority_comparator);
                } else
                    exec_context->priority /= 10;  /* demote the task */
                PARSEC_LIST_ITEM_SINGLETON(exec_context);
                __parsec_schedule(eu_context, exec_context, distance + 1);
                exec_context = NULL;
                break;
            default:
                assert( 0 ); /* Internal error: invalid return value for data_lookup function */
            }

            // subsequent select begins
            PINS(eu_context, SELECT_BEGIN, NULL);
        } else {
            misses_in_a_row++;
        }
    }

    parsec_rusage_per_eu(eu_context, true);

    /* We're all done ? */
    parsec_barrier_wait( &(parsec_context->barrier) );

#if defined(PARSEC_SIM)
    if( PARSEC_THREAD_IS_MASTER(eu_context) ) {
        parsec_vp_t *vp;
        int32_t my_vpid, my_idx;
        int largest_date = 0;
        for(my_vpid = 0; my_vpid < parsec_context->nb_vp; my_vpid++) {
            vp = parsec_context->virtual_processes[my_vpid];
            for(my_idx = 0; my_idx < vp->nb_cores; my_idx++) {
                if( vp->execution_units[my_idx]->largest_simulation_date > largest_date )
                    largest_date = vp->execution_units[my_idx]->largest_simulation_date;
            }
        }
        parsec_context->largest_simulation_date = largest_date;
    }
    parsec_barrier_wait( &(parsec_context->barrier) );
    eu_context->largest_simulation_date = 0;
#endif

    if( !PARSEC_THREAD_IS_MASTER(eu_context) ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
    // final select end - can we mark this as special somehow?
    // actually, it will already be obviously special, since it will be the only select
    // that has no context
    PINS(eu_context, SELECT_END, NULL);

#if defined(PARSEC_SCHED_REPORT_STATISTICS)
    parsec_inform("#Scheduling: th <%3d/%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu",
            eu_context->th_id, eu_context->virtual_process->vp_id, nbiterations, (long long unsigned int)found_local,
            (long long unsigned int)found_remote,
            (long long unsigned int)found_victim,
            (long long unsigned int)miss_local,
            (long long unsigned int)miss_victim );

    if( PARSEC_THREAD_IS_MASTER(eu_context) ) {
        char  priority_trace_fname[64];
        FILE *priority_trace = NULL;
        sprintf(priority_trace_fname, "priority_trace-%d.dat", eu_context->virtual_process->parsec_context->my_rank);
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
        PINS_THREAD_FINI(eu_context);
    }
    return nbiterations;
}

/*  *********** COMPOSITION OF PARSEC_OBJECTS ***************  */
typedef struct parsec_compound_state_t {
    parsec_context_t* ctx;
    int nb_objects;
    int completed_objects;
    parsec_handle_t* objects_array[1];
} parsec_compound_state_t;

static int parsec_composed_cb( parsec_handle_t* o, void* cbdata )
{
    parsec_handle_t* compound = (parsec_handle_t*)cbdata;
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound->functions_array;
    int completed_objects = compound_state->completed_objects++;
    assert( o == compound_state->objects_array[completed_objects] ); (void)o;
    if( compound->nb_pending_actions-- ) {
        assert( NULL != compound_state->objects_array[completed_objects+1] );
        parsec_enqueue(compound_state->ctx,
                      compound_state->objects_array[completed_objects+1]);
    }
    return 0;
}

static void parsec_compound_startup( parsec_context_t *context,
                                    parsec_handle_t *compound_object,
                                    parsec_execution_context_t** startup_list)
{
    parsec_compound_state_t* compound_state = (parsec_compound_state_t*)compound_object->functions_array;
    parsec_handle_t* first = compound_state->objects_array[0];
    int i;

    assert( 0 == compound_object->nb_functions );
    assert( NULL != first );
    first->startup_hook(context, first, startup_list);
    compound_state->ctx = context;
    compound_object->nb_pending_actions = compound_state->nb_objects;
    for( i = 0; i < compound_state->nb_objects; i++ ) {
        parsec_handle_t* o = compound_state->objects_array[i];
        assert( NULL != o );
        o->on_complete      = parsec_composed_cb;
        o->on_complete_data = compound_object;
    }
}

parsec_handle_t* parsec_compose( parsec_handle_t* start,
                               parsec_handle_t* next )
{
    parsec_handle_t* compound = NULL;
    parsec_compound_state_t* compound_state = NULL;

    if( start->nb_functions == 0 ) {  /* start is already a compound object */
        compound = start;
        compound_state = (parsec_compound_state_t*)compound->functions_array;
        compound_state->objects_array[compound_state->nb_objects++] = next;
        /* make room for NULL terminating, if necessary */
        if( 0 == (compound_state->nb_objects % 16) ) {
            compound_state = realloc(compound_state, sizeof(parsec_compound_state_t) +
                            (1 + compound_state->nb_objects / 16) * 16 * sizeof(void*));
            compound->functions_array = (void*)compound_state;
        }
        compound_state->objects_array[compound_state->nb_objects] = NULL;
    } else {
        compound = calloc(1, sizeof(parsec_handle_t));
        compound->functions_array = malloc(sizeof(parsec_compound_state_t) + 16 * sizeof(void*));
        compound_state = (parsec_compound_state_t*)compound->functions_array;
        compound_state->objects_array[0] = start;
        compound_state->objects_array[1] = next;
        compound_state->objects_array[2] = NULL;
        compound_state->completed_objects = 0;
        compound_state->nb_objects = 2;
        compound->startup_hook = parsec_compound_startup;
    }
    return compound;
}
/* END: Composition */

int32_t parsec_set_priority( parsec_handle_t* handle, int32_t new_priority )
{
    int32_t old_priority = handle->priority;
    handle->priority = new_priority;
    return old_priority;
}

int parsec_enqueue( parsec_context_t* context, parsec_handle_t* handle )
{
    if( NULL == current_scheduler) {
        parsec_set_scheduler( context );
    }

    handle->context = context;  /* save the context */

    PINS_HANDLE_INIT(handle);  /* PINS handle initialization */

    /* Update the number of pending objects */
    (void)parsec_atomic_inc_32b( &(context->active_objects) );

    /* If necessary trigger the on_enqueue callback */
    if( NULL != handle->on_enqueue ) {
        handle->on_enqueue(handle, handle->on_enqueue_data);
    }

    if( NULL != handle->startup_hook ) {
        parsec_execution_context_t **startup_list;
        int p;
        /* These pointers need to be initialized to NULL; doing it with calloc */
        startup_list = (parsec_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(parsec_execution_context_t*) );
        if( NULL == startup_list ) {  /* bad bad */
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        handle->startup_hook(context, handle, startup_list);
        for(p = 0; p < vpmap_get_nb_vp(); p++) {
            if( NULL != startup_list[p] ) {
                parsec_list_t temp;

                OBJ_CONSTRUCT( &temp, parsec_list_t );
                /* Order the tasks by priority */
                parsec_list_chain_sorted(&temp, (parsec_list_item_t*)startup_list[p],
                                        parsec_execution_context_priority_comparator);
                startup_list[p] = (parsec_execution_context_t*)parsec_list_nolock_unchain(&temp);
                OBJ_DESTRUCT(&temp);
                /* We should add these tasks on the system queue when there is one */
                __parsec_schedule(context->virtual_processes[p]->execution_units[0],
                                  startup_list[p], 0);
            }
        }
        free(startup_list);
    } else {
        parsec_check_complete_cb(handle, context, handle->nb_pending_actions);
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
 * If there are enqueued handles waiting to be executed launch the other threads
 * and then return. Mark the internal structures in such a way that we can't
 * start the context mutiple times without completions.
 *
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads have
 * been activated.
 */
int parsec_context_start( parsec_context_t* context )
{
    /* No active work */
    if(all_tasks_done(context)) return -2;
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
        (void)parsec_atomic_inc_32b( (uint32_t*)&(context->active_objects) );
        return 0;
    }
    return -1;  /* Someone else start it up */
}

int parsec_context_test( parsec_context_t* context )
{
    return !all_tasks_done(context);
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
    /* Remove the additional active_object to signal the runtime that we
     * are ready to complete a scheduling epoch.
     */
    int active = parsec_atomic_dec_32b( &(context->active_objects) );
    if( active < 0 ) {
        parsec_warning("parsec_context_wait detected on a non-started context\n");
        /* put the context back on it's original state */
        (void)parsec_atomic_inc_32b( &(context->active_objects) );
        return -1;
    }

    /* Here we wait on all dtd handles registered with us */
    parsec_detach_all_dtd_handles_from_context( context );

    ret = __parsec_context_wait( context->virtual_processes[0]->execution_units[0] );

    context->__parsec_internal_finalization_counter++;
    (void)parsec_remote_dep_off(context);
    assert(context->flags & PARSEC_CONTEXT_FLAG_COMM_ACTIVE);
    assert(context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    context->flags ^= (PARSEC_CONTEXT_FLAG_COMM_ACTIVE | PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
    return ret;
}
