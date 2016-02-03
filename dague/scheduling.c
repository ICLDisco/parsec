/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/mca/mca_repository.h"
#include "dague/mca/sched/sched.h"
#include "dague/profiling.h"
#include "datarepo.h"
#include "dague/execution_unit.h"
#include "dague/vpmap.h"
#include "dague/mca/pins/pins.h"
#include "dague/os-spec-timing.h"
#include "dague/remote_dep.h"

#include "dague/debug_marks.h"
#include "dague/ayudame.h"
#include "dague/constants.h"

#include <signal.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif /* defined(HAVE_STRING_H) */
#include <sched.h>
#include <sys/types.h>
#if defined(HAVE_ERRNO_H)
#include <errno.h>
#endif  /* defined(HAVE_ERRNO_H) */
#if defined(HAVE_SCHED_SETAFFINITY)
#include <linux/unistd.h>
#endif  /* defined(HAVE_SCHED_SETAFFINITY) */
#if defined(DAGUE_PROF_TRACE) && defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS)
#define TAKE_TIME(EU_PROFILE, KEY, ID)  DAGUE_PROFILING_TRACE((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
#define DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER 65536
typedef struct {
    int      thread_id;
    int      vp_id;
    int32_t  priority;
    uint32_t step;
} sched_priority_trace_t;
static sched_priority_trace_t sched_priority_trace[DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER];
static uint32_t sched_priority_trace_counter;
#endif


#if defined(DAGUE_PROF_RUSAGE_EU)

#if defined(HAVE_GETRUSAGE) && defined(HAVE_RUSAGE_THREAD)
#include <sys/time.h>
#include <sys/resource.h>

static void dague_statistics_per_eu(char* str, dague_execution_unit_t* eu)
{

    struct rusage current;
    getrusage(RUSAGE_THREAD, &current);
    if( !eu->_eu_rusage_first_call ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - eu->_eu_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - eu->_eu_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - eu->_eu_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - eu->_eu_rusage.ru_stime.tv_usec) / 1000000.0);

        dague_inform(("\n=============================================================\n"
                "%s: Resource Usage Data for Exec. Unit ...\n"
                "VP: %i Thread: %i (Core %i, socket %i)\n"
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
                "=============================================================\n"
                , str, eu->virtual_process->vp_id, eu->th_id, eu->core_id, eu->socket_id,
                usr, sys, usr + sys,
                (current.ru_minflt  - eu->_eu_rusage.ru_minflt), (current.ru_majflt  - eu->_eu_rusage.ru_majflt),
                (current.ru_nswap   - eu->_eu_rusage.ru_nswap) , (current.ru_nvcsw   - eu->_eu_rusage.ru_nvcsw),
                (current.ru_inblock - eu->_eu_rusage.ru_inblock), (current.ru_oublock - eu->_eu_rusage.ru_oublock)));

    }
    eu->_eu_rusage_first_call = !eu->_eu_rusage_first_call;
    eu->_eu_rusage = current;
    return;
}
#else
static void dague_statistics_per_eu(char* str, dague_execution_unit_t* eu) { (void)str; (void)eu; return; }
#endif /* defined(HAVE_GETRUSAGE) */
#endif /* defined(DAGUE_PROF_RUSAGE_EU) */

#if 0
/**
 * Disabled by now.
 */
int __dague_context_wait_task( dague_execution_unit_t* eu_context,
                           dague_execution_context_t* task )
{
    (void)eu_context;
    switch(task->status) {
        case DAGUE_TASK_STATUS_NONE:
#if defined(DAGUE_DEBUG)
            char tmp[MAX_TASK_STRLEN];
            dague_degug_verbose(5, dague_debug_output, "thread %d of VP %d Execute %s\n", eu_context->th_id, eu_context->virtual_process->vp_id,
                   dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, task));
#endif
        return -1;

        case DAGUE_TASK_STATUS_PREPARE_INPUT:
            task->status = DAGUE_TASK_STATUS_EVAL;
            break;
        case DAGUE_TASK_STATUS_EVAL:
            task->status = DAGUE_TASK_STATUS_HOOK;
            break;
        case DAGUE_TASK_STATUS_HOOK:
            task->status = DAGUE_TASK_STATUS_PREPARE_OUTPUT;
            break;
        case DAGUE_TASK_STATUS_PREPARE_OUTPUT:
            task->status = DAGUE_TASK_STATUS_COMPLETE;
            break;
        case DAGUE_TASK_STATUS_COMPLETE:
            break;
    }
    return -1;
}
#endif

int __dague_execute( dague_execution_unit_t* eu_context,
                     dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int rc;
#if defined(DAGUE_DEBUG)
    char tmp[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif

    PINS(eu_context, EXEC_BEGIN, exec_context);
    AYU_TASK_RUN(eu_context->th_id, exec_context);
    /* Let's assume everything goes just fine */
    exec_context->status = DAGUE_TASK_STATUS_COMPLETE;
    /* Try all the incarnations until one agree to execute. */
    do {
#if defined(DAGUE_DEBUG)
        dague_debug_verbose(5, dague_debug_output, "Thread %d of VP %d Execute %s[%d]",
               eu_context->th_id, eu_context->virtual_process->vp_id,
               tmp, function->incarnations[exec_context->chore_id].type);
#endif
        rc = function->incarnations[exec_context->chore_id].hook( eu_context, exec_context );
        if( DAGUE_HOOK_RETURN_NEXT != rc ) {
            PINS(eu_context, EXEC_END, exec_context);
            return rc;
        }
        exec_context->chore_id++;
    } while(NULL != function->incarnations[exec_context->chore_id].hook);
    /* We failed to execute. Give it another chance ... */
    exec_context->status = DAGUE_TASK_STATUS_HOOK;
    /* We're out of luck, no more chores */
    PINS(eu_context, EXEC_END, exec_context);
    return DAGUE_HOOK_RETURN_ERROR;
}

static inline int all_tasks_done(dague_context_t* context)
{
    return (context->active_objects == 0);
}

int dague_check_complete_cb(dague_handle_t *dague_handle, dague_context_t *context, int remaining)
{
   if( 0 == remaining ) {
        /* A dague object has been completed. Call the attached callback if
         * necessary, then update the main engine.
         */
        if( NULL != dague_handle->on_complete ) {
            (void)dague_handle->on_complete( dague_handle, dague_handle->on_complete_data );
        }
        dague_atomic_dec_32b( &(context->active_objects) );
        PINS_HANDLE_FINI(dague_handle);
        return 1;
    }
    return 0;
}

dague_sched_module_t *current_scheduler                  = NULL;
static dague_sched_base_component_t *scheduler_component = NULL;

void dague_remove_scheduler( dague_context_t *dague )
{
    if( NULL != current_scheduler ) {
        current_scheduler->module.remove( dague );
        assert( NULL != scheduler_component );
        mca_component_close( (mca_base_component_t*)scheduler_component );
        current_scheduler = NULL;
        scheduler_component = NULL;
    }
}

int dague_set_scheduler( dague_context_t *dague )
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

    dague_remove_scheduler( dague );
    current_scheduler   = (dague_sched_module_t*)new_scheduler;
    scheduler_component = (dague_sched_base_component_t*)new_component;

    dague_debug_verbose(4, dague_debug_output, " Installing scheduler %s", current_scheduler->component->base_version.mca_component_name);
    PROFILING_SAVE_sINFO("sched", (char *)current_scheduler->component->base_version.mca_component_name);

    current_scheduler->module.install( dague );
    return 1;
}

/**
 * This is where we end up after the release_dep_fct is called and generates a
 * readylist. the new_context IS the readylist.
 */
int __dague_schedule( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* new_context )
{
    int ret;

#if defined(DAGUE_DEBUG_PARANOID) && defined(DAGUE_DEBUG_MOTORMOUTH)
    {
        dague_execution_context_t* context = new_context;
        const struct dague_flow_s* flow;
        int set_parameters, i;
        char tmp[MAX_TASK_STRLEN];

        do {
            for( i = set_parameters = 0; NULL != (flow = context->function->in[i]); i++ ) {
                if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) continue;
                if( NULL != context->data[flow->flow_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[flow->flow_index].data_in ) {
                        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Task %s has flow %s data_repo != NULL but a data == NULL (%s:%d)",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context),
                                flow->name, __FILE__, __LINE__);
                    }
                }
            }
            /*if( set_parameters > 1 ) {
                dague_abort( "Task %s has more than one input flow set (impossible)!! (%s:%d)",
                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context), __FILE__, __LINE__);
            }*/ /* Change it as soon as dtd has a running version */
            DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "thread %d of VP %d Schedules %s",
                    eu_context->th_id, eu_context->virtual_process->vp_id,
                    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context) );
            context = (dague_execution_context_t*)context->super.list_item.list_next;
        } while ( context != new_context );
    }
#endif  /* defined(DAGUE_DEBUG_PARANOID) */

    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);
     */
    ret = current_scheduler->module.schedule(eu_context, new_context);
    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME( eu_context->eu_profile, schedule_push_end, 0);
     */

    return ret;
}

#ifdef  HAVE_SCHED_SETAFFINITY
#define gettid() syscall(__NR_gettid)
#endif /* HAVE_SCHED_SETAFFINITY */

#define TIME_STEP 5410
#define MIN(x, y) ( (x)<(y)?(x):(y) )
static inline unsigned long exponential_backoff(uint64_t k)
{
    unsigned int n = MIN( 64, k );
    unsigned int r = (unsigned int) ((double)n * ((double)rand()/(double)RAND_MAX));
    return r * TIME_STEP;
}

int __dague_complete_execution( dague_execution_unit_t *eu_context,
                                dague_execution_context_t *exec_context )
{
    dague_handle_t *handle;
    int rc = 0;

    /* complete execution==add==push (also includes exec of immediates) */
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
    handle = exec_context->dague_handle;
    exec_context->function->release_task( eu_context, exec_context );

    /* Update the number of remaining tasks */
    (void)dague_handle_update_nbtask(handle, -1);

    return rc;
}

int __dague_context_wait( dague_execution_unit_t* eu_context )
{
    uint64_t misses_in_a_row;
    dague_context_t* dague_context = eu_context->virtual_process->dague_context;
    int32_t my_barrier_counter = dague_context->__dague_internal_finalization_counter;
    dague_execution_context_t* exec_context;
    int rc, nbiterations = 0;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) {
        /* Wait until all threads are done binding themselves
         * (see dague_init) */
        dague_barrier_wait( &(dague_context->barrier) );
        my_barrier_counter = 1;
    } else {
        /* The master thread might not have to trigger the barrier if the other
         * threads have been activated by a previous start.
         */
        if( DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE & dague_context->flags ) {
            goto skip_first_barrier;
        }
        dague_context->flags |= DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE;
    }

#if defined(DAGUE_PROF_RUSAGE_EU)
    dague_statistics_per_eu("EU", eu_context);
#endif
    /* first select begin, right before the wait_for_the... goto label */
    PINS(eu_context, SELECT_BEGIN, NULL);

    /* The main loop where all the threads will spend their time */
  wait_for_the_next_round:
    /* Wait until all threads are here and the main thread signal the begining of the work */
    dague_barrier_wait( &(dague_context->barrier) );

    if( dague_context->__dague_internal_finalization_in_progress ) {
        my_barrier_counter++;
        for(; my_barrier_counter <= dague_context->__dague_internal_finalization_counter; my_barrier_counter++ ) {
            dague_barrier_wait( &(dague_context->barrier) );
        }
        goto finalize_progress;
    }

    if( NULL == current_scheduler ) {
        fprintf(stderr, "DAGuE: Main thread entered dague_context_wait, while scheduler is not selected yet!\n");
        return -1;
    }

  skip_first_barrier:
    while( !all_tasks_done(dague_context) ) {
#if defined(DISTRIBUTED)
        if( (1 == dague_communication_engine_up) &&
            (eu_context->virtual_process[0].dague_context->nb_nodes == 1) &&
            DAGUE_THREAD_IS_MASTER(eu_context) ) {
            /* check for remote deps completion */
            while(dague_remote_dep_progress(eu_context) > 0)  {
                misses_in_a_row = 0;
            }
        }
#endif /* defined(DISTRIBUTED) */

        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            nanosleep(&rqtp, NULL);
        }

        exec_context = current_scheduler->module.select(eu_context);

        if( exec_context != NULL ) {
            PINS(eu_context, SELECT_END, exec_context);
            misses_in_a_row = 0;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
            {
                uint32_t my_idx = dague_atomic_inc_32b(&sched_priority_trace_counter);
                if(my_idx < DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
                    sched_priority_trace[my_idx].step = eu_context->sched_nb_tasks_done++;
                    sched_priority_trace[my_idx].thread_id = eu_context->th_id;
                    sched_priority_trace[my_idx].vp_id     = eu_context->virtual_process->vp_id;
                    sched_priority_trace[my_idx].priority  = exec_context->priority;
                }
            }
#endif

            rc = DAGUE_HOOK_RETURN_DONE;
            if(exec_context->status <= DAGUE_TASK_STATUS_PREPARE_INPUT) {
                PINS(eu_context, PREPARE_INPUT_BEGIN, exec_context);
                rc = exec_context->function->prepare_input(eu_context, exec_context);
                PINS(eu_context, PREPARE_INPUT_END, exec_context);
            }
            switch(rc) {
            case DAGUE_HOOK_RETURN_DONE: {
                if(exec_context->status <= DAGUE_TASK_STATUS_HOOK) {
                    rc = __dague_execute( eu_context, exec_context );
                }
                /* We're good to go ... */
                switch(rc) {
                case DAGUE_HOOK_RETURN_DONE:    /* This execution succeeded */
                    exec_context->status = DAGUE_TASK_STATUS_COMPLETE;
                    __dague_complete_execution( eu_context, exec_context );
                    break;
                case DAGUE_HOOK_RETURN_AGAIN:   /* Reschedule later */
                    exec_context->status = DAGUE_TASK_STATUS_HOOK;
                    if(0 == exec_context->priority) {
                        SET_LOWEST_PRIORITY(exec_context, dague_execution_context_priority_comparator);
                    } else
                        exec_context->priority /= 10;  /* demote the task */
                    DAGUE_LIST_ITEM_SINGLETON(exec_context);
                    __dague_schedule(eu_context, exec_context);
                    exec_context = NULL;
                    break;
                case DAGUE_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                                 * even try to change it's state, the completion
                                                 * will be triggered asynchronously. */
                    break;
                case DAGUE_HOOK_RETURN_NEXT:    /* Try next variant [if any] */
                case DAGUE_HOOK_RETURN_DISABLE: /* Disable the device, something went wrong */
                case DAGUE_HOOK_RETURN_ERROR:   /* Some other major error happened */
                    assert( 0 ); /* Internal error: invalid return value */
                }
                nbiterations++;
                break;
            }
            case DAGUE_HOOK_RETURN_ASYNC:   /* The task is outside our reach we should not
                                             * even try to change it's state, the completion
                                             * will be triggered asynchronously. */
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

#if defined(DAGUE_PROF_RUSAGE_EU)
    dague_statistics_per_eu("EU ", eu_context);
#endif

    /* We're all done ? */
    dague_barrier_wait( &(dague_context->barrier) );

#if defined(DAGUE_SIM)
    if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
        dague_vp_t *vp;
        int32_t my_vpid, my_idx;
        int largest_date = 0;
        for(my_vpid = 0; my_vpid < dague_context->nb_vp; my_vpid++) {
            vp = dague_context->virtual_processes[my_vpid];
            for(my_idx = 0; my_idx < vp->nb_cores; my_idx++) {
                if( vp->execution_units[my_idx]->largest_simulation_date > largest_date )
                    largest_date = vp->execution_units[my_idx]->largest_simulation_date;
            }
        }
        dague_context->largest_simulation_date = largest_date;
    }
    dague_barrier_wait( &(dague_context->barrier) );
    eu_context->largest_simulation_date = 0;
#endif

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
    // final select end - can we mark this as special somehow?
    // actually, it will already be obviously special, since it will be the only select
    // that has no context
    PINS(eu_context, SELECT_END, NULL);

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    dague_inform(("#Scheduling: th <%3d/%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu",
            eu_context->th_id, eu_context->virtual_process->vp_id, nbiterations, (long long unsigned int)found_local,
            (long long unsigned int)found_remote,
            (long long unsigned int)found_victim,
            (long long unsigned int)miss_local,
            (long long unsigned int)miss_victim ));

    if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
        char  priority_trace_fname[64];
        FILE *priority_trace = NULL;
        sprintf(priority_trace_fname, "priority_trace-%d.dat", eu_context->virtual_process->dague_context->my_rank);
        priority_trace = fopen(priority_trace_fname, "w");
        if( NULL != priority_trace ) {
            uint32_t my_idx;
            fprintf(priority_trace,
                    "#Step\tPriority\tThread\tVP\n"
                    "#Tasks are ordered in execution order\n");
            for(my_idx = 0; my_idx < MIN(sched_priority_trace_counter, DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER); my_idx++) {
                fprintf(priority_trace, "%d\t%d\t%d\t%d\n",
                        sched_priority_trace[my_idx].step, sched_priority_trace[my_idx].priority,
                        sched_priority_trace[my_idx].thread_id, sched_priority_trace[my_idx].vp_id);
            }
            fclose(priority_trace);
        }
    }
#endif  /* DAGUE_REPORT_STATISTICS */

    if( dague_context->__dague_internal_finalization_in_progress ) {
        PINS_THREAD_FINI(eu_context);
    }
    return nbiterations;
}

/************ COMPOSITION OF DAGUE_OBJECTS ****************/
typedef struct dague_compound_state_t {
    dague_context_t* ctx;
    int nb_objects;
    int completed_objects;
    dague_handle_t* objects_array[1];
} dague_compound_state_t;

static int dague_composed_cb( dague_handle_t* o, void* cbdata )
{
    dague_handle_t* compound = (dague_handle_t*)cbdata;
    dague_compound_state_t* compound_state = (dague_compound_state_t*)compound->functions_array;
    int completed_objects = compound_state->completed_objects++;
    assert( o == compound_state->objects_array[completed_objects] ); (void)o;
    if( compound->nb_pending_actions-- ) {
        assert( NULL != compound_state->objects_array[completed_objects+1] );
        dague_enqueue(compound_state->ctx,
                      compound_state->objects_array[completed_objects+1]);
    }
    return 0;
}

static void dague_compound_startup( dague_context_t *context,
                                    dague_handle_t *compound_object,
                                    dague_execution_context_t** startup_list)
{
    dague_compound_state_t* compound_state = (dague_compound_state_t*)compound_object->functions_array;
    dague_handle_t* first = compound_state->objects_array[0];
    int i;

    assert( 0 == compound_object->nb_functions );
    assert( NULL != first );
    first->startup_hook(context, first, startup_list);
    compound_state->ctx = context;
    compound_object->nb_pending_actions = compound_state->nb_objects;
    for( i = 0; i < compound_state->nb_objects; i++ ) {
        dague_handle_t* o = compound_state->objects_array[i];
        assert( NULL != o );
        o->on_complete      = dague_composed_cb;
        o->on_complete_data = compound_object;
    }
}

dague_handle_t* dague_compose( dague_handle_t* start,
                               dague_handle_t* next )
{
    dague_handle_t* compound = NULL;
    dague_compound_state_t* compound_state = NULL;

    if( start->nb_functions == 0 ) {  /* start is already a compound object */
        compound = start;
        compound_state = (dague_compound_state_t*)compound->functions_array;
        compound_state->objects_array[compound_state->nb_objects++] = next;
        /* make room for NULL terminating, if necessary */
        if( 0 == (compound_state->nb_objects % 16) ) {
            compound_state = realloc(compound_state, sizeof(dague_compound_state_t) +
                            (1 + compound_state->nb_objects / 16) * 16 * sizeof(void*));
            compound->functions_array = (void*)compound_state;
        }
        compound_state->objects_array[compound_state->nb_objects] = NULL;
    } else {
        compound = calloc(1, sizeof(dague_handle_t));
        compound->functions_array = malloc(sizeof(dague_compound_state_t) + 16 * sizeof(void*));
        compound_state = (dague_compound_state_t*)compound->functions_array;
        compound_state->objects_array[0] = start;
        compound_state->objects_array[1] = next;
        compound_state->objects_array[2] = NULL;
        compound_state->completed_objects = 0;
        compound_state->nb_objects = 2;
        compound->startup_hook = dague_compound_startup;
    }
    return compound;
}
/** END: Composition ***/

int32_t dague_set_priority( dague_handle_t* object, int32_t new_priority )
{
    int32_t old_priority = object->priority;
    object->priority = new_priority;
    return old_priority;
}

int dague_enqueue( dague_context_t* context, dague_handle_t* handle )
{
    if( NULL == current_scheduler) {
        dague_set_scheduler( context );
    }

    handle->context = context;  /* save the context */

    PINS_HANDLE_INIT(handle);  /* PINS handle initialization */

    /* Update the number of pending objects */
    dague_atomic_inc_32b( &(context->active_objects) );

    /* If necessary trigger the on_enqueue callback */
    if( NULL != handle->on_enqueue ) {
        handle->on_enqueue(handle, handle->on_enqueue_data);
    }

    if( NULL != handle->startup_hook ) {
        dague_execution_context_t **startup_list;
        int p;
        /* These pointers need to be initialized to NULL; doing it with calloc */
        startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*) );
        if( NULL == startup_list ) {  /* bad bad */
            return DAGUE_ERR_OUT_OF_RESOURCE;
        }
        handle->startup_hook(context, handle, startup_list);
        for(p = 0; p < vpmap_get_nb_vp(); p++) {
            if( NULL != startup_list[p] ) {
                dague_list_t temp;

                OBJ_CONSTRUCT( &temp, dague_list_t );
                /* Order the tasks by priority */
                dague_list_chain_sorted(&temp, (dague_list_item_t*)startup_list[p],
                                        dague_execution_context_priority_comparator);
                startup_list[p] = (dague_execution_context_t*)dague_list_nolock_unchain(&temp);
                OBJ_DESTRUCT(&temp);
                /* We should add these tasks on the system queue when there is one */
                __dague_schedule( context->virtual_processes[p]->execution_units[0], startup_list[p] );
            }
        }
        free(startup_list);
    } else {
        dague_check_complete_cb(handle, context, handle->nb_pending_actions);
    }
#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    sched_priority_trace_counter = 0;
#endif

    return 0;
}

static inline int
__dague_context_cas_or_flag(dague_context_t* context,
                            uint32_t flags)
{
    uint32_t current_flags = context->flags;
    /* if the flags are already set don't reset them */
    if( flags == (current_flags & flags) ) return 0;
    return dague_atomic_cas(&context->flags,
                            current_flags,
                            current_flags | flags);
}

/**
 * If there are enqueued handles waiting to be executed launch the other threads
 * and then return. Mark the internal structures in such a way that we can't
 * start the context mutiple times without completions.
 *
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads have
 * been activated.
 */
int dague_context_start( dague_context_t* context )
{
    /* No active work */
    if(all_tasks_done(context)) return -2;
    /* Context already active */
    if( DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE & context->flags )
        return -1;
    /* Start up the context */
    if( __dague_context_cas_or_flag(context, DAGUE_CONTEXT_FLAG_COMM_ACTIVE) ) {
        (void)dague_remote_dep_on(context);
        /* Mark the context so that we will skip the initial barrier during the _wait */
        context->flags |= DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE;
        /* Wake up the other threads */
        dague_barrier_wait( &(context->barrier) );
        return 0;
    }
    return -1;  /* Someone else start it up */
}

/**
 * Check the status of a context. No progress on the context is guaranteed.
 *
 * @returns: 0 if the context is active, any other value otherwide.
 */
int dague_context_test( dague_context_t* context )
{
    return !all_tasks_done(context);
}

/**
 * If the context is active the current thread (which must be the thread that
 * created the context will join the other active threads to complete the tasks
 * enqueued on the context. This function is blocking, the return is only
 * possible upon completion of all active handles in the context.
 */
int dague_context_wait( dague_context_t* context )
{
    int ret = 0;

    if( __dague_context_cas_or_flag(context,
                                    DAGUE_CONTEXT_FLAG_COMM_ACTIVE) ) {
        (void)dague_remote_dep_on(context);
    }

    ret = __dague_context_wait( context->virtual_processes[0]->execution_units[0] );

    context->__dague_internal_finalization_counter++;
    (void)dague_remote_dep_off(context);
    assert(context->flags & DAGUE_CONTEXT_FLAG_COMM_ACTIVE);
    assert(context->flags & DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE);
    context->flags ^= (DAGUE_CONTEXT_FLAG_COMM_ACTIVE | DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE);
    return ret;
}
