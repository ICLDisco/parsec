/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "src/mca/mca_repository.h"
#include "src/mca/sched/sched.h"
#include "profiling.h"
#include "stats.h"
#include "datarepo.h"
#include "execution_unit.h"
#include "vpmap.h"
#include "src/mca/pins/pins.h"
#include "os-spec-timing.h"

#include "dague/ayudame.h"

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

#if defined(HAVE_GETRUSAGE) || !defined(__bgp__)
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

        STATUS(("\n=============================================================\n"
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
                "=============================================================\n\n"
                ,str, eu->virtual_process->vp_id, eu->th_id, eu->core_id, eu->socket_id,
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
static void dague_statistics_per_eu(char* str, dague_execution_unit_t* eu) { (void)str; return; }
#endif /* defined(HAVE_GETRUSAGE) */
#endif /* defined(DAGUE_PROF_RUSAGE_EU) */

#if 0
/**
 * Disabled by now.
 */
int __dague_progress_task( dague_execution_unit_t* eu_context,
                           dague_execution_context_t* task )
{
    (void)eu_context;
    switch(task->status) {
        case DAGUE_TASK_STATUS_NONE:
#ifdef DAGUE_DEBUG_VERBOSE1
            char tmp[MAX_TASK_STRLEN];
            DEBUG(("thread %d of VP %d Execute %s\n", eu_context->th_id, eu_context->virtual_process->vp_id,
                   dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, task)));
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

    DAGUE_STAT_DECREASE(counter_nbtasks, 1ULL);
    AYU_TASK_RUN(eu_context->th_id, exec_context);
    /**
     * Try all the incarnation until one agree to execute.
     */
    do {
#ifdef DAGUE_DEBUG_VERBOSE1
        char tmp[MAX_TASK_STRLEN];
        DEBUG(("thread %d of VP %d Execute %s[%d]\n",
               eu_context->th_id, eu_context->virtual_process->vp_id,
               dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context),
               function->incarnations[exec_context->chore_id].type));
#endif
        rc = function->incarnations[exec_context->chore_id].hook( eu_context, exec_context );
        if( DAGUE_HOOK_RETURN_NEXT != rc )
            return rc;
        exec_context->chore_id++;
    } while(NULL != function->incarnations[exec_context->chore_id].hook);
    /* We're out of luck, no more chores */
    return DAGUE_HOOK_RETURN_ERROR;
}

static inline int all_tasks_done(dague_context_t* context)
{
    return (context->active_objects == 0);
}

int __dague_complete_task(dague_handle_t *dague_handle, dague_context_t* context)
{
    int remaining;

    assert( dague_handle->nb_local_tasks != 0 );
    remaining = dague_atomic_dec_32b( &(dague_handle->nb_local_tasks) );

    if( 0 == remaining ) {
        /* A dague object has been completed. Call the attached callback if
         * necessary, then update the main engine.
         */
        if( NULL != dague_handle->complete_cb ) {
            (void)dague_handle->complete_cb( dague_handle, dague_handle->complete_cb_data );
        }
        dague_atomic_dec_32b( &(context->active_objects) );
        return 1;
    }
    return 0;
}

static dague_sched_module_t         *current_scheduler = NULL;
static dague_sched_base_component_t *scheduler_component = NULL;

void dague_remove_scheduler( dague_context_t *dague )
{
    if( NULL != current_scheduler ) {
        current_scheduler->module.remove( dague );
        //        pins_fini_steals(dague); // PETER TODO where does this actually belong!?
        assert( NULL != scheduler_component );
        mca_component_close( (mca_base_component_t*)scheduler_component );
        current_scheduler = NULL;
        scheduler_component = NULL;
    }
}

static int no_scheduler_is_active( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            if( vp->execution_units[t]->scheduler_object != NULL ) {
                return 0;
            }
        }
    }

    return 1;
}

int dague_set_scheduler( dague_context_t *dague )
{
    mca_base_component_t **scheds;
    mca_base_module_t    *new_scheduler = NULL;
    mca_base_component_t *new_component = NULL;

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

    DEBUG((" Installing %s\n", current_scheduler->component->base_version.mca_component_name));
    PROFILING_SAVE_sINFO("sched", (char *)current_scheduler->component->base_version.mca_component_name);

    assert( no_scheduler_is_active(dague) );
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

#if defined(DAGUE_DEBUG_ENABLE)
    {
        dague_execution_context_t* context = new_context;
        const struct dague_flow_s* flow;
        int set_parameters, i;
        char tmp[MAX_TASK_STRLEN];

        do {
            for( i = set_parameters = 0; NULL != (flow = context->function->in[i]); i++ ) {
                if( ACCESS_NONE == flow->access_type ) continue;
                if( NULL != context->data[flow->flow_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[flow->flow_index].data_in ) {
                        DEBUG2(("Task %s has flow %s data_repo != NULL but a data == NULL (%s:%d)\n",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context),
                                flow->name, __FILE__, __LINE__));
                    }
                }
            }
            if( set_parameters > 1 ) {
                ERROR(( "Task %s has more than one input flow set (impossible)!! (%s:%d)\n",
                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context), __FILE__, __LINE__));
            }
            DEBUG2(( "thread %d of VP %d Schedules %s\n",
                    eu_context->th_id, eu_context->virtual_process->vp_id,
                    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context) ));
            context = (dague_execution_context_t*)context->list_item.list_next;
        } while ( context != new_context );
    }
#endif  /* defined(DAGUE_DEBUG_ENABLE) */

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
    int rc = 0;

    if( NULL != exec_context->function->prepare_output ) {
        exec_context->function->prepare_output( eu_context, exec_context );
    }
    if( NULL != exec_context->function->complete_execution )
        rc = exec_context->function->complete_execution( eu_context, exec_context );
    /* Update the number of remaining tasks */
    __dague_complete_task(exec_context->dague_handle, eu_context->virtual_process->dague_context);
    AYU_TASK_COMPLETE(exec_context);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( eu_context->th_id, eu_context->virtual_process->vp_id, exec_context );
    /* Release the execution context */
    DAGUE_STAT_DECREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    dague_thread_mempool_free( eu_context->context_mempool, exec_context );
    return rc;
}

void* __dague_progress( dague_execution_unit_t* eu_context )
{
    uint64_t misses_in_a_row;
    dague_context_t* dague_context = eu_context->virtual_process->dague_context;
    int32_t my_barrier_counter = dague_context->__dague_internal_finalization_counter;
    dague_execution_context_t* exec_context;
    int nbiterations = 0;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    misses_in_a_row = 1;

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) {
        /* Wait until all threads are done binding themselves
         * (see dague_init) */
        dague_barrier_wait( &(dague_context->barrier) );
        my_barrier_counter = 1;
    }

#if defined(DAGUE_PROF_RUSAGE_EU)
    dague_statistics_per_eu("EU", eu_context);
#endif
    /* first select begin, right before the wait_for_the... goto label */
    PINS(SELECT_BEGIN, eu_context, NULL, NULL);

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
        fprintf(stderr, "DAGuE: Main thread entered dague_progress, while scheduler is not selected yet!\n");
        return (void *)-1;
    }

    while( !all_tasks_done(dague_context) ) {
#if defined(DISTRIBUTED)
        if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
            /* check for remote deps completion */
            while(dague_remote_dep_progress(eu_context) > 0)  {
                misses_in_a_row = 0;
            }
        }
#endif /* DISTRIBUTED */

        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            DAGUE_STATACC_ACCUMULATE(time_starved, rqtp.tv_nsec/1000);
            nanosleep(&rqtp, NULL);
        }

        /* time how long it takes to get a task, if indeed we get one */
        dague_time_t select_begin = take_time();
        exec_context = current_scheduler->module.select(eu_context);

        if( exec_context != NULL ) {
            dague_time_t select_end = take_time();
            uint64_t select_time = diff_time(select_begin, select_end);

            PINS(SELECT_END, eu_context, exec_context, (void *)select_time);
            // select end, and record with it the amount of time actually spent selecting
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

            // prepare_input begin
            PINS(PREPARE_INPUT_BEGIN, eu_context, exec_context, NULL);
            switch( exec_context->function->prepare_input(eu_context, exec_context) ) {
            case DAGUE_HOOK_RETURN_DONE:
            {
                PINS(PREPARE_INPUT_END, eu_context, exec_context, NULL);
                // prepare input end
                int rv = 0;
                /* We're good to go ... */
                PINS(EXEC_BEGIN, eu_context, exec_context, NULL);
                rv = __dague_execute( eu_context, exec_context );
                PINS(EXEC_END, eu_context, exec_context, NULL  );
                if( 0 == rv ) {
                    // complete execution==add==push begin
                    PINS(COMPLETE_EXEC_BEGIN, eu_context, exec_context, NULL);
                    __dague_complete_execution( eu_context, exec_context );
                    PINS(COMPLETE_EXEC_END, eu_context, exec_context, NULL);
                }
                nbiterations++;
                break;
            }
            default:
                assert( 0 ); /* Internal error: invalid return value for data_lookup function */
            }

            // subsequent select begins
            PINS(SELECT_BEGIN, eu_context, NULL, NULL);
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
    PINS(SELECT_END, eu_context, NULL, NULL);

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    STATUS(("#Scheduling: th <%3d/%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu\n",
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

    return (void*)((long)nbiterations);
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
    if( compound->nb_local_tasks-- ) {
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
    compound_object->nb_local_tasks = compound_state->nb_objects;
    for( i = 0; i < compound_state->nb_objects; i++ ) {
        dague_handle_t* o = compound_state->objects_array[i];
        assert( NULL != o );
        o->complete_cb = dague_composed_cb;
        o->complete_cb_data = compound_object;
    }
}

dague_handle_t* dague_compose( dague_handle_t* start,
                               dague_handle_t* next )
{
    dague_handle_t* compound = NULL;
    dague_compound_state_t* compound_state = NULL;

    if( start->nb_functions == 0 ) {
        compound = start;
        compound_state = (dague_compound_state_t*)compound->functions_array;
        compound_state->objects_array[compound_state->nb_objects++] = next;
        /* make room for NULL terminating, if necessary */
        if( 0 == (compound_state->nb_objects%16) ) {
            compound_state = realloc(compound_state, sizeof(dague_compound_state_t) +
                            (1 + compound_state->nb_objects/16) * 16 * sizeof(void*));
            compound->functions_array = (void*)compound_state;
        }
        compound_state->objects_array[compound_state->nb_objects] = NULL;
    }
    else {
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

int dague_enqueue( dague_context_t* context, dague_handle_t* object)
{
    dague_execution_context_t **startup_list;
    int p;

    if( NULL == current_scheduler) {
        dague_set_scheduler( context );
    }

    /* These pointers need to be initialized to NULL; doing it with calloc */
    startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*) );

    if( object->nb_local_tasks > 0 ) {
        /* Update the number of pending objects */
        dague_atomic_inc_32b( &(context->active_objects) );

        if( NULL != object->startup_hook ) {
            object->startup_hook(context, object, startup_list);
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
        }
    }

    free(startup_list);

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    sched_priority_trace_counter = 0;
#endif

    return 0;
}

int dague_start( dague_context_t* context )
{
    (void) context; // silence the compiler
    return 0;
}

int dague_test( dague_context_t* context )
{
    (void) context; // silence the compiler
    return -1;  /* Not yet implemented */
}

int dague_wait( dague_context_t* context )
{
    int ret = 0;
    (void)dague_remote_dep_on(context);

    ret = (int)(long)__dague_progress( context->virtual_processes[0]->execution_units[0] );

    context->__dague_internal_finalization_counter++;
    (void)dague_remote_dep_off(context);
    return ret;
}

int dague_progress(dague_context_t* context)
{
    int ret = 0;
    (void)dague_remote_dep_on(context);

    ret = (int)(long)__dague_progress( context->virtual_processes[0]->execution_units[0] );

    context->__dague_internal_finalization_counter++;
    (void)dague_remote_dep_off(context);
    return ret;
}
