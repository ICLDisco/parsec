/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "schedulers.h"
#include "profiling.h"
#include "remote_dep.h"
#include "stats.h"
#include "datarepo.h"
#include "execution_unit.h"
#include "vpmap.h"

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
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID), -1, NULL)
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

int __dague_execute( dague_execution_unit_t* eu_context,
                     dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
#if defined(DAGUE_DEBUG)
    {
        const struct dague_flow* flow;
        int set_parameters, i;

        for( i = set_parameters = 0; NULL != (flow = exec_context->function->in[i]); i++ ) {
            if( (NULL != exec_context->data[flow->flow_index].data_repo) &&
                (ACCESS_NONE != flow->access_type)) {
                set_parameters++;
                assert( NULL != exec_context->data[flow->flow_index].data );
            }
        }
        assert( set_parameters <= 1 );
    }
# endif
#ifdef DAGUE_DEBUG_VERBOSE1
    char tmp[MAX_TASK_STRLEN];
    DEBUG(( "thread %d of VP %d Execute %s\n", eu_context->th_id, eu_context->virtual_process->vp_id, dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context))); 
#endif
    DAGUE_STAT_DECREASE(counter_nbtasks, 1ULL);

    return function->hook( eu_context, exec_context );
}

static inline int all_tasks_done(dague_context_t* context)
{
    return (context->active_objects == 0);
}

int __dague_complete_task(dague_object_t *dague_object, dague_context_t* context)
{
    int remaining;

    assert( dague_object->nb_local_tasks != 0 );
    remaining = dague_atomic_dec_32b( &(dague_object->nb_local_tasks) );
    //    printf("remaining local tasks: %d\n", dague_object->nb_local_tasks);
    if( 0 == remaining ) {
        /* A dague object has been completed. Call the attached callback if
         * necessary, then update the main engine.
         */
        if( NULL != dague_object->complete_cb ) {
            (void)dague_object->complete_cb( dague_object, dague_object->complete_cb_data );
        }
        dague_atomic_dec_32b( &(context->active_objects) );
        return 1;
    }
    return 0;
}


static dague_scheduler_t scheduler = { "None", NULL, NULL, NULL, NULL, NULL };

void dague_set_scheduler( dague_context_t *dague, dague_scheduler_t *s )
{
    if( NULL != scheduler.finalize ) {
        scheduler.finalize( dague );
    }
    if( NULL != s ) {
        memcpy( &scheduler, s, sizeof(dague_scheduler_t) );
        scheduler.init( dague );
    } else {
        memset( &scheduler, 0, sizeof(dague_scheduler_t) );
    }
}

/**
 * This is where we end up after the release_dep_fct is called and generates a
 * readylist. the new_context IS the readylist.
 */
int __dague_schedule( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* new_context )
{
    int ret;

#if defined(DAGUE_DEBUG)
    {
        dague_execution_context_t* context = new_context;
        const struct dague_flow* flow;
        int set_parameters, i;
        char tmp[MAX_TASK_STRLEN];

		  // PETER it seems like this while loop mostly verifies
		  // that nothing is terrible wrong?
        do {
            for( i = set_parameters = 0; NULL != (flow = context->function->in[i]); i++ ) {
                if( ACCESS_NONE == flow->access_type ) continue;
                if( NULL != context->data[flow->flow_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[flow->flow_index].data ) {
                        ERROR(( "Task %s has flow %d data_repo != NULL but a data == NULL (%s:%d)\n",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context), flow->flow_index, __FILE__, __LINE__));
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
# endif

    /* Deactivate this measurement, until the MPI thread has its own execution unit
     *  TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);
     */
    ret = scheduler.schedule_task(eu_context, new_context);
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

inline int __dague_complete_execution( dague_execution_unit_t *eu_context,
                                       dague_execution_context_t *exec_context )
{
    int rc = 0;

    if( NULL != exec_context->function->complete_execution )
        rc = exec_context->function->complete_execution( eu_context, exec_context );
    /* Update the number of remaining tasks */
    __dague_complete_task(exec_context->dague_object, eu_context->virtual_process->dague_context);

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

    if( NULL == scheduler.select_task ||
        NULL == scheduler.schedule_task ) {
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
            TAKE_TIME( eu_context->eu_profile, schedule_sleep_begin, nbiterations);
            nanosleep(&rqtp, NULL);
            TAKE_TIME( eu_context->eu_profile, schedule_sleep_end, nbiterations);
        }

        TAKE_TIME( eu_context->eu_profile, schedule_poll_begin, nbiterations);
        exec_context = scheduler.select_task(eu_context);
        TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);

        if( exec_context != NULL ) {
            // DEBUG PETER
            assert(NULL != exec_context->function);
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

            // MY MODS
            TAKE_TIME(eu_context->eu_profile, queue_remove_begin, 0);
            TAKE_TIME(eu_context->eu_profile, queue_remove_end, 0);
            
            switch( exec_context->function->data_lookup( exec_context ) ) {
            case DAGUE_LOOKUP_DONE:
                /* We're good to go ... */
                if( 0 == __dague_execute( eu_context, exec_context ) ) {
                    __dague_complete_execution( eu_context, exec_context );
                }
                nbiterations++;
                break;
            default:
                assert( 0 ); /* Internal error: invalid return value for data_lookup function */
            }

        } else {
            misses_in_a_row++;
        }
    }

    /* We're all done ? */
    dague_barrier_wait( &(dague_context->barrier) );

#if defined(DAGUE_SIM)
    if( DAGUE_THREAD_IS_MASTER(eu_context) ) {
        int32_t my_idx;
        int largest_date = 0;
        for(my_idx = 0; my_idx < dague_context->nb_cores; my_idx++) {
            if( dague_context->execution_units[my_idx]->largest_simulation_date > largest_date )
                largest_date = dague_context->execution_units[my_idx]->largest_simulation_date;
        }
        dague_context->largest_simulation_date = largest_date;
    }
    dague_barrier_wait( &(dague_context->barrier) );
    eu_context ->largest_simulation_date = 0;
#endif

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
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

int dague_enqueue( dague_context_t* context, dague_object_t* object)
{
    dague_execution_context_t **startup_list;
    int p;

    if( NULL == scheduler.schedule_task ) {
        /* No scheduler selected yet. The default is 0 */
        dague_set_scheduler( context, dague_schedulers_array[DAGUE_SCHEDULER_LFQ] );
    }

    /* These pointers need to be initialized to NULL; doing it with calloc */
    startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*) );

    if( object->nb_local_tasks > 0 ) {
        /* Update the number of pending dague objects */
        dague_atomic_inc_32b( &(context->active_objects) );

        if( NULL != object->startup_hook ) {
            object->startup_hook(context, object, startup_list);
            for(p = 0; p < vpmap_get_nb_vp(); p++) {
                if( NULL != startup_list[p] ) {
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

