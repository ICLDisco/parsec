/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "scheduling.h"
#include "dequeue.h"
#include "profiling.h"
#include "remote_dep.h"
#include "dague.h"
#include "stats.h"
#include "priority_sorted_queue.h"

#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <errno.h>

#ifdef  HAVE_SCHED_SETAFFINITY
#include <linux/unistd.h>
#endif  /* HAVE_SCHED_SETAFFINITY */

#if defined(DAGUE_PROF_TRACE) && 0
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID))
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
#define DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER 65536
typedef struct {
    int      thread_id;
    int32_t  priority;
    uint32_t step;
} sched_priority_trace_t;
static sched_priority_trace_t sched_priority_trace[DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER];
static uint32_t sched_priority_trace_counter;
#endif

static inline int __dague_execute( dague_execution_unit_t* eu_context,
                                   dague_execution_context_t* exec_context )
{
    int rc = 0;
    const dague_t* function = exec_context->function;
#if defined(DAGUE_DEBUG)
    {
        const struct param* param;
        int set_parameters, i;
        char tmp[128];

        DEBUG(( "thread %d Execute %s\n", eu_context->eu_id, dague_service_to_string(exec_context, tmp, 128)));
        for( i = set_parameters = 0; NULL != (param = exec_context->function->in[i]); i++ ) {
            if( NULL != exec_context->data[param->param_index].data_repo ) {
                set_parameters++;
                assert( NULL != exec_context->data[param->param_index].data );
            }
        }
        assert( set_parameters <= 1 );
    }
# endif
    DAGUE_STAT_DECREASE(counter_nbtasks, 1ULL);

    if( NULL != function->hook ) {
        rc = function->hook( eu_context, exec_context );
    }
    return rc; 
}

static inline void set_tasks_todo(dague_context_t* context, uint32_t n)
{
    context->taskstodo = n;
}

static inline int all_tasks_done(dague_context_t* context)
{
    return (context->taskstodo == 0);
}

static inline void done_task(dague_context_t* context)
{
    dague_atomic_dec_32b( &(context->taskstodo) );
}

int __dague_schedule( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* new_context )
{
#if defined(DAGUE_DEBUG)
    {
        dague_execution_context_t* context = new_context;
        const struct param* param;
        int set_parameters, i;
        char tmp[128];

        do {
            for( i = set_parameters = 0; NULL != (param = context->function->in[i]); i++ ) {
                if( NULL != context->data[param->param_index].data_repo ) {
                    set_parameters++;
                    if( NULL == context->data[param->param_index].data ) {
                        DEBUG(( "Task %s has parameters %d data_repo != NULL but a data == NULL (%s:%d)\n",
                                dague_service_to_string(context, tmp, 128), param->param_index, __FILE__, __LINE__));
                        assert( NULL == context->data[param->param_index].data );
                    }
                }
            }
            if( set_parameters > 1 ) {
                DEBUG(( "Task %s has more than one parameter set (impossible)!! (%s:%d)\n",
                        dague_service_to_string(context, tmp, 128), __FILE__, __LINE__));
                assert( set_parameters > 1 );
            }
            DEBUG(( "thread %d Schedules %s\n", eu_context->eu_id, dague_service_to_string(context, tmp, 128)));
            context = (dague_execution_context_t*)context->list_item.list_next;
        } while ( context != new_context );
    }
# endif

    TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);

#  if (DAGUE_SCHEDULER == DAGUE_SCHEDULER_ABSOLUTE_PRIORITIES)
    dague_priority_sorted_list_merge( eu_context->eu_task_queue, (dague_list_item_t*)new_context );
#  elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES)
    dague_hbbuffer_push_all( eu_context->eu_task_queue, (dague_list_item_t*)new_context );
#  elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_GLOBAL_DEQUEUE)
    if( new_context->function->flags & DAGUE_HIGH_PRIORITY_TASK ) {
        dague_dequeue_push_front( eu_context->eu_system_queue, (dague_list_item_t*)new_context);
    } else {
        dague_dequeue_push_back( eu_context->eu_system_queue, (dague_list_item_t*)new_context);
    }
#  else
#    error No scheduler is defined
#  endif

    TAKE_TIME( eu_context->eu_profile, schedule_push_end, 0);

    return 0;
}

void dague_register_nb_tasks(dague_context_t* context, int n)
{
#if defined(DAGUE_PROF_TRACE)
    /* Reset the profiling information */
    dague_profiling_reset();
#endif  /* defined(DAGUE_PROF_TRACE) */
        
    set_tasks_todo(context, (uint32_t)n);
}

#include <math.h>
static void __do_some_computations( void )
{
    const int NB = 256;
    double *A = (double*)malloc(NB*NB*sizeof(double));
    int i, j;

    for( i = 0; i < NB; i++ ) {
        for( j = 0; j < NB; j++ ) {
            A[i*NB+j] = (double)rand() / RAND_MAX;
        }
    }
    free(A);
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

#if (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES)
#if defined(DAGUE_SCHED_CACHE_AWARE)
static  unsigned int ranking_function_bycache(dague_list_item_t *elt, void *param)
{
    unsigned int value;
    cache_t *cache = (cache_t*)param;
    dague_execution_context_t *exec = (dague_execution_context_t*)elt;
    
    /* TODO: fix this, depends on the depth */
    value = exec->function->cache_rank_function(exec, cache, 128);
    DEBUG(("maxvalue of this choice is %u\n", value));
    return value;
}
#  else
static  unsigned int ranking_function_bypriority(dague_list_item_t *elt, void *_)
{
    dague_execution_context_t *exec = (dague_execution_context_t*)elt;
    (void)_;
    return (~(unsigned int)0) - exec->priority;
}
#  endif
#endif

static inline dague_execution_context_t *choose_local_job( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t *exec_context = NULL;

#if   (DAGUE_SCHEDULER == DAGUE_SCHEDULER_ABSOLUTE_PRIORITIES)
    exec_context = (dague_execution_context_t*)dague_priority_sorted_list_pop_front(eu_context->eu_task_queue);
#elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES)
#  if defined(DAGUE_SCHED_CACHE_AWARE)
    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(eu_context->eu_task_queue,
                                                                       ranking_function_bycache,
                                                                       eu_context->closest_cache);
#  else  /* DAGUE_SCHED_CACHE_AWARE */
    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(eu_context->eu_task_queue,
                                                                       ranking_function_bypriority,
                                                                       NULL);
#  endif /* DAGUE_SCHED_CACHE_AWARE */
#elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_GLOBAL_DEQUEUE)
    exec_context = (dague_execution_context_t*)dague_dequeue_pop_front( eu_context->eu_system_queue );
#else
#error DAGUE_SCHEDULER is not defined
#endif
    
    return exec_context;
}

inline int dague_complete_execution( dague_execution_unit_t *eu_context,
                                     dague_execution_context_t *exec_context )
{
    int rc = 0;
    if( NULL != exec_context->function->complete_execution ) 
        rc = exec_context->function->complete_execution( eu_context, exec_context );
    /* Update the number of remaining tasks */
    done_task(eu_context->master_context);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( eu_context->eu_id, exec_context );
    /* Release the execution context */
    DAGUE_STAT_DECREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    dague_thread_mempool_free( eu_context->context_mempool, exec_context );
    return rc;
}

void* __dague_progress( dague_execution_unit_t* eu_context )
{
    uint64_t found_local, miss_local, found_victim, miss_victim, found_remote, system_victim;
    uint64_t misses_in_a_row;
    dague_context_t* master_context = eu_context->master_context;
    int32_t my_barrier_counter = master_context->__dague_internal_finalization_counter;
    dague_execution_context_t* exec_context;
    int nbiterations = 0;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    found_local = miss_local = found_victim = miss_victim = found_remote = system_victim = 0;
    misses_in_a_row = 1;
    
    if( 0 != eu_context->eu_id ) {
        /* Force the kernel to bind me to the expected core */
        __do_some_computations();

        /* Wait until all threads are done binding themselves 
         * (see dague_init) */
        dague_barrier_wait( &(master_context->barrier) );
        my_barrier_counter = 1;
    }

    /* The main loop where all the threads will spend their time */
 wait_for_the_next_round:
    /* Wait until all threads are here and the main thread signal the begining of the work */
    dague_barrier_wait( &(master_context->barrier) );

    if( master_context->__dague_internal_finalization_in_progress ) {
        my_barrier_counter++;
        for(; my_barrier_counter <= master_context->__dague_internal_finalization_counter; my_barrier_counter++ ) {
            dague_barrier_wait( &(master_context->barrier) );
        }
        goto finalize_progress;
    }

    while( !all_tasks_done(master_context) ) {
#if defined(DISTRIBUTED)
        if( eu_context->eu_id == 0) {
            int ret;
            /* check for remote deps completion */
            while((ret = dague_remote_dep_progress(eu_context)) > 0)  {
                found_remote += ret;
                misses_in_a_row = 0;
            }
        }
#else
        (void) found_remote;
#endif /* DISTRIBUTED */
        
        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            DAGUE_STATACC_ACCUMULATE(time_starved, rqtp.tv_nsec/1000);
            TAKE_TIME( eu_context->eu_profile, schedule_sleep_begin, nbiterations);
            nanosleep(&rqtp, NULL);
            TAKE_TIME( eu_context->eu_profile, schedule_sleep_end, nbiterations);
        }
        
        TAKE_TIME( eu_context->eu_profile, schedule_poll_begin, nbiterations);
        exec_context = choose_local_job(eu_context);

        if( exec_context != NULL ) {
            misses_in_a_row = 0;
            found_local++;
#if (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES)
        do_some_work:
#endif

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
            {
                uint32_t my_idx = dague_atomic_inc_32b(&sched_priority_trace_counter);
                if(my_idx < DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER ) {
                    sched_priority_trace[my_idx].step = eu_context->sched_nb_tasks_done++;
                    sched_priority_trace[my_idx].thread_id = eu_context->eu_id;
                    sched_priority_trace[my_idx].priority  = exec_context->priority;
                }
            }
#endif

            TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);
            /* We're good to go ... */
            if( 0 == __dague_execute( eu_context, exec_context ) ) {
                dague_complete_execution( eu_context, exec_context );
            }
            nbiterations++;
        } else {
            miss_local++;

#if (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES)
            /* Work stealing from the other workers */
            {
                unsigned int i;
                                
                for(i = 0; i <  eu_context->eu_nb_hierarch_queues; i++ ) {
#  if defined(DAGUE_SCHED_CACHE_AWARE)
                    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(eu_context->eu_hierarch_queues[i],
                                                                                       ranking_function_bycache,
                                                                                       eu_context->closest_cache);
#  else  /* DAGUE_SCHED_CACHE_AWARE */
                    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(eu_context->eu_hierarch_queues[i],
                                                                                       ranking_function_bypriority,
                                                                                       NULL);
#  endif /* DAGUE_SCHED_CACHE_AWARE */
                    if( NULL != exec_context ) {
                        misses_in_a_row = 0;
                        found_victim++;
                        goto do_some_work;
                    }
                    miss_victim++;
                }
                exec_context = (dague_execution_context_t *)dague_dequeue_pop_front(eu_context->eu_system_queue);
                if( NULL != exec_context ) {
                    misses_in_a_row = 0;
                    system_victim++;
                    goto do_some_work;
                }
                miss_victim++;
            }
#endif
            misses_in_a_row++;
            TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);
        }
    }
    
#if   (DAGUE_SCHEDULER == DAGUE_SCHEDULER_ABSOLUTE_PRIORITIES)
    assert( dague_priority_sorted_list_empty(eu_context->eu_task_queue) );
#elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_HIER_QUEUES) || (DAGUE_SCHEDULER == DAGUE_SCHEDULER_LOCAL_FLAT_QUEUES)
    assert( dague_hbbuffer_is_empty( eu_context->eu_task_queue ) );
    assert( dague_dequeue_is_empty( eu_context->eu_system_queue ) );
#elif (DAGUE_SCHEDULER == DAGUE_SCHEDULER_GLOBAL_DEQUEUE)
    assert( dague_dequeue_is_empty( eu_context->eu_system_queue ) );
#else
#error DAGUE_SCHEDULER is not defined
#endif

    /* We're all done ? */
    dague_barrier_wait( &(master_context->barrier) );

#if defined(DAGUE_SIM)
    if( 0 == eu_context->eu_id ) {
        uint32_t my_idx;
        int largest_date = 0;
        for(my_idx = 0; my_idx < master_context->nb_cores; my_idx++) {
            if( master_context->execution_units[my_idx]->largest_simulation_date > largest_date )
                largest_date = master_context->execution_units[my_idx]->largest_simulation_date;
        }
        printf("Simulated Time: %d\n", largest_date);
    }
    dague_barrier_wait( &(master_context->barrier) );
#endif

    if( 0 != eu_context->eu_id ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    printf("#Scheduling: th <%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu\n",
           eu_context->eu_id, nbiterations, (long long unsigned int)found_local,
           (long long unsigned int)found_remote,
           (long long unsigned int)found_victim,
           (long long unsigned int)miss_local,
           (long long unsigned int)miss_victim );

    if( eu_context->eu_id == 0 ) {
        char  priority_trace_fname[64];
        FILE *priority_trace = NULL;
        sprintf(priority_trace_fname, "priority_trace-%d.dat", eu_context->master_context->my_rank);
        priority_trace = fopen(priority_trace_fname, "w");
        if( NULL != priority_trace ) {
            uint32_t my_idx;
            fprintf(priority_trace, 
                    "#Step\tPriority\tThread\n"
                    "#Tasks are ordered in execution order\n");
            for(my_idx = 0; my_idx < MIN(sched_priority_trace_counter, DAGUE_SCHED_MAX_PRIORITY_TRACE_COUNTER); my_idx++) {
                fprintf(priority_trace, "%d\t%d\t%d\n", sched_priority_trace[my_idx].step, sched_priority_trace[my_idx].priority, sched_priority_trace[my_idx].thread_id);
            }
            fclose(priority_trace);
        }
    }
#endif  /* DAGUE_REPORT_STATISTICS */

    return (void*)(long)nbiterations;
}

int dague_enqueue( dague_context_t* context, dague_object_t* object)
{
    dague_execution_context_t *startup_list = NULL;

    context->taskstodo += object->nb_local_tasks;
    if( NULL != object->startup_hook ) {
        object->startup_hook(context, object, &startup_list);
        if( NULL != startup_list ) {
            /* We should add these tasks on the system queue */
            __dague_schedule( context->execution_units[0], startup_list );
        }
    }
    
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
    int ret;
    (void)dague_remote_dep_on(context);
    
    ret = (int)(long)__dague_progress( context->execution_units[0] );

    context->__dague_internal_finalization_counter++;
    (void)dague_remote_dep_off(context);
    return ret;
}

int dague_progress(dague_context_t* context)
{
    int ret;
    (void)dague_remote_dep_on(context);
    
    ret = (int)(long)__dague_progress( context->execution_units[0] );

    context->__dague_internal_finalization_counter++;
    (void)dague_remote_dep_off(context);
    return ret;
}
