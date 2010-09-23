/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
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

#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <errno.h>

#ifdef  HAVE_SCHED_SETAFFINITY
#include <linux/unistd.h>
#endif  /* HAVE_SCHED_SETAFFINITY */

#if defined(DAGUE_PROFILING) && 0
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID))
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

static inline int __dague_execute( dague_execution_unit_t* eu_context,
                                   dague_execution_context_t* exec_context )
{
    int rc = 0;
    const dague_t* function = exec_context->function;
#if defined(DAGUE_DEBUG)
    char tmp[128]; 
    DEBUG(( "Execute %s\n", dague_service_to_string(exec_context, tmp, 128)));
#endif
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

/**
 * Schedule the instance of the service based on the values of the
 * local variables stored in the execution context, by calling the
 * attached hook if any. At the end of the execution the dependencies
 * are released.
 */
int dague_schedule( dague_context_t* context, const dague_execution_context_t* exec_context )
{
    dague_execution_context_t* new_context;
    dague_execution_unit_t* eu_context;
    dague_thread_mempool_t *mpool;

    eu_context = context->execution_units[0];

    new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
    mpool = new_context->mempool_owner;
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    memcpy( new_context, exec_context, sizeof(dague_execution_context_t) );
    new_context->mempool_owner = mpool;
#if defined(DAGUE_CACHE_AWARE)
    new_context->data[1] = NULL;
#endif
    DAGUE_LIST_ITEM_SINGLETON( new_context );
    return __dague_schedule( eu_context, new_context, 1);
}

int __dague_schedule( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* new_context, int use_placeholder )
{
    TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);

#if defined(DAGUE_DEBUG)
    {
        char tmp[128];
        dague_list_item_t* item = (dague_list_item_t*)new_context, *next;

        do {
            next = (dague_list_item_t*)item->list_next;
            DEBUG(( "thread %d Schedule %s\n", eu_context->eu_id, dague_service_to_string((dague_execution_context_t*)item, tmp, 128)));
            item = next;
        } while ( item != (dague_list_item_t*)new_context );
    }
# endif

#  if defined(DAGUE_USE_LIFO) || defined(DAGUE_USE_GLOBAL_LIFO)
    (void)use_placeholder;
    dague_atomic_lifo_push( eu_context->eu_task_queue, (dague_list_item_t*)new_context );
#  elif defined(HAVE_HWLOC)
    (void)use_placeholder;
    dague_hbbuffer_push_all( eu_context->eu_task_queue, (dague_list_item_t*)new_context );
#  else
#    if PLACEHOLDER_SIZE
    if(use_placeholder) {
        /**
         * The list is supposedly arranged by decreasing order of priority, so we will
         * pick the highest priority tasks and move them into the placeholder.
         */
        while( (((eu_context->placeholder_push + 1) % PLACEHOLDER_SIZE) != eu_context->placeholder_pop) ) {
            eu_context->placeholder[eu_context->placeholder_push] = new_context;
            eu_context->placeholder_push = (eu_context->placeholder_push + 1) % PLACEHOLDER_SIZE;

            if( new_context->list_item.list_next == (dague_list_item_t*)new_context )
                goto done_pushing_tasks;

            new_context->list_item.list_next->list_prev = new_context->list_item.list_prev;
            new_context->list_item.list_prev->list_next = new_context->list_item.list_next;
            new_context = (dague_execution_context_t*)new_context->list_item.list_next;
        }
    }
#    endif  /* PLACEHOLDER_SIZE */
    if( new_context->function->flags & DAGUE_HIGH_PRIORITY_TASK ) {
        dague_dequeue_push_front( eu_context->eu_task_queue, (dague_list_item_t*)new_context);
    } else {
        dague_dequeue_push_back( eu_context->eu_task_queue, (dague_list_item_t*)new_context);
    }
#  endif  /* DAGUE_USE_LIFO */
#if PLACEHOLDER_SIZE && !defined(HAVE_HWLOC) && !defined(DAGUE_USE_LIFO) && !defined(DAGUE_USE_GLOBAL_LIFO)
 done_pushing_tasks:
#endif
    TAKE_TIME( eu_context->eu_profile, schedule_push_end, 0);

    return 0;
}

void dague_register_nb_tasks(dague_context_t* context, int n)
{
#if defined(DAGUE_PROFILING)
    /* Reset the profiling information */
    dague_profiling_reset();
#endif  /* defined(DAGUE_PROFILING) */
        
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

#if defined( HAVE_HWLOC )
#  if defined(DAGUE_CACHE_AWARE)
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
static  unsigned int ranking_function_firstfound(dague_list_item_t *elt, void *_)
{
    (void)elt;
    (void)_;
    return DAGUE_RANKING_FUNCTION_BEST;
}
#  endif
#endif

#if defined(DAGUE_USE_LIFO) || defined(DAGUE_USE_GLOBAL_LIFO)
#  define DAGUE_POP(eu_context, queue_name) \
    (dague_execution_context_t*)dague_atomic_lifo_pop( (eu_context)->queue_name )
#elif defined(HAVE_HWLOC) 
#  if defined(DAGUE_CACHE_AWARE)
#    define DAGUE_POP(eu_context, queue_name) \
    (dague_execution_context_t*)dague_hbbuffer_pop_best((eu_context)->queue_name, \
                                                            ranking_function_bycache, \
                                                            (eu_context)->closest_cache)
#  else
#    define DAGUE_POP(eu_context, queue_name) \
    (dague_execution_context_t*)dague_hbbuffer_pop_best((eu_context)->queue_name, \
                                                            ranking_function_firstfound, \
                                                            NULL)
#  endif /* DAGUE_CACHE_AWARE */
#  define DAGUE_SYSTEM_POP(eu_context, queue_name) (dague_execution_context_t*)dague_dequeue_pop_front( (eu_context)->queue_name )
#else /* Don't use LIFO, Global LIFO or HWLOC (hbbuffer): use dequeue */
#  define DAGUE_POP(eu_context, queue_name) \
    (dague_execution_context_t*)dague_dequeue_pop_front( (eu_context)->queue_name )
#endif

static inline dague_execution_context_t *choose_local_job( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t *exec_context = NULL;

#if !defined(DAGUE_USE_LIFO) && !defined(DAGUE_USE_GLOBAL_LIFO) && !defined(HAVE_HWLOC) && PLACEHOLDER_SIZE
    if( eu_context->placeholder_pop != eu_context->placeholder_push ) {
        exec_context = eu_context->placeholder[eu_context->placeholder_pop];
        eu_context->placeholder_pop = ((eu_context->placeholder_pop + 1) % PLACEHOLDER_SIZE);
    } 
    else
#endif
        exec_context = DAGUE_POP(eu_context, eu_task_queue);
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
        do_some_work:
            TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);
            /* We're good to go ... */
            if( 0 == __dague_execute( eu_context, exec_context ) ) {
                dague_complete_execution( eu_context, exec_context );
            }
            nbiterations++;
        } else {
#if !defined(DAGUE_USE_GLOBAL_LIFO)
            miss_local++;
#endif
#if defined(DISTRIBUTED)
            /* check for remote deps completion */
            if(dague_remote_dep_progress(eu_context) > 0)  {
                found_remote++;
                
                exec_context = choose_local_job(eu_context);
                
                if( NULL != exec_context ) {
                    misses_in_a_row = 0;
                    goto do_some_work;
                }
            }
#else
            (void) found_remote;
#endif /* DISTRIBUTED */

#if !defined(DAGUE_USE_GLOBAL_LIFO)
            /* Work stealing from the other workers */
            {
                unsigned int i;
#  if defined(HAVE_HWLOC) 

                unsigned int max = eu_context->eu_nb_hierarch_queues;

#if !defined(USE_HIERARCHICAL_QUEUES)
                if( (unsigned int)master_context->taskstodo < 2 * (unsigned int)master_context->nb_cores ) {
                    unsigned int nbc = master_context->nb_cores /*dague_hwloc_nb_cores( 1, eu_context->eu_id )*/;
                    max = eu_context->eu_nb_hierarch_queues < nbc ? eu_context->eu_nb_hierarch_queues : nbc;
                }
#endif
                for(i = 0; i < max; i++ ) {
                    exec_context = DAGUE_POP( eu_context, eu_hierarch_queues[i] );
                    if( NULL != exec_context ) {
                        misses_in_a_row = 0;
                        found_victim++;
                        goto do_some_work;
                    }
                    miss_victim++;
                }
                exec_context = DAGUE_SYSTEM_POP( eu_context, eu_system_queue );
                if( NULL != exec_context ) {
                    misses_in_a_row = 0;
                    system_victim++;
                    goto do_some_work;
                }
                miss_victim++;
#  else 
                for(i = 0; i < (unsigned) master_context->nb_cores; i++ ) {
                    exec_context = DAGUE_POP( master_context->execution_units[i], eu_task_queue );
                    if( NULL != exec_context ) {
                        misses_in_a_row = 0;
                        found_victim++;
                        goto do_some_work;
                    }
                    miss_victim++;
                }
#  endif
            }
#endif  /* DAGUE_USE_GLOBAL_LIFO */
            misses_in_a_row++;
        }

        TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);
    }

#if defined(DAGUE_USE_LIFO) || defined(DAGUE_USE_GLOBAL_LIFO)
    assert(dague_atomic_lifo_is_empty(eu_context->eu_task_queue));
#elif defined(HAVE_HWLOC)
    assert(dague_hbbuffer_is_empty(eu_context->eu_task_queue));
#else
    assert(dague_dequeue_is_empty(eu_context->eu_task_queue));
#endif  /* defined(DAGUE_USE_LIFO) || defined(DAGUE_USE_GLOBAL_LIFO) */

    /* We're all done ? */
    dague_barrier_wait( &(master_context->barrier) );

    if( 0 != eu_context->eu_id ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
#if defined(DAGUE_REPORT_STATISTICS)
#if defined(DAGUE_USE_GLOBAL_LIFO)
    printf("# th <%3d> done %d\n", eu_context->eu_id, nbiterations);
#else
    printf("# th <%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu\n",
           eu_context->eu_id, nbiterations, (long long unsigned int)found_local,
           (long long unsigned int)found_remote,
           (long long unsigned int)found_victim,
           (long long unsigned int)miss_local,
           (long long unsigned int)miss_victim );
#endif  /* defined(DAGUE_USE_GLOBAL_LIFO) */
#endif  /* DAGUE_REPORT_STATISTICS */

    return (void*)(long)nbiterations;
}

int dague_enqueue( dague_context_t* context, dague_object_t* object)
{
#if 0
    if( NULL != object->context ) {
        return -1;  /* Already enqueued in another context */
    }
#endif
    dague_execution_context_t *startup_list = NULL;

    context->taskstodo += object->nb_local_tasks;
    if( NULL != object->startup_hook ) {
        object->startup_hook(context->execution_units[0], object, &startup_list);
        if( NULL != startup_list ) {
            /* We should add these tasks on the sstem queue */
            __dague_schedule( context->execution_units[0], startup_list, 0 );
        }
    }
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
