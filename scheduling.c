/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "scheduling.h"
#include "dequeue.h"
#include "profiling.h"
#include "remote_dep.h"
#include "dplasma.h"
#include "stats.h"

#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <errno.h>

#ifdef  HAVE_SCHED_SETAFFINITY
#include <linux/unistd.h>
#endif  /* HAVE_SCHED_SETAFFINITY */

#if defined(DPLASMA_PROFILING) && 0
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dplasma_profiling_trace((EU_PROFILE), (KEY), (ID))
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

static int dplasma_execute( dplasma_execution_unit_t*, dplasma_execution_context_t* );

static inline void set_tasks_todo(dplasma_context_t* context, uint32_t n)
{
    context->taskstodo = n;
}

static inline int all_tasks_done(dplasma_context_t* context)
{
    return (context->taskstodo == 0);
}

static inline void done_task(dplasma_context_t* context)
{
    dplasma_atomic_dec_32b( &(context->taskstodo) );
}

/**
 * Schedule the instance of the service based on the values of the
 * local variables stored in the execution context, by calling the
 * attached hook if any. At the end of the execution the dependencies
 * are released.
 */
int dplasma_schedule( dplasma_context_t* context, const dplasma_execution_context_t* exec_context )
{
    dplasma_execution_context_t* new_context;
    dplasma_execution_unit_t* eu_context;

    eu_context = context->execution_units[0];

    new_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    memcpy( new_context, exec_context, sizeof(dplasma_execution_context_t) );
#if defined(DPLASMA_CACHE_AWARE)
    new_context->pointers[1] = NULL;
#endif
    DPLASMA_LIST_ITEM_SINGLETON( new_context );
    return __dplasma_schedule( eu_context, new_context, 1);
}

int __dplasma_schedule( dplasma_execution_unit_t* eu_context,
                        dplasma_execution_context_t* new_context, int use_placeholder )
{
    TAKE_TIME(eu_context->eu_profile, schedule_push_begin, 0);

#if defined(DPLASMA_DEBUG)
    {
        char tmp[128];
        dplasma_list_item_t* item = (dplasma_list_item_t*)new_context, *next;

        do {
            next = (dplasma_list_item_t*)item->list_next;
            DEBUG(( "thread %d Schedule %s\n", eu_context->eu_id, dplasma_service_to_string((dplasma_execution_context_t*)item, tmp, 128)));
            printf( "thread %d Schedule %s\n", eu_context->eu_id, dplasma_service_to_string((dplasma_execution_context_t*)item, tmp, 128));
            item = next;
        } while ( item != (dplasma_list_item_t*)new_context );
    }
# endif

#  if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    dplasma_atomic_lifo_push( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context );
#  elif defined(HAVE_HWLOC)
    {
#    if defined(USE_HIERARCHICAL_QUEUES)
        int i;
        for(i = 0; i < eu_context->eu_nb_hierarch_queues; i++) {
            /** Be nice: share. Push in the closest buffer that is not ideally filled 
             *  (mine if I'm starving) */
            if( dplasma_hbbuffer_push_ideal_nonrec( eu_context->eu_hierarch_queues[i], 
                                                    (dplasma_list_item_t**)&new_context ) ) {
                /** Every contexts were pushed at this level or below */
                goto done_pushing_tasks;
            }
        }
#    endif /* USE_HIERARCHICAL_QUEUES */
        /** We couldn't push more: everybody above me (and myself) are ideally full, so 
         *  let's overfill, potentially pushing recursively in the system queue
         */
        dplasma_hbbuffer_push_all( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context );
    }
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

            if( new_context->list_item.list_next == (dplasma_list_item_t*)new_context )
                goto done_pushing_tasks;

            new_context->list_item.list_next->list_prev = new_context->list_item.list_prev;
            new_context->list_item.list_prev->list_next = new_context->list_item.list_next;
            new_context = (dplasma_execution_context_t*)new_context->list_item.list_next;
        }
    }
#    endif  /* PLACEHOLDER_SIZE */
    if( new_context->function->flags & DPLASMA_HIGH_PRIORITY_TASK ) {
        dplasma_dequeue_push_front( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context);
    } else {
        dplasma_dequeue_push_back( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context);
    }
#  endif  /* DPLASMA_USE_LIFO */
 done_pushing_tasks:
    TAKE_TIME( eu_context->eu_profile, schedule_push_end, 0);

    return 0;
}

void dplasma_register_nb_tasks(dplasma_context_t* context, int n)
{
#if defined(DPLASMA_PROFILING)
    /* Reset the profiling information */
    dplasma_profiling_reset();
#endif  /* defined(DPLASMA_PROFILING) */
        
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
#  if defined(DPLASMA_CACHE_AWARE)
static  unsigned int ranking_function_bycache(dplasma_list_item_t *elt, void *param)
{
    unsigned int value;
    cache_t *cache = (cache_t*)param;
    dplasma_execution_context_t *exec = (dplasma_execution_context_t*)elt;
    
    /* TODO: fix this, depends on the depth */
    value = exec->function->cache_rank_function(exec, cache, 128);
    DEBUG(("maxvalue of this choice is %u\n", value));
    return value;
}
#  else
static  unsigned int ranking_function_firstfound(dplasma_list_item_t *elt, void *_)
{
    return DPLASMA_RANKING_FUNCTION_BEST;
}
#  endif
#endif

#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
#  define DPLASMA_POP(eu_context, queue_name) \
    (dplasma_execution_context_t*)dplasma_atomic_lifo_pop( (eu_context)->queue_name )
#elif defined(HAVE_HWLOC) 
#  if defined(DPLASMA_CACHE_AWARE)
#    define DPLASMA_POP(eu_context, queue_name) \
    (dplasma_execution_context_t*)dplasma_hbbuffer_pop_best((eu_context)->queue_name, \
                                                            ranking_function_bycache, \
                                                            (eu_context)->closest_cache)
#  else
#    define DPLASMA_POP(eu_context, queue_name) \
    (dplasma_execution_context_t*)dplasma_hbbuffer_pop_best((eu_context)->queue_name, \
                                                            ranking_function_firstfound, \
                                                            NULL)
#  endif /* DPLASMA_CACHE_AWARE */
#  define DPLASMA_SYSTEM_POP(eu_context, queue_name) (dplasma_execution_context_t*)dplasma_dequeue_pop_front( (eu_context)->queue_name )
#else /* Don't use LIFO, Global LIFO or HWLOC (hbbuffer): use dequeue */
#  define DPLASMA_POP(eu_context, queue_name) \
    (dplasma_execution_context_t*)dplasma_dequeue_pop_front( (eu_context)->queue_name )
#endif

static inline dplasma_execution_context_t *choose_local_job( dplasma_execution_unit_t *eu_context )
{
    dplasma_execution_context_t *exec_context = NULL;

#if !defined(DPLASMA_USE_LIFO) && !defined(DPLASMA_USE_GLOBAL_LIFO) && !defined(HAVE_HWLOC) && PLACEHOLDER_SIZE
    if( eu_context->placeholder_pop != eu_context->placeholder_push ) {
        exec_context = eu_context->placeholder[eu_context->placeholder_pop];
        eu_context->placeholder_pop = ((eu_context->placeholder_pop + 1) % PLACEHOLDER_SIZE);
    } 
    else
#endif
        exec_context = DPLASMA_POP(eu_context, eu_task_queue);
    return exec_context;
}

inline int dplasma_complete_execution( dplasma_execution_unit_t *eu_context,
                                       dplasma_execution_context_t *exec_context )
{
    int rc = exec_context->function->complete_execution( eu_context, exec_context );
    /* Update the number of remaining tasks */
    done_task(eu_context->master_context);

    /* Succesfull execution. The context is ready to be released, all
     * dependencies have been marked as completed.
     */
    DEBUG_MARK_EXE( eu_context->eu_id, exec_context );
    /* Release the execution context */
    DPLASMA_STAT_DECREASE(mem_contexts, sizeof(dplasma_execution_context_t) + STAT_MALLOC_OVERHEAD);
    free( exec_context );
    return rc;
}

void* __dplasma_progress( dplasma_execution_unit_t* eu_context )
{
    uint64_t found_local, miss_local, found_victim, miss_victim, found_remote;
    uint64_t misses_in_a_row;
    dplasma_context_t* master_context = eu_context->master_context;
    int32_t my_barrier_counter = master_context->__dplasma_internal_finalization_counter;
    dplasma_execution_context_t* exec_context;
    int nbiterations = 0;
    struct timespec rqtp;

    rqtp.tv_sec = 0;
    found_local = miss_local = found_victim = miss_victim = found_remote = 0;
    misses_in_a_row = 1;
    
    if( 0 != eu_context->eu_id ) {
        /* Force the kernel to bind me to the expected core */
        __do_some_computations();

        /* Wait until all threads are done binding themselves 
         * (see dplasma_init) */
        dplasma_barrier_wait( &(master_context->barrier) );
        my_barrier_counter = 1;
    }

    /* The main loop where all the threads will spend their time */
 wait_for_the_next_round:
    /* Wait until all threads are here and the main thread signal the begining of the work */
    dplasma_barrier_wait( &(master_context->barrier) );

    if( master_context->__dplasma_internal_finalization_in_progress ) {
        my_barrier_counter++;
        for(; my_barrier_counter <= master_context->__dplasma_internal_finalization_counter; my_barrier_counter++ ) {
            dplasma_barrier_wait( &(master_context->barrier) );
        }
        goto finalize_progress;
    }

    while( !all_tasks_done(master_context) ) {

        if( misses_in_a_row > 1 ) {
            rqtp.tv_nsec = exponential_backoff(misses_in_a_row);
            DPLASMA_STATACC_ACCUMULATE(time_starved, rqtp.tv_nsec/1000);
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
            if( 0 == dplasma_execute( eu_context, exec_context ) ) {
                dplasma_complete_execution( eu_context, exec_context );
            }
            nbiterations++;
        } else {
#if !defined(DPLASMA_USE_GLOBAL_LIFO)
            miss_local++;
#endif
#if defined(DISTRIBUTED)
            /* check for remote deps completion */
            if(dplasma_remote_dep_progress(eu_context) > 0)  {
                found_remote++;
                
                exec_context = choose_local_job(eu_context);
                
                if( NULL != exec_context ) {
                    misses_in_a_row = 0;
                    goto do_some_work;
                }
            }
#endif /* DISTRIBUTED */

#if !defined(DPLASMA_USE_GLOBAL_LIFO)
            /* Work stealing from the other workers */
            {
                int i;
#  if defined(HAVE_HWLOC) 

                int max = eu_context->eu_nb_hierarch_queues;

#if !defined(USE_HIERARCHICAL_QUEUES)
                if( master_context->taskstodo < 2 * master_context->nb_cores ) {
                    int nbc = dplasma_hwloc_nb_cores( 1, eu_context->eu_id );
                    max = eu_context->eu_nb_hierarch_queues < nbc ? eu_context->eu_nb_hierarch_queues : nbc;
                }
#endif
                for(i = 0; i < max; i++ ) {
                    exec_context = DPLASMA_POP( eu_context, eu_hierarch_queues[i] );
                    if( NULL != exec_context ) {
                        misses_in_a_row = 0;
                        found_victim++;
                        goto do_some_work;
                    }
                    miss_victim++;
                }
                exec_context = DPLASMA_SYSTEM_POP( eu_context, eu_system_queue );
                if( NULL != exec_context ) {
                    misses_in_a_row = 0;
                    found_victim++;
                    goto do_some_work;
                }
                miss_victim++;
#  else 
                for(i = 0; i < master_context->nb_cores; i++ ) {
                    exec_context = DPLASMA_POP( master_context->execution_units[i], eu_task_queue );
                    if( NULL != exec_context ) {
                        misses_in_a_row = 0;
                        found_victim++;
                        goto do_some_work;
                    }
                    miss_victim++;
                }
#  endif
            }
#endif  /* DPLASMA_USE_GLOBAL_LIFO */
            misses_in_a_row++;
        }

        TAKE_TIME( eu_context->eu_profile, schedule_poll_end, nbiterations);
    }

#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    assert(dplasma_atomic_lifo_is_empty(eu_context->eu_task_queue));
#elif defined(HAVE_HWLOC)
    assert(dplasma_hbbuffer_is_empty(eu_context->eu_task_queue));
#else
    assert(dplasma_dequeue_is_empty(eu_context->eu_task_queue));
#endif  /* defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO) */

    /* We're all done ? */
    dplasma_barrier_wait( &(master_context->barrier) );

    if( 0 != eu_context->eu_id ) {
        my_barrier_counter++;
        goto wait_for_the_next_round;
    }

 finalize_progress:
#if defined(DPLASMA_REPORT_STATISTICS)
#if defined(DPLASMA_USE_GLOBAL_LIFO)
    printf("# th <%3d> done %d\n", eu_context->eu_id, nbiterations);
#else
    printf("# th <%3d> done %6d | local %6llu | remote %6llu | stolen %6llu | starve %6llu | miss %6llu\n",
           eu_context->eu_id, nbiterations, (long long unsigned int)found_local,
           (long long unsigned int)found_remote,
           (long long unsigned int)found_victim,
           (long long unsigned int)miss_local,
           (long long unsigned int)miss_victim );
#endif  /* defined(DPLASMA_USE_GLOBAL_LIFO) */
#endif  /* DPLASMA_REPORT_STATISTICS */

    return (void*)(long)nbiterations;
}

int dplasma_progress(dplasma_context_t* context)
{
    int ret;
    dplasma_remote_dep_on(context);
    
    ret = (int)(long)__dplasma_progress( context->execution_units[0] );

    context->__dplasma_internal_finalization_counter++;
    dplasma_remote_dep_off(context);
    return ret;
}

static int dplasma_execute( dplasma_execution_unit_t* eu_context,
                            dplasma_execution_context_t* exec_context )
{
    dplasma_t* function = exec_context->function;
    int rc = 0;
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif
    
    DEBUG(( "Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    DPLASMA_STAT_DECREASE(counter_nbtasks, 1ULL);

    if( NULL != function->hook ) {
        rc = function->hook( eu_context, exec_context );
    }
#ifdef DEPRECATED
    if( 0 == rc ) {
        return dplasma_trigger_dependencies( eu_context, exec_context, 1 );
    }
#endif
    return rc;
}


int dplasma_trigger_dependencies( dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* exec_context,
                                int forward_remote )
{
    param_t* param;
    dep_t* dep;
    dplasma_t* function = exec_context->function;
    dplasma_execution_context_t new_context;
    int i, j, k, value;    

#if 0 /*DISTRIBUTED this code is outdated, does not work in distributed anymore */
    dplasma_remote_dep_reset_forwarded(eu_context);
#endif

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->inout[i]); i++ ) {
        param = function->inout[i];
        
        if( !(SYM_OUT & param->sym_type) ) {
            continue;  /* this is only an INPUT dependency */
        }
        for( j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != param->dep_out[j]); j++ ) {
            int dont_generate = 0;
            
            dep = param->dep_out[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                (void)expr_eval( dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
                if( 0 == value ) {
                    continue;
                }
            }
            new_context.function = dep->dplasma;
            /* Nothing to do is this is not a real function */
            if( 0 == new_context.function->nb_locals ) continue;

            DEBUG(( " -> %s( ", dep->dplasma->name ));
            /* Check to see if any of the params are conditionals or ranges and if they are
             * if they match. If yes, then set the correct values.
             */
            for( k = 0; (k < MAX_CALL_PARAM_COUNT) && (NULL != dep->call_params[k]); k++ ) {
                new_context.locals[k].sym = dep->dplasma->locals[k];
                if( EXPR_OP_BINARY_RANGE != dep->call_params[k]->op ) {
                    (void)expr_eval( dep->call_params[k], exec_context->locals, MAX_LOCAL_COUNT, &value );
                    new_context.locals[k].min = new_context.locals[k].max = value;
                    DEBUG(( "%d ", value ));
                } else {
                    int min, max;
                    (void)expr_range_to_min_max( dep->call_params[k], exec_context->locals, MAX_LOCAL_COUNT, &min, &max );
                    if( min > max ) {
                        dont_generate = 1;
                        DEBUG(( " -- skipped" ));
                        break;  /* No reason to continue here */
                    }
                    new_context.locals[k].min = min;
                    new_context.locals[k].max = max;
                    if( min == max ) {
                        DEBUG(( "%d ", min ));
                    } else {
                        DEBUG(( "[%d..%d] ", min, max ));
                    }
                }
                new_context.locals[k].value = new_context.locals[k].min;
            }
            DEBUG(( ")\n" ));
            if( dont_generate ) {
                continue;
            }
            
            /* Mark the end of the list */
            if( k < MAX_CALL_PARAM_COUNT ) {
                new_context.locals[k].sym = NULL;
            }
            dplasma_release_OUT_dependencies( eu_context,
                                              exec_context, param,
                                              &new_context, dep->param, forward_remote );
        }
    }
    return 0;
}

