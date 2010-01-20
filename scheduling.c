/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifdef HAVE_CPU_SET_T
#include <linux/unistd.h>
#endif  /* HAVE_CPU_SET_T */
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <errno.h>
#include "scheduling.h"
#include "dequeue.h"
#include "remote_dep.h"
#include "profiling.h"

static int dplasma_execute( dplasma_execution_unit_t*, const dplasma_execution_context_t* );

#define DEPTH_FIRST_SCHEDULE 0

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
    /* Dirty workaround or how to deliberaty generate memory leaks */
    {
        int i, upto = dplasma_nb_elements();

        for( i = 0; i < upto; i++ ) {
            dplasma_t* object = (dplasma_t*)dplasma_element_at(i);
            object->deps = NULL;
        }

#if defined(DPLASMA_PROFILING)
        /* Reset the profiling information */
        for( i = 0; i < context->nb_cores; i++ ) {
            dplasma_profiling_reset( &context->execution_units[i] );
        }
#endif  /* defined(DPLASMA_PROFILING) */
    }

#if !DEPTH_FIRST_SCHEDULE
    {
        dplasma_execution_unit_t* eu_context;

        eu_context = &(context->execution_units[0]);

        return __dplasma_schedule( eu_context, exec_context );
    }
#else
    return dplasma_execute(eu_context, exec_context);
#endif  /* !DEPTH_FIRST_SCHEDULE */
}

int __dplasma_schedule( dplasma_execution_unit_t* eu_context,
                        const dplasma_execution_context_t* exec_context )
{
#if !DEPTH_FIRST_SCHEDULE
    dplasma_execution_context_t* new_context;

    new_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    memcpy( new_context, exec_context, sizeof(dplasma_execution_context_t) );
#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    dplasma_atomic_lifo_push( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context );
#else
    if( NULL == eu_context->placeholder ) {
        eu_context->placeholder = (void*)new_context;
    } else {
        dplasma_dequeue_push_front( eu_context->eu_task_queue, (dplasma_list_item_t*)new_context);
    }
#endif  /* DPLASMA_USE_LIFO */
    return 0;
#else
    printf( "This internal version of the dplasma_schedule is not supposed to be called\n");
    return -1;
#endif  /* !DEPTH_FIRST_SCHEDULE */
}

void dplasma_register_nb_tasks(dplasma_context_t* context, int n)
{
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

#define gettid() syscall(__NR_gettid)

void* __dplasma_progress( dplasma_execution_unit_t* eu_context )
{
    uint64_t found_local, miss_local, found_victim, miss_victim;
    dplasma_context_t* master_context = eu_context->master_context;
    int32_t my_barrier_counter = master_context->__dplasma_internal_finalization_counter;
    dplasma_execution_context_t* exec_context;
    int nbiterations;

    if( 0 != eu_context->eu_id ) {
#ifdef HAVE_CPU_SET_T
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
#if defined(HAVE_HWLOC)
        CPU_SET(eu_context->eu_steal_from[0], &cpuset);
#else
        CPU_SET(eu_context->eu_id, &cpuset);
#endif  /* defined(HAVE_HWLOC) */

        if( -1 == sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) ) {
            printf( "Unable to set the thread affinity (%s)\n", strerror(errno) );
        }
#endif  /* HAVE_CPU_SET_T */

        /* Force the kernel to bind me to the expected core */
        __do_some_computations();

        /* Wait until all threads are done binding themselves */
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
    found_local = miss_local = found_victim = miss_victim = 0;
    nbiterations = 0;

    while( !all_tasks_done(master_context) ) {
#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
        exec_context = (dplasma_execution_context_t*)dplasma_atomic_lifo_pop(eu_context->eu_task_queue);
#else
        if( NULL != eu_context->placeholder ) {
            exec_context = (dplasma_execution_context_t*)eu_context->placeholder;
            eu_context->placeholder = NULL;
        } else {
            /* extract the first execution context from the ready list */
            exec_context = (dplasma_execution_context_t*)dplasma_dequeue_pop_front(eu_context->eu_task_queue);
        }
#endif  /* DPLASMA_USE_LIFO */

        if( exec_context != NULL ) {
            found_local++;
        do_some_work:
            /* We're good to go ... */
            dplasma_execute( eu_context, exec_context );
            done_task(master_context);
            nbiterations++;
            /* Release the execution context */
            free( exec_context );
        } else {
#if defined(USE_MPI)
            /* check for remote deps completion */
            if(dplasma_remote_dep_progress(eu_context) > 0)
                continue;
#endif  /* defined(USE_MPI) */
#if !defined(DPLASMA_USE_GLOBAL_LIFO)
            miss_local++;
            /* Work stealing from the other workers */
#if defined(HAVE_HWLOC)
            int i = 1;
#else
            int i = 0;
#endif  /* defined(HAVE_HWLOC) */
            for( ; i < master_context->nb_cores; i++ ) {
                dplasma_execution_unit_t* victim;
#if defined(HAVE_HWLOC)
                victim = &(master_context->execution_units[eu_context->eu_steal_from[i]]);
#else
                victim = &(master_context->execution_units[i]);
#endif  /* defined(HAVE_HWLOC) */
#if defined(DPLASMA_USE_LIFO)
                exec_context = (dplasma_execution_context_t*)dplasma_atomic_lifo_pop(victim->eu_task_queue);
#else
                exec_context = (dplasma_execution_context_t*)dplasma_dequeue_pop_back(victim->eu_task_queue);
#endif  /* DPLASMA_USE_LIFO */
                if( NULL != exec_context ) {
                    found_victim++;
                    goto do_some_work;
                } else {
                    miss_victim++;
                }
            }
#endif  /* DPLASMA_USE_GLOBAL_LIFO */
        }
    }

#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    while( NULL != (exec_context = (dplasma_execution_context_t*)dplasma_atomic_lifo_pop(eu_context->eu_task_queue)) ) {
        char tmp[128];
        dplasma_service_to_string( exec_context, tmp, 128 );
        printf( "[iteration %d: th %d] Pending task: %s\n", my_barrier_counter, eu_context->eu_id, tmp );
    }
    assert(dplasma_atomic_lifo_is_empty(eu_context->eu_task_queue));
#else
    while( NULL != (exec_context = (dplasma_execution_context_t*)dplasma_dequeue_pop_back(eu_context->eu_task_queue)) ) {
        char tmp[128];
        dplasma_service_to_string( exec_context, tmp, 128 );
        printf( "[iteration %d: th %d] Pending task: %s\n", my_barrier_counter, eu_context->eu_id, tmp );
    }
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
    printf("# th <%3d> done %d local %llu stolen %llu miss %llu failed %llu\n",
           eu_context->eu_id, nbiterations, (long long unsigned int)found_local,
           (long long unsigned int)found_victim,
           (long long unsigned int)miss_local,
           (long long unsigned int)miss_victim );
#endif  /* defined(DPLASMA_USE_GLOBAL_LIFO) */
#endif  /* DPLASMA_REPORT_STATISTICS */

    return (void*)(long)nbiterations;
}

int dplasma_progress(dplasma_context_t* context)
{
    int ret = (int)(long)__dplasma_progress( &(context->execution_units[0]) );

    context->__dplasma_internal_finalization_counter++;
    return ret;
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

#ifdef USE_MPI
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

static int dplasma_execute( dplasma_execution_unit_t* eu_context,
                            const dplasma_execution_context_t* exec_context )
{
    dplasma_t* function = exec_context->function;
#ifdef _DEBUG
    char tmp[128];
#endif

    if( NULL != function->hook ) {
        function->hook( eu_context, exec_context );
    } else {
        DEBUG(( "Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    }
    return dplasma_trigger_dependencies( eu_context, exec_context, 1 );
}
