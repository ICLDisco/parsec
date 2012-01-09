/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"
#include "stats.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include "scheduling.h"
#include "barrier.h"
#include "remote_dep.h"
#include "bindthread.h"
#include "dague_prof_grapher.h"
#include "priority_sorted_queue.h"

#ifdef DAGUE_PROF_TRACE
#include "profiling.h"
#endif

#ifdef HAVE_PAPI
#include <papime.h>
#endif

#ifdef HAVE_HWLOC
#include "hbbuffer.h"
#include "dague_hwloc.h"
#endif

#ifdef HAVE_CUDA
#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime_api.h"
#endif

dague_allocate_data_t dague_data_allocate = malloc;
dague_free_data_t     dague_data_free = free;

#ifdef DAGUE_PROF_TRACE
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
#endif  /* DAGUE_PROF_TRACE */

#ifdef HAVE_PAPI
int eventSet = PAPI_NULL;
int num_events = 0;
char* event_names[MAX_EVENTS];
#endif

#ifdef HAVE_HWLOC
#define MAX_CORE_LIST 128
#endif

#if defined(HAVE_GETRUSAGE)
#include <sys/time.h>
#include <sys/resource.h>

static int _dague_rusage_first_call = 1;
static struct rusage _dague_rusage;

static void dague_object_empty_repository(void);

static void dague_statistics(char* str)
{
    struct rusage current;

    getrusage(RUSAGE_SELF, &current);

    if ( !_dague_rusage_first_call ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - _dague_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - _dague_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - _dague_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - _dague_rusage.ru_stime.tv_usec) / 1000000.0);

        printf("=============================================================\n");
        printf("%s: Resource Usage Data...\n", str);
        printf("-------------------------------------------------------------\n");
        printf("User Time   (secs)          : %10.3f\n", usr);
        printf("System Time (secs)          : %10.3f\n", sys);
        printf("Total Time  (secs)          : %10.3f\n", usr + sys);
        printf("Minor Page Faults           : %10ld\n", (current.ru_minflt  - _dague_rusage.ru_minflt));
        printf("Major Page Faults           : %10ld\n", (current.ru_majflt  - _dague_rusage.ru_majflt));
        printf("Swap Count                  : %10ld\n", (current.ru_nswap   - _dague_rusage.ru_nswap));
        printf("Voluntary Context Switches  : %10ld\n", (current.ru_nvcsw   - _dague_rusage.ru_nvcsw));
        printf("Involuntary Context Switches: %10ld\n", (current.ru_nivcsw  - _dague_rusage.ru_nivcsw));
        printf("Block Input Operations      : %10ld\n", (current.ru_inblock - _dague_rusage.ru_inblock));
        printf("Block Output Operations     : %10ld\n", (current.ru_oublock - _dague_rusage.ru_oublock));
        printf("=============================================================\n");
    }

    _dague_rusage_first_call = !_dague_rusage_first_call;
    _dague_rusage = current;

    return;
}
#else
static void dague_statistics(char* str) { (void)str; return; }
#endif /* defined(HAVE_GETRUSAGE) */

typedef struct __dague_temporary_thread_initialization_t {
    dague_vp_t *virtual_process;
    int th_id;
    int nb_cores;
    int bindto;
} __dague_temporary_thread_initialization_t;

static int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
					 __dague_temporary_thread_initialization_t* startup);
static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context);

const dague_function_t* dague_find(const dague_object_t *dague_object, const char *fname)
{
    unsigned int i;
    const dague_function_t* object;

    for( i = 0; i < dague_object->nb_functions; i++ ) {
        object = dague_object->functions_array[i];
        if( 0 == strcmp( object->name, fname ) ) {
            return object;
        }
    }
    return NULL;
}

static void* __dague_thread_init( __dague_temporary_thread_initialization_t* startup )
{
    dague_execution_unit_t* eu;
    int pi;

    /* Bind to the specified CORE */
    dague_bindthread(startup->bindto);
    DEBUG(("Bind thread %i on core %i\n", startup->th_id, startup->bindto));
    
    eu = (dague_execution_unit_t*)malloc(sizeof(dague_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->th_id           = startup->th_id;
    eu->virtual_process = startup->virtual_process;
    eu->scheduler_object = NULL;
    startup->virtual_process->execution_units[startup->th_id] = eu;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    eu->sched_nb_tasks_done = 0;
#endif

    eu->context_mempool = &(eu->virtual_process->context_mempool.thread_mempools[eu->th_id]);
    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
        eu->datarepo_mempools[pi] = &(eu->virtual_process->datarepo_mempools[pi].thread_mempools[eu->th_id]);

#ifdef DAGUE_PROF_TRACE
    eu->eu_profile = dague_profiling_thread_init( 2*1024*1024, "DAGuE Thread %d of VP %d", eu->th_id, eu->virtual_process->vp_id );
#endif

#if defined(DAGUE_SIM)
    eu->largest_simulation_date = 0;
#endif

    /* The main thread of VP 0 will go back to the user level */
    if( DAGUE_THREAD_IS_MASTER(eu) )
        return NULL;

    return __dague_progress(eu);
}

#ifdef HAVE_PAPI
extern int num_events;
extern char* event_names[];
#endif

static void dague_vp_init( dague_vp_t *vp,
                           int32_t nb_cores,
                           __dague_temporary_thread_initialization_t *startup)
{
    int t, pi;
    dague_execution_context_t fake_context;
    data_repo_entry_t fake_entry;

    vp->nb_cores = nb_cores;
#if defined(DAGUE_SIM)
    vp->largest_simulation_date = 0;
#endif /* DAGUE_SIM */
    
    dague_mempool_construct( &vp->context_mempool, sizeof(dague_execution_context_t),
                             ((char*)&fake_context.mempool_owner) - ((char*)&fake_context), 
                             vp->nb_cores );

    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
        dague_mempool_construct( &vp->datarepo_mempools[pi], 
                                 sizeof(data_repo_entry_t)+(pi-1)*sizeof(dague_arena_chunk_t*),
                                 ((char*)&fake_entry.data_repo_mempool_owner) - ((char*)&fake_entry),
                                 vp->nb_cores);


    /* Prepare the temporary storage for each thread startup */
    for( t = 0; t < vp->nb_cores; t++ ) {
        startup[t].th_id = t;
        startup[t].virtual_process = vp;
        startup[t].nb_cores = nb_cores;
        startup[t].bindto = -1;
    }
}

dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[] )
{
    int argc = (*pargc);
    int nb_vp = 1;
    int p, t, nb_total_comp_threads;
    char** argv = NULL;
    __dague_temporary_thread_initialization_t *startup;
    dague_context_t* context;
    
#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
        {
	    {"dague_help",       no_argument,        NULL, 'h'},
            {"dague_bind",       optional_argument,  NULL, 'b'},
            {"dague_bind_comm",  optional_argument,  NULL, 'c'},
            {0, 0, 0, 0}
        };
#endif  /* defined(HAVE_GETOPT_LONG) */

    context = (dague_context_t*)malloc(sizeof(dague_context_t) + (nb_vp-1) * sizeof(dague_vp_t*));

    context->__dague_internal_finalization_in_progress = 0;
    context->__dague_internal_finalization_counter = 0;
    context->nb_nodes       = 1;
    context->active_objects = 0;
    context->my_rank        = 0;

    /* TODO: nb_cores should depend on the vp_id */
    nb_total_comp_threads = 0;
    for(p = 0; p < nb_vp; p++) {
        nb_total_comp_threads += nb_cores;
    }
    startup = 
        (__dague_temporary_thread_initialization_t*)malloc(nb_total_comp_threads * sizeof(__dague_temporary_thread_initialization_t));

    context->nb_vp = nb_vp;
    t = 0;
    for(p = 0; p < nb_vp; p++) {
        dague_vp_t *vp;
        vp = (dague_vp_t *)malloc(sizeof(dague_vp_t) + (nb_cores-1) * sizeof(dague_execution_unit_t*));
        vp->dague_context = context;
        vp->vp_id = p;
        context->virtual_processes[p] = vp;
        /** This creates startup[t] -> startup[t+nb_cores] */
        dague_vp_init(vp, nb_cores, &(startup[t]));
        t += nb_cores;
    }

#if defined(HAVE_HWLOC)
    dague_hwloc_init();
    context->comm_th_core   = -1;
    context->comm_th_binding_mask = hwloc_bitmap_alloc(); 
    context->core_free_mask = hwloc_bitmap_alloc();
    hwloc_bitmap_set_range(context->core_free_mask, 0, dague_hwloc_nb_real_cores()-1);
#endif  /* defined(HWLOC) */

#ifdef HAVE_PAPI
    num_events = 0;
#endif
    
    {
        int index = 0;
        /* Check for the upper level arguments */
        while(1) {
            if( NULL == (*pargv)[index] )
                break;
            if( 0 == strcmp( "--", (*pargv)[index]) ) {
                argv = &(*pargv)[index];
                break;
            }
            index++;
        }
        argc = (*pargc) - index;
    }

    if( argv != NULL ) {
        optind = 1;
        do {
            int ret;
#if defined(HAVE_GETOPT_LONG)
            int option_index = 0;
            
            ret = getopt_long (argc, argv, "p:b:c:",
                               long_options, &option_index);
#else
            ret = getopt (argc, argv, "p:b:c:");
#endif  /* defined(HAVE_GETOPT_LONG) */
            if( -1 == ret ) break;  /* we're done */

            switch(ret) {
            case 'h': dague_usage(); break;
            case 'c': dague_parse_comm_binding_parameter(optarg, context); break;
            case 'b': dague_parse_binding_parameter(optarg, context, startup); break;
            }
        } while(1);
    }
    
#ifdef HAVE_HWLOC 
    /* update the core_free_mask according to the thread binding defined */ 
    for(t = 0; t < nb_total_comp_threads; t++)
        hwloc_bitmap_clr(context->core_free_mask, startup[t].bindto);     
#endif  	
    
    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_total_comp_threads );
#ifdef DAGUE_PROF_TRACE
    dague_profiling_init( "%s", (*pargv)[0] );

    dague_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                            0, NULL,
                                            &MEMALLOC_start_key, &MEMALLOC_end_key);
    dague_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                            0, NULL,
                                            &schedule_poll_begin, &schedule_poll_end);
    dague_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                            0, NULL,
                                            &schedule_push_begin, &schedule_push_end);
    dague_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                            0, NULL,
                                            &schedule_sleep_begin, &schedule_sleep_end);
#endif  /* DAGUE_PROF_TRACE */


    if( nb_total_comp_threads > 1 ) {
        pthread_attr_t thread_attr;

        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
        pthread_setconcurrency(nb_total_comp_threads);
#endif  /* __linux */

        context->pthreads = (pthread_t*)malloc(nb_total_comp_threads * sizeof(pthread_t));

        /* The first execution unit is for the master thread */
        for( t = 1; t < nb_total_comp_threads; t++ ) {
            pthread_create( &((context)->pthreads[t]),
                            &thread_attr,
                            (void* (*)(void*))__dague_thread_init,
                            (void*)&(startup[t]));
        }
    } else {
        context->pthreads = NULL;
    }

    __dague_thread_init( &startup[0] );

    /* Wait until all threads are done binding themselves */
    dague_barrier_wait( &(context->barrier) );
    context->__dague_internal_finalization_counter++;

    /* Release the temporary array used for starting up the threads */
    free(startup);

    /* Introduce communication thread */
    context->nb_nodes = dague_remote_dep_init(context);
    
    dague_statistics("DAGuE");

    return context;
}

static void dague_vp_fini( dague_vp_t *vp )
{
    int i;
    dague_mempool_destruct( &vp->context_mempool );
    for(i = 0; i <= MAX_PARAM_COUNT; i++)
        dague_mempool_destruct( &vp->datarepo_mempools[i]);

    for(i = 0; i < vp->nb_cores; i++) {
        free(vp->execution_units[i]);
        vp->execution_units[i] = NULL;
    }

}

/**
 *
 */
int dague_fini( dague_context_t** pcontext )
{
    dague_context_t* context = *pcontext;
    int nb_total_comp_threads, t, p;

    nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

    /* Now wait until every thread is back */
    context->__dague_internal_finalization_in_progress = 1;
    dague_barrier_wait( &(context->barrier) );

    /* The first execution unit is for the master thread */
    if( nb_total_comp_threads > 1 ) {
        for(t = 1; t < nb_total_comp_threads; t++) {
            pthread_join( context->pthreads[t], NULL );
        }
        free(context->pthreads);
        context->pthreads = NULL;
    }

    (void) dague_remote_dep_fini( context );

    dague_set_scheduler( context, NULL );

    for(p = 0; p < context->nb_vp; p++) {
        dague_vp_fini(context->virtual_processes[p]);
        free(context->virtual_processes[p]);
        context->virtual_processes[p] = NULL;
    }

#ifdef DAGUE_PROF_TRACE
    dague_profiling_fini( );
#endif  /* DAGUE_PROF_TRACE */

    /* Destroy all resources allocated for the barrier */
    dague_barrier_destroy( &(context->barrier) );

#if defined(HAVE_HWLOC)
    /* Release thread binding masks */
    hwloc_bitmap_free(context->comm_th_binding_mask);
    hwloc_bitmap_free(context->core_free_mask);

    dague_hwloc_fini();
#endif  /* defined(HWLOC) */

#if defined(DAGUE_STATS)
    {
        char filename[64];
        char prefix[32];
# if defined(DISTRIBUTED) && defined(HAVE_MPI)
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        snprintf(filename, 64, "dague-%d.stats", rank);
        snprintf(prefix, 32, "%d/%d", rank, size);
# else
        snprintf(filename, 64, "dague.stats");
        prefix[0] = '\0';
# endif
        dague_stats_dump(filename, prefix);
    }
#endif

    dague_object_empty_repository();

    free(context);
    *pcontext = NULL;
    return 0;
}

/**
 * Convert the execution context to a string.
 */
char* dague_service_to_string( const dague_execution_context_t* exec_context,
                               char* tmp,
                               size_t length )
{
    const dague_function_t* function = exec_context->function;
    unsigned int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s", function->name );
    if( index >= length ) return tmp;
    for( i = 0; i < function->nb_parameters; i++ ) {
        index += snprintf( tmp + index, length - index, "%s%d",
                           (i == 0) ? "(" : ", ",
                           exec_context->locals[i].value );
        if( index >= length ) return tmp;
    }
    index += snprintf(tmp + index, length - index, ")");

    return tmp;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t
dague_check_IN_dependencies( const dague_object_t *dague_object,
                             const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, mask, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];
        /* this param has no dependency condition satisfied */
#if defined(DAGUE_SCHED_DEPS_MASK)
        mask = (1 << flow->flow_index);
#else
        mask = 1;
#endif
        if( ACCESS_NONE == flow->access_type ) {
            active = mask;
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func(dague_object, exec_context->locals) ) {
                        continue;
                    }
                }
                active = 0;
                break;
            }
        } else {
            active = 0;
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( dep->dague->nb_parameters == 0 ) {  /* this is only true for memory locations */
                    if( NULL != dep->cond ) {
                        /* Check if the condition apply on the current setting */
                        assert( dep->cond->op == EXPR_OP_INLINE );
                        if( 0 == dep->cond->inline_func(dague_object, exec_context->locals) ) {
                            continue;
                        }
                    }
                    active = mask;
                    break;
                }
            }
        }
        ret += active;
    }
    return ret;
}

static dague_dependency_t *find_deps(dague_object_t *dague_object,
                                     dague_execution_context_t* restrict exec_context)
{
    dague_dependencies_t *deps;
    int p;

    deps = dague_object->dependencies_array[exec_context->function->deps];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_parameters - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[p].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[exec_context->locals[exec_context->function->nb_parameters - 1].value - deps->min]);
}

/**
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const dague_flow_t* restrict origin_flow,
                                          dague_execution_context_t* restrict exec_context,
                                          const dague_flow_t* restrict dest_flow,
                                          data_repo_entry_t* dest_repo_entry,
                                          dague_execution_context_t** pready_list )
{
    const dague_function_t* function = exec_context->function;
    dague_dependency_t *deps;
    dague_dependency_t dep_new_value, dep_cur_value;
#if defined(DAGUE_DEBUG)
    char tmp[128];
#endif

    (void)eu_context;

    DEBUG(("Activate dependencies for %s priority %d\n",
           dague_service_to_string(exec_context, tmp, 128), exec_context->priority));
    deps = find_deps(dague_object, exec_context);
    
#if !defined(DAGUE_SCHED_DEPS_MASK)

    if( 0 == *deps ) {
        dep_new_value = 1 + dague_check_IN_dependencies( dague_object, exec_context );
        if( dague_atomic_cas( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_inc_32b( deps );
    } else {
        dep_cur_value = dague_atomic_inc_32b( deps );
    }

#if defined(DAGUE_DEBUG)
    if( dep_cur_value > function->dependencies_goal ) {
        DEBUG(("function %s as reached a dependency count of %d, higher than the goal dependencies count of %d\n",
               dague_service_to_string(exec_context, tmp, 128), dep_cur_value, function->dependencies_goal));
        assert(dep_cur_value <= function->dependencies_goal);
    }
#endif /* DAGUE_DEBUG */

    if( dep_cur_value == function->dependencies_goal ) {

#else  /* defined(DAGUE_SCHED_DEPS_MASK) */

#   if defined(DAGUE_DEBUG)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        char tmp2[128];
        DEBUG(("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)\n",
               dest_flow->flow_index, dague_service_to_string(origin, tmp, 128), origin_flow->name,
               *deps,
               dague_service_to_string(exec_context, tmp2, 128),  dest_flow->name ));
    }
#   else
    (void) origin; (void) origin_flow;
#   endif 
    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= dague_check_IN_dependencies( dague_object, exec_context );
#   ifdef DAGUE_DEBUG
        if( dep_new_value != 0 ) {
            DEBUG(("Activate IN dependencies with mask 0x%x\n", dep_new_value));
        }
#   endif /* DAGUE_DEBUG */
    }

    dep_cur_value = dague_atomic_bor( deps, dep_new_value );

    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {

#endif /* defined(DAGUE_SCHED_DEPS_MASK) */

        dague_prof_grapher_dep(origin, exec_context, 1, origin_flow, dest_flow);

#if defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK)
        {
            int success;
            dague_dependency_t tmp_mask;
            tmp_mask = *deps;
            success = dague_atomic_cas( deps,
                                        tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
            if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
                char tmp2[128];
                fprintf(stderr, "I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice (second time by %s)!!!\n",
                        success, tmp_mask, dague_service_to_string(exec_context, tmp, 128),
                        dague_service_to_string(origin, tmp2, 128));
                assert(0);
            }
        }
#endif  /* defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK) */

        /* This service is ready to be executed as all dependencies
         * are solved.  Queue it into the ready_list passed as an
         * argument.
         */
        {
#if defined(DAGUE_DEBUG)
            char tmp2[128];
#endif
            dague_execution_context_t* new_context;
            dague_thread_mempool_t *mpool;
            new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
            /* this should not be copied over from the old execution context */
            mpool = new_context->mempool_owner;
            /* we copy everything but the dague_list_item_t at the beginning, to
             * avoid copying uninitialized stuff from the stack
             */
            assert( (uintptr_t)new_context == (uintptr_t)&new_context->list_item );
            memcpy( ((char*)new_context) + sizeof(dague_list_item_t), 
                    ((char*)exec_context) + sizeof(dague_list_item_t), 
                    sizeof(dague_minimal_execution_context_t) - sizeof(dague_list_item_t) );
            new_context->mempool_owner = mpool;
            DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);

            DEBUG(("%s becomes schedulable from %s with mask 0x%04x on thread %d of VP %d\n", 
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(origin, tmp2, 128),
                   *deps,
                   eu_context->th_id, eu_context->virtual_process->vp_id));

#if defined(DAGUE_SCHED_CACHE_AWARE)
            new_context->data[0].gc_data = NULL;
#endif
            /* TODO: change this to the real number of input dependencies */
            memset( new_context->data, 0, sizeof(dague_data_pair_t) * MAX_PARAM_COUNT );
            assert( dest_flow->flow_index <= MAX_PARAM_COUNT );
            /**
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].data_repo = dest_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data      = origin->data[(int)origin_flow->flow_index].data;
            dague_list_add_single_elem_by_priority( pready_list, new_context );
        }

        DAGUE_STAT_INCREASE(counter_nbtasks, 1ULL);

    } else { /* Service not ready */

        dague_prof_grapher_dep(origin, exec_context, 0, origin_flow, dest_flow);

#if defined(DAGUE_SCHED_DEPS_MASK)
        DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x: real 0x%02x)\n",
               dague_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_goal,
               (int)(dep_cur_value & DAGUE_DEPENDENCIES_BITMASK),
               (int)(dep_cur_value)));
#else
        DEBUG(("  => Service %s not yet ready (requires %d dependencies, %d done)\n",
               dague_service_to_string( exec_context, tmp, 128 ), 
               (int)function->dependencies_goal, dep_cur_value));
#endif
    }

    return 0;
}

#define is_inplace(ctx,flow,dep) NULL
#define is_read_only(ctx,flow,dep) NULL

dague_ontask_iterate_t dague_release_dep_fct(dague_execution_unit_t *eu, 
                                             dague_execution_context_t *newcontext, 
                                             dague_execution_context_t *oldcontext, 
                                             int out_index, int outdep_index, 
                                             int src_rank, int dst_rank,
                                             dague_arena_t* arena,
                                             void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    const dague_flow_t* target = oldcontext->function->out[out_index];

    if( !(arg->action_mask & (1 << out_index)) ) {
#if defined(DAGUE_DEBUG)
        char tmp[128];
        DEBUG(("On task %s out_index %d not on the action_mask %x\n",
               dague_service_to_string(oldcontext, tmp, 128), out_index, arg->action_mask));
#endif
        return DAGUE_ITERATE_CONTINUE;
    }

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {
        if( arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS ) {
            void* data;

            data = is_read_only(oldcontext, out_index, outdep_index);
            if(NULL != data) {
                arg->deps->msg.which &= ~(1 << out_index); /* unmark all data that are RO we already hold from previous tasks */
            } else {
                arg->deps->msg.which |= (1 << out_index); /* mark all data that are not RO */
                data = is_inplace(oldcontext, out_index, outdep_index);  /* Can we do it inplace */
            }
            arg->deps->output[out_index].data = data; /* if still NULL allocate it */
            arg->deps->output[out_index].type = arena;
            if(newcontext->priority > arg->deps->max_priority) arg->deps->max_priority = newcontext->priority;
        }
        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ) {
            int _array_pos, _array_mask;

            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            arg->remote_deps->root = src_rank;
            if( !(arg->remote_deps->output[out_index].rank_bits[_array_pos] & _array_mask) ) {
                arg->remote_deps->output[out_index].type = arena;
                arg->remote_deps->output[out_index].data = oldcontext->data[target->flow_index].data;
                arg->remote_deps->output[out_index].rank_bits[_array_pos] |= _array_mask;
                arg->remote_deps->output[out_index].count++;
                arg->remote_deps_count++;
            }
            if(newcontext->priority > arg->remote_deps->max_priority) arg->remote_deps->max_priority = newcontext->priority;
        }
    }
#else
    (void)src_rank;
    (void)arena;
#endif

    if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
        (eu->virtual_process->dague_context->my_rank == dst_rank) ) {
        if( (NULL != arg->output_entry) && (NULL != oldcontext->data[target->flow_index].data) ) {
            arg->output_entry->data[out_index] = oldcontext->data[target->flow_index].data;
            arg->output_usage++;
            AREF( arg->output_entry->data[out_index] );
        }
        arg->nb_released += dague_release_local_OUT_dependencies(oldcontext->dague_object,
                                                                 eu, oldcontext,
                                                                 oldcontext->function->out[out_index],
                                                                 newcontext,
                                                                 oldcontext->function->out[out_index]->dep_out[outdep_index]->flow,
                                                                 arg->output_entry,
                                                                 &arg->ready_list);
    }
    
    return DAGUE_ITERATE_CONTINUE;
}

void dague_dump_object( dague_object_t* object )
{
    (void) object;
}

void dague_dump_execution_context( dague_execution_context_t* exec_context )
{
    char tmp[128];

    printf( "Task %s\n", dague_service_to_string( exec_context, tmp, 128 ) );
}

void dague_destruct_dependencies(dague_dependencies_t* d)
{
    int i;
    if( (d != NULL) && (d->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) ) {
        for(i = d->min; i <= d->max; i++)
            if( NULL != d->u.next[i-d->min] )
                dague_destruct_dependencies(d->u.next[i-d->min]);
    }
    free(d);
}

/**
 *
 */
int dague_set_complete_callback( dague_object_t* dague_object,
                                 dague_completion_cb_t complete_cb, void* complete_cb_data )
{
    if( NULL == dague_object->complete_cb ) {
        dague_object->complete_cb      = complete_cb;
        dague_object->complete_cb_data = complete_cb_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_get_complete_callback( const dague_object_t* dague_object,
                                 dague_completion_cb_t* complete_cb, void** complete_cb_data )
{
    if( NULL != dague_object->complete_cb ) {
        *complete_cb      = dague_object->complete_cb;
        *complete_cb_data = dague_object->complete_cb_data;
        return 0;
    }
    return -1;
}

/* TODO: Change this code to something better */
static dague_object_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;

static void dague_object_empty_repository(void)
{
    free(object_array);
    object_array = NULL;
    object_array_size = 1;
    object_array_pos = 0;
}

/**< Retrieve the local object attached to a unique object id */
dague_object_t* dague_object_lookup( uint32_t object_id )
{
    if( object_id > object_array_pos ) {
        return NULL;
    }
    return object_array[object_id];
}

/**< Register the object with the engine. Create the unique identifier for the object */
int dague_object_register( dague_object_t* object )
{
    uint32_t index = ++object_array_pos;

    if( index >= object_array_size ) {
        object_array_size *= 2;
        object_array = (dague_object_t**)realloc(object_array, object_array_size * sizeof(dague_object_t*) );
    }
    object_array[index] = object;
    object->object_id = index;
    return (int)index;
}

/**< Print DAGuE usage message */
void dague_usage(void)
{
    fprintf(stderr, "\n"
	    "A DAGuE argument sequence prefixed by \"--\" can end the command line\n"
            " --dague_bind        : define a set of core for the thread binding\n"
	    "                       accepted values:\n"
	    "                        - a core list          (exp: --dague_bind=[+]1,3,5-6)\n"
	    "                        - a hexadecimal mask   (exp: --dague_bind=[+]0xff012)\n"
	    "                        - a binding range expression: [+][start]:[end]:[step] \n"
	    "                          -> define a round-robin one thread per core distribution from start (default 0)\n"
            "                             to end (default physical core number) by step (default 1)\n"
	    "                             (exp: --dague_bind=[+]1:7:2  bind the 6 first threads on the cores 1 3 5 2 4 6\n"
            "                             while extra threads remain unbound)\n"
	    "                       if starts with \"+\", the communication thread will be executed on the core subset\n"
	    " --dague_bind_comm   : define the core the communication thread will be bound on (prevail over --dague_bind)\n"
	    "                       (default: a NUIOA-aware core subset)\n"

	    "\n"
         /* " --dague_verbose     : extra verbose output\n" */
         /* " --dague_papi        : enable PAPI\n" */
	    " --dague_help         : this message\n"
	    "\n"
	);
}

/* Parse --dague_bind */
int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
                                  __dague_temporary_thread_initialization_t* startup)
{
#if defined(HAVE_HWLOC)
    DEBUG(("Parse request for the binding of threads\n"));
    char* option = optarg;
    char* position;
    int p, t, nb_total_comp_threads;
    int nb_real_cores=dague_hwloc_nb_real_cores();
    
    nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++)
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;

    if (option[0]=='+' & context->comm_th_core == -1) {
        /* the communication thread has to be included 
           if no more specific binding is defined */
        context->comm_th_core=-2;
        option++;  /* skip the + */
    }

    if (NULL != (position = strchr(option, 'x'))) {
        /* Hexadecimal mask */
        /* convert mask into hwloc bitmap, used if needed for the communication thread binding */
        position++;
        unsigned long mask = strtoul(position, NULL, 16);
        if( context->comm_th_binding_mask==NULL)
            context->comm_th_binding_mask=hwloc_bitmap_alloc();
        hwloc_bitmap_from_ulong(context->comm_th_binding_mask, mask);
            
        /* compute the bitmap indexes to define the binding. */
        int prev=-1;
        for( t = 0; t < nb_total_comp_threads; t++ ) {
            prev=hwloc_bitmap_next(context->comm_th_binding_mask, prev);
            if(prev==-1) {
                /* reached the last index, start again */
                prev=hwloc_bitmap_next(context->comm_th_binding_mask, prev);
            }
            startup[t].bindto=prev;
        }
            
        char *str2 = NULL;
        hwloc_bitmap_asprintf(&str2, context->comm_th_binding_mask);
        fprintf(stderr, "binding defined by the mask %s\n", str2);
        free(str2);
            
#if defined(DAGUE_DEBUG)
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, context->comm_th_binding_mask);
        DEBUG(( "binding defined by the mask %s\n", str));
        free(str);
#endif
    } else if( NULL != (position = strchr(option, ':'))) {
        /* Range expression such as [start]:[end]:[step]*/
        int arg;
        int start = 0, step = 1;
        int end=nb_real_cores-1;
        if( position != option ) {  /* we have a starting position */
            arg = strtol(option, NULL, 10);
            if( arg < nb_real_cores && arg > -1)
                start = strtol(option, NULL, 10);
            else
                fprintf(stderr,"** WARNING binding start core not valid (restored to default value)\n");
        }
        position++;  /* skip the : */
        if( '\0' != position[0] ) {
            if( ':' != position[0] ) {
                arg = strtol(position, &position, 10);
                if( arg < nb_real_cores && arg > -1 )
                    end = arg;
                else
                    fprintf(stderr,"** WARNING: binding end core not valid (restored to default value)\n");
            }
            position = strchr(position, ':');  /* find the step */
        }
        if( NULL != position )
            position++;  /* skip the : directly into the step */
        if( (NULL != position) && ('\0' != position[0]) ) {
            arg = strtol(position, NULL, 10);
            if( arg < nb_real_cores && arg > -1 )
                step = arg;
            else
                fprintf(stderr,"** WARNING:  binding step not valid (restored to default value)\n");
        }
            
        DEBUG(( "binding defined by core range [%d:%d:%d]\n", start, end, step));
        {
            int where = start, skip = 1;
            for( t = 0; t < nb_total_comp_threads; t++ ) {
                startup[t].bindto = where;
                where += step;
                if( where > end ) {
                    where = start + skip;
                    skip++;
                    if((skip > step) && (t < (nb_total_comp_threads - 1))) {
                        printf( "No more available cores to bind to. The remaining %d threads are not bound\n", nb_total_comp_threads -1-t);
                        int j;
                        for( j = t+1; j < nb_total_comp_threads; j++ )
                            startup[j].bindto = -1;
                        break;
                    }
                }
            }
        }
            
        /* communication thread binding is legal on cores from start to end */
        for(t=start; t <= end; t++)
            hwloc_bitmap_set(context->comm_th_binding_mask, t);
        
    } else {
        /* List of cores */
        int core_tab[MAX_CORE_LIST];
        memset(core_tab, -1, MAX_CORE_LIST*sizeof(int));
        int cmp=0;
        int arg, next_arg;
        
        if(option==NULL) {
            /* default binding,
               no restrinction for the communication thread binding */
            hwloc_bitmap_fill(context->comm_th_binding_mask);
        } else {
            while( option != NULL && option[0] != '\0') {
                /* first core of the remaining list */
                arg = strtol(option, &option, 10);
                if( arg < nb_real_cores && arg > -1 ) {
                    core_tab[cmp]=arg;
                    hwloc_bitmap_set(context->comm_th_binding_mask, arg);
                    cmp++;
                }
                if( NULL != (position = strpbrk(option, ",-"))) {
                    if (position[0] == '-') {
                        /* core range */
                        position++;
                        next_arg = strtol(position, &position, 10);
                        for(t=arg+1; t<=next_arg; t++)
                            if(t < nb_real_cores && t> -1 ) {
                                core_tab[cmp]=t;
                                hwloc_bitmap_set(context->comm_th_binding_mask, t);
                                cmp++;
                            }
                        option++; /* skip the - and folowing number  */
                        option++;
                    }
                }
                if( '\0' == option[0])
                    option=NULL;
                else
                    option++;  /*skip the comma */
            }
        }
        if (core_tab[0]== -1)
            fprintf(stderr,"** WARNING:  bindind arguments are not valid (restored to default value)\n");
        else { /* we have a legal list to defined the binding  */
            cmp=0;
            for(t=0; t < nb_total_comp_threads; t++) {
                startup[t].bindto=core_tab[cmp];
                cmp++;
                if(core_tab[cmp] == -1)
                    cmp=0;
            }
        }
        
#if defined(DAGUE_DEBUG)
        char tmp[MAX_CORE_LIST];
        char* str = tmp;
        size_t offset;
        for(t=0; t<MAX_CORE_LIST; t++){
            if(core_tab[t]==-1)
                break;
            offset = sprintf(str, "%i ", core_tab[t]);
            str += offset;
        }
        DEBUG(( "binding defined by the parsed list: %s \n", tmp));
#endif
    }
    
    return 0;
#else
    fprintf(stderr, "** Warning: the binding defined by --dague_bind has been ignored (HWLOC is required).\n");
    return -1;
#endif
}

static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context){
#if defined(HAVE_HWLOC)
    char* option = optarg;
    if (option[0]!='\0'){
	int core=atoi(optarg);
	if (core > 0 & core < dague_hwloc_nb_real_cores())
	    context->comm_th_core=core;
	else
	    fprintf(stderr,"** Warning: the binding defined by --dague_bind_comm has been ignored (illegal core number)\n");
    } else {
	/* TODO:: Add binding NUIOA aware by default */
	DEBUG(( "default binding for the communication thtread\n", tmp));
    }
    return 0;
#else
    fprintf(stderr, "** Warning: the binding defined by --dague_bind_comm has been ignored (HWLOC is required).\n");
    return -1;
#endif
}
