 /*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include "dague/ayudame.h"

#include "list.h"
#include "scheduling.h"
#include "barrier.h"
#include "remote_dep.h"
#include "datarepo.h"
#include "bindthread.h"
#include "dague_prof_grapher.h"
#include "stats.h"
#include "vpmap.h"

#ifdef DAGUE_PROF_TRACE
#include "profiling.h"
#endif

#ifdef HAVE_HWLOC
#include "hbbuffer.h"
#include "dague_hwloc.h"
#endif

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

dague_data_allocate_t dague_data_allocate = malloc;
dague_data_free_t     dague_data_free = free;

#if defined(DAGUE_PROF_TRACE)
#if defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS)
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
int queue_add_begin, queue_add_end;
int queue_remove_begin, queue_remove_end;
#endif  /* defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS) */
int device_delegate_begin, device_delegate_end;
#endif  /* DAGUE_PROF_TRACE */

#ifdef HAVE_HWLOC
#define MAX_CORE_LIST 128
#endif

#if defined(HAVE_GETRUSAGE) || !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>

static int _dague_rusage_first_call = 1;
static struct rusage _dague_rusage;

static char *dague_enable_dot = NULL;
static char *dague_app_name = NULL;

static void dague_statistics(char* str)
{
    struct rusage current;

    getrusage(RUSAGE_SELF, &current);
    if( !_dague_rusage_first_call ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - _dague_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - _dague_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - _dague_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - _dague_rusage.ru_stime.tv_usec) / 1000000.0);

        STATUS(("=============================================================\n"));
        STATUS(("%s: Resource Usage Data...\n", str));
        STATUS(("-------------------------------------------------------------\n"));
        STATUS(("User Time   (secs)          : %10.3f\n", usr));
        STATUS(("System Time (secs)          : %10.3f\n", sys));
        STATUS(("Total Time  (secs)          : %10.3f\n", usr + sys));
        STATUS(("Minor Page Faults           : %10ld\n", (current.ru_minflt  - _dague_rusage.ru_minflt)));
        STATUS(("Major Page Faults           : %10ld\n", (current.ru_majflt  - _dague_rusage.ru_majflt)));
        STATUS(("Swap Count                  : %10ld\n", (current.ru_nswap   - _dague_rusage.ru_nswap)));
        STATUS(("Voluntary Context Switches  : %10ld\n", (current.ru_nvcsw   - _dague_rusage.ru_nvcsw)));
        STATUS(("Involuntary Context Switches: %10ld\n", (current.ru_nivcsw  - _dague_rusage.ru_nivcsw)));
        STATUS(("Block Input Operations      : %10ld\n", (current.ru_inblock - _dague_rusage.ru_inblock)));
        STATUS(("Block Output Operations     : %10ld\n", (current.ru_oublock - _dague_rusage.ru_oublock)));
        STATUS(("=============================================================\n"));
    }
    _dague_rusage_first_call = !_dague_rusage_first_call;
    _dague_rusage = current;
    return;
}
#else
static void dague_statistics(char* str) { (void)str; return; }
#endif /* defined(HAVE_GETRUSAGE) */

static void dague_object_empty_repository(void);

typedef struct __dague_temporary_thread_initialization_t {
    dague_vp_t *virtual_process;
    int th_id;
    int nb_cores;
    int bindto;
    int bindto_ht;
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
    dague_bindthread(startup->bindto, startup->bindto_ht);
    DEBUG2(("VP %i : bind thread %i.%i on core %i [HT %i]\n", startup->virtual_process->vp_id, startup->virtual_process->vp_id, startup->th_id, startup->bindto, startup->bindto_ht));

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
    eu->eu_profile = dague_profiling_thread_init( 2*1024*1024,
                                                  "DAGuE Thread %d of VP %d",
                                                  eu->th_id,
                                                  eu->virtual_process->vp_id );
#endif

#if defined(DAGUE_SIM)
    eu->largest_simulation_date = 0;
#endif

    /* The main thread of VP 0 will go back to the user level */
    if( DAGUE_THREAD_IS_MASTER(eu) ) {
#if (0 < DAGUE_DEBUG_VERBOSE)
        vpmap_display_map(stderr);
#endif
        return NULL;
    }

    return __dague_progress(eu);
}

static void dague_vp_init( dague_vp_t *vp,
                           int32_t nb_cores,
                           __dague_temporary_thread_initialization_t *startup)
{
    int t, pi;
    dague_execution_context_t fake_context;
    data_repo_entry_t fake_entry;

    vp->nb_cores = nb_cores;

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
        if( vpmap_get_nb_cores_affinity(vp->vp_id, t) == 1 )
            vpmap_get_core_affinity(vp->vp_id, t, &startup[t].bindto, &startup[t].bindto_ht);
        else if( vpmap_get_nb_cores_affinity(vp->vp_id, t) > 1 )
            printf("multiple core to bind on... for now, do nothing\n");
        else{
            startup[t].bindto= -1;
            startup[t].bindto_ht= -1;
        }
    }
}

#define DEFAULT_APPNAME "app_name_%d"

dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[] )
{
    int argc = 0, nb_vp, p, t, nb_total_comp_threads;
    char **argv = NULL;
    __dague_temporary_thread_initialization_t *startup;
    dague_context_t* context;

#if defined(HAVE_HWLOC)
    dague_hwloc_init();
#endif  /* defined(HWLOC) */

    if((NULL == pargc) || (*pargc == 0)) {
        int rc = asprintf( &dague_app_name, DEFAULT_APPNAME, (int)getpid() );
        if (rc == -1) {
	    dague_app_name = strdup( "app_name_XXXXXX" );
        }
    } else {
        dague_app_name = strdup( (*pargv)[0] );
    }

    /* Set a default the number of cores if not defined by parameters
     * - with hwloc if available
     * - with sysconf otherwise (hyperthreaded core number)
     */
    if( nb_cores <= 0 ) {
#if defined(HAVE_HWLOC)
        nb_cores = dague_hwloc_nb_real_cores();
#else
        nb_cores = sysconf(_SC_NPROCESSORS_ONLN);
        if(nb_cores == -1) {
            perror("sysconf(_SC_NPROCESSORS_ONLN)\n");
            nb_cores = 1;
        }
#endif
    }

    /* Default case if vpmap has not been initialized */
    if(vpmap_get_nb_vp() == -1)
        vpmap_init_from_flat(nb_cores);

#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
        {
            {"dague_help",       no_argument,        NULL, 'h'},
            {"dague_bind",       optional_argument,  NULL, 'b'},
            {"dague_bind_comm",  optional_argument,  NULL, 'c'},
#if defined(DAGUE_PROF_GRAPHER)
            {"dague_dot",        optional_argument,  NULL, '.'},
#endif  /* defined(DAGUE_PROF_GRAPHER) */
            {0, 0, 0, 0}
        };
#endif  /* defined(HAVE_GETOPT_LONG) */
    nb_vp = vpmap_get_nb_vp();

    context = (dague_context_t*)malloc(sizeof(dague_context_t) + (nb_vp-1) * sizeof(dague_vp_t*));

    context->__dague_internal_finalization_in_progress = 0;
    context->__dague_internal_finalization_counter = 0;
    context->nb_nodes       = 1;
    context->active_objects = 0;
    context->my_rank        = 0;

#if defined(DAGUE_SIM)
    context->largest_simulation_date = 0;
#endif /* DAGUE_SIM */

    /* TODO: nb_cores should depend on the vp_id */
    nb_total_comp_threads = 0;
    for(p = 0; p < nb_vp; p++) {
        nb_total_comp_threads += vpmap_get_nb_threads_in_vp(p);
    }

    if( nb_cores != nb_total_comp_threads ) {
        fprintf(stderr, "Warning: using %d threads instead of the requested %d (need to change features in VP MAP)\n",
                nb_total_comp_threads, nb_cores);
    }

    startup =
        (__dague_temporary_thread_initialization_t*)malloc(nb_total_comp_threads * sizeof(__dague_temporary_thread_initialization_t));

    context->nb_vp = nb_vp;
    t = 0;
    for(p = 0; p < nb_vp; p++) {
        dague_vp_t *vp;
        vp = (dague_vp_t *)malloc(sizeof(dague_vp_t) + (vpmap_get_nb_threads_in_vp(p)-1) * sizeof(dague_execution_unit_t*));
        vp->dague_context = context;
        vp->vp_id = p;
        context->virtual_processes[p] = vp;
        /** This creates startup[t] -> startup[t+nb_cores] */
        dague_vp_init(vp, vpmap_get_nb_threads_in_vp(p), &(startup[t]));
        t += vpmap_get_nb_threads_in_vp(p);
    }

#if defined(HAVE_HWLOC)
    context->comm_th_core   = -1;
#if defined(HAVE_HWLOC_BITMAP)
    context->comm_th_index_mask = hwloc_bitmap_alloc();
    context->index_core_free_mask = hwloc_bitmap_alloc();
    hwloc_bitmap_set_range(context->index_core_free_mask, 0, dague_hwloc_nb_real_cores()-1);
#endif /* HAVE_HWLOC_BITMAP */
#endif

    if( (NULL != pargc) && (*pargc != 0) ) {
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

            ret = getopt_long (argc, argv, "p:b:c:v:.::",
                               long_options, &option_index);
#else
            ret = getopt (argc, argv, "p:b:c:.::");
#endif  /* defined(HAVE_GETOPT_LONG) */
            if( -1 == ret ) break;  /* we're done */

            switch(ret) {
            case 'h': dague_usage(); break;
            case 'c': dague_parse_comm_binding_parameter(optarg, context); break;
            case 'b': dague_parse_binding_parameter(optarg, context, startup); break;
            case '.':
                if( dague_enable_dot ) free( dague_enable_dot );
                /** Could not make optional_argument work. Recoding its behavior... */ 
                if( strlen( argv[optind-1] ) >= 2 && strncmp( argv[optind-1], "-.", 2) == 0) {
                    /** Case one: using short argument -. */
                    if( strlen( argv[optind-1] ) > 2 ) {
                        dague_enable_dot = strdup( argv[optind-1] + 2 );
                    } else {
                        dague_enable_dot = strdup(dague_app_name);
                    }
                } else {
                   /** Long argument type */
                   if( (strlen( argv[optind-1] ) > 12) &&
                       (strncmp( argv[optind-1], "--dague_dot=", 12 ) == 0 ) ) {
                       dague_enable_dot = strdup( argv[optind-1]+12 );
                   } else {
                       dague_enable_dot = strdup(dague_app_name);
                    }
               }
               break;
            }
        } while(1);
    }

#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    /* update the index_core_free_mask according to the thread binding defined */
    for(t = 0; t < nb_total_comp_threads; t++)
        hwloc_bitmap_clr(context->index_core_free_mask, startup[t].bindto);

#if defined(DAGUE_DEBUG_VERBOSE3)
    {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, context->index_core_free_mask);
        DEBUG3(( "binding core free mask is %s\n", str));
        free(str);
    }
#endif /* DAGUE_DEBUG_VERBOSE3 */
#endif /* HAVE_HWLOC && HAVE_HWLOC_BITMAP */

    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_total_comp_threads );

#if defined(DAGUE_PROF_TRACE)
        dague_profiling_init( "%s", dague_app_name );

#  if defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS)
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
    dague_profiling_add_dictionary_keyword( "Queue ADD", "fill:#767676",
                                            0, NULL,
                                            &queue_add_begin, &queue_add_end);
    dague_profiling_add_dictionary_keyword( "Queue REMOVE", "fill:#B9B243",
                                            0, NULL,
                                            &queue_remove_begin, &queue_remove_end);
#  endif /* DAGUE_PROF_TRACE_SCHEDULING_EVENTS */
    dague_profiling_add_dictionary_keyword( "Device delegate", "fill:#EAE7C6",
                                            0, NULL,
                                            &device_delegate_begin, &device_delegate_end);
#endif  /* DAGUE_PROF_TRACE */

    if(dague_enable_dot) {
#if defined(DAGUE_PROF_GRAPHER)
          dague_prof_grapher_init(dague_enable_dot, nb_total_comp_threads);
#else
        fprintf(stderr,
                "************************************************************************************************\n"
                "*** Warning: dot generation requested, but DAGUE configured with DAGUE_PROF_GRAPHER disabled ***\n"
                "************************************************************************************************\n");
#endif  /* defined(DAGUE_PROF_GRAPHER) */
    }

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

    AYU_INIT();
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

    AYU_FINI();
#ifdef DAGUE_PROF_TRACE
    dague_profiling_fini( );
#endif  /* DAGUE_PROF_TRACE */

    if(dague_enable_dot) {
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_fini();
#endif  /* defined(DAGUE_PROF_GRAPHER) */
        free(dague_enable_dot);
        dague_enable_dot = NULL;
    }
    /* Destroy all resources allocated for the barrier */
    dague_barrier_destroy( &(context->barrier) );


#if defined(HAVE_HWLOC_BITMAP)
    /* Release thread binding masks */
    hwloc_bitmap_free(context->comm_th_index_mask);
    hwloc_bitmap_free(context->index_core_free_mask);

    dague_hwloc_fini();
#endif  /* HAVE_HWLOC_BITMAP */

    if (dague_app_name != NULL ) {
       free(dague_app_name);
       dague_app_name = NULL;
    }

#if defined(DAGUE_STATS)
    {
        char filename[64];
        char prefix[32];
#if defined(DISTRIBUTED) && defined(HAVE_MPI)
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
    debug_mark_purge_all_history();

    free(context);
    *pcontext = NULL;
    return 0;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t
dague_check_IN_dependencies_with_mask( const dague_object_t *dague_object,
                                       const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];

        /**
         * Controls and data have different logic:
         * Flows can depend conditionally on multiple input or control.
         * It is assumed that in the data case, one input will always become true.
         *  So, the Input dependency is already solved if one is found with a true cond,
         *      and depend only on the data.
         *
         * On the other hand, if all conditions for the control are false,
         *  it is assumed that no control should be expected.
         */
        if( ACCESS_NONE == flow->access_type ) {
            active = (1 << flow->flow_index);
            /* Control case: resolved unless we find at least one input control */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(dague_object, exec_context->locals) ) {
                        /* Cannot use control gather magic with the USE_DEPS_MASK */
                        assert( NULL == dep->ctl_gather_nb );
                        continue;
                    }
                }
                active = 0;
                break;
            }
        } else {
            /* Data case: resolved only if we found a data already ready */
            active = 0;
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( dep->function_id == -1 ) {  /* this is only true for memory locations */
                    if( NULL != dep->cond ) {
                        /* Check if the condition apply on the current setting */
                        assert( dep->cond->op == EXPR_OP_INLINE );
                        if( 0 == dep->cond->inline_func32(dague_object, exec_context->locals) ) {
                            continue;
                        }
                    }
                    active = (1 << flow->flow_index);
                    break;
                }
            }
        }
        ret |= active;
    }
    return ret;
}

static dague_dependency_t
dague_check_IN_dependencies_with_counter( const dague_object_t *dague_object,
                                          const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_CTL_GATHER) &&
        !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        /* If the number of goal does not depend on this particular task instance,
         * it is pre-computed by the daguepp compiler
         */
        return function->dependencies_goal;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];

        /**
         * Controls and data have different logic:
         * Flows can depend conditionally on multiple input or control.
         * It is assumed that in the data case, one input will always become true.
         *  So, the Input dependency is already solved if one is found with a true cond,
         *      and depend only on the data.
         *
         * On the other hand, if all conditions for the control are false,
         *  it is assumed that no control should be expected.
         */
        active = 0;
        if( ACCESS_NONE == flow->access_type ) {
            /* Control case: just count how many must be resolved */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( dep->cond->inline_func32(dague_object, exec_context->locals) ) {
                        if( NULL == dep->ctl_gather_nb)
                            active++;
                        else {
                            assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                            active += dep->ctl_gather_nb->inline_func32(dague_object, exec_context->locals);
                        }
                    }
                } else {
                    if( NULL == dep->ctl_gather_nb)
                        active++;
                    else {
                        assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                        active += dep->ctl_gather_nb->inline_func32(dague_object, exec_context->locals);
                    }
                }
            }
        } else {
            /* Data case: count all that do not have a direct dependence on a data */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( dep->function_id != -1 ) {  /* we don't count memory locations */
                    if( NULL != dep->cond ) {
                        /* Check if the condition apply on the current setting */
                        assert( dep->cond->op == EXPR_OP_INLINE );
                        if( dep->cond->inline_func32(dague_object, exec_context->locals) ) {
                            active++;
                        }
                    } else {
                        active++;
                    }
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

    deps = dague_object->dependencies_array[exec_context->function->function_id];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_parameters - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min]);
}

static int dague_update_deps_with_counter( dague_object_t *dague_object,
                                           dague_execution_context_t* restrict exec_context,
                                           dague_dependency_t *deps )
{
    dague_dependency_t dep_new_value, dep_cur_value;

    if( 0 == *deps ) {
        dep_new_value = dague_check_IN_dependencies_with_counter( dague_object, exec_context ) - 1;
        if( dague_atomic_cas( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_dec_32b( deps );
    } else {
        dep_cur_value = dague_atomic_dec_32b( deps );
    }

#if defined(DAGUE_DEBUG)
    {
        char tmp[MAX_TASK_STRLEN];
        if( (uint32_t)dep_cur_value > (uint32_t)-128) {
            ERROR(("function %s as reached an improbable dependency count of %u\n",
                   dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context), dep_cur_value ));
        }

        DEBUG3(("Task %s has a current dependencies count of %d (remaining). It %s using the mask approach\n",
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context),
                dep_cur_value,
                (dep_cur_value == 0) ? "becomes ready" : "stays there waiting"));
    }
#endif /* DAGUE_DEBUG */

    return dep_cur_value == 0;
}

static int dague_update_deps_with_mask( dague_object_t *dague_object,
                                        dague_execution_context_t* restrict exec_context,
                                        dague_dependency_t *deps,
                                        const dague_execution_context_t* restrict origin,
                                        const dague_flow_t* restrict origin_flow,
                                        const dague_flow_t* restrict dest_flow )
{
#if defined(DAGUE_DEBUG_VERBOSE3) || defined(DAGUE_DEBUG)
    char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
#endif
    dague_dependency_t dep_new_value, dep_cur_value;
    const dague_function_t* function = exec_context->function;

#if defined(DAGUE_DEBUG)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        ERROR(("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)\n",
               dest_flow->flow_index, dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, origin), origin_flow->name,
               *deps,
               dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, exec_context),  dest_flow->name ));
    }
#else
    (void) origin; (void) origin_flow;
#endif

    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= dague_check_IN_dependencies_with_mask( dague_object, exec_context );
#ifdef DAGUE_DEBUG_VERBOSE3
        if( dep_new_value != 0 ) {
            DEBUG3(("Activate IN dependencies with mask 0x%x\n", dep_new_value));
        }
#endif /* DAGUE_DEBUG */
    }

    dep_cur_value = dague_atomic_bor( deps, dep_new_value );

#if defined(DAGUE_DEBUG)
    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {
        int success;
        dague_dependency_t tmp_mask;
        tmp_mask = *deps;
        success = dague_atomic_cas( deps,
                                    tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
        if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
            ERROR(("Task %s scheduled twice (second time by %s)!!!\n",
                   dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context),
                   dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, origin)));
        }
    }
#endif

    DEBUG3(("Task %s has a current dependencies of 0x%x and a goal of 0x%x -- It %s using the mask approach\n",
            dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context),
            dep_cur_value, function->dependencies_goal,
            ((dep_cur_value & function->dependencies_goal) == function->dependencies_goal) ?
            "becomes ready" : "stays there waiting"));
    return (dep_cur_value & function->dependencies_goal) == function->dependencies_goal;
}

void dague_dependencies_mark_task_as_startup(dague_execution_context_t* restrict exec_context)
{
    const dague_function_t* function = exec_context->function;
    dague_object_t *dague_object = exec_context->dague_object;
    dague_dependency_t *deps = find_deps(dague_object, exec_context);

    if( function->flags & DAGUE_USE_DEPS_MASK ) {
        *deps = DAGUE_DEPENDENCIES_STARTUP_TASK | function->dependencies_goal;
    } else {
        *deps = 0;
    }
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
                                          dague_execution_context_t** pready_ring)
{
    const dague_function_t* function = exec_context->function;
    dague_dependency_t *deps;
    int completed;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif

    (void)eu_context;
    DEBUG2(("Activate dependencies for %s flags = 0x%04x\n",
            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context), function->flags));
    deps = find_deps(dague_object, exec_context);

    if( function->flags & DAGUE_USE_DEPS_MASK ) {
        completed = dague_update_deps_with_mask(dague_object, exec_context, deps, origin, origin_flow, dest_flow);
    } else {
        completed = dague_update_deps_with_counter(dague_object, exec_context, deps);
    }

    if( completed ) {
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_dep(origin, exec_context, 1, origin_flow, dest_flow);
#endif  /* defined(DAGUE_PROF_GRAPHER) */

#if defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK)
        {
            int success;
            char tmp1[MAX_TASK_STRLEN];
            dague_dependency_t tmp_mask;
            tmp_mask = *deps;
            success = dague_atomic_cas( deps,
                                        tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
            if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
                char tmp2[MAX_TASK_STRLEN];
                ERROR(("I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice (second time by %s)!!!\n",
                        success, tmp_mask, dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context),
                        dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, origin)));
            }
        }
#endif  /* defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK) */

        DAGUE_STAT_INCREASE(counter_nbtasks, 1ULL);

        /* This service is ready to be executed as all dependencies
         * are solved.  Queue it into the ready_list passed as an
         * argument.
         */
        {
#if defined(DAGUE_DEBUG_VERBOSE1)
            char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
#endif
            dague_execution_context_t* new_context;
            dague_thread_mempool_t *mpool;
            new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
            /* this should not be copied over from the old execution context */
            mpool = new_context->mempool_owner;
            /* we copy everything but the dague_list_item_t at the beginning, to
             * avoid copying uninitialized stuff from the stack
             */
            memcpy( ((char*)new_context) + sizeof(dague_list_item_t),
                    ((char*)exec_context) + sizeof(dague_list_item_t),
                    sizeof(struct dague_minimal_execution_context_t) - sizeof(dague_list_item_t) );
            new_context->mempool_owner = mpool;
            DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
            AYU_ADD_TASK(new_context);

            DEBUG(("%s becomes ready from %s on thread %d:%d, with mask 0x%04x and priority %d\n",
                   dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context),
                   dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, origin),
                   eu_context->th_id, eu_context->virtual_process->vp_id,
                   *deps,
                   exec_context->priority));

            assert( dest_flow->flow_index <= new_context->function->nb_flows);
            memset( new_context->data, 0, sizeof(dague_data_pair_t) * new_context->function->nb_flows);
            /**
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].data_repo = dest_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data      = origin->data[(int)origin_flow->flow_index].data;
            AYU_ADD_TASK_DEP(new_context, (int)dest_flow->flow_index);

            if(exec_context->function->flags & DAGUE_IMMEDIATE_TASK) {
                DEBUG3(("  Task %s is immediate and will be executed ASAP\n", dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, new_context)));
                __dague_execute(eu_context, new_context);
                __dague_complete_execution(eu_context, new_context);
            } else {
                *pready_ring = (dague_execution_context_t*)dague_list_item_ring_push_sorted( (dague_list_item_t*)(*pready_ring),
                                                                                             &new_context->list_item,
                                                                                             dague_execution_context_priority_comparator );
            }
        }

    } else { /* Service not ready */
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_dep(origin, exec_context, 0, origin_flow, dest_flow);
#endif  /* defined(DAGUE_PROF_GRAPHER) */

        DEBUG2(("  => Service %s not yet ready\n",
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context)));
    }

    return 0;
}

#define is_inplace(ctx,flow,dep) NULL
#define is_read_only(ctx,flow,dep) NULL

dague_ontask_iterate_t
dague_release_dep_fct(dague_execution_unit_t *eu,
                      dague_execution_context_t *newcontext,
                      dague_execution_context_t *oldcontext,
                      int out_index, int outdep_index,
                      int src_rank, int dst_rank,
                      int dst_vpid,
                      dague_dep_data_description_t* data,
                      void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    const dague_flow_t* target = oldcontext->function->out[out_index];

    if( !(arg->action_mask & (1 << out_index)) ) {
        char tmp[MAX_TASK_STRLEN];
        WARNING(("On task %s out_index %d not on the action_mask %x\n",
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, oldcontext), out_index, arg->action_mask));
        return DAGUE_ITERATE_CONTINUE;
    }

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {
        if( arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS ) {
            void* dataptr = is_read_only(oldcontext, out_index, outdep_index);
            if(NULL != dataptr) {
                arg->deps->msg.which &= ~(1 << out_index); /* unmark all data that are RO we already hold from previous tasks */
            } else {
                arg->deps->msg.which |= (1 << out_index); /* mark all data that are not RO */
                dataptr = is_inplace(oldcontext, out_index, outdep_index);  /* Can we do it inplace */
            }
            arg->deps->output[out_index].data.arena  = data->arena;
            arg->deps->output[out_index].data.ptr    = dataptr; /* if still NULL allocate it */
            arg->deps->output[out_index].data.layout = data->layout;
            arg->deps->output[out_index].data.count  = data->count;
            arg->deps->output[out_index].data.displ  = data->displ;
            if(newcontext->priority > arg->deps->max_priority) arg->deps->max_priority = newcontext->priority;
        }
        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ) {
            int _array_pos, _array_mask;

            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            assert( (-1 == arg->remote_deps->root) || (arg->remote_deps->root == src_rank) );
            arg->remote_deps->root = src_rank;
            if( !(arg->remote_deps->output[out_index].rank_bits[_array_pos] & _array_mask) ) {
                arg->remote_deps->output[out_index].data.arena  = data->arena;
                arg->remote_deps->output[out_index].data.ptr    = oldcontext->data[target->flow_index].data;
                arg->remote_deps->output[out_index].data.layout = data->layout;
                arg->remote_deps->output[out_index].data.count  = data->count;
                arg->remote_deps->output[out_index].data.displ  = data->displ;
                arg->remote_deps->output[out_index].rank_bits[_array_pos] |= _array_mask;
                arg->remote_deps->output[out_index].count_bits++;
                arg->remote_deps_count++;
            } else {
                /* The bit is already flipped. This means either that we reached the same peer
                 * several times with the same operation (broadcast), or that we reached the
                 * same peer with two operations that dispatch the same output dependency
                 * (aka. the same data) using distinct communication paths due to different
                 * outdep index.
                 */
            }
            if(newcontext->priority > arg->remote_deps->max_priority)
                arg->remote_deps->max_priority = newcontext->priority;
        }
    }
#else
    (void)src_rank;
    (void)data;
#endif

    if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
        (eu->virtual_process->dague_context->my_rank == dst_rank) ) {
        if( ACCESS_NONE != target->access_type ) {
            arg->output_entry->data[out_index] = oldcontext->data[target->flow_index].data;
            arg->output_usage++;
            /* BEWARE: This increment is required to be done here. As the target task
             * bits are marked, another thread can now enable the task. Once schedulable
             * the task will try to access its input data and decrement their ref count.
             * Thus, if the ref count is not increased here, the data might dissapear
             * before it become useless.
             */
            AREF( arg->output_entry->data[out_index] );
        }
        arg->nb_released += dague_release_local_OUT_dependencies(oldcontext->dague_object,
                                                                 eu, oldcontext,
                                                                 oldcontext->function->out[out_index],
                                                                 newcontext,
                                                                 oldcontext->function->out[out_index]->dep_out[outdep_index]->flow,
                                                                 arg->output_entry,
                                                                 &arg->ready_lists[dst_vpid]);
    }

    return DAGUE_ITERATE_CONTINUE;
}

/**
 * Convert the execution context to a string.
 */
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task)
{
    const dague_function_t* function = task->function;
    unsigned int i, ip, index = 0, is_param;

    assert( NULL != task->dague_object );
    index += snprintf( str + index, size - index, "%s(", function->name );
    if( index >= size ) return str;
    for( ip = 0; ip < function->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "" : ", ",
                           task->locals[function->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")[");
    if( index >= size ) return str;

    for( i = 0; i < function->nb_locals; i++ ) {
        is_param = 0;
        for( ip = 0; ip < function->nb_parameters; ip++ ) {
            if(function->params[ip]->context_index == function->locals[i]->context_index) {
                is_param = 1;
                break;
            }
        }
        index += snprintf( str + index, size - index,
                           (is_param ? "%s%d" : "[%s%d]"),
                           (i == 0) ? "" : ", ",
                           task->locals[i].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, "]<%d>{%u}", task->priority, task->dague_object->object_id );

    return str;
}
/**
 * Convert assignments to a string.
 */
char* dague_snprintf_assignments( char* str, size_t size,
                                  const struct dague_function_s* function,
                                  const struct assignment* locals)
{
    unsigned int ip, index = 0;

    index += snprintf( str + index, size - index, "%s", function->name );
    if( index >= size ) return str;
    for( ip = 0; ip < function->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "(" : ", ",
                           locals[function->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")" );

    return str;
}


void dague_destruct_dependencies(dague_dependencies_t* d)
{
    int i;
    if( (d != NULL) && (d->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) ) {
        for(i = d->min; i <= d->max; i++)
            if( NULL != d->u.next[i - d->min] )
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
static volatile uint32_t object_array_lock = 0;
static dague_object_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;
#define NOOBJECT ((void*)-1)

static void dague_object_empty_repository(void)
{
    dague_atomic_lock( &object_array_lock );
    free(object_array);
    object_array = NULL;
    object_array_size = 1;
    object_array_pos = 0;
    dague_atomic_unlock( &object_array_lock );
}

/**< Retrieve the local object attached to a unique object id */
dague_object_t* dague_object_lookup( uint32_t object_id )
{
    dague_object_t *r;
    dague_atomic_lock( &object_array_lock );
    if( object_id > object_array_pos ) {
        r = NULL;
    } else {
        r = object_array[object_id];
    }
    dague_atomic_unlock( &object_array_lock );
    return r;
}

/**< Register the object with the engine. Create the unique identifier for the object */
int dague_object_register( dague_object_t* object )
{
    uint32_t index;

    dague_atomic_lock( &object_array_lock );
    index = (uint32_t)++object_array_pos;

    if( index >= object_array_size ) {
        object_array_size *= 2;
        object_array = (dague_object_t**)realloc(object_array, object_array_size * sizeof(dague_object_t*) );
#if defined(DAGUE_DEBUG)
        {
            unsigned int i;
            for(i = index; i < object_array_size; i++)
                object_array[i] = NOOBJECT;
        }
#endif
    }
    object_array[index] = object;
    object->object_id = index;
    dague_atomic_unlock( &object_array_lock );
    (void)dague_remote_dep_new_object( object );
    return (int)index;
}

/**< Unregister the object with the engine. */
void dague_object_unregister( dague_object_t* object )
{
    dague_atomic_lock( &object_array_lock );
    assert( object->object_id < object_array_size );
    assert( object_array[object->object_id] == object );
    assert( object->nb_local_tasks == 0 );
    object_array[object->object_id] = NOOBJECT;
    dague_atomic_unlock( &object_array_lock );
}

/**< This function is called in a body only.
 *   It sets the current task as the last task.
 *   The current implemenation has limitations: it assumes that
 *   no other task of the object is running, and it assumes that
 *   all tasks that are cancelled were never started (i.e. no
 *   input dependency for these tasks was ever true).
 *
 *   Update this comment and the comment in the header file
 *   if some limitations are relieved.
 */
void dague_object_terminate( dague_object_t *object )
{
    fprintf(stderr, "Still %d tasks to execute\n", object->nb_local_tasks );
    object->nb_local_tasks = 1;
}

/**< Unregister the object with the engine. */
void dague_object_dec_nbtask( dague_object_t* object, uint32_t nb_tasks )
{
    assert( object->nb_local_tasks >= nb_tasks );
    dague_atomic_add_32b(&object->nb_local_tasks, -nb_tasks);
}

/**< Print DAGuE usage message */
void dague_usage(void)
{
    fprintf(stderr,"\n"
            "A DAGuE argument sequence prefixed by \"--\" can end the command line\n\n"
            "     --dague_bind_comm   : define the core the communication thread will be bound on\n"
            "\n"
            "     Warning:: The binding options rely on hwloc. The core numerotation is defined between 0 and the number of cores.\n"
            "     Be careful when used with cgroups.\n"
            "\n"
            "     --dague_help         : this message\n"
            "\n"
            );
}




/* Parse --dague_bind parameter (define a set of core for the thread binding)
 * The parameter can be
 * - a core list
 * - a hexadecimal mask
 * - a range expression
 * - a file containing the parameters (list, mask or expression) for each processes
 *
 * The function rely on a version of hwloc which support for bitmap.
 * It redefines the fields "bindto" of the startup structure used to initialize the threads
 */

/* We use the topology core indexes to define the binding, not the core numbers.
 * The index upper/lower bounds are 0 and (number_of_cores - 1).
 * The core_index_mask stores core indexes and will be converted into a core_number_mask
 * for the hwloc binding. It will ensure a homogeneous behavior on topology without a sequential
 * core numeration starting from zero (partial topology returned with control groups).
 */

int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
                                  __dague_temporary_thread_initialization_t* startup)
{
#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    char* option = optarg;
    char* position;
    int p, t, nb_total_comp_threads;

    int nb_real_cores=dague_hwloc_nb_real_cores();

    nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++)
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;


    /* The parameter is a file */
    if( NULL != (position = strstr(option, "file:")) ) {
        /* Read from the file the binding parameter set for the local process and parse it
           (recursive call). */

        char *filename=position+5;
        FILE *f;
        char *line = NULL;
        size_t line_len = 0;

        f = fopen(filename, "r");
        if( NULL == f ) {
            WARNING(("invalid binding file %s.\n", filename));
            return -1;
        }

#if defined(DISTRIBUTED) && defined(HAVE_MPI)
        /* distributed version: first retrieve the parameter for the process */
        int rank, line_num=0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        while (getline(&line, &line_len, f) != -1) {
            if(line_num==rank){
                DEBUG2(("MPI_process %i uses the binding parameters: %s", rank, line));
                break;
            }
            line_num++;
        }

        if( line ){
            if( line_num==rank )
                dague_parse_binding_parameter(line, context, startup);
            else
                DEBUG2(("MPI_process %i uses the default thread binding\n", rank));
            free(line);
        }
#else
        /* Single process, read the first line */
        if( getline(&line, &line_len, f) != -1 ) {
            DEBUG2(("Binding parameters: %s", line));
        }
        if( line ){
            dague_parse_binding_parameter(line, context, startup);
            free(line);
        }
#endif /* DISTRIBUTED && HAVE_MPI */
        else
            WARNING(("default thread binding"));
        fclose(f);
        return -1;
    }


    if( (option[0]=='+') && (context->comm_th_core == -1)) {
        /* The parameter starts with "+" and no specific binding is (yet) defined for the communication thread.
       It is included in the thread mapping. */
        context->comm_th_core=-2;
        option++;  /* skip the + */
    }


    /* Parse  hexadecimal mask, range expression of core list expression */
    if( NULL != (position = strchr(option, 'x')) ) {
        /* The parameter is a hexadecimal mask */
        position++; /* skip the x */

        /* convert the mask into a bitmap (define legal core indexes) */
         unsigned long mask = strtoul(position, NULL, 16);

         if( context->comm_th_index_mask==NULL )
            context->comm_th_index_mask=hwloc_bitmap_alloc();
        hwloc_bitmap_from_ulong(context->comm_th_index_mask, mask);

        /* update binding information in the startup structure */
        int prev=-1;

        for( t = 0; t < nb_total_comp_threads; t++ ) {
            prev=hwloc_bitmap_next(context->comm_th_index_mask, prev);
            if(prev==-1){
                /* reached the last index, start again */
                prev=hwloc_bitmap_next(context->comm_th_index_mask, prev);
            }
            startup[t].bindto=prev;
        }

#if defined(DAGUE_DEBUG_VERBOSE3)
        {
            char *str = NULL;
            hwloc_bitmap_asprintf(&str, context->comm_th_index_mask);
            DEBUG3(( "binding (core indexes) defined by the mask %s\n", str));
            free(str);
        }
#endif /* DAGUE_DEBUG_VERBOSE3 */
    }

    else if( NULL != (position = strchr(option, ':'))) {
        /* The parameter is a range expression such as [start]:[end]:[step] */
        int arg;
        int start = 0, step = 1;
        int end=nb_real_cores-1;
        if( position != option ) {
            /* we have a starting position */
            arg = strtol(option, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                start = strtol(option, NULL, 10);
            else
                WARNING(("binding start core not valid (restored to default value)"));
        }
        position++;  /* skip the : */
        if( '\0' != position[0] ) {
            /* check for the ending position */
            if( ':' != position[0] ) {
                arg = strtol(position, &position, 10);
                if( (arg < nb_real_cores) && (arg > -1) )
                    end = arg;
                else
                    WARNING(("binding end core not valid (restored to default value)\n"));
            }
            position = strchr(position, ':');  /* find the step */
        }
        if( NULL != position )
            position++;  /* skip the : directly into the step */
        if( (NULL != position) && ('\0' != position[0]) ) {
            arg = strtol(position, NULL, 10);
            if( (arg < nb_real_cores) && (arg > -1) )
                step = arg;
            else
                WARNING(("binding step not valid (restored to default value)\n"));
        }
        DEBUG3(("binding defined by core range [%d:%d:%d]\n", start, end, step));

        /* redefine the core according to the trio start/end/step */
        {
            int where = start, skip = 1;
            for( t = 0; t < nb_total_comp_threads; t++ ) {
                startup[t].bindto = where;
                where += step;
                if( where > end ) {
                    where = start + skip;
                    skip++;
                    if((skip > step) && (t < (nb_total_comp_threads - 1))) {
                        STATUS(( "No more available cores to bind to. The remaining %d threads are not bound\n", nb_total_comp_threads -1-t));
                        int j;
                        for( j = t+1; j < nb_total_comp_threads; j++ )
                            startup[j].bindto = -1;
                        break;
                    }
                }
            }
        }

        /* communication thread binding is legal on cores indexes from start to end */
        for(t=start; t <= end; t++)
            hwloc_bitmap_set(context->comm_th_index_mask, t);
    } else {
        /* List of cores */
        int core_tab[MAX_CORE_LIST];
        memset(core_tab, -1, MAX_CORE_LIST*sizeof(int));
        int cmp=0;
        int arg, next_arg;

        if( NULL == option ) {
            /* default binding  no restrinction for the communication thread binding */
            hwloc_bitmap_fill(context->comm_th_index_mask);
        } else {
            while( option != NULL && option[0] != '\0') {
                /* first core of the remaining list */
                arg = strtol(option, &option, 10);
                if( (arg < nb_real_cores) && (arg > -1) ) {
                    core_tab[cmp]=arg;
                    hwloc_bitmap_set(context->comm_th_index_mask, arg);
                    cmp++;
                } else {
                    WARNING(("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n Binding restored to default\n", arg, nb_real_cores-1));
                }

                if( NULL != (position = strpbrk(option, ",-"))) {
                    if( position[0] == '-' ) {
                        /* core range */
                        position++;
                        next_arg = strtol(position, &position, 10);

                        for(t=arg+1; t<=next_arg; t++)
                            if( (t < nb_real_cores) && (t > -1) ) {
                                core_tab[cmp]=t;
                                hwloc_bitmap_set(context->comm_th_index_mask, t);
                                cmp++;
                            }
                        option++; /* skip the - and folowing number  */
                        option++;
                    }
                }
                if( '\0' == option[0])
                    option=NULL;
                else
                    /*skip the comma */
                    option++;
            }
        }
        if( core_tab[0]== -1 )
            WARNING(("bindind arguments are not valid (restored to default value)\n"));
        else { /* we have a legal list to defined the binding  */
            cmp=0;
            for(t=0; t<nb_total_comp_threads; t++) {
                startup[t].bindto=core_tab[cmp];
                cmp++;
                if(core_tab[cmp] == -1)
                    cmp=0;
            }
        }
#if defined(DAGUE_DEBUG_VERBOSE3)
        {
            char tmp[MAX_CORE_LIST];
            char* str = tmp;
            size_t offset;
            int i;
            for(i=0; i<MAX_CORE_LIST; i++) {
                if(core_tab[i]==-1)
                    break;
                offset = sprintf(str, "%i ", core_tab[i]);
                str += offset;
            }
            DEBUG3(("binding defined by the parsed list: %s \n", tmp));
        }
#endif /* DAGUE_DEBUG_VERBOSE3 */

#if defined(DAGUE_DEBUG_VERBOSE)
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
#endif /* DAGUE_DEBUG */
    }
    return 0;
#else
    (void)optarg;
    (void)context;
    (void)startup;
    WARNING(("the binding defined by --dague_bind has been ignored (requires a build with HWLOC with bitmap support).\n"));
    return -1;
#endif /* HAVE_HWLOC && HAVE_HWLOC_BITMAP */
}

static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context)
{
#if defined(HAVE_HWLOC)
    char* option = optarg;
    if( option[0]!='\0' ) {
        int core=atoi(optarg);
        if( (core > -1) && (core < dague_hwloc_nb_real_cores()) )
            context->comm_th_core=core;
        else
            WARNING(("the binding defined by --dague_bind_comm has been ignored (illegal core number)\n"));
    } else {
        /* TODO:: Add binding NUIOA aware by default */
        DEBUG3(("default binding for the communication thread\n"));
    }
    return 0;
#else
    (void)optarg; (void)context;
    WARNING(("The binding defined by --dague_bind has been ignored (requires HWLOC use with bitmap support).\n"));
    return -1;
#endif  /* HAVE_HWLOC */
}

#if defined(DAGUE_SIM)
int dague_getsimulationdate( dague_context_t *dague_context ){
    return dague_context->largest_simulation_date;
}
#endif

/**
 * Array based local data handling.
 */
#include "data_distribution.h"
static uint32_t return_local_u(struct dague_ddesc *mat, ...) { return 0; (void)mat; };
static int32_t  return_local_s(struct dague_ddesc *mat, ...) { return 0; (void)mat; };
static void*    return_null(struct dague_ddesc *mat, ...) { return NULL; (void)mat; };
#ifdef DAGUE_PROF_TRACE
static int key_to_string(struct dague_ddesc *mat, uint32_t datakey, char* buffer, uint32_t buffer_size)
{
  return snprintf( buffer, buffer_size, "%u ", datakey); (void)mat;
}
#endif  /* DAGUE_PROF_TRACE */

const dague_ddesc_t dague_static_local_data_ddesc = {
      0, /* uint32_t myrank */
      1, /* uint32_t cores */
      1, /* uint32_t nodes */
      return_local_u,  /* uint32_t (*rank_of)(struct dague_ddesc *mat, ...) */
      return_null,   /* void *   (*data_of)(struct dague_ddesc *mat, ...) */
      return_local_s,  /* int32_t  (*vpid_of)(struct dague_ddesc *mat, ...) */
      NULL,  /*  moesi_map_t* moesi_map */
#if defined(DAGUE_PROF_TRACE) || defined(HAVE_CUDA)
      return_local_u,  /* uint32_t (*data_key)(struct dague_ddesc *mat, ...) */
#endif
#ifdef DAGUE_PROF_TRACE
      key_to_string, /* int (*key_to_string)(struct dague_ddesc *mat, uint32_t datakey, char * buffer, uint  32_t buffer_size) */
      NULL,  /* char      *key_dim */
      NULL,  /* char      *key */
#endif /* DAGUE_PROF_TRACE */
};

static int32_t dague_expr_eval32(const expr_t *expr, dague_execution_context_t *context)
{
    dague_object_t *object = context->dague_object;

    assert( expr->op == EXPR_OP_INLINE );
    return expr->inline_func32(object, context->locals);
}

static int dague_debug_enumerate_next_in_execution_space(dague_execution_context_t *context,
                                                         int param_depth)
{
    const dague_function_t *function = context->function;
    int cur, max, incr, min;

    if( param_depth == function->nb_parameters )
        return 0;

    if( param_depth < function->nb_parameters ) {
        if( dague_debug_enumerate_next_in_execution_space(context, param_depth+1) )
            return 1;
    }
    cur = context->locals[ function->params[param_depth]->context_index ].value;
    max = dague_expr_eval32(function->params[param_depth]->max, context);
    if( function->params[param_depth]->expr_inc == NULL ) {
        incr = function->params[param_depth]->cst_inc;
    } else {
        incr = dague_expr_eval32(function->params[param_depth]->expr_inc, context);
    }
    if( cur + incr > max ) {
        min = dague_expr_eval32(function->params[param_depth]->min, context);
        context->locals[ function->params[param_depth]->context_index ].value = min;
        return 0;
    }
    context->locals[ function->params[param_depth]->context_index ].value = cur + incr;
    return 1;
}

void dague_debug_print_local_expecting_tasks_for_function( dague_object_t *object,
                                                           const dague_function_t *function,
                                                           int show_remote,
                                                           int show_startup,
                                                           int show_complete,
                                                           int *nlocal,
                                                           int *nreleased,
                                                           int *ntotal)
{
    dague_execution_context_t context;
    dague_dependency_t *dep;
    int pi, li;

    DAGUE_LIST_ITEM_SINGLETON( &context.list_item );
    context.mempool_owner = NULL;
    context.dague_object = object;
    context.function = function;
    context.priority = -1;
    context.status = DAGUE_TASK_STATUS_NONE;
    context.hook_id = 0;
    memset( context.data, 0, MAX_PARAM_COUNT * sizeof(dague_data_pair_t) );

    *nlocal = 0;
    *nreleased = 0;
    *ntotal = 0;

    /* For debugging purposes */
    for(li = 0; li < MAX_LOCAL_COUNT; li++) {
        context.locals[li].value = -1;
    }

    /* Starting point of the context space enumeration */
    for( pi = 0; pi < function->nb_parameters; pi++) {
        context.locals[function->params[pi]->context_index].value = dague_expr_eval32(function->params[pi]->min,
                                                                                      &context);
    }

    do {
        char tmp[MAX_TASK_STRLEN];
        (*ntotal)++;
        if( dague_expr_eval32(function->pred, &context) ) {
            (*nlocal)++;
            dep = find_deps(object, &context);
            if( function->flags & DAGUE_USE_DEPS_MASK ) {
                if( *dep & DAGUE_DEPENDENCIES_STARTUP_TASK ) {
                    (*nreleased)++;
                    if( show_startup )
                        fprintf(stderr, "   Task %s is a local startup task\n",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
                } else {
                    if((*dep & DAGUE_DEPENDENCIES_BITMASK) == function->dependencies_goal) {
                        (*nreleased)++;
                    }
                    if( show_complete ||
                        ((*dep & DAGUE_DEPENDENCIES_BITMASK) != function->dependencies_goal) ) {
                        fprintf(stderr, "   Task %s is a local task with dependency 0x%08x (goal is 0x%08x) -- Flags: %s %s\n",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context),
                                *dep & DAGUE_DEPENDENCIES_BITMASK,
                                function->dependencies_goal,
                                *dep & DAGUE_DEPENDENCIES_TASK_DONE ? "TASK_DONE" : "",
                                *dep & DAGUE_DEPENDENCIES_IN_DONE ? "IN_DONE" : "");
                    }
                }
            } else {
                if( *dep == 0 )
                    (*nreleased)++;

                if( (*dep != 0) || show_complete )
                    fprintf(stderr, "   Task %s is a local task that must wait for %d more dependencies to complete -- using count method for this task (CTL gather)\n",
                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context),
                            *dep);
            }
        } else {
            if( show_remote )
                fprintf(stderr, "   Task %s is a remote task\n",
                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
        }
    } while( dague_debug_enumerate_next_in_execution_space(&context, 0)  );
}

void dague_debug_print_local_expecting_tasks_for_object( dague_object_t *object,
                                                         int show_remote, int show_startup, int show_complete)
{
    uint32_t fi;
    int nlocal, ntotal, nreleased;
    /* The object has not been initialized yet, or it has been completed */
    if( object->dependencies_array == NULL )
        return;

    for(fi = 0; fi < object->nb_functions; fi++) {
        fprintf(stderr, " Tasks of Function %u (%s):\n", fi, object->functions_array[fi]->name);
        dague_debug_print_local_expecting_tasks_for_function( object, object->functions_array[fi],
                                                              show_remote, show_startup, show_complete,
                                                              &nlocal, &nreleased, &ntotal );
        fprintf(stderr, " Total number of Tasks of Class %s: %d\n", object->functions_array[fi]->name, ntotal);
        fprintf(stderr, " Local number of Tasks of Class %s: %d\n", object->functions_array[fi]->name, nlocal);
        fprintf(stderr, " Number of Tasks of Class %s that have been released: %d\n", object->functions_array[fi]->name, nreleased);
    }
}

void dague_debug_print_local_expecting_tasks( int show_remote, int show_startup, int show_complete )
{
    dague_object_t *object;
    uint32_t oi;

    dague_atomic_lock( &object_array_lock );
    for( oi = 1; oi <= object_array_pos; oi++) {
        object = object_array[ oi ];
        if( object == NOOBJECT )
            continue;
        if( object == NULL )
            continue;
        fprintf(stderr, "Tasks of Object %u:\n", oi);
        dague_debug_print_local_expecting_tasks_for_object( object,
                                                            show_remote,
                                                            show_startup,
                                                            show_complete );
    }
    dague_atomic_unlock( &object_array_lock );
}
