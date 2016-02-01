/**
 * Copyright (c) 2009-2016 The University of Tennessee and The University
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
#include <limits.h>
#if defined(HAVE_GEN_H)
#include <libgen.h>
#endif  /* defined(HAVE_GEN_H) */
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <dague/ayudame.h>

#include "dague/mca/pins/pins.h"
#include "dague/mca/sched/sched.h"
#include "dague/utils/output.h"
#include "dague/data_internal.h"
#include "dague/class/list.h"
#include "dague/scheduling.h"
#include "dague/class/barrier.h"
#include "dague/remote_dep.h"
#include "dague/datarepo.h"
#include "dague/bindthread.h"
#include "dague/dague_prof_grapher.h"
#include "dague/vpmap.h"
#include "dague/utils/mca_param.h"
#include "dague/utils/installdirs.h"
#include "dague/devices/device.h"
#include "dague/utils/cmd_line.h"
#include "dague/utils/mca_param_cmd_line.h"

#include "dague/mca/mca_repository.h"

#ifdef DAGUE_PROF_TRACE
#include "dague/profiling.h"
#endif

#include "dague/dague_hwloc.h"
#ifdef HAVE_HWLOC
#include "dague/hbbuffer.h"
#endif

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

/**
 * Global variables.
 */
size_t dague_task_startup_iter = 64;
size_t dague_task_startup_chunk = 256;

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
int arena_memory_alloc_key, arena_memory_free_key;
int arena_memory_used_key, arena_memory_unused_key;
int task_memory_alloc_key, task_memory_free_key;
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
static char *dague_enable_profiling = NULL;  /* profiling file when DAGUE_PROF_TRACE is on */
static dague_device_t* dague_device_cpus = NULL;
static dague_device_t* dague_device_recursive = NULL;

/**
 * Object based task definition (no specialized constructor and destructor) */
OBJ_CLASS_INSTANCE(dague_execution_context_t, dague_hashtable_item_t,
                   NULL, NULL);

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

        dague_inform("==== Resource Usage Data...   %s\n"
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
                     "=============================================================\n",
                     str, usr, sys, usr + sys,
                     current.ru_minflt  - _dague_rusage.ru_minflt, current.ru_majflt  - _dague_rusage.ru_majflt,
                     current.ru_nswap   - _dague_rusage.ru_nswap,
                     current.ru_nvcsw   - _dague_rusage.ru_nvcsw, current.ru_nivcsw  - _dague_rusage.ru_nivcsw,
                     current.ru_inblock - _dague_rusage.ru_inblock, current.ru_oublock - _dague_rusage.ru_oublock);
    }
    _dague_rusage_first_call = !_dague_rusage_first_call;
    _dague_rusage = current;
    return;
}
#else
static void dague_statistics(char* str) { (void)str; return; }
#endif /* defined(HAVE_GETRUSAGE) */

static void dague_handle_empty_repository(void);

typedef struct __dague_temporary_thread_initialization_t {
    dague_vp_t *virtual_process;
    int th_id;
    int nb_cores;
    int bindto;
    int bindto_ht;
    dague_barrier_t*  barrier;       /**< the barrier used to synchronize for the
                                      *   local VP data construction. */
} __dague_temporary_thread_initialization_t;

static int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
                                         __dague_temporary_thread_initialization_t* startup);
static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context);

const dague_function_t* dague_find(const dague_handle_t *dague_handle, const char *fname)
{
    unsigned int i;
    const dague_function_t* object;

    for( i = 0; i < dague_handle->nb_functions; i++ ) {
        object = dague_handle->functions_array[i];
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
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "VP %i : bind thread %i.%i on core %i [HT %i]",
            startup->virtual_process->vp_id, startup->virtual_process->vp_id,
            startup->th_id, startup->bindto, startup->bindto_ht);

    eu = (dague_execution_unit_t*)malloc(sizeof(dague_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->th_id            = startup->th_id;
    eu->virtual_process  = startup->virtual_process;
    eu->scheduler_object = NULL;
    startup->virtual_process->execution_units[startup->th_id] = eu;
    eu->core_id          = startup->bindto;
#if defined(HAVE_HWLOC)
    eu->socket_id        = dague_hwloc_socket_id(startup->bindto);
#else
    eu->socket_id        = 0;
#endif  /* defined(HAVE_HWLOC) */

#if defined(DAGUE_PROF_RUSAGE_EU)
    eu-> _eu_rusage_first_call = 1;
#endif

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    eu->sched_nb_tasks_done = 0;
#endif

    /**
     * A single thread per VP has a little bit more responsability: allocating
     * the memory pools.
     */
    if( startup->th_id == (startup->nb_cores - 1) ) {
        dague_vp_t *vp = startup->virtual_process;
        dague_execution_context_t fake_context;
        data_repo_entry_t fake_entry;
        dague_mempool_construct( &vp->context_mempool,
                                 OBJ_CLASS(dague_execution_context_t), sizeof(dague_execution_context_t),
                                 ((char*)&fake_context.super.mempool_owner) - ((char*)&fake_context),
                                 vp->nb_cores );

        for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
            dague_mempool_construct( &vp->datarepo_mempools[pi],
                                     NULL, sizeof(data_repo_entry_t)+(pi-1)*sizeof(dague_arena_chunk_t*),
                                     ((char*)&fake_entry.data_repo_mempool_owner) - ((char*)&fake_entry),
                                     vp->nb_cores);

    }
    /* Synchronize with the other threads */
    dague_barrier_wait(startup->barrier);

    if( NULL != current_scheduler->module.flow_init )
        current_scheduler->module.flow_init(eu, startup->barrier);

    eu->context_mempool = &(eu->virtual_process->context_mempool.thread_mempools[eu->th_id]);
    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
        eu->datarepo_mempools[pi] = &(eu->virtual_process->datarepo_mempools[pi].thread_mempools[eu->th_id]);

#ifdef DAGUE_PROF_TRACE
    {
        char *binding = dague_hwloc_get_binding();
        eu->eu_profile = dague_profiling_thread_init( 2*1024*1024,
                                                      DAGUE_PROFILE_THREAD_STR,
                                                      eu->th_id,
                                                      eu->virtual_process->vp_id,
                                                      binding);
        free(binding);
    }
    if( NULL != eu->eu_profile ) {
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "boundto", startup->bindto);
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "th_id", eu->th_id);
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "vp_id", eu->virtual_process->vp_id );
    }
#endif /* DAGUE_PROF_TRACE */

    PINS_THREAD_INIT(eu);

#if defined(DAGUE_SIM)
    eu->largest_simulation_date = 0;
#endif

    /* The main thread of VP 0 will go back to the user level */
    if( DAGUE_THREAD_IS_MASTER(eu) ) {
        return NULL;
    }

    return (void*)(long)__dague_context_wait(eu);
}

static void dague_vp_init( dague_vp_t *vp,
                           int32_t nb_cores,
                           __dague_temporary_thread_initialization_t *startup)
{
    int t, pi;
    dague_barrier_t*  barrier;

    vp->nb_cores = nb_cores;

    barrier = (dague_barrier_t*)malloc(sizeof(dague_barrier_t));
    dague_barrier_init(barrier, NULL, vp->nb_cores);

    /* Prepare the temporary storage for each thread startup */
    for( t = 0; t < vp->nb_cores; t++ ) {
        startup[t].th_id = t;
        startup[t].virtual_process = vp;
        startup[t].nb_cores = nb_cores;
        startup[t].bindto = -1;
        startup[t].bindto_ht = -1;
        startup[t].barrier = barrier;
        pi = vpmap_get_nb_cores_affinity(vp->vp_id, t);
        if( 1 == pi )
            vpmap_get_core_affinity(vp->vp_id, t, &startup[t].bindto, &startup[t].bindto_ht);
        else if( 1 < pi )
            dague_warning("multiple core to bind on... for now, do nothing"); //TODO: what does that mean?
    }
}

#define DEFAULT_APPNAME "app_name_%d"

#define GET_INT_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = dague_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        char* __value = dague_cmd_line_get_param((CMD), (ARGV), 0, 0); \
        if( NULL != __value ) \
            (VALUE) = (int)strtol(__value, NULL, 10); \
    } \
} while (0)

#define GET_STR_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = dague_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        (VALUE) = dague_cmd_line_get_param((CMD), (ARGV), 0, 0); \
    } \
} while (0)


dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[] )
{
    int ret, nb_vp, p, t, nb_total_comp_threads, display_vpmap = 0;
    char *comm_binding_parameter = NULL;
    char *binding_parameter = NULL;
    __dague_temporary_thread_initialization_t *startup;
    dague_context_t* context;
    dague_cmd_line_t *cmd_line = NULL;
    char **environ = NULL;
    char **env_variable, *env_name, *env_value;

    dague_installdirs_open();
    dague_mca_param_init();
    dague_output_init();

    /* Extract what we can from the arguments */
    cmd_line = OBJ_NEW(dague_cmd_line_t);
    if( NULL == cmd_line ) {
        return NULL;
    }

    /* Declare the command line for the .dot generation */
    dague_cmd_line_make_opt3(cmd_line, 'h', "help", "help", 0,
                             "Show the usage text.");
    dague_cmd_line_make_opt3(cmd_line, '.', "dot", "dague_dot", 1,
                             "Filename for the .dot file");
    dague_cmd_line_make_opt3(cmd_line, 'b', NULL, "dague_bind", 1,
                             "Execution thread binding");
    dague_cmd_line_make_opt3(cmd_line, 'C', NULL, "dague_bind_comm", 1,
                             "Communication thread binding");
    dague_cmd_line_make_opt3(cmd_line, 'c', "cores", "cores", 1,
                             "Number of cores to used");
    dague_cmd_line_make_opt3(cmd_line, 'g', "gpus", "gpus", 1,
                             "Number of GPU to used (deprecated use MCA instead)");
    dague_cmd_line_make_opt3(cmd_line, 'V', "vpmap", "vpmap", 1,
                             "Virtual process map");
    dague_cmd_line_make_opt3(cmd_line, 'H', "ht", "ht", 1,
                             "Enable hyperthreading");
    dague_mca_cmd_line_setup(cmd_line);

    if( (NULL != pargc) && (0 != *pargc) ) {
        dague_app_name = strdup( (*pargv)[0] );

        ret = dague_cmd_line_parse(cmd_line, true, *pargc, *pargv);
        if (DAGUE_SUCCESS != ret) {
            fprintf(stderr, "%s: command line error (%d)\n", (*pargv)[0], ret);
        }
    } else {
        ret = asprintf( &dague_app_name, DEFAULT_APPNAME, (int)getpid() );
        if (ret == -1) {
            dague_app_name = strdup( "app_name_XXXXXX" );
        }
    }

    ret = dague_mca_cmd_line_process_args(cmd_line, &environ, &environ);
    if( environ != NULL ) {
        for(env_variable = environ;
            *env_variable != NULL;
            env_variable++) {
            env_name = *env_variable;
            for(env_value = env_name; *env_value != '\0' && *env_value != '='; env_value++)
                /* nothing */;
            if(*env_value == '=') {
                *env_value = '\0';
                env_value++;
            }
            setenv(env_name, env_value, 1);
            free(*env_variable);
        }
        free(environ);
    }
    dague_debug_init();
    mca_components_repository_init();

#if defined(HAVE_HWLOC)
    dague_hwloc_init();
#endif  /* defined(HWLOC) */

    if( dague_cmd_line_is_taken(cmd_line, "ht") ) {
#if defined(HAVE_HWLOC)
        int hyperth = 0;
        GET_INT_ARGV(cmd_line, "ht", hyperth);
        dague_hwloc_allow_ht(hyperth);
#else
        dague_inform("Option ht (hyper-threading) is only supported when HWLOC is enabled at compile time.");
#endif  /* defined(HAVE_HWLOC) */
    }

    /* Set a default the number of cores if not defined by parameters
     * - with hwloc if available
     * - with sysconf otherwise (hyperthreaded core number)
     */
    if( nb_cores <= 0 ) {
        nb_cores = dague_hwloc_nb_real_cores();
    }

    if( dague_cmd_line_is_taken(cmd_line, "gpus") ) {
        dague_warning("Option g (for accelerators) is deprecated as an argument. Use the MCA parameter instead.");
    }

    GET_INT_ARGV(cmd_line, "cores", nb_cores);
    GET_STR_ARGV(cmd_line, "dague_bind_comm", comm_binding_parameter);
    GET_STR_ARGV(cmd_line, "dague_bind", binding_parameter);

    /**
     * Initializa the VPMAP, the discrete domains hosting
     * execution flows but where work stealing is prevented.
     */
    {
        /* Change the vpmap choice: first cancel the previous one if any */
        vpmap_fini();

        if( dague_cmd_line_is_taken(cmd_line, "vpmap") ) {
            char* optarg = NULL;
            GET_STR_ARGV(cmd_line, "vpmap", optarg);
            if( !strncmp(optarg, "display", 7 )) {
                display_vpmap = 1;
            } else {
                if( !strncmp(optarg, "flat", 4) ) {
                    /* default case (handled in dague_init) */
                } else if( !strncmp(optarg, "hwloc", 5) ) {
                    vpmap_init_from_hardware_affinity(nb_cores);
                } else if( !strncmp(optarg, "file:", 5) ) {
                    vpmap_init_from_file(optarg + 5);
                } else if( !strncmp(optarg, "rr:", 3) ) {
                    int n, p, co;
                    sscanf(optarg, "rr:%d:%d:%d", &n, &p, &co);
                    vpmap_init_from_parameters(n, p, co);
                } else {
                    dague_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
                }
            }
        }
        nb_vp = vpmap_get_nb_vp();
        if( -1 == nb_vp ) {
            vpmap_init_from_flat(nb_cores);
            nb_vp = vpmap_get_nb_vp();
        }
    }

    if( dague_cmd_line_is_taken(cmd_line, "dot") ) {
        char* optarg = NULL;
        GET_STR_ARGV(cmd_line, "dot", optarg);

        if( dague_enable_dot ) free( dague_enable_dot );
        if( NULL == optarg ) {
            dague_enable_dot = strdup(dague_app_name);
        } else {
            dague_enable_dot = strdup(optarg);
        }
    }

    /* the extra allocation will pertain to the virtual_processes array */
    context = (dague_context_t*)malloc(sizeof(dague_context_t) + (nb_vp-1) * sizeof(dague_vp_t*));

    context->__dague_internal_finalization_in_progress = 0;
    context->__dague_internal_finalization_counter = 0;
    context->active_objects = 0;
    context->flags          = 0;
    context->nb_nodes       = 1;
    context->comm_ctx       = NULL;
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
        dague_inform("Your vpmap uses %d threads when %d cores where available",
                     nb_total_comp_threads, nb_cores);
        nb_cores = nb_total_comp_threads;
    }

    startup = (__dague_temporary_thread_initialization_t*)
        malloc(nb_total_comp_threads * sizeof(__dague_temporary_thread_initialization_t));

    context->nb_vp = nb_vp;
    t = 0;
    for(p = 0; p < nb_vp; p++) {
        dague_vp_t *vp;
        vp = (dague_vp_t *)malloc(sizeof(dague_vp_t) + (vpmap_get_nb_threads_in_vp(p)-1) * sizeof(dague_execution_unit_t*));
        vp->dague_context = context;
        vp->vp_id = p;
        context->virtual_processes[p] = vp;
        /**
         * Set the threads local variables from startup[t] -> startup[t+nb_cores].
         * Do not create or initialize any memory yet, or it will be automatically
         * bound to the allocation context of this thread.
         */
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

#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    /* update the index_core_free_mask according to the thread binding defined */
    for(t = 0; t < nb_total_comp_threads; t++)
        hwloc_bitmap_clr(context->index_core_free_mask, startup[t].bindto);

#if defined(DAGUE_DEBUG_NOISIER)
    {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, context->index_core_free_mask);
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output,  "binding core free mask is %s", str);
        free(str);
    }
#endif /* defined(DAGUE_DEBUG_NOISIER) */
#endif /* HAVE_HWLOC && HAVE_HWLOC_BITMAP */

    /**
     * Parameters defining the default ARENA behavior. Handle with care they can lead to
     * significant memory consumption or to a significant overhead in memory management
     * (allocation/deallocation). These values are only used by ARENAs constructed with
     * the default constructor (not the extended one).
     */
    dague_mca_param_reg_sizet_name("arena", "max_used", "The maxmimum amount of memory each arena can"
                                   " allocate (default unlimited)",
                                   false, false, dague_arena_max_allocated_memory, &dague_arena_max_allocated_memory);
    dague_mca_param_reg_sizet_name("arena", "max_cached", "The maxmimum amount of memory each arena can"
                                   " cache in a freelist (0=no caching)",
                                   false, false, dague_arena_max_cached_memory, &dague_arena_max_cached_memory);

    dague_mca_param_reg_sizet_name("task", "startup_iter", "The number of ready tasks to be generated during the startup "
                                   "before allowing the scheduler to distribute them across the entire execution context.",
                                   false, false, dague_task_startup_iter, &dague_task_startup_iter);
    dague_mca_param_reg_sizet_name("task", "startup_chunk", "The total number of tasks to be generated during the startup "
                                   "before delaying the remaining of the startup. The startup process will be "
                                   "continued at a later moment once the number of ready tasks decreases.",
                                   false, false, dague_task_startup_chunk, &dague_task_startup_chunk);

    dague_mca_param_reg_string_name("profile", "filename",
#if defined(DAGUE_PROF_TRACE)
                                    "Path to the profiling file (<none> to disable, <app> for app name, <*> otherwise)",
                                    false, false,
#else
                                    "Path to the profiling file (unused due to profiling being turned off during building)",
                                    false, true,  /* profiling disabled: read-only */
#endif  /* defined(DAGUE_PROF_TRACE) */
                                    "<none>", &dague_enable_profiling);
#if defined(DAGUE_PROF_TRACE)
    if( (0 != strncasecmp(dague_enable_profiling, "<none>", 6)) && (0 == dague_profiling_init( )) ) {
        int i, l;
        char *cmdline_info = basename(dague_app_name);

        /* Use either the app name (argv[0]) or the user provided filename */
        if( 0 == strncmp(dague_enable_profiling, "<app>", 5) ) {
            ret = dague_profiling_dbp_start( cmdline_info, dague_app_name );
        } else {
            ret = dague_profiling_dbp_start( dague_enable_profiling, dague_app_name );
        }
        if( ret != 0 ) {
            dague_warning("Profiling framework deactivated because of error %s.", dague_profiling_strerror());
        }

        l = 0;
        for(i = 0; i < *pargc; i++) {
            l += strlen( (*pargv)[i] ) + 1;
        }
        cmdline_info = (char*)calloc(sizeof(char), l + 1);
        l = 0;
        for(i = 0; i < *pargc; i++) {
            sprintf(cmdline_info + l, "%s ", (*pargv)[i]);
            l += strlen( (*pargv)[i] ) + 1;
        }
        cmdline_info[l] = '\0';
        dague_profiling_add_information("CMDLINE", cmdline_info);

        /* we should be adding the PaRSEC options to the profile here
         * instead of in common.c/h as we do now. */
        PROFILING_SAVE_iINFO("nb_cores", nb_cores);
        PROFILING_SAVE_iINFO("nb_vps", nb_vp);

        free(cmdline_info);

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
#if defined(DAGUE_PROF_TRACE_ACTIVE_ARENA_SET)
        dague_profiling_add_dictionary_keyword( "ARENA_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_alloc_key, &arena_memory_free_key);
        dague_profiling_add_dictionary_keyword( "ARENA_ACTIVE_SET", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_used_key, &arena_memory_unused_key);
#endif  /* defined(DAGUE_PROF_TRACE_ACTIVE_ARENA_SET) */
        dague_profiling_add_dictionary_keyword( "TASK_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &task_memory_alloc_key, &task_memory_free_key);
        dague_profiling_add_dictionary_keyword( "Device delegate", "fill:#EAE7C6",
                                                0, NULL,
                                                &device_delegate_begin, &device_delegate_end);
        /* Ready to rock! The profiling must be on by default */
        dague_profiling_start();
    }
#endif  /* DAGUE_PROF_TRACE */

    /* Initialize Performance Instrumentation (PINS) */
    PINS_INIT(context);

    dague_devices_init(context);
    /* By now let's add one device for the CPUs */
    {
        dague_device_cpus = (dague_device_t*)calloc(1, sizeof(dague_device_t));
        dague_device_cpus->name = "default";
        dague_device_cpus->type = DAGUE_DEV_CPU;
        dague_devices_add(context, dague_device_cpus);
        /* TODO: This is plain WRONG, but should work by now */
        dague_device_cpus->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        dague_device_cpus->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }
    /* By now let's add one device for the recursive kernels */
    {
        dague_device_recursive = (dague_device_t*)calloc(1, sizeof(dague_device_t));
        dague_device_recursive->name = "recursive";
        dague_device_recursive->type = DAGUE_DEV_RECURSIVE;
        dague_devices_add(context, dague_device_recursive);
        /* TODO: This is plain WRONG, but should work by now */
        dague_device_recursive->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        dague_device_recursive->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }
    dague_devices_select(context);
    dague_devices_freeze(context);

    /* Init the data infrastructure. Must be done only after the freeze of the devices */
    dague_data_init(context);

    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_total_comp_threads );

    /* Load the default scheduler. User can change it afterward,
     * but we need to ensure that one is loadable and here.
     */
    if( 0 == dague_set_scheduler( context ) ) {
        /* TODO: handle memory leak / thread leak here: this is a fatal
         * error for PaRSEC */
        dague_abort("Unable to load any scheduler in init function.");
        return NULL;
    }

    if(dague_enable_dot) {
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_init(dague_enable_dot, nb_total_comp_threads);
#else
        dague_warning("DOT generation requested, but DAGUE_PROF_GRAPHER was not selected during compilation: DOT generation ignored.");
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
    {
        dague_barrier_t* barrier = startup[0].barrier;
        dague_barrier_destroy(barrier);
        free(barrier);
        for(t = 0; t < nb_total_comp_threads; t++) {
            if(barrier != startup[t].barrier) {
                barrier = startup[t].barrier;
                dague_barrier_destroy(barrier);
                free(barrier);
            }
        }
    }
    free(startup);

    /* Introduce communication thread */
    (void)dague_remote_dep_init(context);
    dague_statistics("DAGuE");

    AYU_INIT();

    /* Play with the thread placement */
    if( NULL != comm_binding_parameter )
        dague_parse_comm_binding_parameter(comm_binding_parameter, context);
    if( NULL != binding_parameter )
        dague_parse_binding_parameter(binding_parameter, context, startup);

    if( display_vpmap )
        vpmap_display_map();

    if( dague_cmd_line_is_taken(cmd_line, "help") ||
        dague_cmd_line_is_taken(cmd_line, "h")) {
        char* help_msg = dague_cmd_line_get_usage_msg(cmd_line);
        dague_list_t* l = NULL;

        dague_output(0, "%s\n\nRegistered MCA parameters", help_msg);
        free(help_msg);

        dague_mca_param_dump(&l, 1);
        dague_mca_show_mca_params(l, "all", "all", 1);
        dague_mca_param_dump_release(l);

        dague_fini(&context);
        return NULL;
    }

    if( NULL != cmd_line )
        OBJ_RELEASE(cmd_line);

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
    int nb_total_comp_threads, p;

    /**
     * We need to force the main thread to drain all possible pending messages
     * on the communication layer. This is not an issue in a distributed run,
     * but on a single node run with MPI support, objects can be created (and
     * thus context_id additions might be pending on the communication layer).
     */
#if defined(DISTRIBUTED)
    if( (1 == dague_communication_engine_up) &&  /* engine enabled */
        (context->nb_nodes == 1) &&  /* single node: otherwise the messages will
                                      * be drained by the communication thread */
        DAGUE_THREAD_IS_MASTER(context->virtual_processes[0]->execution_units[0]) ) {
        /* check for remote deps completion */
        dague_remote_dep_progress(context->virtual_processes[0]->execution_units[0]);
    }
#endif /* defined(DISTRIBUTED) */

    /* Now wait until every thread is back */
    context->__dague_internal_finalization_in_progress = 1;
    dague_barrier_wait( &(context->barrier) );

    PINS_THREAD_FINI(context->virtual_processes[0]->execution_units[0]);

    nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

    /* The first execution unit is for the master thread */
    if( nb_total_comp_threads > 1 ) {
        for(p = 1; p < nb_total_comp_threads; p++) {
            pthread_join( context->pthreads[p], NULL );
        }
        free(context->pthreads);
        context->pthreads = NULL;
    }
    /* From now on all the thrteads have been shut-off, and they are supposed to
     * have cleaned all their provate memory. Unleash the global cleaning process.
     */

    PINS_FINI(context);

#ifdef DAGUE_PROF_TRACE
    dague_profiling_dbp_dump();
#endif  /* DAGUE_PROF_TRACE */

    (void) dague_remote_dep_fini(context);

    dague_remove_scheduler( context );

    dague_data_fini(context);

    dague_devices_remove(dague_device_cpus);
    free(dague_device_cpus);
    dague_device_cpus = NULL;

    dague_devices_remove(dague_device_recursive);
    free(dague_device_recursive);
    dague_device_recursive = NULL;

    dague_devices_fini(context);

    for(p = 0; p < context->nb_vp; p++) {
        dague_vp_fini(context->virtual_processes[p]);
        free(context->virtual_processes[p]);
        context->virtual_processes[p] = NULL;
    }

    AYU_FINI();
#ifdef DAGUE_PROF_TRACE
    (void)dague_profiling_fini( );  /* we're leaving, ignore errors */
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

    dague_handle_empty_repository();

    dague_mca_param_finalize();
    dague_installdirs_close();
    dague_output_finalize();

    free(context);
    *pcontext = NULL;

    dague_class_finalize();
    dague_debug_fini();  /* Always last */
    return 0;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t
dague_check_IN_dependencies_with_mask( const dague_handle_t *dague_handle,
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
         * it is assumed that no control should be expected.
         */
        if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) {
            active = (1 << flow->flow_index);
            /* Control case: resolved unless we find at least one input control */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) ) {
                        /* Cannot use control gather magic with the USE_DEPS_MASK */
                        assert( NULL == dep->ctl_gather_nb );
                        continue;
                    }
                }
                active = 0;
                break;
            }
        } else {
            if( !(flow->flow_flags & FLOW_HAS_IN_DEPS) ) continue;
            if( NULL == flow->dep_in[0] ) {
                /** As the flow is tagged with FLOW_HAS_IN_DEPS and there is no
                 * dep_in we are in the case where a write only dependency used
                 * an in dependency to specify the arena where the data should
                 * be allocated.
                 */
                active = (1 << flow->flow_index);
            } else {
                /* Data case: resolved only if we found a data already ready */
                for( active = 0, j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                    dep = flow->dep_in[j];
                    if( NULL != dep->cond ) {
                        /* Check if the condition apply on the current setting */
                        assert( dep->cond->op == EXPR_OP_INLINE );
                        if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) )
                            continue;  /* doesn't match */
                        /* the condition triggered let's check if it's for a data */
                    }  /* otherwise we have an input flow without a condition, it MUST be final */
                    if( 0xFF == dep->function_id )
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
dague_check_IN_dependencies_with_counter( const dague_handle_t *dague_handle,
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
        if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) {
            /* Control case: just count how many must be resolved */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( dep->cond->inline_func32(dague_handle, exec_context->locals) ) {
                        if( NULL == dep->ctl_gather_nb)
                            active++;
                        else {
                            assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                            active += dep->ctl_gather_nb->inline_func32(dague_handle, exec_context->locals);
                        }
                    }
                } else {
                    if( NULL == dep->ctl_gather_nb)
                        active++;
                    else {
                        assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                        active += dep->ctl_gather_nb->inline_func32(dague_handle, exec_context->locals);
                    }
                }
            }
        } else {
            /* Data case: we count how many inputs we must have (the opposite
             * compared with the mask case). We iterate over all the input
             * dependencies of the flow to make sure the flow is expected to
             * hold a valid value.
             */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) )
                        continue;  /* doesn't match */
                    /* the condition triggered let's check if it's for a data */
                } else {
                    /* we have an input flow without a condition, it MUST be final */
                }
                if( 0xFF != dep->function_id )  /* if not a data we must wait for the flow activation */
                    active++;
                break;
            }
        }
        ret += active;
    }
    return ret;
}

dague_dependency_t *dague_default_find_deps(const dague_handle_t *dague_handle,
                                            const dague_execution_context_t* restrict exec_context)
{
    dague_dependencies_t *deps;
    int p;

    deps = dague_handle->dependencies_array[exec_context->function->function_id];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_parameters - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min]);
}

static int dague_update_deps_with_counter(const dague_handle_t *dague_handle,
                                          const dague_execution_context_t* restrict exec_context,
                                          dague_dependency_t *deps)
{
    dague_dependency_t dep_new_value, dep_cur_value;
#if defined(DAGUE_DEBUG_PARANOID) || defined(DAGUE_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif

    if( 0 == *deps ) {
        dep_new_value = dague_check_IN_dependencies_with_counter( dague_handle, exec_context ) - 1;
        if( dague_atomic_cas( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_dec_32b( deps );
    } else {
        dep_cur_value = dague_atomic_dec_32b( deps );
    }
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate counter dependency for %s leftover %d (excluding current)",
            tmp, dep_cur_value);

#if defined(DAGUE_DEBUG_PARANOID)
    {
        char wtmp[MAX_TASK_STRLEN];
        if( (uint32_t)dep_cur_value > (uint32_t)-128) {
            dague_abort("function %s as reached an improbable dependency count of %u",
                  wtmp, dep_cur_value );
        }

        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Task %s has a current dependencies count of %d remaining. %s to go!",
               tmp, dep_cur_value,
               (dep_cur_value == 0) ? "Ready" : "Not ready");
    }
#endif /* DAGUE_DEBUG_PARANOID */

    return dep_cur_value == 0;
}

static int dague_update_deps_with_mask(const dague_handle_t *dague_handle,
                                       const dague_execution_context_t* restrict exec_context,
                                       dague_dependency_t *deps,
                                       const dague_execution_context_t* restrict origin,
                                       const dague_flow_t* restrict origin_flow,
                                       const dague_flow_t* restrict dest_flow)
{
    dague_dependency_t dep_new_value, dep_cur_value;
    const dague_function_t* function = exec_context->function;
#if defined(DAGUE_DEBUG_NOISIER) || defined(DAGUE_DEBUG_PARANOID)
    char tmpo[MAX_TASK_STRLEN], tmpt[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmpo, MAX_TASK_STRLEN, origin);
    dague_snprintf_execution_context(tmpt, MAX_TASK_STRLEN, exec_context);
#endif

    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate mask dep for %s:%s (current 0x%x now 0x%x goal 0x%x) from %s:%s",
            dest_flow->name, tmpt, *deps, (1 << dest_flow->flow_index), function->dependencies_goal,
            origin_flow->name, tmpo);
#if defined(DAGUE_DEBUG_PARANOID)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        dague_abort("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)",
               dest_flow->flow_index, tmpo,
               origin_flow->name, *deps,
               tmpt, dest_flow->name );
    }
#else
    (void) origin; (void) origin_flow;
#endif

    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= dague_check_IN_dependencies_with_mask( dague_handle, exec_context );
#if defined(DAGUE_DEBUG_NOISIER)
        if( dep_new_value != 0 ) {
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Activate IN dependencies with mask 0x%x", dep_new_value);
        }
#endif
    }

    dep_cur_value = dague_atomic_bor( deps, dep_new_value );

#if defined(DAGUE_DEBUG_PARANOID)
    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {
        int success;
        dague_dependency_t tmp_mask;
        tmp_mask = *deps;
        success = dague_atomic_cas( deps,
                                    tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
        if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
            dague_abort("Task %s scheduled twice (second time by %s)!!!",
                   tmpt, tmpo);
        }
    }
#endif  /* defined(DAGUE_DEBUG_PARANOID) */

    DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Task %s has a current dependencies of 0x%x and a goal of 0x%x. %s to go!",
            tmpt, dep_cur_value, function->dependencies_goal,
            ((dep_cur_value & function->dependencies_goal) == function->dependencies_goal) ?
            "Ready" : "Not ready");
    return (dep_cur_value & function->dependencies_goal) == function->dependencies_goal;
}

/**
 * Mark the task as having all it's dependencies satisfied. This is not
 * necessarily required for the startup process, but it leaves traces such that
 * all executed tasks will show consistently (no difference between the startup
 * tasks and later tasks).
 */
void dague_dependencies_mark_task_as_startup(dague_execution_context_t* restrict exec_context)
{
    const dague_function_t* function = exec_context->function;
    dague_handle_t *dague_handle = exec_context->dague_handle;
    dague_dependency_t *deps = function->find_deps(dague_handle, exec_context);

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
int dague_release_local_OUT_dependencies(dague_execution_unit_t* eu_context,
                                         const dague_execution_context_t* restrict origin,
                                         const dague_flow_t* restrict origin_flow,
                                         const dague_execution_context_t* restrict exec_context,
                                         const dague_flow_t* restrict dest_flow,
                                         data_repo_entry_t* dest_repo_entry,
                                         dague_dep_data_description_t* data,
                                         dague_execution_context_t** pready_ring)
{
    const dague_function_t* function = exec_context->function;
    dague_dependency_t *deps;
    int completed;
#if defined(DAGUE_DEBUG_NOISIER)
    char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context);
#endif

    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate dependencies for %s flags = 0x%04x", tmp1, function->flags);
    deps = function->find_deps(origin->dague_handle, exec_context);

    if( function->flags & DAGUE_USE_DEPS_MASK ) {
        completed = dague_update_deps_with_mask(origin->dague_handle, exec_context, deps, origin, origin_flow, dest_flow);
    } else {
        completed = dague_update_deps_with_counter(origin->dague_handle, exec_context, deps);
    }

#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_dep(origin, exec_context, completed, origin_flow, dest_flow);
#endif  /* defined(DAGUE_PROF_GRAPHER) */

    if( completed ) {

        /* This task is ready to be executed as all dependencies are solved.
         * Queue it into the ready_list passed as an argument.
         */
        {
            dague_execution_context_t *new_context = (dague_execution_context_t *) dague_thread_mempool_allocate(eu_context->context_mempool);
            DAGUE_COPY_EXECUTION_CONTEXT(new_context, exec_context);
            new_context->status = DAGUE_TASK_STATUS_NONE;
            AYU_ADD_TASK(new_context);

            DAGUE_DEBUG_VERBOSE(6, dague_debug_output,
                   "%s becomes ready from %s on thread %d:%d, with mask 0x%04x and priority %d",
                   tmp1,
                   dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, origin),
                   eu_context->th_id, eu_context->virtual_process->vp_id,
                   *deps,
                   exec_context->priority);

            assert( dest_flow->flow_index <= new_context->function->nb_flows);
            memset( new_context->data, 0, sizeof(dague_data_pair_t) * new_context->function->nb_flows);
            /**
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].data_repo = dest_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data_in   = origin->data[origin_flow->flow_index].data_out;
            (void)data;
            AYU_ADD_TASK_DEP(new_context, (int)dest_flow->flow_index);

            if(exec_context->function->flags & DAGUE_IMMEDIATE_TASK) {
                DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "  Task %s is immediate and will be executed ASAP", tmp1);
                __dague_execute(eu_context, new_context);
                __dague_complete_execution(eu_context, new_context);
#if 0 /* TODO */
                SET_HIGHEST_PRIORITY(new_context, dague_execution_context_priority_comparator);
                DAGUE_LIST_ITEM_SINGLETON(&(new_context->list_item));
                if( NULL != (*pimmediate_ring) ) {
                    (void)dague_list_item_ring_push( (dague_list_item_t*)(*pimmediate_ring), &new_context->list_item );
                }
                *pimmediate_ring = new_context;
#endif
            } else {
                *pready_ring = (dague_execution_context_t*)
                    dague_list_item_ring_push_sorted( (dague_list_item_t*)(*pready_ring),
                                                      &new_context->super.list_item,
                                                      dague_execution_context_priority_comparator );
            }
        }
    } else { /* Service not ready */
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "  => Service %s not yet ready", tmp1);
    }

    return 0;
}

dague_ontask_iterate_t
dague_release_dep_fct(dague_execution_unit_t *eu,
                      const dague_execution_context_t *newcontext,
                      const dague_execution_context_t *oldcontext,
                      const dep_t* dep,
                      dague_dep_data_description_t* data,
                      int src_rank, int dst_rank, int dst_vpid,
                      void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    const dague_flow_t* src_flow = dep->belongs_to;

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {

        assert( 0 == (arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS) );

        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ){
            struct remote_dep_output_param_s* output;
            int _array_pos, _array_mask;

#if !defined(DAGUE_DIST_COLLECTIVES)
            assert(src_rank == eu->virtual_process->dague_context->my_rank);
#endif
            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            output = &arg->remote_deps->output[dep->dep_datatype_index];
            assert( (-1 == arg->remote_deps->root) || (arg->remote_deps->root == src_rank) );
            arg->remote_deps->root = src_rank;
            arg->remote_deps->outgoing_mask |= (1 << dep->dep_datatype_index);
            if( !(output->rank_bits[_array_pos] & _array_mask) ) {
                output->rank_bits[_array_pos] |= _array_mask;
                output->deps_mask |= (1 << dep->dep_index);
                if( 0 == output->count_bits ) {
                    output->data = *data;
                } else {
                    assert(output->data.data == data->data);
                }
                output->count_bits++;
                if(newcontext->priority > output->priority) {
                    output->priority = newcontext->priority;
                    if(newcontext->priority > arg->remote_deps->max_priority)
                        arg->remote_deps->max_priority = newcontext->priority;
                }
            }  /* otherwise the bit is already flipped, the peer is already part of the propagation. */
        }
    }
#else
    (void)src_rank;
    (void)data;
#endif

    if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
        (eu->virtual_process->dague_context->my_rank == dst_rank) ) {
        /* Old condition */
        /* if( FLOW_ACCESS_NONE != (src_flow->flow_flags & FLOW_ACCESS_MASK) ) { */

        /* Copying data in data-repo if there is data .
         * We are doing this in order for dtd to be able to track control dependences.
         */
        if( oldcontext->data[src_flow->flow_index].data_out != NULL ) {
            arg->output_entry->data[src_flow->flow_index] = oldcontext->data[src_flow->flow_index].data_out;
            arg->output_usage++;
            /* BEWARE: This increment is required to be done here. As the target task
             * bits are marked, another thread can now enable the task. Once schedulable
             * the task will try to access its input data and decrement their ref count.
             * Thus, if the ref count is not increased here, the data might dissapear
             * before this task released it completely.
             */
            OBJ_RETAIN( arg->output_entry->data[src_flow->flow_index] );
        }
        dague_release_local_OUT_dependencies(eu, oldcontext, src_flow,
                                             newcontext, dep->flow,
                                             arg->output_entry,
                                             data,
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

    assert( NULL != task->dague_handle );
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
    index += snprintf(str + index, size - index, "]<%d>{%u}", task->priority, task->dague_handle->handle_id );

    return str;
}
/**
 * Convert assignments to a string.
 */
char* dague_snprintf_assignments( char* str, size_t size,
                                  const dague_function_t* function,
                                  const assignment_t* locals)
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
int dague_set_complete_callback( dague_handle_t* dague_handle,
                                 dague_event_cb_t complete_cb, void* complete_cb_data )
{
    if( NULL == dague_handle->on_complete ) {
        dague_handle->on_complete      = complete_cb;
        dague_handle->on_complete_data = complete_cb_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_get_complete_callback( const dague_handle_t* dague_handle,
                                 dague_event_cb_t* complete_cb, void** complete_cb_data )
{
    if( NULL != dague_handle->on_complete ) {
        *complete_cb      = dague_handle->on_complete;
        *complete_cb_data = dague_handle->on_complete_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_set_enqueue_callback( dague_handle_t* dague_handle,
                                dague_event_cb_t enqueue_cb, void* enqueue_cb_data )
{
    if( NULL == dague_handle->on_enqueue ) {
        dague_handle->on_enqueue      = enqueue_cb;
        dague_handle->on_enqueue_data = enqueue_cb_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_get_enqueue_callback( const dague_handle_t* dague_handle,
                                dague_event_cb_t* enqueue_cb, void** enqueue_cb_data )
{
    if( NULL != dague_handle->on_enqueue ) {
        *enqueue_cb      = dague_handle->on_enqueue;
        *enqueue_cb_data = dague_handle->on_enqueue_data;
        return 0;
    }
    return -1;
}

/* TODO: Change this code to something better */
static volatile uint32_t object_array_lock = 0;
static dague_handle_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;
#define NOOBJECT ((void*)-1)

static void dague_handle_empty_repository(void)
{
    dague_atomic_lock( &object_array_lock );
    free(object_array);
    object_array = NULL;
    object_array_size = 1;
    object_array_pos = 0;
    dague_atomic_unlock( &object_array_lock );
}

/**< Retrieve the local object attached to a unique object id */
dague_handle_t* dague_handle_lookup( uint32_t handle_id )
{
    dague_handle_t *r = NOOBJECT;
    dague_atomic_lock( &object_array_lock );
    if( handle_id <= object_array_pos ) {
        r = object_array[handle_id];
    }
    dague_atomic_unlock( &object_array_lock );
    return (NOOBJECT == r ? NULL : r);
}

/**< Reverse an unique ID for the handle but without adding the object to the management array.
 *   Beware that on a distributed environment the connected objects must have the same ID.
 */
int dague_handle_reserve_id( dague_handle_t* object )
{
    uint32_t idx;

    dague_atomic_lock( &object_array_lock );
    idx = (uint32_t)++object_array_pos;

    if( (NULL == object_array) || (idx >= object_array_size) ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object->handle_id = idx;
    assert( NOOBJECT == object_array[idx] );
    dague_atomic_unlock( &object_array_lock );
    return idx;
}

/**< Register a handle object with the engine. Once enrolled the object can be target
 * for other components of the runtime, such as communications.
 */
int dague_handle_register( dague_handle_t* object)
{
    uint32_t idx = object->handle_id;

    dague_atomic_lock( &object_array_lock );
    if( (NULL == object_array) || (idx >= object_array_size) ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object_array[idx] = object;
    dague_atomic_unlock( &object_array_lock );
    return idx;
}

/**< globally synchronize object id's so that next register generates the same
 * id at all ranks. */
void dague_handle_sync_ids( void )
{
    uint32_t idx;
    dague_atomic_lock( &object_array_lock );
    idx = (int)object_array_pos;
#if defined(DISTRIBUTED) && defined(HAVE_MPI)
    MPI_Allreduce( MPI_IN_PLACE, &idx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
#endif
    if( idx >= object_array_size ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object_array_pos = idx;
    dague_atomic_unlock( &object_array_lock );
}

/**< Unregister the object with the engine. This make the handle available for
 * future handles. Beware that in a distributed environment the connected objects
 * must have the same ID.
 */
void dague_handle_unregister( dague_handle_t* object )
{
    dague_atomic_lock( &object_array_lock );
    assert( object->handle_id < object_array_size );
    assert( object_array[object->handle_id] == object );
    assert( 0 == object->nb_tasks );
    assert( 0 == object->nb_pending_actions );
    object_array[object->handle_id] = NOOBJECT;
    dague_atomic_unlock( &object_array_lock );
}

void dague_handle_free(dague_handle_t *handle)
{
    if( NULL == handle )
        return;
    if( NULL == handle->destructor ) {
        free( handle );
        return;
    }
    /* the destructor calls the appropriate free on the handle */
    handle->destructor( handle );
}

/**
 * The final step of a handle activation. At this point we assume that all the local
 * initializations have been succesfully completed for all components, and that the
 * handle is ready to be registered with the system, and any potential pending tasks
 * ready to go.
 *
 * The local_task allows for concurrent management of the startup_queue, and provide a way
 * to prevent a task from being added to the scheduler. The nb_tasks equal to zero prevents
 * the handle from being registered with the communications engine, as no remote dependencies
 * can exists (because there are no local tasks).
 */
int dague_handle_enable(dague_handle_t* handle,
                        dague_execution_context_t** startup_queue,
                        dague_execution_context_t* local_task,
                        dague_execution_unit_t * eu,
                        int nb_tasks)
{
    if( NULL != startup_queue ) {
        dague_list_item_t *ring = NULL;
        dague_execution_context_t* ttask = (dague_execution_context_t*)*startup_queue;

        while( NULL != (ttask = (dague_execution_context_t*)*startup_queue) ) {
            /* Transform the single linked list into a ring */
            *startup_queue = (dague_execution_context_t*)ttask->super.list_item.list_next;
            if(ttask != local_task) {
                ttask->status = DAGUE_TASK_STATUS_HOOK;
                DAGUE_LIST_ITEM_SINGLETON(ttask);
                if(NULL == ring) ring = (dague_list_item_t *)ttask;
                else dague_list_item_ring_push(ring, &ttask->super.list_item);
            }
        }
        if( NULL != ring ) __dague_schedule(eu, (dague_execution_context_t *)ring);
    }
    /* Always register the handle. This allows the handle_t destructor to unregister it in all cases. */
    dague_handle_register(handle);
    if( 0 != nb_tasks ) {
        (void)dague_remote_dep_new_object(handle);
    }
    return DAGUE_HOOK_RETURN_DONE;
}

/**< Add nb_tasks to the task count of the handle. This update accounts only
 *   for upper-level tasks, and ignore all other runtime related activities.
 */
int dague_handle_update_nbtask( dague_handle_t* handle, int32_t nb_tasks )
{
    int remaining;

    assert( 0 != nb_tasks );
    remaining = dague_atomic_add_32b((int32_t*)&handle->nb_tasks, nb_tasks);
    assert( 0 <= remaining );
    return (0 != remaining) ? 0 : dague_handle_update_runtime_nbtask(handle, -1);
}

/**< Increases the number of runtime associated activities (decreases if
 *   nb_tasks is negative). When this counter reaches zero the handle is
 *   considered as completed, and all resources will be marked for
 *   release.
 */
int dague_handle_update_runtime_nbtask(dague_handle_t *handle, int32_t nb_tasks)
{
    int remaining;

    assert( handle->nb_pending_actions != 0 );
    remaining = dague_atomic_add_32b((int32_t*)&(handle->nb_pending_actions), nb_tasks );
    assert( 0<= remaining );
    return dague_check_complete_cb(handle, handle->context, remaining);
}

/**< Print DAGuE usage message */
void dague_usage(void)
{
    dague_output(0,"\n"
            "A DAGuE argument sequence prefixed by \"--\" can end the command line\n\n"
            "     --dague_bind_comm   : define the core the communication thread will be bound on\n"
            "\n"
            "     Warning:: The binding options rely on hwloc. The core numerotation is defined between 0 and the number of cores.\n"
            "     Be careful when used with cgroups.\n"
            "\n"
            "    --help         : this message\n"
            "\n"
            " -c --cores        : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --gpus         : number of GPU (default: 0)\n"
            " -o --scheduler    : select the scheduler (default: LFQ)\n"
            "                     Accepted values:\n"
            "                       LFQ -- Local Flat Queues\n"
            "                       GD  -- Global Dequeue\n"
            "                       LHQ -- Local Hierarchical Queues\n"
            "                       AP  -- Absolute Priorities\n"
            "                       PBQ -- Priority Based Local Flat Queues\n"
            "                       LTQ -- Local Tree Queues\n"
            "\n"
            "    --dot[=file]   : create a dot output file (default: don't)\n"
            "\n"
            "    --ht nbth      : enable a SMT/HyperThreadind binding using nbth hyper-thread per core.\n"
            "                     This parameter must be declared before the virtual process distribution parameter\n"
            " -V --vpmap        : select the virtual process map (default: flat map)\n"
            "                     Accepted values:\n"
            "                       flat  -- Flat Map: all cores defined with -c are under the same virtual process\n"
            "                       hwloc -- Hardware Locality based: threads up to -c are created and threads\n"
            "                                bound on cores that are under the same socket are also under the same\n"
            "                                virtual process\n"
            "                       rr:n:p:c -- create n virtual processes per real process, each virtual process with p threads\n"
            "                                   bound in a round-robin fashion on the number of cores c (overloads the -c flag)\n"
            "                       file:filename -- uses filename to load the virtual process map. Each entry details a virtual\n"
            "                                        process mapping using the semantic  [mpi_rank]:nb_thread:binding  with:\n"
            "                                        - mpi_rank : the mpi process rank (empty if not relevant)\n"
            "                                        - nb_thread : the number of threads under the virtual process\n"
            "                                                      (overloads the -c flag)\n"
            "                                        - binding : a set of cores for the thread binding. Accepted values are:\n"
            "                                          -- a core list          (exp: 1,3,5-6)\n"
            "                                          -- a hexadecimal mask   (exp: 0xff012)\n"
            "                                          -- a binding range expression: [start];[end];[step] \n"
            "                                             wich defines a round-robin one thread per core distribution from start\n"
            "                                             (default 0) to end (default physical core number) by step (default 1)\n"
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
            dague_warning("invalid binding file %s.", filename);
            return -1;
        }

#if defined(DISTRIBUTED) && defined(HAVE_MPI)
        /* distributed version: first retrieve the parameter for the process */
        int rank, line_num=0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        while (getline(&line, &line_len, f) != -1) {
            if(line_num==rank){
                DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "MPI_process %i uses the binding parameters: %s", rank, line);
                break;
            }
            line_num++;
        }

        if( line ){
            if( line_num==rank ) {
                dague_parse_binding_parameter(line, context, startup);
            }
            else {
                DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "MPI_process %i uses the default thread binding", rank);
            }
            free(line);
        }
#else
        /* Single process, read the first line */
        if( getline(&line, &line_len, f) != -1 ) {
            DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Binding parameters: %s", line);
        }
        if( line ){
            dague_parse_binding_parameter(line, context, startup);
            free(line);
        }
#endif /* DISTRIBUTED && HAVE_MPI */
        else
            dague_warning("default thread binding");
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

#if defined(DAGUE_DEBUG_NOISIER)
        {
            char *str = NULL;
            hwloc_bitmap_asprintf(&str, context->comm_th_index_mask);
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output,  "binding (core indexes) defined by the mask %s", str);
            free(str);
        }
#endif /* defined(DAGUE_DEBUG_NOISIER) */
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
                dague_warning("binding start core not valid (restored to default value)");
        }
        position++;  /* skip the : */
        if( '\0' != position[0] ) {
            /* check for the ending position */
            if( ':' != position[0] ) {
                arg = strtol(position, &position, 10);
                if( (arg < nb_real_cores) && (arg > -1) )
                    end = arg;
                else
                    dague_warning("binding end core not valid (restored to default value)");
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
                dague_warning("binding step not valid (restored to default value)");
        }
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "binding defined by core range [%d:%d:%d]", start, end, step);

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
                        dague_inform("No more available cores to bind to. The remaining %d threads are not bound", nb_total_comp_threads -1-t);
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
                    dague_warning("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n Binding restored to default", arg, nb_real_cores-1);
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
            dague_warning("bindind arguments are not valid (restored to default value)");
        else { /* we have a legal list to defined the binding  */
            cmp=0;
            for(t=0; t<nb_total_comp_threads; t++) {
                startup[t].bindto=core_tab[cmp];
                cmp++;
                if(core_tab[cmp] == -1)
                    cmp=0;
            }
        }
#if defined(DAGUE_DEBUG_NOISIER)
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
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "binding defined by the parsed list: %s ", tmp);
        }
#endif /* defined(DAGUE_DEBUG_NOISIER) */
    }
    return 0;
#else
    (void)optarg;
    (void)context;
    (void)startup;
    dague_warning("the binding defined by --dague_bind has been ignored (requires a build with HWLOC with bitmap support).");
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
            dague_warning("the binding defined by --dague_bind_comm has been ignored (illegal core number)");
    } else {
        /* TODO:: Add binding NUIOA aware by default */
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "default binding for the communication thread");
    }
    return 0;
#else
    (void)optarg; (void)context;
    dague_warning("The binding defined by --dague_bind has been ignored (requires HWLOC use with bitmap support).");
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
#include "dague/data_distribution.h"
static uint32_t return_local_u(dague_ddesc_t *unused, ...) { (void)unused; return 0; };
static int32_t  return_local_s(dague_ddesc_t *unused, ...) { (void)unused; return 0; };
static dague_data_t* return_data(dague_ddesc_t *unused, ...) { (void)unused; return NULL; };
static uint32_t rank_of_key(dague_ddesc_t *unused, dague_data_key_t key)
{ (void)unused; (void)key; return 0; }
static dague_data_t* data_of_key(dague_ddesc_t *unused, dague_data_key_t key)
{ (void)unused; (void)key; return NULL; }
static int32_t  vpid_of_key(dague_ddesc_t *unused, dague_data_key_t key)
{ (void)unused; (void)key; return 0; }
static dague_data_key_t data_key(dague_ddesc_t *mat, ...)
{ (void)mat; return 0; }
#if defined(DAGUE_PROF_TRACE)
static int key_to_string(dague_ddesc_t *unused, dague_data_key_t datakey, char* buffer, uint32_t buffer_size)
{
    (void)unused;
    return snprintf( buffer, buffer_size, "%u ", datakey);
}
#endif /* DAGUE_PROF_TRACE */

const dague_ddesc_t dague_static_local_data_ddesc = {
    0, /* uint32_t myrank */
    1, /* uint32_t nodes */

    data_key,  /* dague_data_key_t (*data_key)(dague_ddesc_t *mat, ...) */

    return_local_u,  /* uint32_t (*rank_of)(struct dague_ddesc *, ...) */
    rank_of_key,

    return_data,   /* dague_data_t*   (*data_of)(struct dague_ddesc *, ...) */
    data_of_key,

    return_local_s,  /* int32_t  (*vpid_of)(struct dague_ddesc *, ...) */
    vpid_of_key,

    NULL,  /* dague_memory_region_management_f register_memory */
    NULL,  /* dague_memory_region_management_f unregister_memory */
    MEMORY_STATUS_UNREGISTERED,  /* memory_registration_status_t memory_registration_status */
    NULL,  /* char      *key_base */

#ifdef DAGUE_PROF_TRACE
    key_to_string, /* int (*key_to_string)(struct dague_ddesc *, uint32_t datakey, char * buffer, uint  32_t buffer_size) */
    NULL,  /* char      *key_dim */
    NULL,  /* char      *key */
#endif /* DAGUE_PROF_TRACE */
};

static int32_t dague_expr_eval32(const expr_t *expr, dague_execution_context_t *context)
{
    dague_handle_t *handle = context->dague_handle;

    assert( expr->op == EXPR_OP_INLINE );
    return expr->inline_func32(handle, context->locals);
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

void dague_debug_print_local_expecting_tasks_for_function( dague_handle_t *handle,
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
    dague_data_ref_t ref;
    int pi, li;

    DAGUE_LIST_ITEM_SINGLETON( &context.super.list_item );
    context.super.mempool_owner = NULL;
    context.dague_handle = handle;
    context.function = function;
    context.priority = -1;
    context.status = DAGUE_TASK_STATUS_NONE;
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
        function->data_affinity(&context, &ref);
        if( ref.ddesc->rank_of_key(ref.ddesc, ref.key) == ref.ddesc->myrank ) {
            (*nlocal)++;
            dep = function->find_deps(handle, &context);
            if( function->flags & DAGUE_USE_DEPS_MASK ) {
                if( *dep & DAGUE_DEPENDENCIES_STARTUP_TASK ) {
                    (*nreleased)++;
                    if( show_startup )
                        dague_debug_verbose(19, dague_debug_output, "  Task %s is a local startup task",
                                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
                } else {
                    if((*dep & DAGUE_DEPENDENCIES_BITMASK) == function->dependencies_goal) {
                        (*nreleased)++;
                    }
                    if( show_complete ||
                        ((*dep & DAGUE_DEPENDENCIES_BITMASK) != function->dependencies_goal) ) {
                        dague_debug_verbose(20, dague_debug_output, "  Task %s is a local task with dependency 0x%08x (goal is 0x%08x) -- Flags: %s %s",
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
                    dague_debug_verbose(20, dague_debug_output, "  Task %s is a local task that must wait for %d more dependencies to complete -- using count method for this task (CTL gather)",
                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context),
                            *dep);
            }
        } else {
            if( show_remote )
                dague_debug_verbose(20, dague_debug_output, "  Task %s is a remote task",
                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
        }
    } while( dague_debug_enumerate_next_in_execution_space(&context, 0)  );
}

void dague_debug_print_local_expecting_tasks_for_handle( dague_handle_t *handle,
                                                         int show_remote, int show_startup, int show_complete)
{
    uint32_t fi;
    int nlocal, ntotal, nreleased;
    /* The handle has not been initialized yet, or it has been completed */
    if( handle->dependencies_array == NULL )
        return;

    for(fi = 0; fi < handle->nb_functions; fi++) {
        dague_debug_verbose(20, dague_debug_output, " Tasks of Function %u (%s):\n", fi, handle->functions_array[fi]->name);
        dague_debug_print_local_expecting_tasks_for_function( handle, handle->functions_array[fi],
                                                              show_remote, show_startup, show_complete,
                                                              &nlocal, &nreleased, &ntotal );
        dague_debug_verbose(20, dague_debug_output, " Total number of Tasks of Class %s: %d\n", handle->functions_array[fi]->name, ntotal);
        dague_debug_verbose(20, dague_debug_output, " Local number of Tasks of Class %s: %d\n", handle->functions_array[fi]->name, nlocal);
        dague_debug_verbose(20, dague_debug_output, " Number of Tasks of Class %s that have been released: %d\n", handle->functions_array[fi]->name, nreleased);
    }
}

void dague_debug_print_local_expecting_tasks( int show_remote, int show_startup, int show_complete )
{
    dague_handle_t *handle;
    uint32_t oi;

    dague_atomic_lock( &object_array_lock );
    for( oi = 1; oi <= object_array_pos; oi++) {
        handle = object_array[ oi ];
        if( handle == NOOBJECT )
            continue;
        if( handle == NULL )
            continue;
        dague_debug_verbose(20, dague_debug_output, "Tasks of Handle %u:\n", oi);
        dague_debug_print_local_expecting_tasks_for_handle( handle,
                                                            show_remote,
                                                            show_startup,
                                                            show_complete );
    }
    dague_atomic_unlock( &object_array_lock );
}

/** deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int dague_task_deps_with_final_output(const dague_execution_context_t *task,
                                      const dep_t **deps)
{
    const dague_function_t *f;
    const dague_flow_t  *flow;
    const dep_t          *dep;
    int fi, di, nbout = 0;

    f = task->function;
    for(fi = 0; fi < f->nb_flows && f->out[fi] != NULL; fi++) {
        flow = f->out[fi];
        if( ! (SYM_OUT & flow->sym_type ) )
            continue;
        for(di = 0; di < MAX_DEP_OUT_COUNT && flow->dep_out[di] != NULL; di++) {
            dep = flow->dep_out[di];
            if( dep->function_id != (uint8_t)-1 )
                continue;
            if( NULL != dep->cond ) {
                assert( EXPR_OP_INLINE == dep->cond->op );
                if( dep->cond->inline_func32(task->dague_handle, task->locals) )
                    continue;
            }
            deps[nbout] = dep;
            nbout++;
        }
    }

    return nbout;
}

/* Function to push back tasks in their mempool once the execution are done */
dague_hook_return_t
dague_release_task_to_mempool(dague_execution_unit_t *eu,
                              dague_execution_context_t *this_task)
{
    (void)eu;
    dague_thread_mempool_free( this_task->super.mempool_owner, this_task );
    return DAGUE_HOOK_RETURN_DONE;
}


/**
 * Special function for generating the dynamic startup functions. The startup
 * code will be of the type of this function, allowing the PaRSEC infrastructure
 * to work. Instead of generating a list of ready tasks during the enqueue and
 * then struggle to spread these tasks over the available resources, we can
 * simply generate a single task (of this type), and allow the runtime to
 * schedule it at its convenience. During the execution of this special task we
 * will generate all the startup tasks, in a manner similar to what we did
 * before. However, we now have the opportunity to build a stateful reentrant
 * function, that can generate only a certain number of tasks and then
 * re-enqueue itself for later execution.
 */

static inline int
priority_of_generic_startup_as_expr_fct(const dague_handle_t * __dague_handle,
                                        const assignment_t * locals)
{
    (void)__dague_handle;
    (void)locals;
    return INT_MIN;
}

static const expr_t priority_of_generic_startup_as_expr = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = (expr_op_int32_inline_func_t)priority_of_generic_startup_as_expr_fct}
};

static inline uint64_t
__dague_generic_startup_hash(const dague_handle_t * __dague_handle,
                             const assignment_t * assignments)
{
    (void)__dague_handle;
    (void)assignments;
    return 0ULL;
}

/**
 * This function is to be used instead of NULL in the generic startup
 * function_t.
 */
static int dague_empty_function_without_arguments(void)
{
    return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dague_generic_startup_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = (dague_hook_t *) dague_empty_function_without_arguments},  /* To be replaced at runtime with the correct point to the startup tasks */
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = (dague_hook_t *) NULL},	/* End marker */
};

const dague_function_t __dague_generic_startup = {
    .name = "Generic Startup",
    .function_id = -1,  /* To be replaced in all copies */
    .nb_flows = 0,
    .nb_parameters = 0,
    .nb_locals = 0,
    .params = {NULL},
    .locals = {NULL},
    .data_affinity = (dague_data_ref_fn_t *) NULL,
    .initial_data = (dague_data_ref_fn_t *) NULL,
    .final_data = (dague_data_ref_fn_t *) NULL,
    .priority = &priority_of_generic_startup_as_expr,
    .in = {NULL},
    .out = {NULL},
    .flags = DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x0,
    .key = (dague_functionkey_fn_t *) __dague_generic_startup_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __dague_generic_startup_chores,
    .iterate_successors = (dague_traverse_function_t *) NULL,
    .iterate_predecessors = (dague_traverse_function_t *) NULL,
    .release_deps = (dague_release_deps_t *) NULL,
    .prepare_input = (dague_hook_t *) dague_empty_function_without_arguments,
    .prepare_output = (dague_hook_t *) NULL,
    .complete_execution = (dague_hook_t *)NULL,
    .release_task = (dague_hook_t *) dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
    .sim_cost_fct = (dague_sim_cost_fct_t *) NULL,
#endif
};
