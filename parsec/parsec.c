/**
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#if defined(PARSEC_HAVE_GEN_H)
#include <libgen.h>
#endif  /* defined(PARSEC_HAVE_GEN_H) */
#if defined(PARSEC_HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(PARSEC_HAVE_GETOPT_H) */
#include "parsec/ayudame.h"

#include "parsec/mca/pins/pins.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/output.h"
#include "parsec/utils/show_help.h"
#include "parsec/data_internal.h"
#include "parsec/class/list.h"
#include "parsec/scheduling.h"
#include "parsec/class/barrier.h"
#include "parsec/remote_dep.h"
#include "parsec/datarepo.h"
#include "parsec/bindthread.h"
#include "parsec/parsec_prof_grapher.h"
#include "parsec/vpmap.h"
#include "parsec/class/info.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/installdirs.h"
#include "parsec/utils/cmd_line.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/parsec_environ.h"
#include "parsec/utils/mca_param_cmd_line.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/interfaces/interface.h"
#include "parsec/sys/tls.h"
#include "parsec/data_distribution.h"
#include "parsec/papi_sde.h"

#include "parsec/mca/mca_repository.h"

#ifdef PARSEC_PROF_TRACE
#include "parsec/profiling.h"
#endif

#include "parsec/parsec_hwloc.h"
#ifdef PARSEC_HAVE_HWLOC
#include "parsec/hbbuffer.h"
#endif

#ifdef PARSEC_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

/*
 * Global variables.
 */

parsec_external_fini_t *external_fini_cbs = NULL;
int                     n_external_fini_cbs = 0;

char parsec_hostname_array[128] = "not yet initialized";
const char* parsec_hostname = parsec_hostname_array;

size_t parsec_task_startup_iter = 64;
size_t parsec_task_startup_chunk = 256;

parsec_data_allocate_t parsec_data_allocate = malloc;
parsec_data_free_t     parsec_data_free = free;
void (*parsec_weaksym_exit)(int status) = _Exit;

#if defined(PARSEC_PROF_TRACE)
#if defined(PARSEC_PROF_TRACE_SCHEDULING_EVENTS)
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
int queue_remove_begin, queue_remove_end;
#endif  /* defined(PARSEC_PROF_TRACE_SCHEDULING_EVENTS) */
int device_delegate_begin, device_delegate_end;
int arena_memory_alloc_key, arena_memory_free_key;
int arena_memory_used_key, arena_memory_unused_key;
int task_memory_alloc_key, task_memory_free_key;
#endif  /* PARSEC_PROF_TRACE */

parsec_info_t parsec_per_device_infos;
parsec_info_t parsec_per_stream_infos;

static int slow_bind_warning = 1;

int parsec_want_rusage = 0;
#if defined(PARSEC_HAVE_GETRUSAGE) && !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>

static struct rusage _parsec_rusage;

static void parsec_rusage(bool print)
{
    struct rusage current;
    getrusage(RUSAGE_SELF, &current);
    if( print ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - _parsec_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - _parsec_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - _parsec_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - _parsec_rusage.ru_stime.tv_usec) / 1000000.0);

        parsec_inform("==== Resource Usage Data...\n"
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
                     "Maximum Resident set size   : %10ld\n"
                     "-------------------------------------------------------------\n",
                     usr, sys, usr + sys,
                     current.ru_minflt  - _parsec_rusage.ru_minflt, current.ru_majflt  - _parsec_rusage.ru_majflt,
                     current.ru_nswap   - _parsec_rusage.ru_nswap,
                     current.ru_nvcsw   - _parsec_rusage.ru_nvcsw, current.ru_nivcsw  - _parsec_rusage.ru_nivcsw,
                     current.ru_inblock - _parsec_rusage.ru_inblock, current.ru_oublock - _parsec_rusage.ru_oublock,
                     current.ru_maxrss);
    }
    _parsec_rusage = current;
}
#define parsec_rusage(b) do { if(parsec_want_rusage > 0) parsec_rusage(b); } while(0)
#else
#define parsec_rusage(b) do {} while(0)
#endif /* defined(PARSEC_HAVE_GETRUSAGE) */

static char *parsec_enable_dot = NULL;
static char *parsec_app_name = NULL;

static int parsec_runtime_max_number_of_cores = -1;
static int parsec_runtime_bind_main_thread = 1;
static int parsec_runtime_bind_threads     = 1;

PARSEC_TLS_DECLARE(parsec_tls_execution_stream);

#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
static void parsec_mpi_exit(int status) {
    MPI_Abort(MPI_COMM_WORLD, status);
}
#endif

/* Create the basic object of class parsec_taskpool_t that inherits parsec_list_t
 * class
 */
static void __parsec_taskpool_constructor(parsec_taskpool_t* tp)
{
    tp->taskpool_id = -1;
    tp->taskpool_name = NULL;
    tp->nb_tasks = 0;
    tp->taskpool_type = 0;
    tp->devices_index_mask = 0;  /* no support for any device. Requires initialization */
    tp->nb_task_classes = 0;
    tp->priority = 0;
    tp->nb_pending_actions = 0;
    tp->context = NULL;  /* not atached to any context */
    tp->startup_hook = NULL;
    tp->task_classes_array = NULL;
    tp->on_enqueue = NULL;
    tp->on_enqueue_data = NULL;
    tp->on_complete = NULL;
    tp->on_complete_data = NULL;
    tp->update_nb_runtime_task = NULL;
    tp->dependencies_array = NULL;
    tp->repo_array = NULL;
}

static void __parsec_taskpool_destructor(parsec_taskpool_t* tp)
{
    if( NULL != tp->taskpool_name ) {
        free(tp->taskpool_name);
    }
}

/* To create object of class parsec_taskpool_t that inherits parsec_list_t
 * class
 */
PARSEC_OBJ_CLASS_INSTANCE(parsec_taskpool_t, parsec_list_item_t,
                          __parsec_taskpool_constructor, __parsec_taskpool_destructor);

/*
 * Taskpool based task definition (no specialized constructor and destructor) */
PARSEC_OBJ_CLASS_INSTANCE(parsec_task_t, parsec_list_item_t,
                   NULL, NULL);

static void parsec_taskpool_release_resources(void);

typedef struct __parsec_temporary_thread_initialization_t {
    parsec_vp_t *virtual_process;
    int th_id;
    int bindto;
    int bindto_ht;
    parsec_barrier_t*  barrier;       /*< the barrier used to synchronize for the
                                       *   local VP data construction. */
} __parsec_temporary_thread_initialization_t;

static int parsec_parse_binding_parameter(const char* option, parsec_context_t* context,
                                         __parsec_temporary_thread_initialization_t* startup);
static int parsec_parse_comm_binding_parameter(const char* option, parsec_context_t* context);

static void* __parsec_thread_init( __parsec_temporary_thread_initialization_t* startup )
{
    parsec_execution_stream_t* es;
    struct timeval tv_now;
    int pi;

    /* don't use PARSEC_THREAD_IS_MASTER, it is too early and we cannot yet allocate the es struct */
    if( parsec_runtime_bind_threads &&
        ((0 != startup->virtual_process->vp_id) || (0 != startup->th_id) || parsec_runtime_bind_main_thread) ) {
        /* Bind to the specified CORE */
        parsec_bindthread(startup->bindto, startup->bindto_ht);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Bind thread %i.%i on core %i [HT %i]",
                            startup->virtual_process->vp_id, startup->th_id,
                            startup->bindto, startup->bindto_ht);
    } else {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Binding disabled for thread %i.%i",
                            startup->virtual_process->vp_id, startup->th_id);
    }

    PARSEC_PAPI_SDE_THREAD_INIT();

    es = (parsec_execution_stream_t*)malloc(sizeof(parsec_execution_stream_t));
    if( NULL == es ) {
        return NULL;
    }
    gettimeofday(&tv_now, NULL);

    PARSEC_TLS_SET_SPECIFIC(parsec_tls_execution_stream, es);

    es->th_id            = startup->th_id;
    es->virtual_process  = startup->virtual_process;
    es->rand_seed        = tv_now.tv_usec + startup->th_id;
    es->scheduler_object = NULL;
    startup->virtual_process->execution_streams[startup->th_id] = es;
    es->core_id          = startup->bindto;
#if defined(PARSEC_HAVE_HWLOC)
    es->socket_id        = parsec_hwloc_socket_id(startup->bindto);
#else
    es->socket_id        = 0;
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    /*
     * A single thread per VP has a little bit more responsability: allocating
     * the memory pools.
     */
    if( startup->th_id == (startup->virtual_process->nb_cores - 1) ) {
        parsec_vp_t *vp = startup->virtual_process;

        parsec_mempool_construct( &vp->context_mempool,
                                  PARSEC_OBJ_CLASS(parsec_task_t), sizeof(parsec_task_t),
                                  offsetof(parsec_task_t, mempool_owner),
                                  vp->nb_cores );

        for(pi = 0; pi <= MAX_PARAM_COUNT; pi++) {
            parsec_mempool_construct( &vp->datarepo_mempools[pi],
                                      NULL, sizeof(data_repo_entry_t)+(pi-1)*sizeof(parsec_arena_chunk_t*),
                                      offsetof(data_repo_entry_t, data_repo_mempool_owner),
                                      vp->nb_cores);
        }
        parsec_mempool_construct( &vp->dependencies_mempool,
                                  NULL, sizeof(parsec_hashable_dependency_t),
                                  offsetof(parsec_hashable_dependency_t, mempool_owner),
                                  vp->nb_cores);
    }
    /* Synchronize with the other threads */
    parsec_barrier_wait(startup->barrier);

    if( NULL != parsec_current_scheduler->module.flow_init )
        parsec_current_scheduler->module.flow_init(es, startup->barrier);

    es->context_mempool = &(es->virtual_process->context_mempool.thread_mempools[es->th_id]);
    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++) {
        es->datarepo_mempools[pi] = &(es->virtual_process->datarepo_mempools[pi].thread_mempools[es->th_id]);
    }
    es->dependencies_mempool = &(es->virtual_process->dependencies_mempool.thread_mempools[es->th_id]);

#ifdef PARSEC_PROF_TRACE
    {
        char *binding = parsec_hwloc_get_binding();
        es->es_profile = parsec_profiling_stream_init( 2*1024*1024,
                                                       PARSEC_PROFILE_THREAD_STR,
                                                       es->th_id,
                                                       es->virtual_process->vp_id,
                                                       NULL == binding ? "(No Binding Information)" : binding);
        parsec_profiling_set_default_thread( es->es_profile );
        if(NULL != binding) free(binding);
    }
    if( NULL != es->es_profile ) {
        PROFILING_STREAM_SAVE_iINFO(es->es_profile, "boundto", startup->bindto);
        PROFILING_STREAM_SAVE_iINFO(es->es_profile, "th_id", es->th_id);
        PROFILING_STREAM_SAVE_iINFO(es->es_profile, "vp_id", es->virtual_process->vp_id );
    }
#endif /* PARSEC_PROF_TRACE */

    PARSEC_PINS_THREAD_INIT(es);

#if defined(PARSEC_SIM)
    es->largest_simulation_date = 0;
#endif

    /* The main thread of VP 0 will go back to the user level */
    if( PARSEC_THREAD_IS_MASTER(es) ) {
        return NULL;
    }

    void *ret = (void*)(long)__parsec_context_wait(es);
    PARSEC_PAPI_SDE_THREAD_FINI();
    return ret;
}

static void parsec_vp_init( parsec_vp_t *vp,
                            int32_t vp_cores,
                           __parsec_temporary_thread_initialization_t *startup)
{
    int t, pi;
    parsec_barrier_t*  barrier;

    assert(vp_cores > 0);
    vp->nb_cores = vp_cores;

    barrier = (parsec_barrier_t*)malloc(sizeof(parsec_barrier_t));
    parsec_barrier_init(barrier, NULL, vp->nb_cores);

    /* Prepare the temporary storage for each thread startup */
    for( t = 0; t < vp->nb_cores; t++ ) {
        startup[t].th_id = t;
        startup[t].virtual_process = vp;
        startup[t].bindto = -1;
        startup[t].bindto_ht = -1;
        startup[t].barrier = barrier;
        pi = vpmap_get_nb_cores_affinity(vp->vp_id, t);
        if( 1 == pi )
            vpmap_get_core_affinity(vp->vp_id, t, &startup[t].bindto, &startup[t].bindto_ht);
        else if( 1 < pi )
            parsec_warning("multiple core to bind on... for now, do nothing"); //TODO: what does that mean?
    }
}

static int check_overlapping_binding(parsec_context_t *context);

#define DEFAULT_APP_NAME "app_name"

#define GET_INT_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = parsec_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        char* __value = parsec_cmd_line_get_param((CMD), (ARGV), 0, 0); \
        if( NULL != __value ) \
            (VALUE) = (int)strtol(__value, NULL, 10); \
    } \
} while (0)

#define GET_STR_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = parsec_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        (VALUE) = parsec_cmd_line_get_param((CMD), (ARGV), 0, 0); \
    } \
} while (0)

parsec_context_t* parsec_init( int nb_cores, int* pargc, char** pargv[] )
{
    int ret, nb_vp, p, t, nb_total_comp_threads, display_vpmap = 0;
    char *comm_binding_parameter = NULL;
    char *binding_parameter = NULL;
    __parsec_temporary_thread_initialization_t *startup;
    parsec_context_t* context;
    parsec_cmd_line_t *cmd_line = NULL;
    char **ctx_environ = NULL;
    char **env_variable, *env_name, *env_value;
    char *parsec_enable_profiling = NULL;  /* profiling file prefix when PARSEC_PROF_TRACE is on */
    int slow_option_used = 0;
#if defined(PARSEC_PROF_TRACE)
    int profiling_id = 0;
#endif
    int profiling_enabled = 0;

    gethostname(parsec_hostname_array, sizeof(parsec_hostname_array));

    PARSEC_PAPI_SDE_INIT();

    parsec_installdirs_open();
    parsec_mca_param_init();
    parsec_output_init();
    parsec_show_help_init();

    /* Extract what we can from the arguments */
    cmd_line = PARSEC_OBJ_NEW(parsec_cmd_line_t);
    if( NULL == cmd_line ) {
        return NULL;
    }

    /* Declare the command line for the .dot generation */
    parsec_cmd_line_make_opt3(cmd_line, 'h', "help", "help", 0,
                             "Show the usage text.");
    parsec_cmd_line_make_opt3(cmd_line, '.', "dot", "parsec_dot", 1,
                             "Filename for the .dot file");
    parsec_cmd_line_make_opt3(cmd_line, 'b', NULL, "parsec_bind", 1,
                             "Execution thread binding");
    parsec_cmd_line_make_opt3(cmd_line, 'C', NULL, "parsec_bind_comm", 1,
                             "Communication thread binding");
    parsec_cmd_line_make_opt3(cmd_line, 'c', "cores", "cores", 1,
                             "Number of cores to used");
    parsec_cmd_line_make_opt3(cmd_line, 'g', "gpus", "gpus", 1,
                             "Number of GPU to used (deprecated use MCA instead)");
    parsec_cmd_line_make_opt3(cmd_line, 'V', "vpmap", "vpmap", 1,
                             "Virtual process map");
    parsec_cmd_line_make_opt3(cmd_line, 'H', "ht", "ht", 1,
                             "Enable hyperthreading");
    parsec_mca_cmd_line_setup(cmd_line);


    if( (NULL != pargc) && (0 != *pargc) ) {
        parsec_app_name = strdup( (*pargv)[0] );

        ret = parsec_cmd_line_parse(cmd_line, true, *pargc, *pargv);
        if (PARSEC_SUCCESS != ret) {
            fprintf(stderr, "%s: command line error (%d)\n", (*pargv)[0], ret);
        }
    } else {
        parsec_app_name = strdup( DEFAULT_APP_NAME );
    }

    ret = parsec_mca_cmd_line_process_args(cmd_line, &ctx_environ, &environ);
    if( ctx_environ != NULL ) {
        for(env_variable = ctx_environ;
            *env_variable != NULL;
            env_variable++) {
            env_name = *env_variable;
            for(env_value = env_name; *env_value != '\0' && *env_value != '='; env_value++)
                /* nothing */;
            if(*env_value == '=') {
                *env_value = '\0';
                env_value++;
            }
            parsec_setenv(env_name, env_value, true, &environ);
            free(*env_variable);
        }
        free(ctx_environ);
    }

#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
    int mpi_is_up;
    MPI_Initialized(&mpi_is_up);
    if( mpi_is_up ) {
        MPI_Comm_rank(MPI_COMM_WORLD, &parsec_debug_rank);
#if defined(PARSEC_PROF_TRACE)
        profiling_id = parsec_debug_rank;
#endif
        parsec_weaksym_exit = parsec_mpi_exit;
    }
#endif
    parsec_debug_init();
    mca_components_repository_init();

    parsec_mca_param_reg_int_name("runtime", "warn_slow_binding", "Disable warnings about the runtime detecting poorly performing binding configuration", false, false, slow_bind_warning, &slow_bind_warning);

#if defined(PARSEC_HAVE_HWLOC)
    parsec_hwloc_init();
#endif  /* defined(HWLOC) */

    if( parsec_cmd_line_is_taken(cmd_line, "ht") ) {
#if defined(PARSEC_HAVE_HWLOC)
        int hyperth = 0;
        GET_INT_ARGV(cmd_line, "ht", hyperth);
        parsec_hwloc_allow_ht(hyperth);
#else
        if( 0 == parsec_debug_rank )
            parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                           "Option ht (hyper-threading) is only supported when HWLOC is enabled at compile time.\n");
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    }

    /* Set a default the number of cores if not defined by parameters
     * - with hwloc if available
     * - with sysconf otherwise (hyperthreaded core number)
     */
    parsec_mca_param_reg_int_name("runtime", "num_cores", "The total number of cores to be used by the runtime (-1 for all available)",
                                 false, false, parsec_runtime_max_number_of_cores, &parsec_runtime_max_number_of_cores);
    if( nb_cores <= 0 ) {
        if( -1 == parsec_runtime_max_number_of_cores )
            nb_cores = parsec_hwloc_nb_real_cores();
        else {
            nb_cores = parsec_runtime_max_number_of_cores;
            if( parsec_runtime_max_number_of_cores > parsec_hwloc_nb_real_cores() ) {
                if( slow_bind_warning )
                    parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                                   "Requested binding %d threads, which is more than the physical number of cores %d.\n"
                                   "\tOversubscribing cores is often slow. You should change the value of the `runtime_num_cores` parameter.\n",
                                   parsec_runtime_max_number_of_cores, parsec_hwloc_nb_real_cores());
            }
        }
    }
    parsec_mca_param_reg_int_name("runtime", "bind_main_thread", "Force the binding of the thread calling parsec_init",
                                 false, false, parsec_runtime_bind_main_thread, &parsec_runtime_bind_main_thread);

    parsec_mca_param_reg_int_name("runtime", "bind_threads", "Bind main and worker threads", false, false,
                                  parsec_runtime_bind_threads, &parsec_runtime_bind_threads);

    if( parsec_cmd_line_is_taken(cmd_line, "gpus") ) {
        parsec_warning("Option g (for accelerators) is deprecated as an argument. Use the MCA parameter instead.");
    }

    /* Allow the parsec_init arguments to overwrite all the previous behaviors */
    GET_INT_ARGV(cmd_line, "cores", nb_cores);
    GET_STR_ARGV(cmd_line, "parsec_bind_comm", comm_binding_parameter);
    GET_STR_ARGV(cmd_line, "parsec_bind", binding_parameter);

    /*
     * Initialize the VPMAP, the discrete domains hosting
     * execution flows but where work stealing is prevented.
     */
    {
        /* Change the vpmap choice: first cancel the previous one if any */
        vpmap_fini();

        if( parsec_cmd_line_is_taken(cmd_line, "vpmap") ) {
            char* optarg = NULL;
            GET_STR_ARGV(cmd_line, "vpmap", optarg);
            if(NULL == optarg) {
                parsec_warning("VPMAP choice (-V argument): expected argument. Falling back to default!");
            } else {
                /* We accept a vpmap that starts with "display" as a mean to show the mapping */
                if( !strncmp(optarg, "display", 7 )) {
                    display_vpmap = 1;
                    if( ':' != optarg[strlen("display")] ) {
                        parsec_warning("Display thread mapping requested but vpmap argument incorrect "
                                       "(must start with display: to print the mapping)");
                    } else {
                        optarg += strlen("display:");
                    }
                }
                if( !strncmp(optarg, "flat", 4) ) {
                    /* default case (handled in parsec_init) */
                } else if( !strncmp(optarg, "hwloc", 5) ) {
                    vpmap_init_from_hardware_affinity(nb_cores);
                } else if( !strncmp(optarg, "file:", 5) ) {
                    vpmap_init_from_file(optarg + 5);
                } else if( !strncmp(optarg, "rr:", 3) ) {
                    int n, p, co;
                    if( sscanf(optarg, "rr:%d:%d:%d", &n, &p, &co) == 3 ) {
                        vpmap_init_from_parameters(n, p, co);
                    } else {
                        parsec_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
                    }
                } else {
                    parsec_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
                }
            }
        }
        nb_vp = vpmap_get_nb_vp();
        if( -1 == nb_vp ) {
            vpmap_init_from_flat(nb_cores);
            nb_vp = vpmap_get_nb_vp();
        }
    }

    parsec_hash_tables_init();

#if defined(PARSEC_PROF_GRAPHER)
    char *parsec_mca_enable_dot = parsec_enable_dot;
    parsec_mca_param_reg_string_name("parsec", "dot", "Create a DOT file from the DAGs executed by parsec (one file per rank)",
                                     false, false, parsec_mca_enable_dot, &parsec_mca_enable_dot);
    if( NULL != parsec_mca_enable_dot ) {
        asprintf(&parsec_enable_dot, "%s-%d.dot", parsec_mca_enable_dot, parsec_debug_rank);
    }
    if( parsec_cmd_line_is_taken(cmd_line, "dot") ) {
        // command-line has priority over MCA parameter
        char* optarg = NULL;
        GET_STR_ARGV(cmd_line, "dot", optarg);

        if( parsec_enable_dot ) free( parsec_enable_dot );
        if( NULL == optarg ) {
            asprintf(&parsec_enable_dot, "%s-%d.dot", parsec_app_name, parsec_debug_rank);
        } else {
            asprintf(&parsec_enable_dot, "%s-%d.dot", optarg, parsec_debug_rank);
        }
    }
#endif

    /* the extra allocation will pertain to the virtual_processes array */
    context = (parsec_context_t*)malloc(sizeof(parsec_context_t) + (nb_vp-1) * sizeof(parsec_vp_t*));

    context->__parsec_internal_finalization_in_progress = 0;
    context->__parsec_internal_finalization_counter = 0;
    context->active_taskpools      = 0;
    context->flags               = 0;
    context->nb_nodes            = 1;
    context->comm_ctx            = -1;
    context->my_rank             = 0;
    context->nb_vp               = nb_vp;
    /* initialize dtd taskpool list */
    context->taskpool_list       = NULL;
    parsec_hash_table_init(&context->dtd_arena_datatypes_hash_table, offsetof(parsec_arena_datatype_t, ht_item),
                           8, parsec_hash_table_generic_key_fn, NULL);
    context->dtd_arena_datatypes_next_id = 0;
#if defined(PARSEC_SIM)
    context->largest_simulation_date = 0;
#endif /* PARSEC_SIM */
#if defined(PARSEC_HAVE_HWLOC)
    context->cpuset_allowed_mask = NULL;
    context->cpuset_free_mask    = NULL;
    context->comm_th_core        = -1;
#endif  /* defined(PARSEC_HAVE_HWLOC) */

    /* TODO: nb_cores should depend on the vp_id */
    nb_total_comp_threads = 0;
    for(p = 0; p < nb_vp; p++) {
        nb_total_comp_threads += vpmap_get_nb_threads_in_vp(p);
    }

    if( nb_cores != nb_total_comp_threads ) {
        if( slow_bind_warning )
            parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                           "Your vpmap uses %d threads when %d cores where available\n",
                           nb_total_comp_threads, nb_cores);
        nb_cores = nb_total_comp_threads;
    }

    startup = (__parsec_temporary_thread_initialization_t*)
        malloc(nb_total_comp_threads * sizeof(__parsec_temporary_thread_initialization_t));

    t = 0;
    for( p = 0; p < nb_vp; p++ ) {
        parsec_vp_t *vp;
        vp = (parsec_vp_t *)malloc(sizeof(parsec_vp_t) + (vpmap_get_nb_threads_in_vp(p)-1) * sizeof(parsec_execution_stream_t*));
        vp->parsec_context = context;
        vp->vp_id = p;
        context->virtual_processes[p] = vp;
        /*
         * Set the threads local variables from startup[t] -> startup[t+nb_cores].
         * Do not create or initialize any memory yet, or it will be automatically
         * bound to the allocation context of this thread.
         */
        parsec_vp_init(vp, vpmap_get_nb_threads_in_vp(p), &(startup[t]));
        t += vp->nb_cores;
    }

    /*
     * Parameters defining the default ARENA behavior. Handle with care they can lead to
     * significant memory consumption or to a significant overhead in memory management
     * (allocation/deallocation). These values are only used by ARENAs constructed with
     * the default constructor (not the extended one).
     */
    parsec_mca_param_reg_sizet_name("arena", "max_used", "The maximum amount of memory each arena can"
                                   " allocate (default unlimited)",
                                   false, false, parsec_arena_max_allocated_memory, &parsec_arena_max_allocated_memory);
    parsec_mca_param_reg_sizet_name("arena", "max_cached", "The maxmimum amount of memory each arena can"
                                   " cache in a freelist (0=no caching)",
                                   false, false, parsec_arena_max_cached_memory, &parsec_arena_max_cached_memory);

    parsec_mca_param_reg_sizet_name("task", "startup_iter", "The number of ready tasks to be generated during the startup "
                                   "before allowing the scheduler to distribute them across the entire execution context.",
                                   false, false, parsec_task_startup_iter, &parsec_task_startup_iter);
    parsec_mca_param_reg_sizet_name("task", "startup_chunk", "The total number of tasks to be generated during the startup "
                                   "before delaying the remaining of the startup. The startup process will be "
                                   "continued at a later moment once the number of ready tasks decreases.",
                                   false, false, parsec_task_startup_chunk, &parsec_task_startup_chunk);

    parsec_mca_param_reg_string_name("profile", "filename",
#if defined(PARSEC_PROF_TRACE)
                                    "Path to the profiling file (<none> to disable, <app> for app name, <*> otherwise)",
                                    false, false,
#else
                                    "Path to the profiling file (unused due to profiling being turned off during building)",
                                    false, true,  /* profiling disabled: read-only */
#endif  /* defined(PARSEC_PROF_TRACE) */
                                    "<none>", &parsec_enable_profiling);
#if defined(PARSEC_PROF_TRACE)
    if( (0 != strncasecmp(parsec_enable_profiling, "<none>", 6)) && (0 == parsec_profiling_init( profiling_id )) ) {
        int i, l;
        char *cmdline_info = NULL;

        /* Use either the app name (argv[0]) or the user provided filename */
        if( 0 == strncmp(parsec_enable_profiling, "<app>", 5) ) {
            /* Specialize the profiling filename to avoid collision with other instances */
            ret = asprintf( &cmdline_info, "%s_%d", basename(parsec_app_name), (int)getpid() );
            if (ret < 0) {
                cmdline_info = strdup(DEFAULT_APP_NAME);
            }
            ret = parsec_profiling_dbp_start( cmdline_info, parsec_app_name );
            free(cmdline_info);
        } else {
            ret = parsec_profiling_dbp_start( parsec_enable_profiling, parsec_app_name );
        }
        if( ret != 0 ) {
            parsec_warning("Profiling framework deactivated because of error %s.", parsec_profiling_strerror());
        } else {
            profiling_enabled = 1;
        }

        l = strlen(parsec_app_name);  /* use the known application name */
        if( NULL != pargc ) {
            for(i = 1; i < *pargc; i++) {
                l += strlen( (*pargv)[i] ) + 1;
            }
        }
        cmdline_info = (char*)malloc(l + 1);
        sprintf(cmdline_info, "%s", parsec_app_name);
        l = strlen(parsec_app_name);
        if( NULL != pargc ) {
            for(i = 1; i < *pargc; i++) {
                sprintf(cmdline_info + l, " %s", (*pargv)[i]);
                l += strlen( (*pargv)[i] ) + 1;
            }
        }
        cmdline_info[l] = '\0';
        parsec_profiling_add_information("CMDLINE", cmdline_info);

        /* we should be adding the PaRSEC options to the profile here
         * instead of in common.c/h as we do now. */
        PROFILING_SAVE_iINFO("nb_cores", nb_cores);
        PROFILING_SAVE_iINFO("nb_vps", nb_vp);
        PROFILING_SAVE_sINFO("GIT_BRANCH", PARSEC_GIT_BRANCH);
        PROFILING_SAVE_sINFO("GIT_HASH", PARSEC_GIT_HASH);
        free(cmdline_info);

#  if defined(PARSEC_PROF_TRACE_SCHEDULING_EVENTS)
        parsec_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                                0, NULL,
                                                &MEMALLOC_start_key, &MEMALLOC_end_key);
        parsec_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                                0, NULL,
                                                &schedule_poll_begin, &schedule_poll_end);
        parsec_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                                0, NULL,
                                                &schedule_push_begin, &schedule_push_end);
        parsec_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                                0, NULL,
                                                &schedule_sleep_begin, &schedule_sleep_end);
        parsec_profiling_add_dictionary_keyword( "Queue REMOVE", "fill:#B9B243",
                                                0, NULL,
                                                &queue_remove_begin, &queue_remove_end);
#  endif /* PARSEC_PROF_TRACE_SCHEDULING_EVENTS */
#if defined(PARSEC_PROF_TRACE_ACTIVE_ARENA_SET)
        parsec_profiling_add_dictionary_keyword( "ARENA_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_alloc_key, &arena_memory_free_key);
        parsec_profiling_add_dictionary_keyword( "ARENA_ACTIVE_SET", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_used_key, &arena_memory_unused_key);
#endif  /* defined(PARSEC_PROF_TRACE_ACTIVE_ARENA_SET) */
        parsec_profiling_add_dictionary_keyword( "TASK_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &task_memory_alloc_key, &task_memory_free_key);
        parsec_profiling_add_dictionary_keyword( "Device delegate", "fill:#EAE7C6",
                                                0, NULL,
                                                &device_delegate_begin, &device_delegate_end);
    }
#endif  /* PARSEC_PROF_TRACE */
    assert (NULL != parsec_enable_profiling);
    free(parsec_enable_profiling);

    /* Extract the expected thread placement */
    if( NULL != comm_binding_parameter )
        parsec_parse_comm_binding_parameter(comm_binding_parameter, context);
    parsec_parse_binding_parameter(binding_parameter, context, startup);

    /* Introduce communication engine */
    (void)parsec_remote_dep_init(context);

    (void)check_overlapping_binding(context);

    PARSEC_PINS_INIT(context);
    if(profiling_enabled && (0 == parsec_pins_nb_modules_enabled())) {
        if(parsec_debug_rank == 0)
            parsec_warning("*** PaRSEC Profiling warning: creating profile file as requested,\n"
                           "*** but no PINS module is enabled, so the file will probably be empty\n"
                           "*** Activate the MCA PINS Module task_profiler to get the previous behavior\n"
                           "***   ( --mca mca_pins task_profiler )\n");
    }

#if defined(PARSEC_PROF_GRAPHER)
    if(parsec_enable_dot) {
        parsec_prof_grapher_init(context, parsec_enable_dot);
        slow_option_used = 1;
    }
#endif  /* defined(PARSEC_PROF_GRAPHER) */

#if defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)
    slow_option_used = 1;
#endif
    if( slow_option_used && 0 == parsec_debug_rank ) {
        parsec_warning("/!\\ DEBUG LEVEL WILL PROBABLY REDUCE THE PERFORMANCE OF THIS RUN /!\\.\n");
        parsec_debug_verbose(4, parsec_debug_output, "--- compiled with DEBUG_NOISIER, DEBUG_PARANOID, or DOT generation requested.");
    }

    if(0 == parsec_debug_rank && parsec_debug_verbose >= 3) {
        char version_info[4096];
        parsec_version_ex(4096, version_info);
        parsec_inform("== PaRSEC Runtime Compilation Configurations ===============================");
        parsec_output(parsec_debug_output, "%s", version_info);
        parsec_inform("============================================================================");
    }

    parsec_mca_device_init();
    /* Init data distribution structure */
    parsec_data_dist_init();

    parsec_mca_device_attach(context);
    parsec_mca_device_registration_complete(context);

    /* Init the data infrastructure. Must be done only after the freeze of the devices */
    parsec_data_init(context);

    /* Initialize the barriers */
    parsec_barrier_init( &(context->barrier), NULL, nb_total_comp_threads );

    /* Load the default scheduler. User can change it afterward,
     * but we need to ensure that one is loadable and here.
     */
    if( PARSEC_SUCCESS > parsec_set_scheduler( context ) ) {
        /* TODO: handle memory leak / thread leak here: this is a fatal
         * error for PaRSEC */
        parsec_fatal("Unable to load any scheduler in init function.");
        return NULL;
    }

    PARSEC_TLS_KEY_CREATE(parsec_tls_execution_stream);

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
                            (void* (*)(void*))__parsec_thread_init,
                            (void*)&(startup[t]));
        }
    } else {
        context->pthreads = NULL;
    }

    __parsec_thread_init( &startup[0] );

    /* Wait until all threads are done binding themselves */
    parsec_barrier_wait( &(context->barrier) );
    context->__parsec_internal_finalization_counter++;

    /* Release the temporary array used for starting up the threads */
    {
        parsec_barrier_t* barrier = startup[0].barrier;
        parsec_barrier_destroy(barrier);
        free(barrier);
        for(t = 0; t < nb_total_comp_threads; t++) {
            if(barrier != startup[t].barrier) {
                barrier = startup[t].barrier;
                parsec_barrier_destroy(barrier);
                free(barrier);
            }
        }
    }
    free(startup);

    if( display_vpmap )
        vpmap_display_map();

    parsec_mca_param_reg_int_name("profile", "rusage", "Report 'getrusage' satistics.\n"
            "0: no report, 1: per process report, 2: per thread report (if available).\n",
            false, false, parsec_want_rusage, &parsec_want_rusage);
    parsec_rusage(false);

    PARSEC_AYU_INIT();

    if( parsec_cmd_line_is_taken(cmd_line, "help") ||
        parsec_cmd_line_is_taken(cmd_line, "h")) {
        if( 0 == context->my_rank ) {
            char* help_msg = parsec_cmd_line_get_usage_msg(cmd_line);
            parsec_list_t* l = NULL;

            parsec_output(0, "%s\n\nRegistered MCA parameters", help_msg);
            free(help_msg);

            parsec_mca_param_dump(&l, 1);
            parsec_mca_show_mca_params(l, "all", "all", 1);
            parsec_mca_param_dump_release(l);
        }
        parsec_fini(&context);
        return NULL;
    }

    if( NULL != cmd_line )
        PARSEC_OBJ_RELEASE(cmd_line);

    return context;
}

int parsec_version( int* version_major, int* version_minor, int* version_release) {
    *version_major = PARSEC_VERSION_MAJOR;
    *version_minor = PARSEC_VERSION_MINOR;
    *version_release = PARSEC_VERSION_RELEASE;
    return PARSEC_SUCCESS;
}

int parsec_version_ex( size_t len, char* version_string) {
    int ret;
    char *sched_components = mca_components_list_compiled("sched");
    char *device_components = mca_components_list_compiled("device");
    char *pins_components = mca_components_list_compiled("pins");

    ret = snprintf(version_string, len,
        "version\t\t= %d.%d.%d\n"
        "git_hash\t= %s\n"
        "git_tag\t\t= %s\n"
        "git_dirty\t= %s\n"
        "git_date\t= %s\n"
        "compile_date\t= %s\n"
        "debug\t\t= %s\n"
        "profiling\t= %s\n"
#if defined(PARSEC_PROF_TRACE)
        "pins\t\t= %s\n"
#endif
        "comms\t\t= %s\n"
        "devices\t\t= %s\n"
        "scheds\t\t= %s\n"
        "hwloc\t\t= %s\n"
        "bits\t\t= %s\n"
        "atomics\t\t= %s\n"
        "c_compiler\t= %s\n"
        "c_flags\t\t= %s\n",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,
        PARSEC_VERSION_RELEASE,
        PARSEC_GIT_HASH,
        PARSEC_GIT_BRANCH,
        PARSEC_GIT_DIRTY,
        PARSEC_GIT_DATE,
        PARSEC_COMPILE_DATE,
#if defined(PARSEC_DEBUG)
        "yes"
#if defined(PARSEC_DEBUG_PARANOID)
        "+paranoid"
#endif
#if defined(PARSEC_DEBUG_NOISIER)
        "+noisier"
#endif
#if defined(PARSEC_DEBUG_HISTORY)
        "+history"
#endif
#else
        "no"
#endif /*PARSEC_DEBUG*/
        ,
#if defined(PARSEC_PROF_TRACE)
        "yes"
#if defined(PARSEC_PROF_DRY_RUN)
        "+dryrun"
#endif
#if defined(PARSEC_PROF_DRY_BODY)
        "+drybody"
#endif
#if defined(PARSEC_PROF_DRY_DEP)
        "+drydep"
#endif
#if defined(PARSEC_PROF_GRAPHER)
        "+grapher"
#endif
#if defined(PARSEC_SIM)
        "+sim"
#endif
        ,
        pins_components
#else
        "no"
#endif /*PARSEC_PROF_TRACE*/
        ,
#if defined(PARSEC_HAVE_MPI)
        "mpi"
#if defined(PARSEC_HAVE_MPI_20)
        "2"
#endif
#if defined(PARSEC_DIST_THREAD)
        "+thread_multiple"
#endif
#else  /* defined(PARSEC_HAVE_MPI) */
        "single process only"
#endif
        ,
        device_components,
        sched_components,
#if defined(PARSEC_HAVE_HWLOC)
        "yes"
#else
        "no"
#endif
        ,
#if 8 == PARSEC_SIZEOF_VOID_P
#if 0 == ULONG_MAX>>63
        "llp64"
#elif 0 == UINT_MAX>>63
        "lp64"
#else
        "ilp64"
#endif
#else
        "ilp32"
#endif
        ,
        /* these tests in the same order as in atomic.h */
#if defined(PARSEC_ATOMIC_USE_C11_ATOMICS)
        "c11"
#elif defined(PARSEC_ATOMIC_USE_XLC_32_BUILTINS)
        "xlc_builtins"
#elif defined(PARSEC_ATOMIC_USE_PPC_BGP)
        "ppc_bgp"
#elif defined(PARSEC_ATOMIC_USE_PPC)
        "ppc"
#elif defined(PARSEC_ATOMIC_USE_GCC_32_BUILTINS)
        "gcc_builtins"
#elif defined(PARSEC_ARCH_X86)
        "asm_x86_32"
#elif defined(PARSEC_ARCH_X86_64)
        "asm_x86_64"
#else
#error "No safe atomics available" /*should never happen due to similar check in atomic.h*/
#endif
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
        "+cas128"
#endif
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR)
        "+llsc"
#endif
        ,
        CMAKE_PARSEC_C_COMPILER,
        CMAKE_PARSEC_C_FLAGS
    );
    free(device_components);
    free(sched_components);
    free(pins_components);
    return len > (size_t)ret? PARSEC_SUCCESS: PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
}

void parsec_abort(parsec_context_t* ctx, int status)
{
    /* ATM, MPI_Abort aborts the whole job, in the future it would be nice to
     * abort only the @ctx */
    parsec_weaksym_exit(status);
    (void)ctx;
}

#if defined(PARSEC_PROF_TRACE)
static void parsec_mempool_stats(parsec_context_t *context)
{
    int i, p;
    unsigned int t;
    size_t m_usage;
    char meminfo[128];
    parsec_vp_t *vp;
    parsec_mempool_t *mp;

    m_usage = 0;
    for(p = 0; p < context->nb_vp; p++) {
        vp = context->virtual_processes[p];
        mp = &vp->context_mempool;
        for(t = 0; t < mp->nb_thread_mempools; t++)
            m_usage += mp->thread_mempools[t].nb_elt * mp->elt_size;
    }
    snprintf(meminfo, 128, "MEMPOOL - Contexts - %zu bytes", m_usage);
    parsec_profiling_add_information("MEMORY_USAGE", meminfo);

    m_usage = 0;
    for(p = 0; p < context->nb_vp; p++) {
        vp = context->virtual_processes[p];
        for(i = 0; i <= MAX_PARAM_COUNT; i++) {
            mp = &vp->datarepo_mempools[i];
            for(t = 0; t < mp->nb_thread_mempools; t++)
                m_usage += mp->thread_mempools[t].nb_elt * mp->elt_size;
        }
    }
    snprintf(meminfo, 128, "MEMPOOL - DataRepos - %zu bytes", m_usage);
    parsec_profiling_add_information("MEMORY_USAGE", meminfo);

    m_usage = 0;
    for(p = 0; p < context->nb_vp; p++) {
        vp = context->virtual_processes[p];
        mp = &vp->dependencies_mempool;
        for(t = 0; t < mp->nb_thread_mempools; t++)
            m_usage += mp->thread_mempools[t].nb_elt * mp->elt_size;
    }
    snprintf(meminfo, 128, "MEMPOOL - Dependencies - %zu bytes", m_usage);
    parsec_profiling_add_information("MEMORY_USAGE", meminfo);
}
#endif

static void parsec_vp_fini( parsec_vp_t *vp )
{
    int i;

    parsec_mempool_destruct( &vp->context_mempool );
    parsec_mempool_destruct( &vp->dependencies_mempool );
    for(i = 0; i <= MAX_PARAM_COUNT; i++) {
        parsec_mempool_destruct( &vp->datarepo_mempools[i]);
    }

    for(i = 0; i < vp->nb_cores; i++) {
        free(vp->execution_streams[i]);
        vp->execution_streams[i] = NULL;
    }
}

void parsec_context_at_fini(parsec_external_fini_cb_t cb, void *data)
{
    n_external_fini_cbs++;
    external_fini_cbs = (parsec_external_fini_t *)realloc(
            external_fini_cbs, sizeof(parsec_external_fini_t)*n_external_fini_cbs);
    external_fini_cbs[n_external_fini_cbs-1].cb = cb;
    external_fini_cbs[n_external_fini_cbs-1].data = data;
}

static void parsec_clean_and_warn_dtd_arena_datatypes(void *elt, void *dta)
{
    void **params = (void **)dta;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)params[0];
    parsec_arena_datatype_t *adt = (parsec_arena_datatype_t*)elt;
    int *count = (int*)params[1];
    (*count)++;
    parsec_hash_table_remove(ht, adt->ht_item.key);
}

int parsec_fini( parsec_context_t** pcontext )
{
    parsec_context_t* context = *pcontext;
    int nb_total_comp_threads, p, nb_items;
    void *params[2] = {&context->dtd_arena_datatypes_hash_table, &nb_items};

    /* if dtd environment is set-up, we clean */
    if( __parsec_dtd_is_initialized ) {
        parsec_dtd_fini();
        /* clean dtd taskpool array */
        PARSEC_OBJ_RELEASE(context->taskpool_list);
        context->taskpool_list = NULL;
    }
    nb_items = 0;
    parsec_hash_table_for_all(&context->dtd_arena_datatypes_hash_table, parsec_clean_and_warn_dtd_arena_datatypes,
                              params);
    if(0 != nb_items) {
        parsec_warning("/!\\ Warning: %d DTD arena datatypes are still registered with this parsec context at "
                       "release time\n", nb_items);
    }
    parsec_hash_table_fini(&context->dtd_arena_datatypes_hash_table);

    /**
     * We need to force the main thread to drain all possible pending messages
     * on the communication layer. This is not an issue in a distributed run,
     * but on a single node run with MPI support, taskpools can be created (and
     * thus context_id additions might be pending on the communication layer).
     */
#if defined(DISTRIBUTED)
    if( (1 == parsec_communication_engine_up) &&  /* engine enabled */
        (context->nb_nodes == 1) &&  /* single node: otherwise the messages will
                                      * be drained by the communication thread */
        PARSEC_THREAD_IS_MASTER(context->virtual_processes[0]->execution_streams[0]) ) {
        /* check for remote deps completion */
        parsec_remote_dep_progress(context->virtual_processes[0]->execution_streams[0]);
    }
#endif /* defined(DISTRIBUTED) */

    /* Now wait until every thread is back */
    context->__parsec_internal_finalization_in_progress = 1;
    parsec_barrier_wait( &(context->barrier) );

    /**
     * The registered at_fini callbacks should be called as early as possible in the
     * context finalization, but not before any actions visible to the outside world
     * has been completed (aka. not until the communication engine is up).
     */
    if( NULL != external_fini_cbs ) {
        for(int i = 0; i < n_external_fini_cbs; i++){
            external_fini_cbs[i].cb( external_fini_cbs[i].data ) ;
        }
        free(external_fini_cbs); external_fini_cbs = NULL;
        n_external_fini_cbs = 0;
    }

    parsec_rusage(true);

    PARSEC_PINS_THREAD_FINI(context->virtual_processes[0]->execution_streams[0]);

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

    PARSEC_PINS_FINI(context);

#ifdef PARSEC_PROF_TRACE
    parsec_mempool_stats(context);
#endif  /* PARSEC_PROF_TRACE */

    (void) parsec_remote_dep_fini(context);

    parsec_remove_scheduler( context );

    parsec_data_fini(context);

    parsec_data_dist_fini();

    parsec_mca_device_fini();

    for(p = 0; p < context->nb_vp; p++) {
        parsec_vp_fini(context->virtual_processes[p]);
        free(context->virtual_processes[p]);
        context->virtual_processes[p] = NULL;
    }

    PARSEC_AYU_FINI();
#ifdef PARSEC_PROF_TRACE
    parsec_profiling_dbp_dump();
    (void)parsec_profiling_fini( );  /* we're leaving, ignore errors */
#endif  /* PARSEC_PROF_TRACE */

    if(parsec_enable_dot) {
#if defined(PARSEC_PROF_GRAPHER)
        parsec_prof_grapher_fini();
#endif  /* defined(PARSEC_PROF_GRAPHER) */
        free(parsec_enable_dot);
        parsec_enable_dot = NULL;
    }
    /* Destroy all resources allocated for the barrier */
    parsec_barrier_destroy( &(context->barrier) );

#if defined(PARSEC_HAVE_HWLOC_BITMAP)
    /* Release thread binding masks */
    hwloc_bitmap_free(context->cpuset_allowed_mask);
    hwloc_bitmap_free(context->cpuset_free_mask);

    parsec_hwloc_fini();
#endif  /* PARSEC_HAVE_HWLOC_BITMAP */

    PARSEC_PAPI_SDE_FINI();

    if (parsec_app_name != NULL ) {
        free(parsec_app_name);
        parsec_app_name = NULL;
    }

    parsec_taskpool_release_resources();

    parsec_show_help_finalize();
    parsec_output_finalize();
    parsec_mca_param_finalize();
    parsec_installdirs_close();

    free(context);
    *pcontext = NULL;

    parsec_class_finalize();
    parsec_debug_fini();  /* Always last */
    return PARSEC_SUCCESS;
}

#define rop1          u_expr.range.op1
#define rop2          u_expr.range.op2
#define rcstinc       u_expr.range.increment.cst
#define rexprinc      u_expr.range.increment.expr
#define return_type   u_expr.v_func.type
#define inline_func32 u_expr.v_func.func.inline_func_int32
#define inline_func64 u_expr.v_func.func.inline_func_int64
#define inline_funcfl u_expr.v_func.func.inline_func_float
#define inline_funcdb u_expr.v_func.func.inline_func_double

/*
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static parsec_dependency_t
parsec_check_IN_dependencies_with_mask(const parsec_taskpool_t *tp,
                                       const parsec_task_t* task)
{
    const parsec_task_class_t* tc = task->task_class;
    int i, j, active;
    const parsec_flow_t* flow;
    const parsec_dep_t* dep;
    parsec_dependency_t ret = 0;

    if( !(tc->flags & PARSEC_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != tc->in[i]); i++ ) {
        flow = tc->in[i];

        /*
         * Controls and data have different logic:
         * Flows can depend conditionally on multiple input or control.
         * It is assumed that in the data case, one input will always become true.
         *  So, the Input dependency is already solved if one is found with a true cond,
         *      and depend only on the data.
         *
         * On the other hand, if all conditions for the control are false,
         * it is assumed that no control should be expected.
         */
        if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) {
            active = (1 << flow->flow_index);
            /* Control case: resolved unless we find at least one input control */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == PARSEC_EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(tp, task->locals) ) {
                        /* Cannot use control gather magic with the USE_DEPS_MASK */
                        assert( NULL == dep->ctl_gather_nb );
                        continue;
                    }
                }
                active = 0;
                break;
            }
        } else {
            if( !(flow->flow_flags & PARSEC_FLOW_HAS_IN_DEPS) ) continue;
            if( NULL == flow->dep_in[0] ) {
                /* As the flow is tagged with PARSEC_FLOW_HAS_IN_DEPS and there is no
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
                        assert( dep->cond->op == PARSEC_EXPR_OP_INLINE );
                        if( 0 == dep->cond->inline_func32(tp, task->locals) )
                            continue;  /* doesn't match */
                        /* the condition triggered let's check if it's for a data */
                    }  /* otherwise we have an input flow without a condition, it MUST be final */
                    if( PARSEC_LOCAL_DATA_TASK_CLASS_ID == dep->task_class_id ) {
                        active = (1 << flow->flow_index);
                    }
                    break;
                }
            }
        }
        ret |= active;
    }
    return ret;
}

static parsec_ontask_iterate_t count_deps_fct(struct parsec_execution_stream_s* es,
                                              const parsec_task_t *newcontext,
                                              const parsec_task_t *oldcontext,
                                              const parsec_dep_t* dep,
                                              parsec_dep_data_description_t *data,
                                              int rank_src, int rank_dst, int vpid_dst,
                                              data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                                              void *param)
{
    int *pactive = (int*)param;
    (void)es;
    (void)newcontext;
    (void)oldcontext;
    (void)dep;
    (void)data;
    (void)rank_src;
    (void)rank_dst;
    (void)vpid_dst;
    (void)successor_repo; (void) successor_repo_key;
    *pactive = *pactive+1;
    return PARSEC_ITERATE_CONTINUE;
}

static parsec_dependency_t
parsec_check_IN_dependencies_with_counter( const parsec_taskpool_t *tp,
                                           const parsec_task_t* task )
{
    const parsec_task_class_t* tc = task->task_class;
    int i, j, active;
    const parsec_flow_t* flow;
    const parsec_dep_t* dep;
    parsec_dependency_t ret = 0;

    if( !(tc->flags & PARSEC_HAS_CTL_GATHER) &&
        !(tc->flags & PARSEC_HAS_IN_IN_DEPENDENCIES) ) {
        /* If the number of goal does not depend on this particular task instance,
         * it is pre-computed by the parsec_ptgpp compiler
         */
        return tc->dependencies_goal;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != tc->in[i]); i++ ) {
        flow = tc->in[i];

        /*
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
        if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) {
            /* Control case: just count how many must be resolved */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    if( dep->cond->op == PARSEC_EXPR_OP_INLINE ) {
                        if( dep->cond->inline_func32(tp, task->locals) ) {
                            if( NULL == dep->ctl_gather_nb)
                                active++;
                            else {
                                assert( dep->ctl_gather_nb->op == PARSEC_EXPR_OP_INLINE );
                                active += dep->ctl_gather_nb->inline_func32(tp, task->locals);
                            }
                        }
                    } else {
                        /* Complicated case: fall back to iterate_predecessors with a counter */
                        task->task_class->iterate_predecessors(NULL, task, 1 << flow->flow_index,  count_deps_fct, &active);
                    }
                } else {
                    if( NULL == dep->ctl_gather_nb)
                        active++;
                    else {
                        assert( dep->ctl_gather_nb->op == PARSEC_EXPR_OP_INLINE );
                        active += dep->ctl_gather_nb->inline_func32(tp, task->locals);
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
                    assert( dep->cond->op == PARSEC_EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(tp, task->locals) )
                        continue;  /* doesn't match */
                    /* the condition triggered let's check if it's for a data */
                } else {
                    /* we have an input flow without a condition, it MUST be final */
                }
                if( PARSEC_LOCAL_DATA_TASK_CLASS_ID != dep->task_class_id )  /* if not a data we must wait for the flow activation */
                    active++;
                break;
            }
        }
        ret += active;
    }
    return ret;
}

parsec_dependency_t*
parsec_default_find_deps(const parsec_taskpool_t *tp,
                         parsec_execution_stream_t *es,
                         const parsec_task_t* restrict task)
{
    parsec_dependencies_t *deps;
    int p;

    (void)es;

    deps = tp->dependencies_array[task->task_class->task_class_id];
    assert( NULL != deps );

    for(p = 0; p < task->task_class->nb_parameters - 1; p++) {
        assert( (deps->flags & PARSEC_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[task->locals[task->task_class->params[p]->context_index].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[task->locals[task->task_class->params[p]->context_index].value - deps->min]);
}

parsec_dependency_t*
parsec_hash_find_deps(const parsec_taskpool_t *tp,
                      parsec_execution_stream_t *es,
                      const parsec_task_t* restrict task)
{
    parsec_hashable_dependency_t *hd;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)tp->dependencies_array[task->task_class->task_class_id];

    if( NULL == es ) {
        /* This is a call for debugging purpose, but we cannot tell anything about this task,
         * and we certainly don't want to have a side effect on the hash table */
        return NULL;
    }
    parsec_key_t key = task->task_class->make_key(tp, task->locals);
    assert(NULL != ht);
    parsec_hash_table_lock_bucket(ht, key);
    hd = parsec_hash_table_nolock_find(ht, key);
    if( NULL == hd ) {
        hd = (parsec_hashable_dependency_t *) parsec_thread_mempool_allocate(es->dependencies_mempool);
        hd->dependency = (parsec_dependency_t)0;
        hd->mempool_owner = es->dependencies_mempool;
        hd->ht_item.key = task->task_class->make_key(tp, task->locals);
        parsec_hash_table_nolock_insert(ht, &hd->ht_item);
    }
    parsec_hash_table_unlock_bucket(ht, key);
    return &hd->dependency;
}

static int
parsec_update_deps_with_counter(const parsec_taskpool_t *tp,
                                const parsec_task_t* restrict task,
                                parsec_dependency_t *deps)
{
    parsec_dependency_t dep_new_value, dep_cur_value;
#if defined(PARSEC_DEBUG_PARANOID) || defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task);
#endif

    if( 0 == *deps ) {
        dep_new_value = parsec_check_IN_dependencies_with_counter(tp, task) - 1;
        if( parsec_atomic_cas_int32( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = parsec_atomic_fetch_dec_int32( deps ) - 1;
    } else {
        dep_cur_value = parsec_atomic_fetch_dec_int32( deps ) - 1;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Activate counter dependency for %s leftover %d (excluding current)",
                         tmp, dep_cur_value);

#if defined(PARSEC_DEBUG_PARANOID)
    {
        char wtmp[MAX_TASK_STRLEN];
        if( dep_cur_value > INT_MAX-128) {
            parsec_fatal("task %s as reached an improbable dependency count of %u",
                  wtmp, dep_cur_value );
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Task %s has a current dependencies count of %d remaining. %s to go!",
                             tmp, dep_cur_value,
                             (dep_cur_value == 0) ? "Ready" : "Not ready");
    }
#endif /* PARSEC_DEBUG_PARANOID */

    return dep_cur_value == 0;
}

static int
parsec_update_deps_with_mask(const parsec_taskpool_t *tp,
                             const parsec_task_t* restrict task,
                             parsec_dependency_t *deps,
                             const parsec_task_t* restrict origin,
                             const parsec_flow_t* restrict origin_flow,
                             const parsec_flow_t* restrict dest_flow)
{
    parsec_dependency_t dep_new_value, dep_cur_value;
    const parsec_task_class_t* tc = task->task_class;
#if defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)
    char tmpo[MAX_TASK_STRLEN], tmpt[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmpo, MAX_TASK_STRLEN, origin);
    parsec_task_snprintf(tmpt, MAX_TASK_STRLEN, task);
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Activate mask dep for %s:%s (current 0x%x now 0x%x goal 0x%x) from %s:%s",
                         dest_flow->name, tmpt, *deps, (1 << dest_flow->flow_index), tc->dependencies_goal,
                         origin_flow->name, tmpo);
#if defined(PARSEC_DEBUG_PARANOID)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        parsec_fatal("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)",
                     dest_flow->flow_index, tmpo,
                     origin_flow->name, *deps,
                     tmpt, dest_flow->name );
    }
#else
    (void) origin; (void) origin_flow;
#endif

    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );

    dep_new_value = PARSEC_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(PARSEC_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= parsec_check_IN_dependencies_with_mask(tp, task);
#if defined(PARSEC_DEBUG_NOISIER)
        if( dep_new_value != 0 ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Activate IN dependencies with mask 0x%x", dep_new_value);
        }
#endif
    }

    dep_cur_value = parsec_atomic_fetch_or_int32( deps, dep_new_value ) | dep_new_value;

#if defined(PARSEC_DEBUG_PARANOID)
    if( (dep_cur_value & tc->dependencies_goal) == tc->dependencies_goal ) {
        int success;
        parsec_dependency_t tmp_mask;
        tmp_mask = *deps;
        success = parsec_atomic_cas_int32(deps,
                                          tmp_mask, (tmp_mask | PARSEC_DEPENDENCIES_TASK_DONE));
        if( !success || (tmp_mask & PARSEC_DEPENDENCIES_TASK_DONE) ) {
            parsec_fatal("Task %s scheduled twice (second time by %s)!!!",
                   tmpt, tmpo);
        }
    }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Task %s has a current dependencies of 0x%x and a goal of 0x%x. %s to go!",
                         tmpt, dep_cur_value, tc->dependencies_goal,
                         ((dep_cur_value & tc->dependencies_goal) == tc->dependencies_goal) ?
                         "Ready" : "Not ready");
    return (dep_cur_value & tc->dependencies_goal) == tc->dependencies_goal;
}

/*
 * Mark the task as having all it's dependencies satisfied. This is not
 * necessarily required for the startup process, but it leaves traces such that
 * all executed tasks will show consistently (no difference between the startup
 * tasks and later tasks).
 * Since data -> task grapher logging is detected during dependency resolving,
 * and startup tasks don't have an input dependency, we also resolve this here.
 */
void parsec_dependencies_mark_task_as_startup(parsec_task_t* restrict task,
                                              parsec_execution_stream_t *es)
{
    const parsec_task_class_t* tc = task->task_class;
    parsec_taskpool_t *tp = task->taskpool;
    parsec_dependency_t *deps = tc->find_deps(tp, es, task);

    if( tc->flags & PARSEC_USE_DEPS_MASK ) {
        *deps = PARSEC_DEPENDENCIES_STARTUP_TASK | tc->dependencies_goal;
    } else {
        *deps = 0;
    }
}

/*
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int
parsec_release_local_OUT_dependencies(parsec_execution_stream_t* es,
                                      const parsec_task_t* restrict origin,
                                      const parsec_flow_t* restrict origin_flow,
                                      const parsec_task_t* restrict task,
                                      const parsec_flow_t* restrict dest_flow,
                                      parsec_dep_data_description_t* data,
                                      parsec_task_t** pready_ring,
                                      data_repo_t* target_repo,
                                      parsec_data_copy_t* target_dc,
                                      data_repo_entry_t* target_repo_entry)
{
    const parsec_task_class_t* tc = task->task_class;
    parsec_dependency_t *deps;
    int completed;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
    parsec_task_snprintf(tmp1, MAX_TASK_STRLEN, task);
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Activate dependencies for %s flags = 0x%04x", tmp1, tc->flags);
    deps = tc->find_deps(origin->taskpool, es, task);

    if( tc->flags & PARSEC_USE_DEPS_MASK ) {
        completed = parsec_update_deps_with_mask(origin->taskpool, task, deps, origin, origin_flow, dest_flow);
    } else {
        completed = parsec_update_deps_with_counter(origin->taskpool, task, deps);
    }

#if defined(PARSEC_PROF_GRAPHER)
    parsec_prof_grapher_dep(origin, task, completed, origin_flow, dest_flow);
#endif  /* defined(PARSEC_PROF_GRAPHER) */

    if( completed ) {

        /* This task is ready to be executed as all dependencies are solved.
         * Queue it into the ready_list passed as an argument.
         */
        {
            parsec_task_t *new_context = (parsec_task_t *) parsec_thread_mempool_allocate(es->context_mempool);

            PARSEC_COPY_EXECUTION_CONTEXT(new_context, task);
            new_context->status = PARSEC_TASK_STATUS_NONE;
            PARSEC_AYU_ADD_TASK(new_context);

            PARSEC_DEBUG_VERBOSE(6, parsec_debug_output,
                   "%s becomes ready from %s on thread %d:%d, with mask 0x%04x and priority %d",
                   tmp1,
                   parsec_task_snprintf(tmp2, MAX_TASK_STRLEN, origin),
                   es->th_id, es->virtual_process->vp_id,
                   *deps,
                   task->priority);

            assert( dest_flow->flow_index <= new_context->task_class->nb_flows);
            memset( new_context->data, 0, sizeof(parsec_data_pair_t) * new_context->task_class->nb_flows);
            new_context->repo_entry = NULL;
            /*
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].source_repo = target_repo;
            new_context->data[(int)dest_flow->flow_index].source_repo_entry = target_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data_in   = target_dc;
            (void)data;
            PARSEC_AYU_ADD_TASK_DEP(new_context, (int)dest_flow->flow_index);

            if(task->task_class->flags & PARSEC_IMMEDIATE_TASK) {
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "  Task %s is immediate and will be executed ASAP", tmp1);
                __parsec_execute(es, new_context);
                __parsec_complete_execution(es, new_context);
#if 0 /* TODO */
                SET_HIGHEST_PRIORITY(new_context, parsec_execution_context_priority_comparator);
                PARSEC_LIST_ITEM_SINGLETON(&(new_context->list_item));
                if( NULL != (*pimmediate_ring) ) {
                    (void)parsec_list_item_ring_push( (parsec_list_item_t*)(*pimmediate_ring), &new_context->list_item );
                }
                *pimmediate_ring = new_context;
#endif
            } else {
                *pready_ring = (parsec_task_t*)
                    parsec_list_item_ring_push_sorted( (parsec_list_item_t*)(*pready_ring),
                                                       &new_context->super,
                                                       parsec_execution_context_priority_comparator );
            }
        }
    } else { /* Service not ready */
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "  => Service %s not yet ready", tmp1);
    }

    return PARSEC_SUCCESS;
}

parsec_ontask_iterate_t
parsec_release_dep_fct(parsec_execution_stream_t *es,
                      const parsec_task_t *newcontext,
                      const parsec_task_t *oldcontext,
                      const parsec_dep_t* dep,
                      parsec_dep_data_description_t* data,
                      int src_rank, int dst_rank, int dst_vpid,
                      data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                      void *param)
{
    parsec_release_dep_fct_arg_t *arg = (parsec_release_dep_fct_arg_t *)param;
    const parsec_flow_t* src_flow = dep->belongs_to;
    const parsec_flow_t* dst_flow = dep->flow;


    data_repo_t        *target_repo = arg->output_repo;
    data_repo_entry_t  *target_repo_entry = arg->output_entry;
    parsec_data_copy_t *target_dc = target_repo_entry->data[src_flow->flow_index];
    data_repo_entry_t  *entry_for_reshapping =
            data_repo_lookup_entry(successor_repo, successor_repo_key);
    /* If the successor repo has been advanced with a reshape promise,
     * that one is selected for release_deps, otherwise the one on the
     * predecessor repo is selected.
     * (On the predecessor repo there may be a fulfilled or unfulfilled future,
     * on the successor repo is always unfulfilled).
     */
    if( (entry_for_reshapping != NULL) && (entry_for_reshapping->data[dst_flow->flow_index] != NULL) ){
        target_repo = successor_repo;
        target_repo_entry = entry_for_reshapping;
        target_dc = entry_for_reshapping->data[dst_flow->flow_index];
    }

    /*
     * Check that we don't forward a NULL data to someone else. This
     * can be done only on the src node, since the dst node can
     * check for datatypes without knowing the data yet.
     * By checking now, we allow for the data to be created any time bfore we
     * actually try to transfer it.
     */
    if( PARSEC_UNLIKELY((data->data == NULL) &&
                       (es->virtual_process->parsec_context->my_rank == src_rank) &&
                       ((dep->belongs_to->flow_flags & PARSEC_FLOW_ACCESS_MASK) != PARSEC_FLOW_ACCESS_NONE)) ) {
        char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
        parsec_fatal("A NULL is forwarded\n"
                    "\tfrom: %s flow %s\n"
                    "\tto:   %s flow %s",
                    parsec_task_snprintf(tmp1, MAX_TASK_STRLEN, oldcontext), dep->belongs_to->name,
                    parsec_task_snprintf(tmp2, MAX_TASK_STRLEN, newcontext), dep->flow->name);
    }

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {
        assert( 0 == (arg->action_mask & PARSEC_ACTION_RECV_INIT_REMOTE_DEPS) );

        if( arg->action_mask & PARSEC_ACTION_SEND_INIT_REMOTE_DEPS ){
            struct remote_dep_output_param_s* output;
            int _array_pos, _array_mask;

#if !defined(PARSEC_DIST_COLLECTIVES)
            assert(src_rank == es->virtual_process->parsec_context->my_rank);
#endif
            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            PARSEC_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            output = &arg->remote_deps->output[dep->dep_datatype_index];
            assert( (-1 == arg->remote_deps->root) || (arg->remote_deps->root == src_rank) );
            arg->remote_deps->root = src_rank;
            arg->remote_deps->outgoing_mask |= (1 << dep->dep_datatype_index);
            if( !(output->rank_bits[_array_pos] & _array_mask) ) {
                output->rank_bits[_array_pos] |= _array_mask;
                output->deps_mask |= (1 << dep->dep_index);
                if( 0 == output->count_bits ) {
                    output->data = *data;
                    assert(output->data.data_future == NULL);
#ifdef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
                    /* Now everything is a reshaping entry */
                    /* Check if we need to reshape before sending */
                    if(parsec_is_CTL_dep(output->data)){ /* CTL DEP */
                        output->data.data_future = NULL;
                        output->data.repo = NULL;
                        output->data.repo_key = -1;
                    }else{
                        /* Get reshape from whatever repo it has been set up into */
                        output->data.data_future = (parsec_datacopy_future_t*)target_dc;
                        output->data.repo = target_repo;
                        output->data.repo_key = target_repo_entry->ht_item.key;
                        PARSEC_DEBUG_VERBOSE(4, parsec_debug_output,
                                         "th%d RESHAPE_PROMISE SETUP FOR REMOTE DEPS [%p:%p] for INLINE REMOTE %s fut %p",
                                         es->th_id, output->data.data, (output->data.data)->dtt,
                                         (target_repo == successor_repo? "UNFULFILLED" : "FULFILLED"),
                                         output->data.data_future);
                    }
#endif
                } else {
                    assert(output->data.data == data->data);
#ifdef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
                    /* There's a reshape entry that is not being managed. */
                    assert( !((entry_for_reshapping != NULL) && (entry_for_reshapping->data[dst_flow->flow_index] != NULL)) );
#endif
                }
                output->count_bits++;
                if(newcontext->priority > output->priority) {
                    output->priority = newcontext->priority;
                    if(newcontext->priority > arg->remote_deps->max_priority)
                        arg->remote_deps->max_priority = newcontext->priority;
                }
            }  /* otherwise the bit is already flipped, the peer is already part of the propagation. */
            else{
                assert(output->data.data == data->data);
#ifdef PARSEC_RESHAPE_BEFORE_SEND_TO_REMOTE
                /* There's a reshape entry that is not being managed. */
                assert( !((entry_for_reshapping != NULL) && (entry_for_reshapping->data[dst_flow->flow_index] != NULL)) );
#endif
            }

        }
    }
#else
    (void)src_rank;
    (void)data;
#endif

    if( (arg->action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) &&
        (es->virtual_process->parsec_context->my_rank == dst_rank) ) {
        /* Copying data in data-repo if there is data .
         * We are doing this in order for dtd to be able to track control dependences.
         * Usage count of the repo is dealt with when setting up reshape promises.
         */
        parsec_release_local_OUT_dependencies(es,
                                              oldcontext,
                                              src_flow,
                                              newcontext,
                                              dep->flow,
                                              data,
                                              &arg->ready_lists[dst_vpid],
                                              target_repo, target_dc, target_repo_entry);
    }

    return PARSEC_ITERATE_CONTINUE;
}

/*
 * Convert the execution context to a string.
 */
char*
parsec_task_snprintf( char* str, size_t size,
                      const parsec_task_t* task)
{
    const parsec_task_class_t* tc = task->task_class;
    unsigned int i, ip, index = 0, is_param;

    index += snprintf( str + index, size - index, "%s(", tc->name );
    if( index >= size ) return str;
    for( ip = 0; ip < tc->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "" : ", ",
                           task->locals[tc->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")[");
    if( index >= size ) return str;

    for( i = 0; i < tc->nb_locals; i++ ) {
        is_param = 0;
        for( ip = 0; ip < tc->nb_parameters; ip++ ) {
            if(tc->params[ip]->context_index == tc->locals[i]->context_index) {
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
    index += snprintf(str + index, size - index, "]<%d>", task->priority );
    if( index >= size ) return str;
    if( NULL != task->taskpool ) {
        index += snprintf(str + index, size - index, "{%u}", task->taskpool->taskpool_id );
        if( index >= size ) return str;
    }
    return str;
}
/*
 * Convert assignments to a string.
 */
char* parsec_snprintf_assignments( char* str, size_t size,
                                  const parsec_task_class_t* tc,
                                  const parsec_assignment_t* locals)
{
    unsigned int ip, index = 0;

    index += snprintf( str + index, size - index, "%s", tc->name );
    if( index >= size ) return str;
    for( ip = 0; ip < tc->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "(" : ", ",
                           locals[tc->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")" );

    return str;
}

size_t parsec_destruct_dependencies(parsec_dependencies_t* d)
{
    int i;
    if( NULL == d ) return 0;
    size_t ret = sizeof(parsec_dependencies_t) + (d->max-d->min) * sizeof(parsec_dependencies_union_t);
    if( (d != NULL) && (d->flags & PARSEC_DEPENDENCIES_FLAG_NEXT) ) {
        for(i = d->min; i <= d->max; i++) {
            if( NULL != d->u.next[i - d->min] ) {
                ret += parsec_destruct_dependencies(d->u.next[i-d->min]);
            }
        }
    }
    free(d);
    return ret;
}

int
parsec_taskpool_set_complete_callback( parsec_taskpool_t* tp,
                                       parsec_event_cb_t complete_cb,
                                       void* complete_cb_data )
{
    if( NULL == tp->on_complete ) {
        tp->on_complete      = complete_cb;
        tp->on_complete_data = complete_cb_data;
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERR_EXISTS;
}

int
parsec_taskpool_get_complete_callback( const parsec_taskpool_t* tp,
                                       parsec_event_cb_t* complete_cb,
                                       void** complete_cb_data )
{
    if( NULL != tp->on_complete ) {
        *complete_cb      = tp->on_complete;
        *complete_cb_data = tp->on_complete_data;
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERR_NOT_FOUND;
}

int
parsec_taskpool_set_enqueue_callback( parsec_taskpool_t* tp,
                                      parsec_event_cb_t enqueue_cb,
                                      void* enqueue_cb_data )
{
    if( NULL == tp->on_enqueue ) {
        tp->on_enqueue      = enqueue_cb;
        tp->on_enqueue_data = enqueue_cb_data;
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERR_EXISTS;
}

int
parsec_taskpool_get_enqueue_callback( const parsec_taskpool_t* tp,
                                      parsec_event_cb_t* enqueue_cb,
                                      void** enqueue_cb_data )
{
    if( NULL != tp->on_enqueue ) {
        *enqueue_cb      = tp->on_enqueue;
        *enqueue_cb_data = tp->on_enqueue_data;
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERR_NOT_FOUND;
}

int32_t
parsec_taskpool_set_priority( parsec_taskpool_t* tp, int32_t new_priority )
{
    int32_t old_priority = tp->priority;
    tp->priority = new_priority;
    return old_priority;
}

/* TODO: Change this code to something better */
static parsec_atomic_lock_t taskpool_array_lock = PARSEC_ATOMIC_UNLOCKED;
static parsec_taskpool_t** taskpool_array = NULL;
static uint32_t taskpool_array_size = 1, taskpool_array_pos = 0;
#define NOTASKPOOL ((void*)-1)

static void parsec_taskpool_release_resources(void)
{
    parsec_atomic_lock( &taskpool_array_lock );
    free(taskpool_array);
    taskpool_array = NULL;
    taskpool_array_size = 1;
    taskpool_array_pos = 0;
    parsec_atomic_unlock( &taskpool_array_lock );
}

/* Retrieve the local taskpool attached to a unique taskpool id */
parsec_taskpool_t* parsec_taskpool_lookup( uint32_t taskpool_id )
{
    parsec_taskpool_t *r = NOTASKPOOL;
    parsec_atomic_lock( &taskpool_array_lock );
    if( taskpool_id <= taskpool_array_pos ) {
        r = taskpool_array[taskpool_id];
    }
    parsec_atomic_unlock( &taskpool_array_lock );
    return (NOTASKPOOL == r ? NULL : r);
}

/* Reverse an unique ID for the taskpool but without adding the taskpool to the management array.
 *   Beware that on a distributed environment the connected taskpools must have the same ID.
 */
int parsec_taskpool_reserve_id( parsec_taskpool_t* tp )
{
    uint32_t idx;

    parsec_atomic_lock( &taskpool_array_lock );
    idx = (uint32_t)++taskpool_array_pos;

    if( (NULL == taskpool_array) || (idx >= taskpool_array_size) ) {
        taskpool_array_size <<= 1;
        taskpool_array = (parsec_taskpool_t**)realloc(taskpool_array, taskpool_array_size * sizeof(parsec_taskpool_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (taskpool_array_size>>1); i < taskpool_array_size;
             taskpool_array[i++] = NOTASKPOOL );
    }
    tp->taskpool_id = idx;
    assert( NOTASKPOOL == taskpool_array[idx] );
    parsec_atomic_unlock( &taskpool_array_lock );
    return idx;
}

/* Register a taskpool taskpool with the engine. Once enrolled the taskpool can be target
 * for other components of the runtime, such as communications.
 */
int parsec_taskpool_register( parsec_taskpool_t* tp )
{
    uint32_t idx = tp->taskpool_id;

    parsec_atomic_lock( &taskpool_array_lock );
    if( (NULL == taskpool_array) || (idx >= taskpool_array_size) ) {
        taskpool_array_size <<= 1;
        taskpool_array = (parsec_taskpool_t**)realloc(taskpool_array, taskpool_array_size * sizeof(parsec_taskpool_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (taskpool_array_size>>1); i < taskpool_array_size;
             taskpool_array[i++] = NOTASKPOOL );
    }
    taskpool_array[idx] = tp;
    parsec_atomic_unlock( &taskpool_array_lock );
    return idx;
}

/* globally synchronize taskpool id's so that next register generates the same
 * id at all ranks on a given communicator. */
void parsec_taskpool_sync_ids_context( intptr_t comm )
{
    uint32_t idx,msz;
    parsec_atomic_lock( &taskpool_array_lock );
    idx = (int)taskpool_array_pos;
    msz = (int)taskpool_array_size;
#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
    int mpi_is_on;
    MPI_Initialized(&mpi_is_on);
    if( mpi_is_on ) {
        MPI_Allreduce( MPI_IN_PLACE, &idx, 1, MPI_INT, MPI_MAX, (MPI_Comm)comm );
        while (idx >= msz){
            msz <<= 1;
        }
    }
#endif
    if( msz > taskpool_array_size ) {
        taskpool_array = (parsec_taskpool_t**)realloc(taskpool_array, msz * sizeof(parsec_taskpool_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = taskpool_array_size; i < msz;
             taskpool_array[i++] = NOTASKPOOL );
    }
    taskpool_array_size = msz;
    taskpool_array_pos = idx;
    parsec_atomic_unlock( &taskpool_array_lock );
}

/* globally synchronize taskpool id's so that next register generates the same
 * id at all ranks. */
void parsec_taskpool_sync_ids( void )
{
  parsec_taskpool_sync_ids_context( (intptr_t)MPI_COMM_WORLD );
}

/* Unregister the taskpool with the engine. This make the taskpool_id available for
 * future taskpools. Beware that in a distributed environment the connected taskpools
 * must have the same ID.
 */
void parsec_taskpool_unregister( parsec_taskpool_t* tp )
{
    parsec_atomic_lock( &taskpool_array_lock );
    assert( tp->taskpool_id < taskpool_array_size );
    assert( taskpool_array[tp->taskpool_id] == tp );
    assert( PARSEC_RUNTIME_RESERVED_NB_TASKS == tp->nb_tasks );
    assert( 0 == tp->nb_pending_actions );
    taskpool_array[tp->taskpool_id] = NOTASKPOOL;
    parsec_atomic_unlock( &taskpool_array_lock );
}

void parsec_taskpool_free(parsec_taskpool_t *tp)
{
    assert(NULL != tp);
    PARSEC_OBJ_RELEASE(tp);
}

/*
 * The final step of a taskpool activation. At this point we assume that all the local
 * initializations have been successfully completed for all components, and that the
 * taskpool is ready to be registered with the system, and any potential pending tasks
 * ready to go. If distributed is non 0, then the runtime assumes that the taskpool has
 * a distributed scope and should be registered with the communication engine.
 *
 * The local_task allows for concurrent management of the startup_queue, and provide a way
 * to prevent a task from being added to the scheduler. As the different tasks classes are
 * initialized concurrently, we need a way to prevent the beginning of the tasks generation until
 * all the tasks classes associated with a DAG are completed. Thus, until the synchronization
 * is complete, the task generators are put on hold in the startup_queue. Once the taskpool
 * is ready to advance, and this is the same moment as when the taskpool is ready to be enabled,
 * we reactivate all pending tasks, starting the tasks generation step for all type classes.
 */
int parsec_taskpool_enable(parsec_taskpool_t* tp,
                           parsec_task_t** startup_queue,
                           parsec_task_t* local_task,
                           parsec_execution_stream_t * es,
                           int distributed)
{
    if( NULL != startup_queue ) {
        parsec_list_item_t *ring = NULL;
        parsec_task_t* ttask = (parsec_task_t*)*startup_queue;

        while( NULL != (ttask = (parsec_task_t*)*startup_queue) ) {
            /* Transform the single linked list into a ring */
            *startup_queue = (parsec_task_t*)ttask->super.list_next;
            if(ttask != local_task) {
                ttask->status = PARSEC_TASK_STATUS_HOOK;
                PARSEC_LIST_ITEM_SINGLETON(ttask);
                if(NULL == ring) ring = (parsec_list_item_t *)ttask;
                else parsec_list_item_ring_push(ring, &ttask->super);
            }
        }
        if( NULL != ring ) __parsec_schedule(es, (parsec_task_t *)ring, 0);
    }
    /* Always register the taskpool. This allows the taskpool destructor to unregister it in all cases. */
    parsec_taskpool_register(tp);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Register a new taskpool %p: %d", tp, tp->taskpool_id);
    if( 0 != distributed ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Register a new taskpool %p: %d with the comm engine", tp, tp->taskpool_id);
        (void)parsec_remote_dep_new_taskpool(tp);
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* Print PaRSEC usage message */
void parsec_usage(void)
{
    parsec_output(0,"\n"
            "A PaRSEC argument sequence prefixed by \"--\" can end the command line\n\n"
            "     --parsec_bind_comm   : define the core the communication thread will be bound on\n"
            "\n"
            "     Warning:: The binding options rely on HWLOC. The core numbering is defined between 0 and the number of cores.\n"
            "     Be careful when used with cgroups.\n"
            "\n"
            "    --help         : this message\n"
            "\n"
            " -c --cores        : number of concurrent threads (default: number of physical hyper-threads)\n"
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
            "                                        - mpi_rank : the mpi process rank (in MPI_COMM_WORLD; empty if not relevant)\n"
            "                                        - nb_thread : the number of threads under the virtual process\n"
            "                                                      (overloads the -c flag)\n"
            "                                        - binding : a set of cores for the thread binding. Accepted values are:\n"
            "                                          -- a core list          (exp: 1,3,5-6)\n"
            "                                          -- a hexadecimal mask   (exp: 0xff012)\n"
            "                                          -- a binding range expression: [start];[end];[step] \n"
            "                                             which defines a round-robin one thread per core distribution from start\n"
            "                                             (default 0) to end (default physical core number) by step (default 1)\n"
            "\n"
            );
}




/* Parse --parsec_bind parameter (define a set of cores for the thread binding)
 * The parameter can be
 * - a file containing the parameters (list, mask or expression) for each processes
 * - or a comma separated list of
 *   - a core
 *   - a hexadecimal mask
 *   - a range expression (a:[b[:c]])
 *
 * The function rely on a version of hwloc which support for bitmap.
 * It redefines the fields "bindto" of the startup structure used to initialize the threads
 *
 * We use the topology core indexes to define the binding, not the core numbers.
 * The index upper/lower bounds are 0 and (number_of_cores - 1).
 * The core_index_mask stores core indexes and will be converted into a core_number_mask
 * for the hwloc binding.
 */

#if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
#define PARSEC_BIND_THREAD(THR, WHERE)                                   \
    do {                                                                \
        int __where = (WHERE);                                          \
        if( (THR) < nb_total_comp_threads ) {                           \
            startup[(THR)].bindto = __where;  /* set the thread binding if legit */ \
            (THR)++;                                                    \
            if( hwloc_bitmap_isset(context->cpuset_allowed_mask, __where) ) { \
                parsec_warning("Oversubscription on core %d detected\n", __where); \
            }                                                           \
        }                                                               \
        hwloc_bitmap_set(context->cpuset_allowed_mask, __where);  /* update the mask */ \
    } while (0)
#endif  /* defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP) */

int parsec_parse_binding_parameter(const char * option, parsec_context_t* context,
                                  __parsec_temporary_thread_initialization_t* startup)
{
#if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    char *position, *endptr;
    int i, thr_idx = 0, nb_total_comp_threads = 0, where;
    int nb_real_cores = parsec_hwloc_nb_real_cores();

    if( NULL == context->cpuset_allowed_mask )
        context->cpuset_allowed_mask = hwloc_bitmap_alloc();

    for(i = 0; i < context->nb_vp; i++)
        nb_total_comp_threads += context->virtual_processes[i]->nb_cores;
    if( NULL == option ) {
        for( thr_idx = 0; thr_idx < nb_total_comp_threads; ) {
            PARSEC_BIND_THREAD(thr_idx, (thr_idx % nb_real_cores));
        }
        if( nb_total_comp_threads < nb_real_cores )
            hwloc_bitmap_set_range(context->cpuset_allowed_mask, nb_total_comp_threads, nb_real_cores-1);
        goto compute_free_mask;
    }
    /* The parameter is a file */
    if( NULL != (position = strstr(option, "file:")) ) {
        /* Read from the file the binding parameter set for the local process and parse it
         (recursive call). */

        char *filename = position + strlen("file:");
        FILE *f;
        char *line = NULL;
        size_t line_len = 0;

        f = fopen(filename, "r");
        if( NULL == f ) {
            parsec_warning("invalid binding file %s.", filename);
            return PARSEC_ERR_NOT_FOUND;
        }

        int rank = parsec_debug_rank, line_num = 0;
        while (getline(&line, &line_len, f) != -1) {
            if(line_num == rank) {
                PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI_process %i uses the binding parameters: %s", rank, line);
                parsec_parse_binding_parameter(line, context, startup);
                break;
            }
            line_num++;
        }
        if( NULL != line )
            free(line);

        fclose(f);
        return PARSEC_SUCCESS;
    }

    if( (option[0] == '+') && (context->comm_th_core == -1)) {
        /* The parameter starts with "+" and no specific binding is (yet) defined
         * for the communication thread. The communication thread is then included
         * in the thread mapping. */
        context->comm_th_core = -2;
        option++;  /* skip the + */
    }

    if( NULL == context->cpuset_allowed_mask )
        context->cpuset_allowed_mask = hwloc_bitmap_alloc();
    /* From now on the option is a comma separated list of entities that can be
     * either single numbers, hexadecimal masks or [::] ranges with steps.
     */
    while( NULL != option ) {
        if( NULL != (position = strchr(option, 'x')) ) {
            option = position + 1;  /* skip the x */
            /* find the end of the hexa mask and parse it in reverse */
            position = strchr(option, ',');
            if( NULL == position )  /* we reached the end of the string, the last char is the one right in front */
                position = (char*)option + strlen(option);
            position--; /* Start with the last character, not the '\0' or the ',' */
            where = 0;
            while( 1 ) {
                long int mask;
                if( *position >= '0' && *position <= '9') mask = *position - '0';
                else if( *position >= 'a' && *position <= 'f') mask = *position + 10 - 'a';
                else if( *position >= 'A' && *position <= 'F') mask = *position + 10 - 'A';
                else {
                    parsec_warning("binding: invalid char (%c) in hexadecimal mask. skip\n", *position);
                    goto next_iteration;
                }
                for( i = 0; i < 4; i++ ) {
                    if( mask & (1<<i) ) {  /* bit is set */
                        PARSEC_BIND_THREAD(thr_idx, where);
                    }
                    where++;
                }
                if( position == option )
                    break;
                position--;       /* reverse parsing to maintain the natural order of bits */
            }
            goto next_iteration;
        }

        if( NULL != (position = strchr(option, ':'))) {
            /* The parameter is a range expression such as [start]:[end]:[step] */
            int start = 0, step, end = nb_real_cores;
            if( position != option ) {
                /* we have a starting position */
                start = strtol(option, NULL, 10);
                if( (start >= nb_real_cores) || (start < 0) ) {
                    start = 0;
                    parsec_warning("binding start core not valid (restored to %d)", start);
                }
            }
            position++;  /* skip the : */
            if( '\0' != position[0] ) {
                /* check for the ending position */
                if( ':' != position[0] ) {
                    end = strtol(position, &position, 10);
                    if( (end >= nb_real_cores) || (end < 0) ) {
                        end = nb_real_cores;
                        parsec_warning("binding end core not valid (restored to default %d)", end);
                    }
                }
                position = strchr(position, ':');  /* find the step */
            }
            step = (start < end ? 1 : -1);
            if( NULL != position ) {
                position++;  /* skip the : directly into the step */
                if( '\0' != position[0] ) {
                    step = strtol(position, &endptr, 10); /* allow all numbers but 0 */
                    if( (0 == step) && (position == endptr) ) {
                        step = (start < end ? 1 : -1);
                    }
                }
            }
            if( (0 == step) || ((step > 0) && (start > end)) || ((step < 0) && (start < end)) ) {
                parsec_warning("user provided binding step (%d) invalid. corrected\n", step);
                step = (start < end ? 1 : -1);
            }
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "binding defined by core range [%d:%d:%d]",
                                start, end, step);

            /* redefine the core according to the trio start/end/step */
            where = start;
            while( ((step > 0) && (where <= end)) || ((step < 0) && (where >= end)) ) {
                PARSEC_BIND_THREAD(thr_idx, where);
                where += step;
            }
        }

        else {  /* List of cores */
            where = strtol(option, (char**)&option, 10);
            if( !((where < nb_real_cores) && (where > -1)) ) {
                parsec_warning("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n",
                              where, nb_real_cores-1);
                goto next_iteration;
            }
            PARSEC_BIND_THREAD(thr_idx, where);
        }
      next_iteration:
        option = strchr(option, ',');  /* skip to the next comma */
        if( NULL != option ) option++;
    }
    /* All not-bounded threads will be unleashed */
    for( ; thr_idx < nb_total_comp_threads; thr_idx++ )
        startup[thr_idx].bindto = -1;

  compute_free_mask:
    /*
     * Compute the cpuset_free_mask bitmap, by excluding all the cores with
     * bound threads from the cpuset_allowed_mask.
     */
    context->cpuset_free_mask = hwloc_bitmap_dup(context->cpuset_allowed_mask);
    /* update the cpuset_free_mask according to the thread binding defined */
    for(thr_idx = 0; thr_idx < nb_total_comp_threads; thr_idx++)
        if( -1 != startup[thr_idx].bindto )
            hwloc_bitmap_clr(context->cpuset_free_mask, startup[thr_idx].bindto);

#if defined(PARSEC_DEBUG_NOISIER)
    {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, context->cpuset_allowed_mask);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                            "Thread binding: cpuset [ALLOWED  ]: %s", str);
        free(str);
        hwloc_bitmap_asprintf(&str, context->cpuset_free_mask);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                            "Thread binding: cpuset [AVAILABLE]: %s", str);
        free(str);
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    return PARSEC_SUCCESS;
#else
    (void)option;
    (void)context;
    (void)startup;
    if( 0 == parsec_debug_rank )
        parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                       "The binding defined by --parsec_bind has been ignored!\n"
                       "\tThis option requires a build with HWLOC with bitmap support.");
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif /* PARSEC_HAVE_HWLOC && PARSEC_HAVE_HWLOC_BITMAP */
}

static int check_overlapping_binding(parsec_context_t *context) {
#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI) && defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    MPI_Comm comml = MPI_COMM_NULL; int i, nl = 0, rl = MPI_PROC_NULL;
    MPI_Comm commw = (MPI_Comm)context->comm_ctx;
    assert(-1 != context->comm_ctx);
    MPI_Comm_split_type(commw, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comml);
    MPI_Comm_size(comml, &nl);
    if( 1 < nl && slow_bind_warning ) {
        /* Hu-ho, double check that our binding is not conflicting with other
         * local procs */
        MPI_Comm_rank(comml, &rl);
        char *myset = NULL, *allsets = NULL;

        if( 0 != hwloc_bitmap_list_asprintf(&myset, context->cpuset_allowed_mask) ) {
        }
        int setlen = strlen(myset);
        int *setlens = NULL;
        if( 0 == rl ) {
            setlens = calloc(nl, sizeof(int));
        }
        MPI_Gather(&setlen, 1, MPI_INT, setlens, 1, MPI_INT, 0, comml);

        int *displs = NULL;
        if( 0 == rl ) {
            displs = calloc(nl, sizeof(int));
            displs[0] = 0;
            for( i = 1; i < nl; i++ ) {
                displs[i] = displs[i-1]+setlens[i-1];
            }
            allsets = calloc(displs[nl-1]+setlens[nl-1], sizeof(char));
        }
        MPI_Gatherv(myset, setlen, MPI_CHAR, allsets, setlens, displs, MPI_CHAR, 0, comml);
        free(myset);

        if( 0 == rl ) {
            int notgood = false;
            for( i = 1; i < nl; i++ ) {
                hwloc_bitmap_t other = hwloc_bitmap_alloc();
                hwloc_bitmap_list_sscanf(other, &allsets[displs[i]]);
                if(hwloc_bitmap_intersects(context->cpuset_allowed_mask, other)) {
                    notgood = true;
                }
                hwloc_bitmap_free(other);
            }
            if( notgood ) {
                parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                               "Multiple PaRSEC processes on the same node may share the same physical core(s);\n"
                               "\tThis is often unintentional, and will perform poorly.\n"
                               "\tNote that in managed environments (e.g., ALPS, jsrun), the launcher may set `cgroups`\n"
                               "\tand hide the real binding from PaRSEC; if you verified that the binding is correct,\n"
                               "\tthis message can be silenced using the MCA argument `runtime_warn_slow_binding`.\n");
            }
            free(setlens);
            free(allsets);
            free(displs);
        }
    }
    return PARSEC_SUCCESS;
#else
    (void)context;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif
}

static int parsec_parse_comm_binding_parameter(const char* option, parsec_context_t* context)
{
#if defined(PARSEC_HAVE_HWLOC)
    if( option[0]!='\0' ) {
        int core = atoi(option);
        if( (core > -1) && (core < parsec_hwloc_nb_real_cores()) )
            context->comm_th_core = core;
        else
            parsec_warning("the binding defined by --parsec_bind_comm has been ignored (illegal core number)");
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "default binding for the communication thread");
    }
    return PARSEC_SUCCESS;
#else
    (void)option; (void)context;
    if( 0 == parsec_debug_rank )
        parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                       "The binding defined by --parsec_bind_comm has been ignored!\n"
                       "\tThis option requires a build with HWLOC with bitmap support.");
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* PARSEC_HAVE_HWLOC */
}

#if defined(PARSEC_SIM)
int parsec_getsimulationdate( parsec_context_t *parsec_context ){
    return parsec_context->largest_simulation_date;
}
#endif

static int32_t parsec_expr_eval32(const parsec_expr_t *expr, parsec_task_t *context)
{
    parsec_taskpool_t *tp = context->taskpool;

    assert( expr->op == PARSEC_EXPR_OP_INLINE );
    return expr->inline_func32(tp, context->locals);
}

static int parsec_debug_enumerate_next_in_execution_space(parsec_task_t *context,
                                                         int init, int li)
{
    const parsec_task_class_t *tc = context->task_class;
    int cur, max, incr, min;

    if( li == tc->nb_locals )
        return init; /* We did not find a new context */

    min = parsec_expr_eval32(tc->locals[li]->min, context);

    max = parsec_expr_eval32(tc->locals[li]->max, context);
    if ( min > max ) {
        return 0; /* There is no context starting with these locals */
    }

    if( init ) {
        context->locals[li].value = min;
    }

    do {
        if( parsec_debug_enumerate_next_in_execution_space(context, init, li+1) )
            return 1; /* We did find a new context */

        if( min == max )
            return 0; /* We can't change this local */

        cur = context->locals[li].value;
        if( tc->locals[li]->expr_inc == NULL ) {
            incr = tc->locals[li]->cst_inc;
        } else {
            incr = parsec_expr_eval32(tc->locals[li]->expr_inc, context);
        }

        if( cur + incr > max ) {
            return 0;
        }
        context->locals[li].value = cur + incr;
        init = 1;
    } while(1);
}

/**
 * @brief Debugging helper
 *
 * @details
 *  This function is intended to be called at runtime from a debugger (e.g. gdb)
 *
 *    @param[IN] tp: the taskpool to explore
 *    @param[IN] tc: the taskclass of taskpool to explore
 *    @param[IN] show_remote: boolean, to decide if we show information about remote tasks (progress is not accurate)
 *    @param[IN] show_startup: boolean, to decide if startup tasks should be treated as normal tasks or not when
 *                             displaying the tasks
 *    @param[IN] show_complete: boolean, to decide if completed tasks are shown or not
 */
static void
parsec_debug_taskpool_count_local_tasks( parsec_taskpool_t *tp,
                                         const parsec_task_class_t *tc,
                                         int show_remote,
                                         int show_startup,
                                         int show_complete,
                                         int *nlocal,
                                         int *nreleased,
                                         int *ntotal)
{
    parsec_task_t task;
    parsec_dependency_t *dep;
    parsec_data_ref_t ref;
    int li, init;

    PARSEC_OBJ_CONSTRUCT(&task, parsec_task_t);
    PARSEC_LIST_ITEM_SINGLETON( &task.super );
    task.mempool_owner = NULL;
    task.taskpool = tp;
    task.task_class = tc;
    task.priority = -1;
    task.status = PARSEC_TASK_STATUS_NONE;
    memset( task.data, 0, MAX_PARAM_COUNT * sizeof(parsec_data_pair_t) );

    *nlocal = 0;
    *nreleased = 0;
    *ntotal = 0;

    /* For debugging purposes */
    for(li = 0; li < MAX_LOCAL_COUNT; li++) {
        task.locals[li].value = -1;
    }

    init = 1;
    while( parsec_debug_enumerate_next_in_execution_space(&task, init, 0) ) {
        char tmp[MAX_TASK_STRLEN];
        init = 0;

        (*ntotal)++;
        tc->data_affinity(&task, &ref);
        if( ref.dc->rank_of_key(ref.dc, ref.key) == ref.dc->myrank ) {
            (*nlocal)++;
            dep = tc->find_deps(tp, NULL, &task);
            if( NULL == dep ) {
                parsec_debug_verbose(0, parsec_debug_output,
                                     "  Task %s uses a dependency lookup mechanism that does not allow it to remember executed / waiting / ready tasks\n",
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, &task));
                (*nlocal)--;
                continue;
            }
            if( tc->flags & PARSEC_USE_DEPS_MASK ) {
                if( *dep & PARSEC_DEPENDENCIES_STARTUP_TASK ) {
                    (*nreleased)++;
                    if( show_startup )
                        parsec_debug_verbose(0, parsec_debug_output, "  Task %s is a local startup task",
                                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, &task));
                } else {
                    if((*dep & PARSEC_DEPENDENCIES_BITMASK) == tc->dependencies_goal) {
                        (*nreleased)++;
                    }
                    if( show_complete ||
                        ((*dep & PARSEC_DEPENDENCIES_BITMASK) != tc->dependencies_goal) ) {
                        parsec_debug_verbose(0, parsec_debug_output, "  Task %s is a local task with dependency 0x%08x (goal is 0x%08x) -- Flags: %s %s",
                                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, &task),
                                            *dep & PARSEC_DEPENDENCIES_BITMASK,
                                            tc->dependencies_goal,
                                            *dep & PARSEC_DEPENDENCIES_TASK_DONE ? "TASK_DONE" : "",
                                            *dep & PARSEC_DEPENDENCIES_IN_DONE ? "IN_DONE" : "");
                    }
                }
            } else {
                if( *dep == 0 )
                    (*nreleased)++;

                if( (*dep != 0) || show_complete )
                    parsec_debug_verbose(0, parsec_debug_output, "  Task %s is a local task that must wait for %d more dependencies to complete -- using count method for this task (CTL gather)",
                                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, &task),
                                        *dep);
            }
        } else {
            if( show_remote )
                parsec_debug_verbose(0, parsec_debug_output, "  Task %s is a remote task",
                                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, &task));
        }
    }
}

/**
 * @brief Debugging helper
 *
 * @details
 *  This function is intended to be called at runtime from a debugger (e.g. gdb)
 *  It is called by parsec_debug_taskpool_local_tasks on each taskpool
 *
 *  See help for parsec_debug_taskpool_local_tasks. Only additional parameter
 *  is which taskpool to use.
 *
 *    @param[IN] taskpool: the taskpool to explore
 *    @param[IN] show_remote: boolean, to decide if we show information about remote tasks (progress is not accurate)
 *    @param[IN] show_startup: boolean, to decide if startup tasks should be treated as normal tasks or not when
 *                             displaying the tasks
 *    @param[IN] show_complete: boolean, to decide if completed tasks are shown or not
 */
void parsec_debug_taskpool_local_tasks( parsec_taskpool_t *tp,
                                        int show_remote, int show_startup, int show_complete)
{
    uint32_t fi;
    int nlocal, ntotal, nreleased;
    /* The taskpool has not been initialized yet, or it has been completed */
    if( tp->dependencies_array == NULL )
        return;

    for(fi = 0; fi < tp->nb_task_classes; fi++) {
        parsec_debug_verbose(0, parsec_debug_output, " Tasks of Class %u (%s):\n", fi, tp->task_classes_array[fi]->name);
        parsec_debug_taskpool_count_local_tasks( tp, tp->task_classes_array[fi],
                                                 show_remote, show_startup, show_complete,
                                                 &nlocal, &nreleased, &ntotal );
        parsec_debug_verbose(0, parsec_debug_output, " Total number of Tasks of Class %s: %d\n", tp->task_classes_array[fi]->name, ntotal);
        parsec_debug_verbose(0, parsec_debug_output, " Local number of Tasks of Class %s: %d\n", tp->task_classes_array[fi]->name, nlocal);
        parsec_debug_verbose(0, parsec_debug_output, " Number of Tasks of Class %s that have been released: %d\n", tp->task_classes_array[fi]->name, nreleased);
    }
}

/**
 * @brief Debugging helper
 *
 * @details
 *  This function is intended to be called at runtime from a debugger (e.g. gdb)
 *  It is completely unsafe and should never be used directly in the code. Instead
 *  it provides a nice facility to dump all tasks from a debugger attached to the
 *  process.
 *
 *  This function prints on the debug output information on the current progress:
 *  it will show tasks that executed, tasks that are known to be ready, or that have
 *  been discovered. Depending on the interface used (e.g. PTG or dtd), and the
 *  dependency tracking mechanism (e.g. hash tables, multi dimensional arrays, user-defined
 *  dependency tracking), information printed might be complete or partial.
 *
 *    @param[IN] show_remote: boolean, to decide if we show information about remote tasks (progress is not accurate)
 *    @param[IN] show_startup: boolean, to decide if startup tasks should be treated as normal tasks or not when
 *                             displaying the tasks
 *    @param[IN] show_complete: boolean, to decide if completed tasks are shown or not
 */
void parsec_debug_all_taskpools_local_tasks( int show_remote, int show_startup, int show_complete )
{
    parsec_taskpool_t *tp;
    uint32_t oi;

    parsec_atomic_lock( &taskpool_array_lock );
    for( oi = 1; oi <= taskpool_array_pos; oi++) {
        tp = taskpool_array[ oi ];
        if( tp == NOTASKPOOL )
            continue;
        if( tp == NULL )
            continue;
        parsec_debug_verbose(0, parsec_debug_output, "Tasks of Taskpool %u:\n", oi);
        parsec_debug_taskpool_local_tasks(tp, show_remote,
                                          show_startup,
                                          show_complete);
    }
    parsec_atomic_unlock( &taskpool_array_lock );
}

/* deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int parsec_task_deps_with_final_output(const parsec_task_t *task,
                                       const parsec_dep_t **deps)
{
    const parsec_task_class_t *tc = task->task_class;
    const parsec_flow_t *flow;
    const parsec_dep_t *dep;
    int fi, di, nbout = 0;

    for(fi = 0; fi < tc->nb_flows && tc->out[fi] != NULL; fi++) {
        flow = tc->out[fi];
        if( ! (PARSEC_SYM_OUT & flow->sym_type ) )
            continue;
        for(di = 0; di < MAX_DEP_OUT_COUNT && flow->dep_out[di] != NULL; di++) {
            dep = flow->dep_out[di];
            if( dep->task_class_id != PARSEC_LOCAL_DATA_TASK_CLASS_ID )
                continue;
            if( NULL != dep->cond ) {
                assert( PARSEC_EXPR_OP_INLINE == dep->cond->op );
                if( dep->cond->inline_func32(task->taskpool, task->locals) )
                    continue;
            }
            deps[nbout] = dep;
            nbout++;
        }
    }

    return nbout;
}

int32_t
parsec_add_fetch_runtime_task( parsec_taskpool_t *tp, int32_t nb_tasks )
{
    return parsec_atomic_fetch_add_int32(&tp->nb_pending_actions, nb_tasks ) + nb_tasks;
}

parsec_execution_stream_t *parsec_my_execution_stream(void)
{
    return (parsec_execution_stream_t*)PARSEC_TLS_GET_SPECIFIC(parsec_tls_execution_stream);
}
