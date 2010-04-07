/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "dplasma.h"
#include "stats.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#include "scheduling.h"
#include "dequeue.h"
#include "barrier.h"
#include "remote_dep.h"

#ifdef HAVE_SCHED_SETAFFINITY
#include <linux/unistd.h>
#endif  /* HAVE_SCHED_SETAFFINITY */

#ifdef DPLASMA_PROFILING
#include "profiling.h"
#endif

#ifdef HAVE_PAPI
#include "papi.h"
#endif

#ifdef HAVE_HWLOC
#include "hbbuffer.h"
#endif

FILE *__dplasma_graph_file = NULL;

#ifdef DPLASMA_PROFILING
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
#endif  /* DPLASMA_PROFILING */

static const dplasma_t** dplasma_array = NULL;
static int dplasma_array_size = 0, dplasma_array_count = 0;
#ifdef HAVE_PAPI
int eventSet = PAPI_NULL;
int num_events = 0;
char* event_names[MAX_EVENTS];
#endif

int DPLASMA_TILE_SIZE = 0;

void dplasma_dump(const dplasma_t *d, const char *prefix)
{
    char *pref2 = malloc(strlen(prefix)+3);
    int i;

    sprintf(pref2, "%s  ", prefix);
    printf("%sDplasma Function: %s\n", prefix, d->name);

    printf("%s Parameter Variables:\n", prefix);
    for(i = 0; i < d->nb_params; i++) {
        symbol_dump(d->params[i], pref2);
    }

    printf("%s Local Variables:\n", prefix);
    for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
        symbol_dump(d->locals[i], pref2);
    }

    printf("%s Predicates:\n", prefix);
    for(i = 0; i < MAX_PRED_COUNT && NULL != d->preds[i]; i++) {
        printf("%s", pref2);
        expr_dump(stdout, d->preds[i]);
        printf("\n");
    }

    printf("%s Parameters and Dependencies:\n", prefix);
    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
        param_dump(d->inout[i], pref2);
    }

    printf("%s Required dependencies mask: 0x%x (%s/%s/%s)\n", prefix,
           (int)d->dependencies_mask, (d->flags & DPLASMA_HAS_IN_IN_DEPENDENCIES ? "I" : "N"),
           (d->flags & DPLASMA_HAS_OUT_OUT_DEPENDENCIES ? "O" : "N"),
           (d->flags & DPLASMA_HAS_IN_STRONG_DEPENDENCIES ? "S" : "N"));
    printf("%s Body:\n", prefix);
    printf("%s  %s\n", prefix, d->body);

    if( NULL != d->deps ) {
        
        printf( "Current dependencies\n" );
    }

    free(pref2);
}

int dplasma_dplasma_index( const dplasma_t *d )
{
    int i;
    for(i = 0; i < dplasma_array_count; i++) {
        if( dplasma_array[i] == d ) {
            return i;
        }
    }
    return -1;
}

void dplasma_dump_all( void )
{
    int i;

    for( i = 0; i < dplasma_array_count; i++ ) {
        printf("/**\n * dplasma_t object named %s index %d\n */\n", dplasma_array[i]->name, i );
        dplasma_dump( dplasma_array[i], "" );
    }
}

int dplasma_push( const dplasma_t* d )
{
    if( dplasma_array_count >= dplasma_array_size ) {
        if( 0 == dplasma_array_size ) {
            dplasma_array_size = 4;
        } else {
            dplasma_array_size *= 2;
        }
        dplasma_array = (const dplasma_t**)realloc( dplasma_array, dplasma_array_size * sizeof(dplasma_t*) );
        if( NULL == dplasma_array ) {
            return -1;  /* No more available memory */
        }
    }
    dplasma_array[dplasma_array_count] = d;
    dplasma_array_count++;
    return 0;
}

const dplasma_t* dplasma_find( const char* name )
{
    int i;
    const dplasma_t* object;

    for( i = 0; i < dplasma_array_count; i++ ) {
        object = dplasma_array[i];
        if( 0 == strcmp( object->name, name ) ) {
            return object;
        }
    }
    return NULL;
}

dplasma_t* dplasma_find_or_create( const char* name )
{
    dplasma_t* object;

    object = (dplasma_t*)dplasma_find(name);
    if( NULL != object ) {
        return object;
    }
    object = (dplasma_t*)calloc(1, sizeof(dplasma_t));
    object->name = strdup(name);
    if( 0 == dplasma_push(object) ) {
        return object;
    }
    free(object);
    return NULL;
}

void dplasma_load_array( dplasma_t *array, int size )
{
    int i;

    dplasma_array_size = size;
    dplasma_array_count = size;
    dplasma_array = (const dplasma_t**)calloc(size, sizeof(dplasma_t*));
    for(i = 0; i < size; i++) {
        dplasma_array[i] = &(array[i]);
    }
}

const dplasma_t* dplasma_element_at( int i )
{
    if( i < dplasma_array_count ){
        return dplasma_array[i];
    }
    return NULL;
}

int dplasma_nb_elements( void )
{
    return dplasma_array_count;
}

/**
 *
 */
#if defined(DPLASMA_USE_GLOBAL_LIFO)
dplasma_atomic_lifo_t ready_list;
#endif  /* defined(DPLASMA_USE_GLOBAL_LIFO) */

#ifdef HAVE_SCHED_SETAFFINITY
#define gettid() syscall(__NR_gettid)
#endif /* HAVE_SCHED_SETAFFINITY */

typedef struct __dplasma_temporary_thread_initialization_t {
    dplasma_context_t* master_context;
    int th_id;
    int nb_cores;
} __dplasma_temporary_thread_initialization_t;

#if !defined(DPLASMA_USE_GLOBAL_LIFO) && defined(HAVE_HWLOC)
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
static void push_in_buffer_wrapper(void *store, dplasma_list_item_t *elt)
{ 
    /* Store is a hbbbuffer */
    dplasma_hbbuffer_push_all( (dplasma_hbbuffer_t*)store, elt );
}

static void push_in_queue_wrapper(void *store, dplasma_list_item_t *elt)
{
    /* Store is a lifo or a dequeue */
#if defined(DPLASMA_USE_LIFO)
    dplasma_atomic_lifo_push( (dplasma_atomic_lifo_t*)store, elt );
#else
    dplasma_dequeue_push_back( (dplasma_dequeue_t*)store, elt );
#endif
}
#endif

static void* __dplasma_thread_init( __dplasma_temporary_thread_initialization_t* startup )
{
    dplasma_execution_unit_t* eu;
#if defined(HAVE_HWLOC) || defined(HAVE_SCHED_SETAFFINITY)
    int bind_to_proc = startup->th_id;
#endif  /* defined(HAVE_HWLOC) || defined(HAVE_SCHED_SETAFFINITY) */

#if !defined(DPLASMA_USE_GLOBAL_LIFO) && defined(HAVE_HWLOC)
#if defined(ON_ZOOT)
    bind_to_proc = distance[startup->th_id];
#endif
#endif  /* !defined(DPLASMA_USE_GLOBAL_LIFO)  && defined(HAVE_HWLOC)*/

#ifdef HAVE_SCHED_SETAFFINITY
    {
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
        CPU_SET(bind_to_proc, &cpuset);

        if( -1 == sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) ) {
            printf( "Unable to set the thread affinity (%s)\n", strerror(errno) );
        }
    }
#endif  /* HAVE_SCHED_SETAFFINITY */

    eu = (dplasma_execution_unit_t*)malloc(sizeof(dplasma_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->eu_id          = startup->th_id;
    eu->master_context = startup->master_context;
    (startup->master_context)->execution_units[startup->th_id] = eu;

#ifdef DPLASMA_PROFILING
    eu->eu_profile = dplasma_profiling_thread_init( 8192, "DPLASMA Thread %d", eu->eu_id );
#endif
#ifdef DPLASMA_USE_LIFO
    eu->eu_task_queue = (dplasma_atomic_lifo_t*)malloc( sizeof(dplasma_atomic_lifo_t) );
    if( NULL == eu->eu_task_queue ) {
        free(eu);
        return NULL;
    }
    dplasma_atomic_lifo_construct( eu->eu_task_queue );
#elif defined(DPLASMA_USE_GLOBAL_LIFO)
    /* Everybody share the same global LIFO */
    eu->eu_task_queue = &ready_list;
#elif defined(HAVE_HWLOC)
    /* we set the eu_task_queue later */
#else
    eu->eu_task_queue = (dplasma_dequeue_t*)malloc( sizeof(dplasma_dequeue_t) );
    if( NULL == eu->eu_task_queue ) {
        free(eu);
        return NULL;
    }
    dplasma_dequeue_construct( eu->eu_task_queue );
#if PLACEHOLDER_SIZE
    eu->placeholder_pop  = 0;
    eu->placeholder_push = 0;
#endif  /* PLACEHOLDER_SIZE */
#endif  /* DPLASMA_USE_LIFO */

#if defined(DPLASMA_CACHE_AWARENESS)
    eu->closest_cache = NULL;
#endif

#if defined(HAVE_HWLOC)
    {
        int level, master, idx;
        if( eu->eu_id == 0 ) {
            eu->eu_system_queue = (dplasma_dequeue_t*)malloc(sizeof(dplasma_dequeue_t));
            dplasma_dequeue_construct( eu->eu_system_queue );
            dplasma_barrier_wait( &startup->master_context->barrier );
        } else {
            dplasma_barrier_wait( &startup->master_context->barrier );
            eu->eu_system_queue = startup->master_context->execution_units[0]->eu_system_queue;
        }

#if defined(USE_HIERARCHICAL_QUEUES)
        eu->eu_nb_hierarch_queues = dplasma_hwloc_nb_levels(startup->master_context);
        assert(eu->eu_nb_hierarch_queues > 0 /* Must have at least a system queue and a socket queue to work with hwloc */ );

        eu->eu_hierarch_queues = (dplasma_hbbuffer_t **)malloc(eu->eu_nb_hierarch_queues * sizeof(dplasma_hbbuffer_t*) );

        for(level = 0; level < eu->eu_nb_hierarch_queues; level++) {
            idx = eu->eu_nb_hierarch_queues - 1 - level;
            master = dplasma_hwloc_master_id(startup->master_context, level, eu->eu_id);
            if( eu->eu_id == master ) {
                int nbcores = dplasma_hwloc_nb_cores(startup->master_context, level, master);
                int queue_size = 96 * (level+1) / nbcores;
                if( queue_size < nbcores ) queue_size = nbcores;

                /* The master(s) create the shared queues */               
                eu->eu_hierarch_queues[idx] = dplasma_hbbuffer_new( queue_size, nbcores,
                                                                    level == 0 ? push_in_queue_wrapper : push_in_buffer_wrapper,
                                                                    level == 0 ? (void*)eu->eu_system_queue : (void*)eu->eu_hierarch_queues[idx+1]);
                DEBUG(("%d creates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)\n",
                       eu->eu_id, queue_size, nbcores,
                       level, idx, eu->eu_hierarch_queues[idx],
                       level == 0 ? (void*)eu->eu_system_queue : (void*)eu->eu_hierarch_queues[idx+1],
                       level == 0 ? "System queue" : "upper level hhbuffer"));
                
                /* The master(s) unblock all waiting slaves */
                dplasma_barrier_wait( &startup->master_context->barrier );
            } else {
                /* Be a slave: wait that the master(s) unblock me */
                dplasma_barrier_wait( &startup->master_context->barrier );
                
                DEBUG(("%d takes the buffer of %d at level %d stored in %d: %p\n",
                       eu->eu_id, master, level, idx, startup->master_context->execution_units[master]->eu_hierarch_queues[idx]));
                /* The slaves take their queue for this level from their master */
                eu->eu_hierarch_queues[idx] = startup->master_context->execution_units[master]->eu_hierarch_queues[idx];
            }
        }
        eu->eu_task_queue = eu->eu_hierarch_queues[0];
#else /* Don't USE_HIERARCHICAL_QUEUES: USE_FLAT_QUEUES */
        {
            int queue_size = startup->master_context->nb_cores * 4;
            int nq = 0;
            int id;

            eu->eu_nb_hierarch_queues = startup->master_context->nb_cores;
            eu->eu_hierarch_queues = (dplasma_hbbuffer_t **)malloc(eu->eu_nb_hierarch_queues * sizeof(dplasma_hbbuffer_t*) );
            /* Each thread creates its own "local" queue, connected to the shared dequeue */
            eu->eu_task_queue = dplasma_hbbuffer_new( queue_size, 1, push_in_queue_wrapper, 
                                                      (void*)eu->eu_system_queue);
            eu->eu_hierarch_queues[0] =  eu->eu_task_queue;

            dplasma_barrier_wait( &startup->master_context->barrier );

            /* Then, they know about all other queues, from the closest to the farthest */
            nq = 1;
            for(level = 0; level <= dplasma_hwloc_nb_levels(); level++) {
                for(id = (eu->eu_id + 1) % startup->master_context->nb_cores; 
                    id != eu->eu_id; 
                    id = (id + 1) %  startup->master_context->nb_cores) {
                    int d;

                    d = dplasma_hwloc_distance(eu->eu_id, id);
                    if( d == 2*level || d == 2*level + 1 ) {
                        eu->eu_hierarch_queues[nq] = startup->master_context->execution_units[id]->eu_task_queue;
                        DEBUG(("%d: my %d preferred queue is the task queue of %d (%p)\n",
                               eu->eu_id, nq, id, eu->eu_hierarch_queues[nq]));
                        nq++;
                    }
                }
            }
        }
#endif

#if defined(DPLASMA_CACHE_AWARENESS)
#define TILE_SIZE (120*120*sizeof(double))
        for(level = 0; level < dplasma_hwloc_nb_levels(); level++) {
            master = dplasma_hwloc_master_id(level, eu->eu_id);
            if( eu->eu_id == master ) {
                int nbtiles = (dplasma_hwloc_cache_size(level, master) / TILE_SIZE)-1;
                int nbcores = dplasma_hwloc_nb_cores(level, master);

                /* The master(s) create the cache explorer, using their current closest cache as its father */
                eu->closest_cache = cache_create( nbcores, eu->closest_cache, nbtiles);
                DEBUG(("%d creates cache of size %d for level %d: %p (parent: %p)\n",
                       eu->eu_id, nbtiles,
                       level, eu->closest_cache,
                       eu->closest_cache != NULL ? eu->closest_cache->parent : NULL));
                
                /* The master(s) unblock all waiting slaves */
                dplasma_barrier_wait( &startup->master_context->barrier );
            } else {
                /* Be a slave: wait that the master(s) unblock me */
                dplasma_barrier_wait( &startup->master_context->barrier );
                
                /* The closest cache has been created by my master. Thank you, master */
                eu->closest_cache = startup->master_context->execution_units[master]->closest_cache;
                DEBUG(("%d takes the closest cache of %d at level %d: %p\n",
                       eu->eu_id, master, level,  eu->closest_cache));
            }
        }
#endif /* DPLASMA_CACHE_AWARENESS */
    }
#endif  /* defined(HAVE_HWLOC)*/

    /* The main thread will go back to the user level */
    if( 0 == eu->eu_id )
        return NULL;

    return __dplasma_progress(eu);
}

#ifdef USE_PAPI
extern int num_events;
extern char* event_names[];
#endif

dplasma_context_t* dplasma_init( int nb_cores, int* pargc, char** pargv[], int tile_size )
{
    dplasma_context_t* context = (dplasma_context_t*)malloc(sizeof(dplasma_context_t) +
                                                            nb_cores * sizeof(dplasma_execution_unit_t*));
    __dplasma_temporary_thread_initialization_t* startup = 
        (__dplasma_temporary_thread_initialization_t*)malloc(nb_cores * sizeof(__dplasma_temporary_thread_initialization_t));
    int i;

    DPLASMA_TILE_SIZE = tile_size;

#if defined(USE_MPI)
    /* Change this to pass the MPI Datatype as parameter to dplasma_init, or 
     * at least authorize to pass something different that MPI_DOUBLE?
     */
    remote_dep_mpi_create_default_datatype(tile_size, MPI_DOUBLE);
#endif

    context->nb_cores = (int32_t) nb_cores;
    context->__dplasma_internal_finalization_in_progress = 0;
    context->__dplasma_internal_finalization_counter = 0;

#ifdef USE_PAPI
    num_events = 0;
#endif
    
    for( i = 0; i < *pargc; i++ ) {
        if( 0 == strcmp( (*pargv)[i], "-dot" ) ) {
            if( NULL == __dplasma_graph_file ) {
                int len = strlen((*pargv)[i+1]) + 32;
                char filename[len];
#if defined(DISTRIBUTED) && defined(USE_MPI)
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                snprintf(filename, len, "%s%d", (*pargv)[i+1], rank);
#else
                snprintf(filename, len, "%s", (*pargv)[i+1]);
#endif
                __dplasma_graph_file = fopen( filename, "w");
                i++;
            }      
        }
        else if( 0 == strcmp( (*pargv)[i], "-papi" ) ) {
#ifdef USE_PAPI
            char* dup;
            char* ptr;
            ptr = dup = strdup((*pargv)[i+1]);
            while(NULL != (ptr = strrchr(dup, ',')))
            {
                if(num_events >= 2)
                {
                    fprintf(stderr, "-papi accepts only up to 3 events\n");
                    break;
                }
                *ptr = '\0';
                events_names[num_events] = strdup(ptr + 1);
                num_events++;
            }
            free(dup);
#else 
            fprintf(stderr, "-papi is pointless for this PAPI disabled build\n");
#endif
        }
    }
    
    /* Initialize the barriers */
    dplasma_barrier_init( &(context->barrier), NULL, nb_cores );

    if( NULL != __dplasma_graph_file ) {
        fprintf(__dplasma_graph_file, "digraph G {\n");
        fflush(__dplasma_graph_file);
    }
#ifdef DPLASMA_PROFILING
    dplasma_profiling_init( "%s", (*pargv)[0] );

    dplasma_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                              &MEMALLOC_start_key, &MEMALLOC_end_key);
    dplasma_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                              &schedule_poll_begin, &schedule_poll_end);
    dplasma_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                              &schedule_push_begin, &schedule_push_end);
    dplasma_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                              &schedule_sleep_begin, &schedule_sleep_end);
#endif  /* DPLASMA_PROFILING */

#if defined(DPLASMA_USE_GLOBAL_LIFO)
    dplasma_atomic_lifo_construct(&ready_list);
#endif  /* defined(DPLASMA_USE_GLOBAL_LIFO) */

    /* Prepare the LIFO task queue for each execution unit */
    for( i = 0; i < nb_cores; i++ ) {
        startup[i].th_id = i;
        startup[i].master_context = context;
        startup[i].nb_cores = nb_cores;
    }

    if( nb_cores > 1 ) {
        pthread_attr_t thread_attr;

        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
        pthread_setconcurrency(nb_cores);
#endif  /* __linux */

        context->pthreads = (pthread_t*)malloc(nb_cores * sizeof(pthread_t));

        /* The first execution unit is for the master thread */
        for( i = 1; i < context->nb_cores; i++ ) {
            pthread_create( &((context)->pthreads[i]),
                            &thread_attr,
                            (void* (*)(void*))__dplasma_thread_init,
                            (void*)&(startup[i]));
        }
    }

    __dplasma_thread_init( &startup[0] );

    /* Wait until all threads are done binding themselves */
    dplasma_barrier_wait( &(context->barrier) );
    context->__dplasma_internal_finalization_counter++;

    /* Release the temporary array used for starting up the threads */
    free(startup);

    /* Wait until threads are bound before introducing progress threads */
    context->nb_nodes = dplasma_remote_dep_init(context);
    
#ifdef HAVE_PAPI
    if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        printf("PAPI library initialization error! \n");
    else {
        if (PAPI_create_eventset(&eventSet) != PAPI_OK)
            printf("PAPI unable to create event set! \n");
        else {
            for( i = 0; i < num_events; ++i ) {
                int event;
                PAPI_event_name_to_code(event_names[i], &event);

                if (PAPI_add_event(eventSet, event) != PAPI_OK) 
                    printf("PAPI unable to add event: %s \n", event_names[i]);
            }
        }
    }
#endif

    return context;
}

/**
 *
 */
int dplasma_fini( dplasma_context_t** pcontext )
{
    dplasma_context_t* context = *pcontext;
    int i;

#ifdef HAVE_PAPI
    PAPI_shutdown();
#endif

    /* Now wait until every thread is back */
    context->__dplasma_internal_finalization_in_progress = 1;
    dplasma_barrier_wait( &(context->barrier) );

    /* The first execution unit is for the master thread */
    for(i = 1; i < context->nb_cores; i++) {
        pthread_join( context->pthreads[i], NULL );
    }

    (void) dplasma_remote_dep_fini( context );
    
    for(i = 1; i < context->nb_cores; i++) {
#if defined(DPLASMA_USE_LIFO) && !defined(DPLASMA_USE_GLOBAL_LIFO)
        free( context->execution_units[i]->eu_task_queue );
        context->execution_units[i]->eu_task_queue = NULL;
#endif  /* defined(DPLASMA_USE_LIFO) && !defined(DPLASMA_USE_GLOBAL_LIFO) */
#if defined(HAVE_HWLOC)
        /**
         * TODO: use HWLOC to know who is responsible to free this, 
         * and free the inside too 
         */
        free(context->execution_units[i]->eu_hierarch_queues);
        context->execution_units[i]->eu_hierarch_queues = NULL;
        context->execution_units[i]->eu_nb_hierarch_queues = 0;
        //free(context->execution_units[i]->eu_system_queue);
        context->execution_units[i]->eu_system_queue = NULL;
#endif  /* !defined(DPLASMA_USE_GLOBAL_LIFO)  && defined(HAVE_HWLOC)*/
    }
    
#ifdef DPLASMA_PROFILING
    dplasma_profiling_fini( );
#endif  /* DPLASMA_PROFILING */

    /* Destroy all resources allocated for the barrier */
    dplasma_barrier_destroy( &(context->barrier) );

    if( context->nb_cores > 1 ) {
        free(context->pthreads);
    }

    if( NULL != __dplasma_graph_file ) {
        fprintf(__dplasma_graph_file, "}\n");
        fclose(__dplasma_graph_file);
        __dplasma_graph_file = NULL;
    }

#if defined(DPLASMA_STATS)
    {
        char filename[64];
        char prefix[32];
# if defined(DISTRIBUTED) && defined(USE_MPI)
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        snprintf(filename, 64, "dplasma-%d.stats", rank);
        snprintf(prefix, 32, "%d/%d", rank, size);
# else
        snprintf(filename, 64, "dplasma.stats");
        snprintf(prefix, 32, "");
# endif
        dplasma_stats_dump(filename, prefix);
    }
#endif

    free(context);
    *pcontext = NULL;
    return 0;
}

/**
 * Compute the correct initial values for an execution context. These values
 * are in the range and validate all possible predicates. If such values do
 * not exist this function returns -1.
 */
int dplasma_set_initial_execution_context( dplasma_execution_context_t* exec_context )
{
    int i, min, rc;
    const dplasma_t* object = exec_context->function;
    const expr_t** predicates = (const expr_t**)object->preds;

    /* Compute the number of local values */
    if( 0 == object->nb_locals ) {
        /* special case for the IN/OUT objects */
        return 0;
    }

    /**
     * Find the minimum values for all locals. Note this is done
     * with NULL predicates, so no validation on the values
     * is performed.
     */
    for( i = 0; i < object->nb_locals; i++ ) {
        exec_context->locals[i].sym = object->locals[i];
        rc = dplasma_symbol_get_first_value(object->locals[i], NULL,
                                            exec_context->locals, &min);
        exec_context->locals[i].value = min;
    }
    /**
     * Now fix these values, by walking up and down the locals
     * stack and validate the selected values through the
     * predicates.
     */
    for( i = 0; i < object->nb_locals; i++ ) {
        rc = dplasma_symbol_get_first_value(object->locals[i], predicates,
                                            exec_context->locals, &min);
        while ( rc != EXPR_SUCCESS ) {
            i--;
            if( i < 0 ) {
                printf( "Impossible to find initial values. Giving up\n" );
                return -1;
            }

            rc = dplasma_symbol_get_next_value(object->locals[i], predicates,
                                               exec_context->locals, &min );
        }
    }
    if( i < MAX_LOCAL_COUNT ) {
        exec_context->locals[i].sym = NULL;
    }
    return 0;
}

/**
 * Check is there is any of the input parameters that do depend on some
 * other service. 
 */
int dplasma_service_can_be_startup( dplasma_execution_context_t* exec_context )
{
    const dplasma_t* function = exec_context->function;
    param_t* param;
    dep_t* dep;
    int i, j, value;

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->inout[i]); i++ ) {
        param = function->inout[i];
        if( !(SYM_IN & param->sym_type) ) {
            continue;
        }

        for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != param->dep_in[j]); j++ ) {
            dep = param->dep_in[j];

            if( NULL == dep->cond ) {
                if( dep->dplasma->nb_locals != 0 ) {
                    /* Strict dependency on another service. No chance to be a starter */
                    return -1;
                }
                continue;
            }
            /* TODO: Check to see if the condition can be applied in the current context */
            (void)expr_eval( dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
            if( value == 1 ) {
                if( dep->dplasma->nb_locals != 0 ) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

/**
 * Convert the execution context to a string.
 */
char* dplasma_service_to_string( const dplasma_execution_context_t* exec_context,
                                 char* tmp,
                                 size_t length )
{
    const dplasma_t* function = exec_context->function;
    int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s", function->name );
    if( index >= length ) return tmp;
    for( i = 0; i < function->nb_locals; i++ ) {
        index += snprintf( tmp + index, length - index, "_%d",
                           exec_context->locals[i].value );
        if( index >= length ) return tmp;
    }

    return tmp;
}

/**
 * Convert a dependency to a string under the format X(...) -> Y(...).
 */
char* dplasma_dependency_to_string( const dplasma_execution_context_t* from,
                                    const dplasma_execution_context_t* to,
                                    char* tmp,
                                    size_t length )
{
    int index = 0;

    dplasma_service_to_string( from, tmp, length );
    index = strlen(tmp);
    index += snprintf( tmp + index, length - index, " -> " );
    dplasma_service_to_string( to, tmp + index, length - index );
    return tmp;
}

/**
 * This function generate all possible execution context for a given function with
 * respect to the predicates.
 */
int dplasma_compute_nb_tasks( const dplasma_t* object, int use_predicates )
{
    dplasma_execution_context_t* exec_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    const expr_t** predicates = (const expr_t**)object->preds;
    int rc, actual_loop, nb_tasks = 0;

    DPLASMA_STAT_INCREASE(mem_contexts, sizeof(dplasma_execution_context_t) + STAT_MALLOC_OVERHEAD);

    exec_context->function = (dplasma_t*)object;

    DEBUG(( "Function %s (loops %d)\n", object->name, object->nb_locals ));
    if( 0 == object->nb_locals ) {
        /* special case for the IN/OUT obejcts */
        return 0;
    }

    if( 0 != dplasma_set_initial_execution_context(exec_context) ) {
        /* if we can't initialize the execution context then there is no reason to
         * continue.
         */
        return -1;
    }

    /* Clear the predicates if not needed */
    if( !use_predicates ) predicates = NULL;

    actual_loop = object->nb_locals - 1;
    while(1) {
        int value;

        /* Do whatever we have to do for this context */
        nb_tasks++;

        /* Go to the next valid value for this loop context */
        rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                            exec_context->locals, &value );

        /* If no more valid values, go to the previous loop,
         * compute the next valid value and redo and reinitialize all other loops.
         */
        if( rc != EXPR_SUCCESS ) {
            int current_loop = actual_loop;
        one_loop_up:
            DEBUG(("Loop index %d based on %s failed to get next value. Going up ...\n",
                   actual_loop, object->locals[actual_loop]->name));
            if( 0 == actual_loop ) {  /* we're done */
                goto end_of_all_loops;
            }
            actual_loop--;  /* one level up */
            rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                                exec_context->locals, &value );
            if( rc != EXPR_SUCCESS ) {
                goto one_loop_up;
            }
            DEBUG(("Keep going on the loop level %d (symbol %s value %d)\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            for( actual_loop++; actual_loop <= current_loop; actual_loop++ ) {
                rc = dplasma_symbol_get_first_value(object->locals[actual_loop], predicates,
                                                    exec_context->locals, &value );
                if( rc != EXPR_SUCCESS ) {  /* no values for this symbol in this context */
                    goto one_loop_up;
                }
                DEBUG(("Loop index %d based on %s get first value %d\n", actual_loop,
                       object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            }
            actual_loop = current_loop;  /* go back to the original loop */
        } else {
            DEBUG(("Loop index %d based on %s get next value %d\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
        }
    }
 end_of_all_loops:

    return nb_tasks;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static int dplasma_check_IN_dependencies( const dplasma_execution_context_t* exec_context )
{
    const dplasma_t* function = exec_context->function;
    int i, j, value, mask = 0;
    param_t* param;
    dep_t* dep;

    if( !(function->flags & DPLASMA_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->inout[i]); i++ ) {
        param = function->inout[i];

        if( !(SYM_IN & param->sym_type) ) {
            continue;  /* this is only an OUTPUT dependency */
        }
        for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != param->dep_in[j]); j++ ) {
            dep = param->dep_in[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                (void)expr_eval( dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
                if( 0 == value ) {
                    continue;
                }
            }
            if( dep->dplasma->nb_locals == 0 ) {
                mask |= param->param_mask;
            }
        }
    }
    return mask;
}

#define CURRENT_DEPS_INDEX(K)  (exec_context->locals[(K)].value - deps->min)

static void malloc_deps(dplasma_execution_unit_t* eu_context, 
                        dplasma_execution_context_t* exec_context, 
                        dplasma_dependencies_t** deps_location)
{
    dplasma_t* function = exec_context->function;
    deps_location = &(function->deps);
    dplasma_dependencies_t* deps = *deps_location;
    dplasma_dependencies_t* last_deps = NULL;
    int i;
    
#ifdef DPLASMA_PROFILING
    dplasma_profiling_trace(eu_context->eu_profile, MEMALLOC_start_key, 0);
#endif
    
    for( i = 0; i < function->nb_locals; i++ ) {
        if( NULL == (*deps_location) ) {
            int min, max, number;
            /* TODO: optimize this section (and the similar one few tens of lines down
             * the code) to work on local ranges instead of absolute ones.
             */
            dplasma_symbol_get_absolute_minimum_value( function->locals[i], &min );
            dplasma_symbol_get_absolute_maximum_value( function->locals[i], &max );
            number = max - min;
            DEBUG(("Allocate %d spaces for loop %s (min %d max %d)\n",
                   number, function->locals[i]->name, min, max));
            deps = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) +
                                                   number * sizeof(dplasma_dependencies_union_t));
            DPLASMA_STAT_INCREASE(mem_contexts, sizeof(dplasma_dependencies_t) +
                  number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD); 
            deps->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL;
            deps->symbol = function->locals[i];
            deps->min = min;
            deps->max = max;
            deps->prev = last_deps; /* chain them backward */
            if( 0 == dplasma_atomic_cas(deps_location, (uintptr_t) NULL, (uintptr_t) deps) ) {
                /* Some other thread manage to set it before us. Not a big deal. */
                free(deps);
                DPLASMA_STAT_DECREASE(mem_contexts,  sizeof(dplasma_dependencies_t) +
                                      number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD);
                goto deps_created_by_another_thread;
            }
            if( NULL != last_deps ) {
                last_deps->flags = DPLASMA_DEPENDENCIES_FLAG_NEXT | DPLASMA_DEPENDENCIES_FLAG_ALLOCATED;
            }
        } else {
        deps_created_by_another_thread:
            deps = *deps_location;
        }
        
        DEBUG(("Prepare storage for next loop variable (value=%d) at %d\n",
               exec_context->locals[i].value, CURRENT_DEPS_INDEX(i)));
        deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(i)]);
        last_deps = deps;
    }
#ifdef DPLASMA_PROFILING
    dplasma_profiling_trace(eu_context->eu_profile, MEMALLOC_end_key, 0);
#endif    
}


/**
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int dplasma_release_local_OUT_dependencies( dplasma_execution_unit_t* eu_context,
                                            const dplasma_execution_context_t* restrict origin,
                                            const param_t* restrict origin_param,
                                            dplasma_execution_context_t* restrict exec_context,
                                            const param_t* restrict dest_param,
                                            dplasma_dependencies_t **deps_location,
                                            dplasma_execution_context_t** pready_list )
{
    dplasma_t* function = exec_context->function;
    dplasma_dependencies_t *deps;
    int i, updated_deps, mask;
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif

    DEBUG(("Activate dependencies for %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    if( NULL == *deps_location ) {
        malloc_deps(eu_context, exec_context, deps_location);
    }
    deps = *deps_location;
    
    i = function->nb_locals - 1;

#if !defined(NDEBUG)
    if( deps->u.dependencies[CURRENT_DEPS_INDEX(i)] & dest_param->param_mask ) {
        char tmp[128], tmp1[128];
        fprintf( stderr, "Output dependencies %2x from %s (param %s) activate an already existing dependency %2x on %s (param %s)\n",
                 dest_param->param_mask, dplasma_service_to_string(origin, tmp, 128), origin_param->name,
                 deps->u.dependencies[CURRENT_DEPS_INDEX(i)],
                 dplasma_service_to_string(exec_context, tmp1, 128),  dest_param->name );
    }
    assert( 0 == (deps->u.dependencies[CURRENT_DEPS_INDEX(i)] & dest_param->param_mask) );
#endif  /* !defined(NDEBUG) */
    mask = DPLASMA_DEPENDENCIES_HACK_IN | dest_param->param_mask;
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DPLASMA_DEPENDENCIES_HACK_IN & deps->u.dependencies[CURRENT_DEPS_INDEX(i)]) ) {
        mask |= dplasma_check_IN_dependencies( exec_context );
#ifdef DPLASMA_DEBUG
        if( mask > 0 ) {
            DEBUG(("Activate IN dependencies with mask 0x%02x\n", mask));
        }
#endif /* DPLASMA_DEBUG */
    }

    updated_deps = dplasma_atomic_bor( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)], mask);

#if defined(DPLASMA_GRAPHER) || 1
    if( NULL != __dplasma_graph_file ) {
        char tmp[128];
        fprintf(__dplasma_graph_file, 
                "%s [label=\"%s=>%s\" color=\"%s\" style=\"%s\"]\n", dplasma_dependency_to_string(origin, exec_context, tmp, 128),
                origin_param->name, dest_param->name, (updated_deps == mask ? "#00FF00" : "#FF0000"),
                ((updated_deps & function->dependencies_mask) == function->dependencies_mask) ? "solid" : "dashed");
        fflush(__dplasma_graph_file);
    }
#endif  /* defined(DPLASMA_GRAPHER) */

    if( (updated_deps & function->dependencies_mask) == function->dependencies_mask ) {

#if !defined(NDEBUG)
        {
            int success, tmp_mask;
            do {
                tmp_mask = deps->u.dependencies[CURRENT_DEPS_INDEX(i)];
                success = dplasma_atomic_cas( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)],
                                              tmp_mask, (tmp_mask | (1<<30)) );
                if( !success || (tmp_mask & (1<<30)) ) {
                    char tmp[128];
                    fprintf(stderr, "I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice !!!\n",
                            success, tmp_mask, dplasma_service_to_string(exec_context, tmp, 128));
                    assert(0);
                }
            } while (0);
        }
#endif  /* !defined(NDEBUG) */
        /* This service is ready to be executed as all dependencies
         * are solved.  Queue it into the ready_list passed as an
         * argument.
         */
        {
            dplasma_execution_context_t* new_context;
            new_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
            DPLASMA_STAT_INCREASE(mem_contexts, sizeof(dplasma_execution_context_t) + STAT_MALLOC_OVERHEAD);
            memcpy( new_context, exec_context, sizeof(dplasma_execution_context_t) );
#if defined(DPLASMA_CACHE_AWARENESS)
            new_context->pointers[1] = NULL;
#endif

            if( NULL == *pready_list ) {
                DPLASMA_LIST_ITEM_SINGLETON(new_context);
                *pready_list = new_context;
            } else {
                new_context->list_item.list_next = (dplasma_list_item_t*)*pready_list;
                new_context->list_item.list_prev = (*pready_list)->list_item.list_prev;
                new_context->list_item.list_next->list_prev = (dplasma_list_item_t*)new_context;
                new_context->list_item.list_prev->list_next = (dplasma_list_item_t*)new_context;
            }
        }

        DPLASMA_STAT_INCREASE(counter_nbtasks, 1ULL);

    } else {
        DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x: real 0x%02x)\n",
               dplasma_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_mask,
               (int)(updated_deps & (~DPLASMA_DEPENDENCIES_HACK_IN)),
               (int)(updated_deps)));
    }

    return 0;
}

/**
 * Check if a particular instance of the service can be executed based on the
 * values of the arguments and the ranges specified.
 */
static int dplasma_is_valid( dplasma_execution_context_t* exec_context )
{
    dplasma_t* function = exec_context->function;
    int i, rc, min, max;
    
    for( i = 0; i < function->nb_locals; i++ ) {
        symbol_t* symbol = function->locals[i];
        
        rc = expr_eval( symbol->min, exec_context->locals, MAX_LOCAL_COUNT, &min );
        if( EXPR_SUCCESS != rc ) {
            fprintf(stderr, " Cannot evaluate the min expression for symbol %s\n", symbol->name);
            return rc;
        }
        rc = expr_eval( symbol->max, exec_context->locals, MAX_LOCAL_COUNT, &max );
        if( EXPR_SUCCESS != rc ) {
            fprintf(stderr, " Cannot evaluate the max expression for symbol %s\n", symbol->name);
            return rc;
        }
        if( (exec_context->locals[i].value < min) ||
           (exec_context->locals[i].value > max) ) {
            char tmp[128];
            fprintf( stderr, "Function %s is not a valid instance.\n",
                    dplasma_service_to_string(exec_context, tmp, 128) );
            return -1;
        }
    }
    return 0;
}

/**
 * Release all OUT dependencies for this particular instance of the service.
 */
int dplasma_release_OUT_dependencies( dplasma_execution_unit_t* eu_context,
                                      const dplasma_execution_context_t* restrict origin,
                                      const param_t* restrict origin_param,
                                      dplasma_execution_context_t* restrict exec_context,
                                      const param_t* restrict dest_param,
                                      int forward_remote )
{
    dplasma_t* function = exec_context->function;
    dplasma_dependencies_t *deps, **deps_location, *last_deps;
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif
    int i, actual_loop, rc;
    static int execution_step = 2;

    if( 0 == function->nb_locals ) {
        /* special case for the IN/OUT objects */
        return 0;
    }

    DEBUG(("Activate dependencies for %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    deps_location = &(function->deps);
    deps = *deps_location;
    last_deps = NULL;

    for( i = 0; i < function->nb_locals; i++ ) {
    restart_validation:
        rc = dplasma_symbol_validate_value( function->locals[i],
                                            (const expr_t**)function->preds,
                                            exec_context->locals );
        if( 0 != rc ) {
#if defined(DISTRIBUTED) && 0
            /* This is a valid value for this parameter, but it is executed 
             * on a remote resource according to the data mapping 
             */
            if(EXPR_FAILURE_CANNOT_EVALUATE_RANGE == rc)
            {
                if(forward_remote)
                {
                   dplasma_remote_dep_activate(eu_context, origin, origin_param, exec_context, dest_param);
                }
            }
#endif
            /* This is not a valid value for this parameter on this host. 
             * Try the next one */
        pick_next_value:
            exec_context->locals[i].value++;
            if( exec_context->locals[i].value > exec_context->locals[i].max ) {
                exec_context->locals[i].value = exec_context->locals[i].min;
                if( --i < 0 ) {
                    /* No valid value has been found. Return ! */
                    return -1;
                }
                deps = deps->prev;
                last_deps = deps;
                goto pick_next_value;
            }
            if( 0 == i ) {
                deps_location = &(function->deps);
            } else {
                deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(i)]);
            }
            goto restart_validation;
        }

        if( NULL == (*deps_location) ) {
            int min, max, number;
            /* TODO: optimize this section (and the similar one few tens of lines down
             * the code) to work on local ranges instead of absolute ones.
             */
            dplasma_symbol_get_absolute_minimum_value( function->locals[i], &min );
            dplasma_symbol_get_absolute_maximum_value( function->locals[i], &max );
            /* Make sure we stay in the expected ranges */
            if( exec_context->locals[i].min < min ) {
                DEBUG(("Readjust the minimum range in function %s for argument %s from %d to %d\n",
                       function->name, exec_context->locals[i].sym->name, exec_context->locals[i].min, min));
                exec_context->locals[i].min = min;
                exec_context->locals[i].value = min;
            }
            if( exec_context->locals[i].max > max ) {
                DEBUG(("Readjust the maximum range in function %s for argument %s from %d to %d\n",
                       function->name, exec_context->locals[i].sym->name, exec_context->locals[i].max, max));
                exec_context->locals[i].max = max;
            }
            assert( (min <= exec_context->locals[i].value) && (max >= exec_context->locals[i].value) );
            number = max - min;
            DEBUG(("Allocate %d spaces for loop %s (min %d max %d)\n",
                   number, function->locals[i]->name, min, max));
            deps = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) +
                                                   number * sizeof(dplasma_dependencies_union_t));
            DPLASMA_STAT_INCREASE(mem_contexts,  sizeof(dplasma_dependencies_t) +
                                  number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD);
            deps->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL;
            deps->symbol = function->locals[i];
            deps->min = min;
            deps->max = max;
            deps->prev = last_deps; /* chain them backward */
            if( 0 == dplasma_atomic_cas(deps_location, (uintptr_t) NULL, (uintptr_t) deps) ) {
                /* Some other thread manage to set it before us. Not a big deal. */
                free(deps);
                DPLASMA_STAT_DECREASE(mem_contexts,  sizeof(dplasma_dependencies_t) +
                                      number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD);
                goto deps_created_by_another_thread;
            }
            if( NULL != last_deps ) {
                last_deps->flags = DPLASMA_DEPENDENCIES_FLAG_NEXT | DPLASMA_DEPENDENCIES_FLAG_ALLOCATED;
            }
        } else {
        deps_created_by_another_thread:
            deps = *deps_location;
            /* Make sure we stay in bounds */
            if( exec_context->locals[i].min < deps->min ) {
                DEBUG(("Readjust the minimum range in function %s for argument %s from %d to %d\n",
                       function->name, exec_context->locals[i].sym->name, exec_context->locals[i].min, deps->min));
                exec_context->locals[i].min = deps->min;
                exec_context->locals[i].value = deps->min;
            }
            if( exec_context->locals[i].max > deps->max ) {
                DEBUG(("Readjust the maximum range in function %s for argument %s from %d to %d\n",
                       function->name, exec_context->locals[i].sym->name, exec_context->locals[i].max, deps->max));
                exec_context->locals[i].max = deps->max;
            }
        }

        DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
               exec_context->locals[i].value, CURRENT_DEPS_INDEX(i)));
        deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(i)]);
        last_deps = deps;
    }

    actual_loop = function->nb_locals - 1;
    while(1) {
#if defined(DPLASMA_GRAPHER) || 1
        int first_encounter = 0;
#endif  /* defined(DPLASMA_GRAPHER) */
        int updated_deps, mask;

        if( 0 != dplasma_is_valid(exec_context) ) {
            char tmp[128], tmp1[128];
            dplasma_service_to_string(origin, tmp, 128);
            dplasma_service_to_string(exec_context, tmp1, 128);
            fprintf( stderr, "Output dependencies of %s generate an invalid call to %s for param %s\n",
                     tmp, tmp1, dest_param->name );
            goto next_value;
        }

#if !defined(NDEBUG)
        if( deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] & dest_param->param_mask ) {
            char tmp[128], tmp1[128];
            fprintf( stderr, "Output dependencies %2x from %s (param %s) activate an already existing dependency %2x on %s (param %s)\n",
                     dest_param->param_mask, dplasma_service_to_string(origin, tmp, 128), origin_param->name,
                     deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)],
                     dplasma_service_to_string(exec_context, tmp1, 128),  dest_param->name );
        }
        assert( 0 == (deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] & dest_param->param_mask) );
#endif  /* !defined(NDEBUG) */
        mask = DPLASMA_DEPENDENCIES_HACK_IN | dest_param->param_mask;
        /* Mark the dependencies and check if this particular instance can be executed */
        if( !(DPLASMA_DEPENDENCIES_HACK_IN & deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)]) ) {
            mask |= dplasma_check_IN_dependencies( exec_context );
            if( mask > 0 ) {
                DEBUG(("Activate IN dependencies with mask 0x%02x\n", mask));
            }
#if defined(DPLASMA_GRAPHER) || 1
            first_encounter = 1;
#endif  /* defined(DPLASMA_GRAPHER) */
        }

        updated_deps = dplasma_atomic_bor( &deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)],
                                           mask);

        if( (updated_deps & function->dependencies_mask) == function->dependencies_mask ) {
#if defined(DPLASMA_GRAPHER) || 1
            if( NULL != __dplasma_graph_file ) {
                char tmp[128];
                fprintf(__dplasma_graph_file,
                        "%s [label=\"%s=>%s\" color=\"%s\" style=\"%s\" headlabel=%d]\n", dplasma_dependency_to_string(origin, exec_context, tmp, 128),
                        origin_param->name, dest_param->name, (first_encounter ? "#00FF00" : "#FF0000"), "solid", execution_step);
                fflush(__dplasma_graph_file);
            }
#endif  /* defined(DPLASMA_GRAPHER) */
            execution_step++;

#if !defined(NDEBUG)
            {
                int success, tmp_mask;
                do {
                    tmp_mask = deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)];
                    success = dplasma_atomic_cas( &deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)],
                                                  tmp_mask, (tmp_mask | (1<<30)) );
                    if( !success || (tmp_mask & (1<<30)) ) {
                        char tmp[128];
                        fprintf(stderr, "I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice !!!\n",
                                success, tmp_mask, dplasma_service_to_string(exec_context, tmp, 128));
                        assert(0);
                    }
                } while (0);
            }
#endif  /* !defined(NDEBUG) */
            /* This service is ready to be executed as all dependencies are solved. Let the
             * scheduler knows about this and keep going.
             */
            dplasma_schedule(eu_context->master_context, exec_context);
        } else {
            DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x: real 0x%02x)\n",
                   dplasma_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_mask,
                   (int)(updated_deps & (~DPLASMA_DEPENDENCIES_HACK_IN)),
                   (int)(updated_deps)));
#if defined(DPLASMA_GRAPHER) || 1
            if( NULL != __dplasma_graph_file ) {
                char tmp[128];
                fprintf(__dplasma_graph_file,
                        "%s [label=\"%s=>%s\" color=\"%s\" style=\"%s\"]\n", dplasma_dependency_to_string(origin, exec_context, tmp, 128),
                        origin_param->name, dest_param->name, (first_encounter ? "#00FF00" : "#FF0000"), "dashed");
                fflush(__dplasma_graph_file);
            }
#endif  /* defined(DPLASMA_GRAPHER) */
        }

    next_value:
        /* Go to the next valid value for this loop context */
        exec_context->locals[actual_loop].value++;
        if( exec_context->locals[actual_loop].max < exec_context->locals[actual_loop].value ) {
            /* We're out of the range for this variable */
            int current_loop = actual_loop;
        one_loop_up:
            DEBUG(("Loop index %d based on %s failed to get next value. Going up ...\n",
                   actual_loop, function->locals[actual_loop]->name));
            if( 0 == actual_loop ) {  /* we're done */
                goto end_of_all_loops;
            }
            actual_loop--;  /* one level up */
            deps = deps->prev;

            exec_context->locals[actual_loop].value++;
            if( exec_context->locals[actual_loop].max < exec_context->locals[actual_loop].value ) {
                goto one_loop_up;
            }
            DEBUG(("Keep going on the loop level %d (symbol %s value %d)\n", actual_loop,
                   function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(actual_loop)]);
            DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
                   exec_context->locals[actual_loop].value, CURRENT_DEPS_INDEX(actual_loop)));
            for( actual_loop++; actual_loop <= current_loop; actual_loop++ ) {
                exec_context->locals[actual_loop].value = exec_context->locals[actual_loop].min;
                last_deps = deps;  /* save the deps */
                if( NULL == *deps_location ) {
                    int min, max, number;
                    dplasma_symbol_get_absolute_minimum_value( function->locals[actual_loop], &min );
                    dplasma_symbol_get_absolute_maximum_value( function->locals[actual_loop], &max );
                    number = max - min;
                    DEBUG(("Allocate %d spaces for loop %s index %d value %d (min %d max %d)\n",
                           number, function->locals[actual_loop]->name, CURRENT_DEPS_INDEX(actual_loop-1),
                           exec_context->locals[actual_loop].value, min, max));
                    deps = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) +
                                                           number * sizeof(dplasma_dependencies_union_t));
                    DPLASMA_STAT_INCREASE(mem_contexts, sizeof(dplasma_dependencies_t) +
                                          number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD);
                    deps->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL;
                    deps->symbol = function->locals[actual_loop];
                    deps->min = min;
                    deps->max = max;
                    deps->prev = last_deps; /* chain them backward */
                    /**
                     * If we fail then the dependencies array has been allocated by another
                     * thread. Keep going.
                     */
                    if( !dplasma_atomic_cas(deps_location, (uintptr_t) NULL, (uintptr_t) deps) ) {
                        DPLASMA_STAT_DECREASE(mem_contexts, sizeof(dplasma_dependencies_t) +
                                              number * sizeof(dplasma_dependencies_union_t) + STAT_MALLOC_OVERHEAD);
                        free(deps);
                    }
                }
                deps = *deps_location;
                deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(actual_loop)]);
                DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
                       exec_context->locals[actual_loop].value, CURRENT_DEPS_INDEX(actual_loop)));
                last_deps = deps;

                DEBUG(("Loop index %d based on %s get first value %d\n", actual_loop,
                       function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            }
            actual_loop = current_loop;  /* go back to the original loop */
        } else {
            DEBUG(("Loop index %d based on %s get next value %d\n", actual_loop,
                   function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
        }
    }
 end_of_all_loops:

    return 0;
}

