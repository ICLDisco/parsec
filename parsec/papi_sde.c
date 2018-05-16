#include "parsec/parsec_config.h"

#include <stdlib.h>

#include "parsec/papi_sde.h"
#include "parsec/sys/tls.h"
#include "parsec/papi_sde_interface.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

papi_handle_t parsec_papi_sde_handle = NULL;

PARSEC_TLS_DECLARE(parsec_papi_sde_basic_counters_tls);
static long long int **parsec_papi_sde_basic_counters;
static int parsec_papi_sde_basic_counters_nb_threads;
static volatile int32_t parsec_papi_sde_basic_counters_next_tid = 0;

typedef struct {
    char                          *name;
    char                          *description;
    int                            basic;
    int                            instant;
} hl_counter_type_t;

hl_counter_type_t hl_counters[PARSEC_PAPI_SDE_NB_HL_EVENTS] = {
    { "PARSEC::MEM_ALLOC",
      "the amount of temporary memory allocated by PaRSEC to communicate "
      " user's data (typically to receive a data sent by a remote task)",
      1, 1 },
    { "PARSEC::MEM_USED",
      "the amount of temporary memory currently used by PaRSEC to communicate"
      " user's data (typically how much bytes have been allocated to host"
      " data sent by a remote task that are currently needed by active or pending tasks)",
      1, 1 },
    { "PARSEC::SCHEDULER::PENDING_TASKS",
      "the number of pending tasks. A task is said pending if it is "
      "ready to execute but waits for execution in one of the scheduler queues.",
      0, 1 },
    { "PARSEC::TASKS_ENABLED",
      "the number of tasks that became ready at this time",
      1, 0 },
    { "PARSEC::TASKS_RETIRED",
      "the numbre of tasks that completed at this time",
      1, 0}
};

static void parsec_papi_sde_define_events(void);
static long long int parsec_papi_sde_simple_event(void *arg);

void parsec_papi_sde_init(void)
{
    parsec_papi_sde_handle = papi_sde_init("PARSEC");
    parsec_papi_sde_define_events();
}

void parsec_papi_sde_enable_basic_events(int nb_threads)
{    
    parsec_papi_sde_hl_counters_t cnt;

    PARSEC_TLS_KEY_CREATE(parsec_papi_sde_basic_counters_tls);
    parsec_papi_sde_basic_counters = (long long int**)calloc( sizeof(long long int*), nb_threads );
    parsec_papi_sde_basic_counters_nb_threads = nb_threads;

    for(cnt = PARSEC_PAPI_SDE_FIRST_EVENT; cnt <= PARSEC_PAPI_SDE_LAST_EVENT; cnt++) {
        if( hl_counters[cnt].basic ) {
            if( hl_counters[cnt].instant ) {
                papi_sde_register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                             PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_simple_event, (void*)cnt);
            } else {
                papi_sde_register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO,
                                             PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_simple_event, (void*)cnt);
            }
        }
    }
}

void parsec_papi_sde_thread_init(void)
{
    void *tls_events;
    int32_t tid;
    
    /* Manage gracefully the case where the communication thread is not created
     * but communications interleave with scheduling and computation */
    if( PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls) != NULL )
        return;
    
    /* General case: allocate the thread's TLS storage, and associate the pointer
     * to a place in the global array and the TLS key */
    tid = parsec_atomic_fetch_inc_int32(&parsec_papi_sde_basic_counters_next_tid);
    assert( tid >= 0 );
    if( tid >= parsec_papi_sde_basic_counters_nb_threads ) {
        parsec_warning("parsec::papi_sde\texpected up to %d threads, but %d are joining. Events from this threads will be ignored\n",
                       parsec_papi_sde_basic_counters_nb_threads, tid+1);
        return;
    }
    tls_events = calloc( sizeof(long long int), PARSEC_PAPI_SDE_NB_HL_EVENTS );
    PARSEC_TLS_SET_SPECIFIC(parsec_papi_sde_basic_counters_tls, tls_events);
    assert( NULL == parsec_papi_sde_basic_counters[tid] );
    parsec_papi_sde_basic_counters[tid] = tls_events;
}

void parsec_papi_sde_counter_set(parsec_papi_sde_hl_counters_t cnt, long long int value)
{
    long long int *tls_events = PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls);
    if( NULL == tls_events )
        return;
    tls_events[cnt] = value;
}

void parsec_papi_sde_counter_add(parsec_papi_sde_hl_counters_t cnt, long long int value)
{
    long long int *tls_events = PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls);
    if( NULL == tls_events )
        return;
    tls_events[cnt] += value;
}

static long long int parsec_papi_sde_simple_event(void *arg)
{
    parsec_papi_sde_hl_counters_t cnt = (parsec_papi_sde_hl_counters_t)(uintptr_t)arg;
    long long int sum = 0;
    int tid;

    for(tid = 0; tid < parsec_papi_sde_basic_counters_nb_threads; tid++) {
        if( NULL != parsec_papi_sde_basic_counters[tid] )
            sum += parsec_papi_sde_basic_counters[tid][cnt];
    }
    return sum;
}

static void parsec_papi_sde_define_events(void)
{
    parsec_papi_sde_hl_counters_t cnt;

    for(cnt = PARSEC_PAPI_SDE_FIRST_EVENT; cnt <= PARSEC_PAPI_SDE_LAST_EVENT; cnt++) {
        papi_sde_describe_counter(parsec_papi_sde_handle, hl_counters[cnt].name, hl_counters[cnt].description);
    }
}
