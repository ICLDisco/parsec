/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <string.h>

#include "parsec/sys/tls.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"
#include "parsec/class/list.h"
#include "parsec/class/parsec_rwlock.h"

#include "parsec/papi_sde.h"

#include "parsec/utils/mca_param.h"
#include "parsec/mca/mca_repository.h"

static papi_handle_t parsec_papi_sde_handle = NULL;

typedef struct {
    parsec_list_item_t super;
    long long int      counters[PARSEC_PAPI_SDE_NB_BASIC_COUNTERS];
} parsec_thread_sde_counters_t;

PARSEC_OBJ_CLASS_DECLARATION(parsec_thread_sde_counters_t);
PARSEC_OBJ_CLASS_INSTANCE(parsec_thread_sde_counters_t, parsec_list_item_t, NULL, NULL);

static PARSEC_TLS_DECLARE(parsec_papi_sde_basic_counters_tls);
/* We protect the sde_threads list with an external rwlock,
   because most operations do not modify the list and can be done in parallel */
static parsec_list_t sde_threads;
static parsec_atomic_rwlock_t sde_threads_lock = PARSEC_RWLOCK_UNLOCKED;

typedef struct {
    char                          *name;
    char                          *description;
    int                            basic;
    int                            instant;
} hl_counter_type_t;

hl_counter_type_t hl_counters[PARSEC_PAPI_SDE_NB_HL_COUNTERS] = {
    { "MEM_ALLOC",
      "the amount of temporary memory allocated by PaRSEC to communicate "
      " user's data (typically to receive a data sent by a remote task)",
      1, 1 },
    { "MEM_USED",
      "the amount of temporary memory currently used by PaRSEC to communicate"
      " user's data (typically how much bytes have been allocated to host"
      " data sent by a remote task that are currently needed by active or pending tasks)",
      1, 1 },
    { "TASKS_ENABLED",
      "the number of tasks that became ready at this time",
      1, 0 },
    { "TASKS_RETIRED",
      "the numbre of tasks that completed at this time",
      1, 0},
    { "SCHEDULER::PENDING_TASKS",
      "the number of pending tasks. A task is said pending if it is "
      "ready to execute but waits for execution in one of the scheduler queues.",
      0, 1 }
};

static long long int parsec_papi_sde_base_counter_cb(void *arg);

void parsec_papi_sde_init(void)
{
    parsec_papi_sde_hl_counters_t cnt;

    parsec_papi_sde_handle = papi_sde_init("PARSEC");

    PARSEC_TLS_KEY_CREATE(parsec_papi_sde_basic_counters_tls);
    PARSEC_OBJ_CONSTRUCT(&sde_threads, parsec_list_t);
    parsec_atomic_rwlock_init( &sde_threads_lock );

    for(cnt = PARSEC_PAPI_SDE_FIRST_BASIC_COUNTER; cnt <= PARSEC_PAPI_SDE_LAST_BASIC_COUNTER; cnt++) {
        if( hl_counters[cnt].basic ) {
            if( hl_counters[cnt].instant ) {
                papi_sde_register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                             PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_base_counter_cb, (void*)cnt);
            } else {
                papi_sde_register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO,
                                             PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_base_counter_cb, (void*)cnt);
            }
            papi_sde_describe_counter(parsec_papi_sde_handle, hl_counters[cnt].name, hl_counters[cnt].description);
        }
    }
}

static papi_sde_fptr_struct_t *parsec_papi_sde_fptr = NULL;

papi_handle_t papi_sde_hook_list_events(papi_sde_fptr_struct_t *fptr_struct){
    parsec_papi_sde_hl_counters_t cnt;

    parsec_papi_sde_handle = fptr_struct->init("PARSEC");
    for(cnt = PARSEC_PAPI_SDE_FIRST_BASIC_COUNTER; cnt < PARSEC_PAPI_SDE_NB_HL_COUNTERS; cnt++) {
        /* It's fine to pass the wrong callback when registering in hook_list_events, because 
         * these callbacks will not be called by papi_native_avail. In order to expose all events,
         * we use parsec_papi_sde_base_counter_cb everywhere. */
        if( hl_counters[cnt].instant ) {
            fptr_struct->register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                                PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_base_counter_cb, (void*)cnt);
        } else {
            fptr_struct->register_fp_counter(parsec_papi_sde_handle, hl_counters[cnt].name, PAPI_SDE_RO,
                                                PAPI_SDE_int, (papi_sde_fptr_t)parsec_papi_sde_base_counter_cb, (void*)cnt);
        }
        fptr_struct->describe_counter(parsec_papi_sde_handle, hl_counters[cnt].name, hl_counters[cnt].description);
    }
    parsec_papi_sde_fptr = fptr_struct;
    parsec_mca_param_init();
    mca_components_repository_init();

    mca_base_component_t **scheds;
    mca_base_module_t    *new_scheduler = NULL;
    mca_base_component_t *new_component = NULL;

    scheds = mca_components_open_bytype( "sched" );
    mca_components_query(scheds,
                         &new_scheduler,
                         &new_component);
    mca_components_close(scheds);

    parsec_papi_sde_fptr = NULL;
    return parsec_papi_sde_handle;
}

void parsec_papi_sde_fini(void)
{
    parsec_list_item_t *it;
    parsec_papi_sde_hl_counters_t cnt;

    for(cnt = PARSEC_PAPI_SDE_FIRST_BASIC_COUNTER; cnt < PARSEC_PAPI_SDE_NB_HL_COUNTERS; cnt++) {
        papi_sde_unregister_counter(parsec_papi_sde_handle, hl_counters[cnt].name);
    }

    papi_sde_shutdown(parsec_papi_sde_handle);
    
    parsec_atomic_rwlock_wrlock( &sde_threads_lock );
    while(NULL != (it = parsec_list_nolock_pop_front(&sde_threads)) ) {
        PARSEC_OBJ_RELEASE(it);
    }
    PARSEC_OBJ_DESTRUCT(&sde_threads);
    parsec_papi_sde_handle = NULL;
}

void parsec_papi_sde_thread_init(void)
{
    parsec_thread_sde_counters_t *new_counters;
    
    /* Manage gracefully the case where the communication thread is not created
     * but communications interleave with scheduling and computation */
    if( PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls) != NULL )
        return;

    /* General case: allocate the thread's TLS storage, and associate the pointer
     * to a place in the global array and the TLS key */
    new_counters = PARSEC_OBJ_NEW(parsec_thread_sde_counters_t);
    memset(new_counters->counters, 0, sizeof(long long int) * PARSEC_PAPI_SDE_NB_BASIC_COUNTERS);

    parsec_atomic_rwlock_wrlock( &sde_threads_lock );
    PARSEC_TLS_SET_SPECIFIC(parsec_papi_sde_basic_counters_tls, new_counters);
    parsec_list_nolock_push_back( &sde_threads, &new_counters->super );
    parsec_atomic_rwlock_wrunlock( &sde_threads_lock );
}

void parsec_papi_sde_thread_fini(void)
{
    parsec_thread_sde_counters_t *my_counters;

    my_counters = PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls);
    if( NULL == my_counters )
        return;
    
    parsec_atomic_rwlock_wrlock( &sde_threads_lock );
    parsec_list_nolock_remove(&sde_threads, &my_counters->super);
    PARSEC_TLS_SET_SPECIFIC(parsec_papi_sde_basic_counters_tls, NULL);
    parsec_atomic_rwlock_wrunlock( &sde_threads_lock );
    PARSEC_OBJ_RELEASE(my_counters);
}

void parsec_papi_sde_counter_set(parsec_papi_sde_hl_counters_t cnt, long long int value)
{
    parsec_thread_sde_counters_t *tls_counters;

    parsec_atomic_rwlock_rdlock( &sde_threads_lock );
    tls_counters = PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls);
    if( NULL == tls_counters ) {
        parsec_atomic_rwlock_rdunlock( &sde_threads_lock );
        return;
    }
    tls_counters->counters[cnt] = value;
    parsec_atomic_rwlock_rdunlock( &sde_threads_lock );
}

void parsec_papi_sde_counter_add(parsec_papi_sde_hl_counters_t cnt, long long int value)
{
    parsec_thread_sde_counters_t *tls_counters;

    parsec_atomic_rwlock_rdlock( &sde_threads_lock );
    tls_counters = PARSEC_TLS_GET_SPECIFIC(parsec_papi_sde_basic_counters_tls);
    if( NULL == tls_counters ) {
        parsec_atomic_rwlock_rdunlock( &sde_threads_lock );
        return;
    }
    tls_counters->counters[cnt] += value;
    parsec_atomic_rwlock_rdunlock( &sde_threads_lock );
}

static long long int parsec_papi_sde_base_counter_cb(void *arg)
{
    parsec_papi_sde_hl_counters_t cnt = (parsec_papi_sde_hl_counters_t)(uintptr_t)arg;
    long long int sum = 0;
    
    parsec_atomic_rwlock_rdlock( &sde_threads_lock );
    for(parsec_list_item_t *it = PARSEC_LIST_ITERATOR_FIRST(&sde_threads);
        it != PARSEC_LIST_ITERATOR_LAST(&sde_threads);
        it = PARSEC_LIST_ITERATOR_NEXT(it)) {
        parsec_thread_sde_counters_t *c = (parsec_thread_sde_counters_t*)it;
        sum += c->counters[cnt];
    }
    parsec_atomic_rwlock_rdunlock( &sde_threads_lock );
    return sum;
}

void parsec_papi_sde_unregister_counter(const char *format, ...)
{
    va_list ap;
    char name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
    va_start(ap, format);
    vsnprintf(name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, format, ap);
    va_end(ap);

    papi_sde_unregister_counter(parsec_papi_sde_handle, name);
}

void parsec_papi_sde_register_fp_counter(const char *event_name, int flags, int type, papi_sde_fptr_t fn, void *data)
{
    papi_sde_register_fp_counter(parsec_papi_sde_handle, event_name, (flags), (type), fn, data);
}

void parsec_papi_sde_register_counter(const char *event_name, int flags, int type, long long int *ptr)
{
    papi_sde_register_counter(parsec_papi_sde_handle, event_name, (flags), (type), ptr);
}

void parsec_papi_sde_add_counter_to_group(const char *event_name, const char *group_name, int operand)
{
    papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name, group_name, (operand));
}

void parsec_papi_sde_describe_counter(const char *event_name, const char *description)
{
    if(NULL == parsec_papi_sde_fptr)
        papi_sde_describe_counter(parsec_papi_sde_handle, event_name, description);
    else {
        parsec_papi_sde_fptr->register_fp_counter(parsec_papi_sde_handle, event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_int, 
                                                  (papi_sde_fptr_t)NULL, NULL);
        parsec_papi_sde_fptr->describe_counter(parsec_papi_sde_handle, event_name, description);
    }
}
