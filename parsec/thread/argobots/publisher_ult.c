/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <netdb.h>
#include <sys/types.h>

#include "parsec_config.h"
#include "parsec_internal.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/profiling.h"
#include "parsec/datarepo.h"
#include "parsec/bindthread.h"
#include "parsec/execution_unit.h"
#include "parsec/vpmap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/os-spec-timing.h"
#include "remote_dep.h"

#include "parsec/ayudame.h"
#include "parsec/constants.h"
#include "parsec/thread/thread.h"
#include "parsec/mca/pins/alperf/pins_alperf.h"


static inline void handle_error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}


static inline size_t check_alperf_contrib(char *message, size_t size, int n) {
    (void) n;
    if ( pins_alperf_counter_store.nb_counters > 0 ) {
        uint64_t t;
        double ts;
        size_t blen;

        t = parsec_profiling_get_time();
#if defined(HAVE_CLOCK_GETTIME)
        ts = (double)t / 1000000000.0;
#elif defined(__IA64) || defined(__X86) || defined(__bgp__)
        /*time scale in GTics*/
        ts = (double)t / 1000000000.0;
#else
        ts = (double)t / 1000000.0;
#endif

        *PINS_ALPERF_DATE = ts;
        assert( pins_alperf_counter_store_size() <= size );
        blen = size < pins_alperf_counter_store_size() ? size : pins_alperf_counter_store_size();
        memcpy(message, pins_alperf_counter_store.counters, blen);

        int i;
        char unit[16];
        memset(unit, 0, 16);
        sprintf(unit, "%s", "PaRSEC.");
        size_t size = strlen(unit);
        for (i = 0; i < pins_alperf_counter_store.nb_counters; ++i) {
            sprintf(unit+size, "%s", PINS_ALPERF_COUNTER(i)->name);
            int j;
            uint64_t val = 0;
            for ( j = 0; j < pins_alperf_counter_store.nb_counters; ++j )
                val += PINS_ALPERF_COUNTER(i)->value_per_eu[j];

#if defined( PUBLISHER_THREAD )
            ABT_event_prof_publish(unit, val, 0.);
#endif
        }
        return blen; /*change the value returned if you pack more than one key*/
    }
    return 0;
}

static inline void check_stats(parsec_context_t *context, parsec_time_t *last_update, int *quit, int force)
{
    parsec_time_t now = take_time();
    uint64_t interval = diff_time( *last_update, now );
    uint64_t threshold = 1;
    (void)threshold;
    (void)quit;
    char time_unit;
    time_unit = 's';

#if defined(HAVE_CLOCK_GETTIME)
    threshold *= 1000000000;
#elif defined(__IA64) || defined(__X86) || defined(__bgp__)
    time_unit = 'c';
    threshold *= 1000000000;
#else
    threshold *= 1000000;
#endif

    threshold *= 1;

    if( threshold < interval || force ) {
#if defined( PUBLISHER_THREAD )
        ABT_event_prof_stop();
#endif
        int i, n = 0;
        for(i=0; i<context->nb_vp; ++i)
            n += context->virtual_processes[i]->nb_cores;

        int k = 0;
        char *message = (char*)malloc(4096);
        memset(message, 0, 4096);

        k += check_alperf_contrib(message+k, 4096-k, n);

        *last_update = now;
#if defined( PUBLISHER_THREAD )
        if ( !force )
            ABT_event_prof_start();
#endif

    }
}


void* __publisher_thread(void* arguments)
{
    (void)arguments;
    parsec_context_t *context = (parsec_context_t*)arguments;

    int quit = 0;

#if defined ( PUBLISHER_THREAD )
    ABT_event_prof_start();
#endif

    parsec_time_t last_update = take_time();
    while(!quit) {
        check_stats(context, &last_update, &quit, 0);
        ABT_thread_yield();
    }

    check_stats(context, &last_update, &quit, 1);

    return 0;
}
