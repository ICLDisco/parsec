#include <errno.h>
#include <stdio.h>
#include "parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "pins_alperf.h"
#include "parsec/profiling.h"
#include "parsec/execution_unit.h"
#include "parsec/utils/mca_param.h"
#include <string.h>
#include "parsec/sys/atomic.h"

/* init functions */
static void pins_init_alperf(parsec_context_t * master_context);
static void pins_fini_alperf(parsec_context_t * master_context);

/* PINS callbacks */
static void alperf_exec_count_end(parsec_execution_unit_t * exec_unit,
                                         parsec_execution_context_t * exec_context,
                                         void * data);

const parsec_pins_module_t parsec_pins_alperf_module = {
    &parsec_pins_alperf_component,
    {
        pins_init_alperf,
        pins_fini_alperf,
        NULL,
        NULL,
        NULL,
        NULL
    }
};

static parsec_pins_callback * exec_end_prev; /* courtesy calls to previously-registered cbs */

static void pins_init_alperf(parsec_context_t * master_context) {
    char* mca_param_string, *c, *s;
    int nb_eu = 0;
    int vpid;
    (void)master_context;

    parsec_mca_param_reg_string_name("pins", "alperf_events",
                                    "Application Level Performance events to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);
    pins_alperf_counter_store.nb_counters = (*mca_param_string != '\0');
    for(c = mca_param_string; *c != '\0'; c++) {
        if( *c == ',' )
            pins_alperf_counter_store.nb_counters++;
    }
    if( pins_alperf_counter_store.nb_counters > 0 ) {
        for(vpid = 0; vpid < master_context->nb_vp; vpid++) {
            nb_eu += master_context->virtual_processes[vpid]->nb_cores;
        }
        pins_alperf_counter_store.ct_size = sizeof(pins_alperf_counter_t) + (nb_eu - 1) * sizeof(uint64_t);
        pins_alperf_counter_store.counters = (pins_alperf_counter_t*)realloc(
            pins_alperf_counter_store.counters,
            sizeof(double) + sizeof(uint16_t) + pins_alperf_counter_store.ct_size * pins_alperf_counter_store.nb_counters);
        *PINS_ALPERF_DATE = 0; /**< Just to reset the timer */
        *PINS_ALPERF_NBCOUNTERS = pins_alperf_counter_store.nb_counters; /**< We copy this to make the message self-contained */
        pins_alperf_counter_store.nb_counters = 0;
        s = mca_param_string;
        while( ( c = strtok_r(s, ",", &s) ) != NULL ) {
            strncpy( PINS_ALPERF_COUNTER(pins_alperf_counter_store.nb_counters)->name, c, PINS_ALPERF_COUNTER_NAME_MAX);
            memset(PINS_ALPERF_COUNTER(pins_alperf_counter_store.nb_counters)->value_per_eu, 0, nb_eu*sizeof(uint64_t));
            pins_alperf_counter_store.nb_counters++;
        }
        exec_end_prev   = PINS_REGISTER(EXEC_END, alperf_exec_count_end);
    }
}

static void pins_fini_alperf(parsec_context_t * master_context) {
    (void)master_context;
    // replace original registrants
    if( pins_alperf_counter_store.nb_counters ) {
        pins_alperf_counter_store.nb_counters = 0;
        free(pins_alperf_counter_store.counters);
        PINS_REGISTER(EXEC_END,   exec_end_prev);
    }
}

/*
 PINS CALLBACKS
 */

static void alperf_exec_count_end(parsec_execution_unit_t * exec_unit,
                                  parsec_execution_context_t * exec_context,
                                  void * data) {
    int i;
    const parsec_property_t *dp;
    uint64_t ta;

    for(dp = exec_context->function->properties;
        dp->name != NULL;
        dp++) {
        /** Slow version to test */
        for(i = 0; i < pins_alperf_counter_store.nb_counters; i++) {
            if( !strcmp(PINS_ALPERF_COUNTER(i)->name, "task") ) {
                PINS_ALPERF_COUNTER(i)->value_per_eu[ exec_unit->th_id ]++;
            } else if( !strcmp(dp->name, PINS_ALPERF_COUNTER(i)->name) ) {
                assert( dp->expr->op == EXPR_OP_INLINE );
                ta = dp->expr->inline_func32(exec_context->parsec_handle, exec_context->locals);
                PINS_ALPERF_COUNTER(i)->value_per_eu[ exec_unit->th_id ] += ta;
            }
        }
    }

    // keep the contract with the previous registrant
    if (exec_end_prev != NULL) {
        (*exec_end_prev)(exec_unit, exec_context, data);
    }
}
