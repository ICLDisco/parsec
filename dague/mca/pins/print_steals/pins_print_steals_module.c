#include "pins_print_steals.h"
#include "dague/mca/pins/pins.h"
#ifdef HAVE_PAPI
#include <papi.h>
#endif
#include "debug.h"
#include "execution_unit.h"

static void pins_init_print_steals(dague_context_t* master_context);
static void pins_thread_init_print_steals(dague_execution_unit_t* exec_unit);
static void pins_thread_fini_print_steals(dague_execution_unit_t* exec_unit);

const dague_pins_module_t dague_pins_print_steals_module = {
    &dague_pins_print_steals_component,
    {
        pins_init_print_steals,
        NULL,
        NULL,
        NULL,
        pins_thread_init_print_steals,
        pins_thread_fini_print_steals
    }
};

typedef struct parsec_pins_print_steals_data_s {
    parsec_pins_next_callback_t cb_data;
    long steal_counters[1];
} parsec_pins_print_steals_data_t;

static void stop_print_steals_count(dague_execution_unit_t* exec_unit,
                                    dague_execution_context_t* exec_context,
                                    parsec_pins_next_callback_t* data);

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *      \
                               exec_unit->virtual_process->dague_context->nb_vp + \
                               exec_unit->th_id )

static int total_cores;

static void pins_init_print_steals(dague_context_t* master)
{
    total_cores = master->nb_vp * master->virtual_processes[0]->nb_cores;
}

static void pins_thread_init_print_steals(dague_execution_unit_t* exec_unit)
{
    parsec_pins_print_steals_data_t* event_cb =
        (parsec_pins_print_steals_data_t*)calloc(1, sizeof(parsec_pins_print_steals_data_t) +
                                                 (total_cores + 2) * sizeof(long));
    PINS_REGISTER(exec_unit, SELECT_END, stop_print_steals_count,
                  (parsec_pins_next_callback_t*)event_cb);
}

static void pins_thread_fini_print_steals(dague_execution_unit_t* exec_unit)
{
    parsec_pins_print_steals_data_t* event_cb;
    PINS_UNREGISTER(exec_unit, SELECT_END, stop_print_steals_count,
                  (parsec_pins_next_callback_t**)&event_cb);

    for (int k = 0; k < total_cores + 2; k++)
        printf("%7ld ", event_cb->steal_counters[k]);
    printf("\n");
    free(event_cb);
}

static void stop_print_steals_count(dague_execution_unit_t* exec_unit,
                                    dague_execution_context_t* exec_context,
                                    parsec_pins_next_callback_t* data)
{
    parsec_pins_print_steals_data_t* event_cb = (parsec_pins_print_steals_data_t*)data;
    /**
     * This is plain wrong and it could not have been working.
     */
    unsigned long long victim_core_num = 0;

    if (exec_context != NULL)
        event_cb->steal_counters[victim_core_num] += 1;
    else
        event_cb->steal_counters[victim_core_num + 1] += 1;
    (void)exec_unit;
}
