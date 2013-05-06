#include "pins_print_steals.h"
#include "dague/mca/pins/pins.h"
#include <papi.h>
#include "debug.h"
#include "execution_unit.h"

static void pins_init_print_steals(dague_context_t * master_context);
static void pins_fini_print_steals(dague_context_t * master_context);
static void pins_thread_init_print_steals(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_print_steals(dague_execution_unit_t * exec_unit);

const dague_pins_module_t dague_pins_print_steals_module = {
    &dague_pins_print_steals_component,
    {
	    pins_init_print_steals,
	    pins_fini_print_steals,
	    NULL,
	    NULL,
	    pins_thread_init_print_steals,
	    pins_thread_fini_print_steals
    }
};

static void stop_print_steals_count(dague_execution_unit_t * exec_unit, 
                                   dague_execution_context_t * exec_context, void * data);

static parsec_pins_callback * select_end_prev; // courtesy call to previously-registered callback

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *       \
                              exec_unit->virtual_process->dague_context->nb_vp + \
                              exec_unit->th_id )

static int total_cores;

static void pins_init_print_steals(dague_context_t * master) {
	select_end_prev   = PINS_REGISTER(SELECT_END,   stop_print_steals_count);
	total_cores = master->nb_vp * master->virtual_processes[0]->nb_cores;
}

static void pins_fini_print_steals(dague_context_t * master) {
	PINS_REGISTER(SELECT_END,   select_end_prev);
}

static void pins_thread_init_print_steals(dague_execution_unit_t * exec_unit) {
    exec_unit->steal_counters = calloc(sizeof(long), total_cores + 2);
}

static void pins_thread_fini_print_steals(dague_execution_unit_t * exec_unit) {
    for (int k = 0; k < total_cores + 2; k++)
        printf("%7ld ", exec_unit->steal_counters[k]);
    printf("\n");
    free(exec_unit->steal_counters);
}

static void stop_print_steals_count(dague_execution_unit_t * exec_unit, 
                                   dague_execution_context_t * exec_context, 
                                   void * data) {
	unsigned long long victim_core_num = (unsigned long long)data;
	unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp 
	                            * exec_unit->virtual_process->nb_cores);

	if (exec_context != NULL)
		exec_unit->steal_counters[victim_core_num] += 1;
	else
		exec_unit->steal_counters[victim_core_num + 1] += 1;

	// keep the contract with the previous registrant
	if (select_end_prev != NULL) {
		(*select_end_prev)(exec_unit, exec_context, data);
	}
}

