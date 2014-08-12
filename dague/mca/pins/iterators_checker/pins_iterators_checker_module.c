#include <errno.h>
#include <stdio.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "pins_iterators_checker.h"
#include "profiling.h"
#include "execution_unit.h"

/* init functions */
static void pins_init_iterators_checker(dague_context_t * master_context);
static void pins_fini_iterators_checker(dague_context_t * master_context);

/* PINS callbacks */
static void iterators_checker_exec_count_begin(dague_execution_unit_t * exec_unit,
                                               dague_execution_context_t * exec_context,
                                               void * data);
static void iterators_checker_exec_count_end(dague_execution_unit_t * exec_unit,
                                             dague_execution_context_t * exec_context,
                                             void * data);

const dague_pins_module_t dague_pins_iterators_checker_module = {
    &dague_pins_iterators_checker_component,
    {
        pins_init_iterators_checker,
        pins_fini_iterators_checker,
        NULL,
        NULL,
        NULL,
        NULL
    }
};

static parsec_pins_callback * exec_begin_prev; /* courtesy calls to previously-registered cbs */

static void pins_init_iterators_checker(dague_context_t * master_context) {
    (void)master_context;

    exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, iterators_checker_exec_count_begin);
}

static void pins_fini_iterators_checker(dague_context_t * master_context) {
    (void)master_context;
    // replace original registrants
    PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
}

/*
 PINS CALLBACKS
 */

#define TASK_STR_LEN 256

static dague_ontask_iterate_t print_link(dague_execution_unit_t *eu,
                                         const dague_execution_context_t *newcontext,
                                         const dague_execution_context_t *oldcontext,
                                         const dep_t* dep,
                                         dague_dep_data_description_t* data,
                                         int src_rank, int dst_rank, int dst_vpid,
                                         void *param)
{
    char  new_str[TASK_STR_LEN];
    char  old_str[TASK_STR_LEN];
    char *info = (char*)param;

    dague_snprintf_execution_context(old_str, TASK_STR_LEN, oldcontext);
    dague_snprintf_execution_context(new_str, TASK_STR_LEN, newcontext);

    fprintf(stderr, "PINS ITERATORS CHECKER::   %s that runs on rank %d, vpid %d is a %s of %s that runs on rank %d\n",
            new_str, dst_rank, dst_vpid, info, old_str, src_rank);

    return DAGUE_ITERATE_CONTINUE;
}

static void iterators_checker_exec_count_begin(dague_execution_unit_t * exec_unit,
                                               dague_execution_context_t * exec_context,
                                               void * data) 
{
    char  str[TASK_STR_LEN];

    if( exec_context->function->iterate_successors )
        exec_context->function->iterate_successors(exec_unit, exec_context, DAGUE_DEPENDENCIES_BITMASK, print_link, "successor");
    else {
        char  str[TASK_STR_LEN];
        dague_snprintf_execution_context(str, TASK_STR_LEN, exec_context);

        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no successor\n", str);
    }

    if( exec_context->function->iterate_predecessors )
        exec_context->function->iterate_predecessors(exec_unit, exec_context, DAGUE_DEPENDENCIES_BITMASK, print_link, "predecessor");
    else {
        char  str[TASK_STR_LEN];
        dague_snprintf_execution_context(str, TASK_STR_LEN, exec_context);

        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no predecessor\n", str);
    }

    // keep the contract with the previous registrant
    if (exec_begin_prev != NULL) {
        (*exec_begin_prev)(exec_unit, exec_context, data);
    }
}
