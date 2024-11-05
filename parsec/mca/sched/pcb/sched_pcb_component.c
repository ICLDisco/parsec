/*
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "parsec/parsec_config.h"
#include "parsec/runtime.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/pcb/sched_pcb.h"
#include "parsec/papi_sde.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/mca_param.h"

#if defined(PARSEC_HAVE_HWLOC)
#include "parsec/parsec_hwloc.h"
#endif

/*
 * Local function
 */
static int sched_pcb_component_query(mca_base_module_t **module, int *priority);
static int sched_pcb_component_register(void);

int sched_pcb_sharing_level = 1;
int sched_pcb_group_mask = 0x7f000000;
int sched_pcb_group_shift = 24;
/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_sched_base_component_t parsec_sched_pcb_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_SCHED_BASE_VERSION_2_0_0,

        /* Component name and version */
        "pcb",
        "", /* options */
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL, /*< No open: sched_pcb is always available, no need to check at runtime */
        NULL, /*< No close: open did not allocate any resource, no need to release them */
        sched_pcb_component_query, 
        /*< specific query to return the module and add it to the list of available modules */
        sched_pcb_component_register, /*< Register at least the SDE events */
        "", /*< no reserve */
    },
    {
        /* The component has no metada */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};
mca_base_component_t *sched_pcb_static_component(void)
{
    return (mca_base_component_t *)&parsec_sched_pcb_component;
}

static int sched_pcb_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_sched_pcb_module;
    *priority = 2;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}

static int sched_pcb_component_register(void)
{
    PARSEC_PAPI_SDE_DESCRIBE_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=PCB",
                              "the number of pending tasks for the PCB scheduler");
    PARSEC_PAPI_SDE_DESCRIBE_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=<VPID>::SCHED=PCB",
                              "the number of pending tasks that end up in the virtual process <VPID> for the LFQ scheduler");
    sched_pcb_sharing_level = 1;
#if defined(PARSEC_HAVE_HWLOC)
    sched_pcb_sharing_level = parsec_hwloc_nb_levels()-1;
    parsec_mca_param_reg_int_name("sched_pcb", "sharing_level",
                                  "Defines at what level threads share the same task list for the Priority Controlled Binding scheduler. "
                                  "Level 1 means each thread has its own task list, level 2 looks one level above in the HWLOC hierarchy, etc...",
                                  false, false, parsec_hwloc_nb_levels()-1, &sched_pcb_sharing_level);
    if(sched_pcb_sharing_level <= 0)
        sched_pcb_sharing_level = 1;
    if(sched_pcb_sharing_level >= parsec_hwloc_nb_levels())
        sched_pcb_sharing_level = parsec_hwloc_nb_levels()-1;
#endif
    parsec_mca_param_reg_int_name("sched_pcb", "group_mask",
                                  "Defines what bits of the priority are used to designate a process group. Other bits are of the priority value "
                                  "are used to define the priority of the task within that group.",
                                  false, false, 0x7f000000, &sched_pcb_group_mask);
    if(sched_pcb_group_mask != 0x7f000000) {
        sched_pcb_group_shift = 0;
        while( (unsigned int)sched_pcb_group_shift < 8*sizeof(int) &&
               (((sched_pcb_group_mask >> sched_pcb_group_shift) & 1) == 0) )
            sched_pcb_group_shift++;
        if(sched_pcb_group_shift == 8*sizeof(int)) {
            parsec_warning("Priority Controlled Binding Scheduler (sched_pcb): sched_pcb_group_mask is set to 0. Scheduler might not work as intended.");
        }
    }

    return MCA_SUCCESS;
}
