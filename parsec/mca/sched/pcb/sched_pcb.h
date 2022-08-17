/*
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

/**
 * @file
 *
 * Priority Controlled Binding scheduler
 * 
 *   This scheduler uses some bits of the priority word attached to each
 *   task to define which set of threads can execute the task.
 * 
 *   The bits that are used in the priority are defined using the MCA 
 *   parameter sched_pcb_priority_mask, which should contain enough
 *   consecutive bits to express a number between 0 and N (inclusive) where
 *   N is the number of 'thread groups'.
 * 
 *   Each computing thread of a given process belongs to a thread group.
 *   If PaRSEC is compiled with HWLOC, which threads belong to which group
 *   are defined using the HWLOC tree hierarchy and the MCA parameter
 *   sched_pcb_sharing_level: a sched_pcb_sharing_level of L means that all 
 *   threads bound to a core that is under the same node of depth L in the HWLOC 
 *   tree belong to the same group. Setting sched_pcb_sharing_level to 0 means
 *   that all threads are in the same group (they are under the root of the tree),
 *   and setting it to parsec_hwloc_nb_levels()-1 means that each thread is
 *   in its own group, by itself. Intermediate values have different results
 *   depending on the machine hierarchy.
 * 
 *   If PaRSEC is compiled without HWLOC, the MCA parameter is not exposed,
 *   and there is a single behavior: each thread belongs to its own group,
 *   by itself.
 * 
 *   There is a 'special' group: the group 0 (other groups are named 1 to N).
 *   Tasks that are bound to that group are in fact shared between all threads
 *   (as is usual for other schedulers).
 * 
 *   So, tasks with a priority 0 are always scheduled opportunistically on
 *   any thread. Because startup tasks also initialize their priority to -1,
 *   tasks with priority -1 are also handled specially by allowing any thread
 *   to execute them (they are assigned the group 0 despite their priority).
 *
 *   Last, the scheduler uses the priority value to order all the tasks that
 *   are bound to a given group.
 * 
 *   At task selection time, a thread compares the priority of the highest
 *   priority task of the group 0 and the highest priority task of its own
 *   group, and selects the task with the highest priority.
 * 
 *   Access to the task lists are protected with locks.
 */

#ifndef MCA_SCHED_PCB_H
#define MCA_SCHED_PCB_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/sched/sched.h"


BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_sched_base_component_t parsec_sched_pcb_component;
PARSEC_DECLSPEC extern const parsec_sched_module_t parsec_sched_pcb_module;
/* static accessor */
mca_base_component_t *sched_pcb_static_component(void);
extern int sched_pcb_sharing_level;
extern int sched_pcb_group_mask;
extern int sched_pcb_group_shift;

END_C_DECLS
#endif /* MCA_SCHED_PCB_H */
