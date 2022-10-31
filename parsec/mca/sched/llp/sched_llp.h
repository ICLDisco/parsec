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
 * Local LIFO with support for Priorities scheduler.
 * Each thread maintains a stack of tasks (implemented using parsec_lifo)
 * in which tasks are sorted by priority.
 * Upon insertion, if all tasks in the ring have higher priorities than
 * the current head, the the ring is pushed onto the stack.
 * Otherwise, the stack is detached, the tasks in the ring are inserted
 * at the right position, and the stack is reattached.
 * Threads generally steal from each other by popping from the stack
 * and push only into their stack. An exception is the communication
 * thread, which pushes into Thread 0. This needs some special attention,
 * see the single_writer parameter for lifo_chain_sorted.
 */


#ifndef MCA_SCHED_LLP_H
#define MCA_SCHED_LLP_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/sched/sched.h"


BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_sched_base_component_t parsec_sched_llp_component;
PARSEC_DECLSPEC extern const parsec_sched_module_t parsec_sched_llp_module;
/* static accessor */
mca_base_component_t *sched_llp_static_component(void);


END_C_DECLS
#endif /* MCA_SCHED_LLP_H */
