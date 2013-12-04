/*
 * Copyright (c) 2013      The University of Tennessee and The University
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
 * Global Dequeue Scheduler
 *
 */


#ifndef MCA_SCHED_IP_H
#define MCA_SCHED_IP_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/sched/sched.h"


BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_sched_base_component_t dague_sched_ip_component;
DAGUE_DECLSPEC extern const dague_sched_module_t dague_sched_ip_module;
/* static accessor */
mca_base_component_t *sched_ip_static_component(void);


END_C_DECLS
#endif /* MCA_SCHED_IP_H */
