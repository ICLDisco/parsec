/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
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
 * Local Flat Queue Scheduler
 *
 */


#ifndef MCA_SCHED_LFQ_H
#define MCA_SCHED_LFQ_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/sched/sched.h"


BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_sched_base_component_t parsec_sched_lfq_component;
PARSEC_DECLSPEC extern const parsec_sched_module_t parsec_sched_lfq_module;
/* static accessor */
mca_base_component_t *sched_lfq_static_component(void);


END_C_DECLS
#endif /* MCA_SCHED_LFQ_H */
