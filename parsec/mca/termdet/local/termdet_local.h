/*
 * Copyright (c) 2018      The University of Tennessee and The University
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
 * Dijsktra-Matter Termination Detection Algorithm, four-counters variant
 *   (see TBD)
 *
 */

#ifndef MCA_TERMDET_LOCAL_H
#define MCA_TERMDET_LOCAL_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/termdet/termdet.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_termdet_base_component_t parsec_termdet_local_component;
PARSEC_DECLSPEC extern const parsec_termdet_module_t parsec_termdet_local_module;
/* static accessor */
mca_base_component_t *termdet_local_static_component(void);

END_C_DECLS
#endif /* MCA_TERMDET_LOCAL_H */

