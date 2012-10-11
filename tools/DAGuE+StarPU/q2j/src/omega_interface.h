/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _OMEGA_INTERFACE_
#define _OMEGA_INTERFACE_

#include "dague_config.h"

BEGIN_C_DECLS

void interrogate_omega(node_t *node, var_t *head);
void add_colocated_data_info(char *a, char *b);
void store_global_invariant(node_t *node);

END_C_DECLS

#endif
