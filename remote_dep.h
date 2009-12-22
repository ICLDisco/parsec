/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifdef __USE_REMOTE_DEP_H__
#define __USE_REMOTE_DEP_H__

#include "dplasma.h"

int dplasma_remote_dep_init(dplasma_execution_unit_t* eu_context);
int dplasma_remote_dep_fini(dplasma_execution_unit_t* eu_context);

int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                dplasma_execution_context_t* exec_context,
                                const param_t* dest_param );

/*int dependency_management_satisfy(DPLASMA_desc * Ddesc, int tm, int tn, int lm, int ln);*/

#endif /* __USE_REMOTE_DEP_H__ */