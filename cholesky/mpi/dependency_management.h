/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifdef __USE_DEPENDENCY_MANAGEMENT_H__
#define __USE_DEPENDENCY_MANAGEMENT_H__

#include "data_management.h"

int dependency_management_init(void);
int dependency_management_fini(void);

int dependency_management_satisfy(DPLASMA_desc * Ddesc, int tm, int tn, int lm, int ln);

#endif /* __USE_DEPENDENCY_MANAGEMENT_H__ */