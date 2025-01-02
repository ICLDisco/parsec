#ifndef _NVLINK_WRAPPER_H
#define _NVLINK_WRAPPER_H
/*
 * Copyright (c) 2019-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"

parsec_taskpool_t* testing_stress_New( parsec_context_t *ctx, int depth, int mb );

#endif /* _NVLINK_WRAPPER_H */
