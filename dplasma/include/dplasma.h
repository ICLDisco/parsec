/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#define dplasma_error(__func, __msg) fprintf(stderr, "%s: %s\n", (__func), (__msg))

#include "data_dist/matrix/matrix.h"
 

#include "generated/dplasma_s.h"
#include "generated/dplasma_d.h"
#include "generated/dplasma_c.h"
#include "generated/dplasma_z.h"

#endif /* _DPLASMA_H_ */
