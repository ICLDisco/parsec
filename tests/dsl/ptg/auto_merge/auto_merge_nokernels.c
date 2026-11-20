/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Builds without a CUDA compiler: the driver keeps every task on the CPU, so
 * the CUDA bodies of the JDFs, if PaRSEC compiled them, are never run. */

#include "parsec/parsec_config.h"
#include "auto_merge_ops.h"

#include <stdio.h>
#include <stdlib.h>

const int am_have_cuda_kernels = 0;

void am_cuda_affine(int *dev_tile, int count, uint32_t a, uint32_t b, int spin_us, void *stream)
{
    (void)dev_tile; (void)count; (void)a; (void)b; (void)spin_us; (void)stream;
    fprintf(stderr, "auto_merge: a CUDA body ran in a build without CUDA kernels\n");
    abort();
}
