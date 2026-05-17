/*
 * Copyright (c) 2023-2026 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <hip/hip_runtime.h>

extern "C" {
void hip_pong_kernel(int *dev_data, int idx, void *stream);
}

__global__ void pong_kernel(int *dev_data, int idx)
{
    dev_data[idx] += idx;
}

void hip_pong_kernel(int *dev_data, int idx, void *stream)
{
    hipLaunchKernelGGL(pong_kernel, 1, 1, 0, (hipStream_t)stream, dev_data, idx);
}
