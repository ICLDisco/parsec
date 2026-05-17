/*
 * Copyright (c) 2023-2026 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <cuda_runtime.h>

extern "C" {
void cuda_pong_kernel(int *dev_data, int idx, void *stream);
}

__global__ void pong_kernel(int *dev_data, int idx)
{
    dev_data[idx] += idx;
}

void cuda_pong_kernel(int *dev_data, int idx, void *stream)
{
    pong_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(dev_data, idx);
}
