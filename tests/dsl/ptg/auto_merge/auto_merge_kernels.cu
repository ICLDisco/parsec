/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {
extern const int am_have_cuda_kernels;
const int am_have_cuda_kernels = 1;
void am_cuda_affine(int *dev_tile, int count, uint32_t a, uint32_t b, int spin_us, void *stream);
}

__global__ void am_affine_kernel(uint32_t *tile, int count, uint32_t a, uint32_t b, long long spin_cycles)
{
    if( spin_cycles > 0 ) {
        /* Keep the device busy for a while, so a later taskpool that is not
         * ordered after this one has a chance to read the tile first. */
        long long start = clock64();
        while( clock64() - start < spin_cycles ) { }
    }
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x ) {
        tile[i] = a * tile[i] + b;
    }
}

void am_cuda_affine(int *dev_tile, int count, uint32_t a, uint32_t b, int spin_us, void *stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    /* About one microsecond per thousand cycles is enough for a delay */
    long long spin_cycles = (long long)spin_us * 1000;
    am_affine_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>((uint32_t*)dev_tile, count, a, b, spin_cycles);
}
