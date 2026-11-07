/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include <stdint.h>
#include <cuda_runtime.h>

/* Must match shared_t in dtd_test_atomic_write_cuda.c */
#define NB_SLOTS 28
typedef struct {
    int32_t counter;
    int32_t spare[3];
    int32_t slot[NB_SLOTS];
} shared_t;

extern "C" {
void dtd_test_atomic_write_cuda_init(void *dev_data, void *stream);
void dtd_test_atomic_write_cuda_write(void *dev_data, int slot, int count, void *stream);
void dtd_test_atomic_write_cuda_verify(void *dev_data, int expected_counter, int nb_slots, void *stream);
}

__global__ void aw_init(shared_t *shared)
{
    shared->counter = 0;
    for( int i = 0; i < 3; i++ ) shared->spare[i] = 0;
    for( int i = 0; i < NB_SLOTS; i++ ) shared->slot[i] = 0;
}

void dtd_test_atomic_write_cuda_init(void *dev_data, void *stream)
{
    aw_init<<<1, 1, 0, (cudaStream_t)stream>>>((shared_t *)dev_data);
}

/* The two ways ATOMIC_WRITE is meant to be used: each task writes its own part of the
 * data, and the tasks that are allowed to read the previous value update a counter. */
__global__ void aw_write(shared_t *shared, int slot, int count)
{
    shared->slot[slot] = slot + 1;
    if( count )
        atomicAdd(&shared->counter, 1);
}

void dtd_test_atomic_write_cuda_write(void *dev_data, int slot, int count, void *stream)
{
    aw_write<<<1, 1, 0, (cudaStream_t)stream>>>((shared_t *)dev_data, slot, count);
}

/* Checks what the run left in the data, and reports in the spare fields of the data itself:
 * the number of mismatches, and the counter it found */
__global__ void aw_verify(shared_t *shared, int expected_counter, int nb_slots)
{
    int mismatches = 0;
    if( expected_counter >= 0 && shared->counter != expected_counter ) mismatches++;
    for( int i = 0; i < nb_slots; i++ ) {
        if( shared->slot[i] != i + 1 ) mismatches++;
    }
    shared->spare[0] = mismatches;
    shared->spare[1] = shared->counter;
}

void dtd_test_atomic_write_cuda_verify(void *dev_data, int expected_counter, int nb_slots, void *stream)
{
    aw_verify<<<1, 1, 0, (cudaStream_t)stream>>>((shared_t *)dev_data, expected_counter, nb_slots);
}
