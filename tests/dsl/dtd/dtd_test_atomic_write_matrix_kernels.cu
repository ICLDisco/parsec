/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include <stdint.h>
#include <cuda_runtime.h>

/* Must match shared_t in dtd_test_atomic_write_matrix.c */
#define NB_SLOTS 28
typedef struct {
    int32_t counter;
    int32_t spare[3];
    int32_t slot[NB_SLOTS];
} shared_t;

extern "C" {
void dtd_test_atomic_write_matrix_check(void *dev_data, int expected_counter, int expected_slots,
                                        int set_marker, int marker, int *errors, void *stream);
}

/* Counts what is wrong in the data in the (managed) error counter, and sets the marker of the
 * tasks that write the data (a check that only reads it leaves the data alone). */
__global__ void aw_matrix_check(shared_t *shared, int expected_counter, int expected_slots,
                                int set_marker, int marker, int *errors)
{
    int mismatches = 0;
    if( shared->counter != expected_counter ) mismatches++;
    for( int i = 0; i < NB_SLOTS; i++ ) {
        int expected = (i < expected_slots) ? i + 1 : 0;
        if( shared->slot[i] != expected ) mismatches++;
    }
    if( mismatches ) atomicAdd(errors, mismatches);
    if( set_marker ) shared->spare[0] = marker;
}

void dtd_test_atomic_write_matrix_check(void *dev_data, int expected_counter, int expected_slots,
                                        int set_marker, int marker, int *errors, void *stream)
{
    aw_matrix_check<<<1, 1, 0, (cudaStream_t)stream>>>((shared_t *)dev_data, expected_counter, expected_slots,
                                                       set_marker, marker, errors);
}
