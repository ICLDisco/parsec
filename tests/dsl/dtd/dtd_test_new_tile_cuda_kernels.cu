#include <stdio.h>

extern "C" {
void dtd_test_new_tile_init(int *dev_data, int nb, int idx);
void dtd_test_new_tile_sum_add(int *dev_data, int nb, int idx, int *acc, int verbose);
void dtd_test_new_tile_multiply_by_two(int *dev_data, int nb, int idx);
}

__global__ void dtnt_init(int *dev_data, int nb, int idx)
{
    for(int i = 0; i < nb; i++)
        dev_data[i] = i;
    (void)idx;
}

void dtd_test_new_tile_init(int *dev_data, int nb, int idx)
{
    dtnt_init<<<1, 1>>>(dev_data, nb, idx);
}

__global__ void dtnt_sum_add(int *dev_data, int nb, int idx, int *acc, int verbose)
{
    int sum = 0;
    for(int i = 0; i < nb; i++) {
        if( dev_data[i] != 2*i )
            printf("Error in cuda_accumulate(%d) at index %d: expected %d, got %d\n", idx, i, 2*i, dev_data[i]);
        sum += dev_data[i];
    }
    if(verbose)
        printf("cuda_accumulate(%d) contributes with %d\n", idx, sum);
    atomicAdd(acc, sum);
}

void dtd_test_new_tile_sum_add(int *dev_data, int nb, int idx, int *acc, int verbose)
{
    dtnt_sum_add<<<1, 1>>>(dev_data, nb, idx, acc, verbose);
}

__global__ void dtnt_multiply_by_two(int *dev_data, int nb, int idx)
{
    for(int i = 0; i < nb; i++) {
        if(dev_data[i] != i)
            printf("Error in cuda_multiply_by_2(%d) at index %d: expected %d, got %d\n", idx, i, i, dev_data[i]);
        dev_data[i] *= 2;
    }
}

void dtd_test_new_tile_multiply_by_two(int *dev_data, int nb, int idx)
{
    dtnt_multiply_by_two<<<1, 1>>>(dev_data, nb, idx);
}

