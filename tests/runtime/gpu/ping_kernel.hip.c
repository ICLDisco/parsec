#include <hip/hip_runtime.h>

extern "C" {
void hip_pong_kernel(int *dev_data, int idx);
}

__global__ void pong_kernel(int *dev_data, int idx)
{
    dev_data[idx] += idx;
}

void hip_pong_kernel(int *dev_data, int idx)
{
    hipLaunchKernelGGL(pong_kernel, 1, 1, 0, 0, dev_data, idx);
}
