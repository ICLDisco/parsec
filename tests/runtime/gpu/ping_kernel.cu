extern "C" {
void cuda_pong_kernel(int *dev_data, int idx);
}

__global__ void pong_kernel(int *dev_data, int idx)
{
    dev_data[idx] += idx;
}

void cuda_pong_kernel(int *dev_data, int idx)
{
    pong_kernel<<<1, 1>>>(dev_data, idx);
}
