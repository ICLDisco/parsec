__global__ void kernel_task1_cuda(int n, int *A1, int *A2, int *A3) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < n; i += stride) {
    A1[i] += 1;
    A3[i] = A2[i];
  }
}

__global__ void kernel_task2_cuda(int n, int *A1, int *A2) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < n; i += stride) {
    A2[i] += A1[i];
  }
}

extern "C" int task1_cuda(int n, int *A1, int *A2, int *A3) {
  int nb = 256;
  int nt = (n + nb - 1) / nb;
  kernel_task1_cuda<<<nt,nb>>>(n, A1, A2, A3);
  return 0;
}

extern "C" int task2_cuda(int n, int *A1, int *A2) {
  int nb = 256;
  int nt = (n + nb -1) / nb;
  kernel_task2_cuda<<<nt,nb>>>(n, A1, A2);
  return 0;
}
