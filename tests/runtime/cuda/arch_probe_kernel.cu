/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
int cuda_arch_probe_run(void);
}

__global__ void arch_probe_kernel(int *dev_data)
{
    dev_data[0] = 1;
}

/* What nvcc actually put in this object, as opposed to what the build asked
 * for. Absent before CUDA 11.5, which is old enough to just say so. */
static const char *compiled_for(void)
{
#if defined(__CUDA_ARCH_LIST__)
/* A build for more than one architecture makes this a comma separated list,
 * so the stringification has to take however many arguments that is. */
#define PROBE_STR(...) #__VA_ARGS__
#define PROBE_XSTR(...) PROBE_STR(__VA_ARGS__)
    return PROBE_XSTR(__CUDA_ARCH_LIST__);
#else
    return "unknown";
#endif
}

/*
 * Launch a kernel from this build on this device and see whether it runs. A
 * mismatch between the architectures the .cu files were compiled for and the
 * hardware does not announce itself: the launch fails, every kernel in the
 * test suite silently does nothing, and what the tests report instead is
 * wrong results and deadlocks far from the cause.
 *
 * Returns 0 when a kernel runs, 10 (-PARSEC_ERR_DEVICE) when there is no
 * device to run it on, and 1 when the build cannot run on the device.
 */
int cuda_arch_probe_run(void)
{
    int nb_devices = 0, *dev_data = NULL, host_data = 0;
    cudaError_t err;

    err = cudaGetDeviceCount(&nb_devices);
    if( (cudaSuccess != err) || (0 == nb_devices) ) {
        fprintf(stderr, "No CUDA device available (%s), skipping\n", cudaGetErrorString(err));
        return 10;
    }

    for( int device = 0; device < nb_devices; device++ ) {
        struct cudaDeviceProp prop;

        if( cudaSuccess != cudaSetDevice(device) ) continue;
        if( cudaSuccess != cudaGetDeviceProperties(&prop, device) ) continue;
        if( cudaSuccess != cudaMalloc((void**)&dev_data, sizeof(int)) ) continue;

        host_data = 0;
        cudaMemcpy(dev_data, &host_data, sizeof(int), cudaMemcpyHostToDevice);
        arch_probe_kernel<<<1, 1>>>(dev_data);
        err = cudaDeviceSynchronize();
        if( cudaSuccess == err ) err = cudaGetLastError();
        if( cudaSuccess == err )
            cudaMemcpy(&host_data, dev_data, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev_data);

        if( (cudaSuccess != err) || (1 != host_data) ) {
            fprintf(stderr,
                    "A kernel from this build cannot run on device %d (%s, compute capability %d.%d):\n"
                    "\t%s\n"
                    "\tthis build compiled its kernels for architecture(s) %s\n"
                    "\trebuild with -DCMAKE_CUDA_ARCHITECTURES=native, or name %d%d explicitly.\n"
                    "Every kernel in the test suite fails the same way, and the tests report it as\n"
                    "wrong results or hangs rather than as a build that does not match the hardware.\n",
                    device, prop.name, prop.major, prop.minor,
                    (cudaSuccess != err) ? cudaGetErrorString(err) : "the kernel ran but wrote nothing",
                    compiled_for(), prop.major, prop.minor);
            return 1;
        }
        printf("Device %d (%s, compute capability %d.%d) runs kernels compiled for %s\n",
               device, prop.name, prop.major, prop.minor, compiled_for());
    }
    return 0;
}
