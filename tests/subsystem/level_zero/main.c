#include <stdio.h> 
#include <stdlib.h>
<<<<<<< HEAD
=======
#include <assert.h>
#include <unistd.h>
#include <time.h>
>>>>>>> f5f749053... Fix the subsystem test. Need to backport fixes in the MCA device

#include <level_zero/ze_api.h>
#include "interface.dpcpp.h"

int dpcpp_kernel_GEMM(sycl_wrapper_driver_t *swp,
		       sycl_wrapper_device_t *swd,
		       sycl_wrapper_queue_t *swq,
                       const double *A,
                       double *C,
                       int mb);

struct driver_s;
struct device_s;
struct stream_s;

#define NB_STREAMS 4
#define MAX_FENCES 2

typedef struct stream_s {
    int                       immediate;
    void                     *sq;
    ze_command_queue_handle_t cq;
    ze_command_list_handle_t  cl;
    struct device_s          *device;
    ze_fence_handle_t         fences[MAX_FENCES];
} stream_t;

typedef struct device_s {
    ze_device_handle_t       device;
    struct driver_s         *driver;
    sycl_wrapper_device_t   *swd;
    stream_t                 streams[NB_STREAMS];
} device_t;

typedef struct driver_s {
    ze_driver_handle_t     driver;
    ze_context_handle_t    context;
    sycl_wrapper_driver_t *swp;
    int                    nb_devices;
    device_t              *devices;
} driver_t;

#define LEVEL_ZERO_CHECK_ERROR(STR, ERROR, CODE)                       \
    do {                                                                      \
        if( ZE_RESULT_SUCCESS != (ERROR) ) {                                  \
            fprintf(stderr, "%s:%d %s returns Error 0x%x", __FILE__, __LINE__,\
                            (STR), (ERROR) );                                 \
            CODE;                                                             \
        }                                                                     \
    } while(0)

static int init_device(device_t *device, ze_device_handle_t gpuDevice)
{
    // Discover all command queue groups
    uint32_t cmdqueueGroupCount = 0;
    ze_result_t ze_rc;

    ze_rc = zeDeviceGetCommandQueueGroupProperties(gpuDevice, &cmdqueueGroupCount, NULL);
    LEVEL_ZERO_CHECK_ERROR( "zeDeviceGetCommandQueueGroupProperties (count) ", ze_rc, { return -1; } );

    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)
                malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
    ze_rc = zeDeviceGetCommandQueueGroupProperties(gpuDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
    LEVEL_ZERO_CHECK_ERROR( "zeDeviceGetCommandQueueGroupProperties (populate) ", ze_rc, { return -1; } );

    // Find a command queue type that support compute
    //TODO: it might be more in line with the design to create different command queues for copy
    //      and compute than using the existing queues.
    uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
    uint32_t copyQueueGroupOrdinal = cmdqueueGroupCount;
    for( uint32_t i = 0; i < cmdqueueGroupCount &&
                         (computeQueueGroupOrdinal == cmdqueueGroupCount ||
                          copyQueueGroupOrdinal == cmdqueueGroupCount); ++i ) {
        if( computeQueueGroupOrdinal == cmdqueueGroupCount && cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
            computeQueueGroupOrdinal = i;
        }
        if( copyQueueGroupOrdinal == cmdqueueGroupCount && cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY ) {
            copyQueueGroupOrdinal = i;
        }
    }
    if( computeQueueGroupOrdinal == cmdqueueGroupCount ) {
        fprintf(stderr, "level zero device: unable to find a Queue Group with COMPUTE flag");
        return -1;
    }
    if( copyQueueGroupOrdinal == cmdqueueGroupCount ) {
        fprintf(stderr,  "level zero device: unable to find a Queue Group with COMPUTE flag");
        return -1;
    }

    for(int j = 0; j < NB_STREAMS; j++ ) {
        ze_command_queue_desc_t commandQueueDesc = {
            ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            NULL,
            (uint32_t)-1,
            0, // index
            0, // flags
            ZE_COMMAND_QUEUE_MODE_DEFAULT,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL
        };
        device->streams[j].device = device;
        if( j < 2 ) {
            device->streams[j].immediate = 1;
            commandQueueDesc.ordinal = copyQueueGroupOrdinal;
            ze_rc = zeCommandListCreateImmediate(device->driver->context, gpuDevice,
                                                 &commandQueueDesc,
                                                 &device->streams[j].cl);
            LEVEL_ZERO_CHECK_ERROR( "zeCommandListCreateImmediate ", ze_rc, { return -1;} );
        } else {
            device->streams[j].immediate = 0;
            commandQueueDesc.ordinal = computeQueueGroupOrdinal;
            ze_rc = zeCommandQueueCreate(device->driver->context, gpuDevice,
                                         &commandQueueDesc, &device->streams[j].cq);
            LEVEL_ZERO_CHECK_ERROR( "zeCommandQueueCreate ", ze_rc, { return -1;} );
            ze_command_list_desc_t commandListDesc = {
                    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                    NULL,
                    computeQueueGroupOrdinal,
                    0 // flags
            };
            ze_rc = zeCommandListCreate(device->driver->context, gpuDevice,
                                        &commandListDesc, &device->streams[j].cl);
            LEVEL_ZERO_CHECK_ERROR( "zeCommandListCreate ", ze_rc, { return -1;} );
            device->streams[j].sq = sycl_queue_create(device->driver->driver, gpuDevice, device->driver->context, device->streams[j].cq);
            if(NULL == device->streams[j].sq)
                return -1;
        }

        for(int k = 0; k < MAX_FENCES; k++ ) {
            ze_fence_desc_t fence_desc = {
                .stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
                .pNext = NULL,
                .flags = 0
            };
            device->streams[j].fences[k]   = NULL;
            ze_rc = zeFenceCreate(device->streams[j].cq, &fence_desc, &(device->streams[j].fences[k]));
            LEVEL_ZERO_CHECK_ERROR( "zeFenceCreate ", ze_rc, { continue; } );
        }
    }
    device->device = gpuDevice;
    return 0;
}

static int init_driver(driver_t *driver, int maxDevices) 
{
    uint32_t deviceCount = 0;
    ze_device_handle_t *allDevices;
    ze_device_handle_t *gpuDevices;
    ze_result_t ze_rc;

    ze_rc = zeDeviceGet(driver->driver, &deviceCount, NULL);
    LEVEL_ZERO_CHECK_ERROR( "zeDeviceGet (count) ", ze_rc, { return -1; } );

    driver->nb_devices = 0;
    driver->devices = NULL;

    if(deviceCount == 0)
        return 0;

    allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
    gpuDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
    ze_rc = zeDeviceGet(driver->driver, &deviceCount, allDevices);
    LEVEL_ZERO_CHECK_ERROR( "zeDeviceGet (populate) ", ze_rc, { return -1; } );

    int deviceId = 0;
    for(int did = 0; did < (int)deviceCount; did++) {
        ze_device_properties_t device_properties;
        zeDeviceGetProperties(allDevices[did], &device_properties);
        if( ZE_DEVICE_TYPE_GPU != device_properties.type) { continue; }
        gpuDevices[deviceId++] = allDevices[did];
        if( deviceId > maxDevices ) {
            break;
        }
    }
    free(allDevices);
    allDevices = NULL;

    if( deviceId == 0) {
        free(gpuDevices);
        return 0;
    }
    deviceCount = deviceId;

    driver->devices = (device_t*)malloc(deviceCount * sizeof(device_t));

    // Create context
    ze_context_desc_t ctxtDesc = {
        ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        NULL,
        0
    };
    ze_rc = zeContextCreate(driver->driver, &ctxtDesc, &driver->context);
    LEVEL_ZERO_CHECK_ERROR( "zeContextCreate ", ze_rc, { continue; } );

    int dpos = 0;
    for(int did = 0; did < (int)deviceCount; did++) {
        driver->devices[dpos].driver = driver;
        if( init_device(&driver->devices[dpos], gpuDevices[did]) < 0 ) {
            continue;
        }
        dpos++;
    }
    deviceCount = dpos;

    driver->nb_devices = deviceCount;
    if(deviceCount == 0) {
        free(driver->devices);
        driver->devices = NULL;
	return 0;
    }
    free(gpuDevices);

    driver->swp = sycl_wrapper_platform_create(driver->driver);
    sycl_wrapper_device_t *swd[deviceCount];
    for(int did = 0; did < (int)deviceCount; did++) {
        driver->devices[did].swd = sycl_wrapper_device_create(driver->devices[did].device);
        swd[did] = driver->devices[did].swd;
    }
    sycl_wrapper_platform_add_context(driver->swp, driver->context, swd, deviceCount);
    for(int did = 0; did < (int)deviceCount; did++) {
        for(int sid = 0; sid < NB_STREAMS; sid++) {
            driver->devices[did].streams[sid].swq = sycl_wrapper_queue_create(driver->swp, driver->devices[did].swd, driver->devices[did].streams[sid].cq);
	}
    }

    return deviceCount;
}

static void *allocate_workspace(device_t *device, size_t size)
{
    ze_result_t status;
    ze_device_properties_t devProperties;
    ze_device_memory_properties_t *devMemProperties;
    ze_device_memory_access_properties_t memAccessProperties;
    void *device_ptr;
    uint32_t count = 0;
    int memIndex = -1;

    status = zeDeviceGetMemoryAccessProperties(device->device, &memAccessProperties);
    LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryAccessProperties ", status, { return NULL; });
    if( 0 == (ZE_MEMORY_ACCESS_CAP_FLAG_RW & memAccessProperties.deviceAllocCapabilities) ) {
        fprintf(stderr, "Device does not have memory allocation capabilities with RW access\n");
        return NULL;
    }
    status = zeDeviceGetProperties(device->device, &devProperties);
    LEVEL_ZERO_CHECK_ERROR("zeDeviceGetProperties ", status, { return NULL; });
    status = zeDeviceGetMemoryProperties(device->device, &count, NULL);
    LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryProperties (count) ", status, { return NULL; });
    devMemProperties = (ze_device_memory_properties_t*)malloc(count * sizeof(ze_device_memory_properties_t));
    status = zeDeviceGetMemoryProperties(device->device, &count, devMemProperties);
    LEVEL_ZERO_CHECK_ERROR("zeDeviceGetMemoryProperties (populate) ", status, { free(devMemProperties); return NULL; });
    for(int i = 0; i < (int)count; i++) {
        // TODO: better approach would be to keep a list of pointers?
        //   for now we just take the memory that has the highest amount of memory available
        if( memIndex == -1 || devMemProperties[memIndex].totalSize < devMemProperties[i].totalSize)
            memIndex = i;
    }
    free(devMemProperties); devMemProperties = NULL;

    if( size > devMemProperties[memIndex].totalSize ) {
        /** Handle the case of jokers who require more than 100% of memory,
         *  and eleventh case of computer scientists who don't know how
         *  to divide a number by another
         */
        fprintf(stderr, "Requested %zd bytes on LEVEL_ZERO device, but only %zd bytes are available -- Returning NULL\n",
                size, devMemProperties[memIndex].totalSize);
        return NULL;
    }
    ze_device_mem_alloc_desc_t memAllocDesc = {
            .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            .pNext = NULL,
            .flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED,
            .ordinal = memIndex
    };

    status = zeMemAllocDevice(device->driver->context, &memAllocDesc, size, 128,
                              device->device, &device_ptr);
    LEVEL_ZERO_CHECK_ERROR( "zeMemAllocDevice ", status, { return NULL; } );
    return device_ptr;
}

typedef struct {
    struct timespec start;
    struct timespec enter_wait;
    struct timespec end; 
} run_timing_t;

typedef struct {
    struct timespec start;
    struct timespec start_alloc;
    struct timespec end_alloc;
    struct timespec start_cpu2gpu;
    struct timespec enter_wait_cpu2gpu;
    struct timespec end_cpu2gpu;
    struct timespec start_gpu2cpu;
    struct timespec enter_wait_gpu2cpu;
    struct timespec end_gpu2cpu;
    run_timing_t   *runs;
} device_timing_t;

double diff_timespec(const struct timespec *time1, const struct timespec *time0) {
  return (time1->tv_sec - time0->tv_sec)
      + (time1->tv_nsec - time0->tv_nsec) / 1000000000.0;
}

static void usage(const char *name) {
    fprintf(stderr, "USAGE: %s [<nunber of Level Zero GPUs>] [Problem size] [Number of iterations]\n", name);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) 
{
    ze_result_t ze_rc;
    uint32_t driverCount = 0;
    driver_t *drivers;
    ze_driver_handle_t *allDrivers;
    int max_devices = 1024*1024, nb_devices = 0;
    void **device_workspaceA, **device_workspaceC;
    int errors = 0;
    int N = 4096;
    int NRUNS = 3;
    device_timing_t *timings;

    if(argc > 1) {
        max_devices = atoi(argv[1]);
        if(max_devices <= 0) {
            usage(argv[0]);
        }

        if(argc > 2) {
            N = atoi(argv[2]);
            if(N <= 0) {
                usage(argv[0]);
            }

            if(argc >3) {
                NRUNS = atoi(argv[3]);
                if(NRUNS <= 0) {
                    usage(argv[0]);
                }
            }
        }
    }

    // Discover all the driver instances
    ze_rc = zeDriverGet(&driverCount, NULL);
    LEVEL_ZERO_CHECK_ERROR( "zeDriverGet (count) ", ze_rc, { return 1; } );
    fprintf(stderr, "STATUS: found %d drivers\n", driverCount);
    drivers = malloc(driverCount * sizeof(driver_t));
    allDrivers = malloc(driverCount * sizeof(ze_driver_handle_t));
    ze_rc = zeDriverGet(&driverCount, allDrivers);
    LEVEL_ZERO_CHECK_ERROR( "zeDriverGet (populate) ", ze_rc, { return 1; } );
    for(int driverId = 0; driverId < (int)driverCount; driverId++) {
        int nb;
        drivers[driverId].driver = allDrivers[driverId];
        if( (nb = init_driver(&drivers[driverId], max_devices)) <= 0 ) {
            fprintf(stderr, "%d device found in driver %d... Bailing out\n", nb_devices, driverId);
            return 1;
        } else {
	        fprintf(stderr, "STATUS: there are %d GPU devices in driver %d\n", nb, driverId);
            nb_devices += nb;
            nb_devices = nb_devices <= max_devices ? nb_devices : max_devices;
        }
    }
    free(allDrivers);
    fprintf(stderr, "%d devices found and initialized\n", nb_devices);

    timings = (device_timing_t*)calloc(sizeof(device_timing_t), nb_devices);
    for(int i = 0; i < nb_devices; i++) {
        clock_gettime(CLOCK_REALTIME, &timings[i].start);
        timings[i].runs = (run_timing_t*)calloc(sizeof(run_timing_t), NRUNS);
    }

    //Allocate GPU memory for each device
    device_workspaceA = (void**)malloc(sizeof(void*)*nb_devices);
    device_workspaceC = (void**)malloc(sizeof(void*)*nb_devices);
    int did = 0;
    for(int driverId = 0; driverId < (int)driverCount; driverId++) {
        for(int deviceId = 0; deviceId < drivers[driverId].nb_devices; deviceId++) {
            clock_gettime(CLOCK_REALTIME, &timings[did].start_alloc);
            device_workspaceA[did] = allocate_workspace(&drivers[driverId].devices[deviceId], sizeof(double)*N*N*2);
            device_workspaceC[did] = allocate_workspace(&drivers[driverId].devices[deviceId], sizeof(double)*N*N);
            clock_gettime(CLOCK_REALTIME, &timings[did].end_alloc);
            did++;
        }
    }

    //Initialize the GPU memory for each device
    double *cpuA = (double*)calloc(sizeof(double), N*N);
    for(int i = 0; i < N; i++) {
        cpuA[i*N+i] = 1.0;
    }
    double *cpuC = (double*)calloc(sizeof(double), N*N);
    did=0;
    for(int driverId = 0; driverId < (int)driverCount; driverId++) {
        for(int deviceId = 0; deviceId < drivers[driverId].nb_devices; deviceId++) {
            device_t *device = &drivers[driverId].devices[deviceId];
            if(NULL != device_workspaceA[did] && NULL != device_workspaceC[did]) {
                fprintf(stderr, "STATUS: Initialize A and C on device %d of driver %d\n", deviceId, driverId);
                void *copy_in = (void*)cpuA;
                void *copy_out = (void*)device_workspaceA[did];
                size_t count = N*N*sizeof(double);
                clock_gettime(CLOCK_REALTIME, &timings[did].start_cpu2gpu);
                ze_rc = zeCommandListReset(device->streams[0].cl);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListReset ", ze_rc, { continue; } );
                ze_rc = (ze_result_t)zeCommandListAppendMemoryCopy(device->streams[0].cl, copy_out, copy_in, count, NULL, 0, NULL);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListAppendMemoryCopy(A->GPU) ", ze_rc, { continue; } );
                copy_in = (void*)cpuC;
                copy_out = (void*)device_workspaceC[did];
                count = N*N*sizeof(double);
                ze_rc = (ze_result_t)zeCommandListAppendMemoryCopy(device->streams[0].cl, copy_out, copy_in, count, NULL, 0, NULL);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListAppendMemoryCopy(C->GPU) ", ze_rc, { continue; } );
                
                ze_rc = zeCommandListClose(device->streams[0].cl);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListClose ", ze_rc, { continue; } );
                ze_rc = zeCommandQueueExecuteCommandLists(device->streams[0].cq, 1, &device->streams[0].cl, device->streams[0].fences[0]);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandQueueExecuteCommandLists ", ze_rc, { continue; } );

                clock_gettime(CLOCK_REALTIME, &timings[did].enter_wait_cpu2gpu);
                do {
                    ze_rc = zeFenceQueryStatus(device->streams[0].fences[0]);
                    if( ZE_RESULT_SUCCESS == ze_rc ) {
                        fprintf(stderr, "STATUS: copies ended on device %d of driver %d\n", deviceId, driverId);
			            break;
                    } else  if( ZE_RESULT_NOT_READY != ze_rc ) {
                        LEVEL_ZERO_CHECK_ERROR( "(progress_stream) zeFenceQueryStatus ", ze_rc, { continue; } );
                    } else {
                        usleep(10);
                    }
                } while(1);
                clock_gettime(CLOCK_REALTIME, &timings[did].end_cpu2gpu);
                fprintf(stderr, "STATUS: CPU->GPU copies on device %d of driver %d completed!\n", deviceId, driverId);
            }
            did++;
        }
    }

    //Do a GEMM (blocking) on each device, and wait for its completion -- yes, memory is not initialized.
    did = 0;
    for(int driverId = 0; driverId < (int)driverCount; driverId++) {
        for(int deviceId = 0; deviceId < drivers[driverId].nb_devices; deviceId++) {
            device_t *device = &drivers[driverId].devices[deviceId];
            if(NULL != device_workspaceA[did] && NULL != device_workspaceC[did]) {
                fprintf(stderr, "STATUS: Get C back from device %d of driver %d\n", deviceId, driverId);
                void *copy_in = (void*)device_workspaceC[did];
                void *copy_out = (void*)cpuC;
                size_t count = N*N*sizeof(double);

                for(int i = 0; i < N*N; i++) cpuC[i] = -1.0;

                clock_gettime(CLOCK_REALTIME, &timings[did].start_gpu2cpu);
                ze_rc = zeCommandListReset(device->streams[1].cl);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListReset ", ze_rc, { continue; } );
                ze_rc = (ze_result_t)zeCommandListAppendMemoryCopy(device->streams[1].cl, copy_out, copy_in, count, NULL, 0, NULL);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListAppendMemoryCopy(GPU->A) ", ze_rc, { continue; } );
                
                ze_rc = zeCommandListClose(device->streams[1].cl);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandListClose ", ze_rc, { continue; } );
                ze_rc = zeFenceReset(device->streams[1].fences[0]);
                LEVEL_ZERO_CHECK_ERROR( "zeFenceReset ", ze_rc, { continue; } );
                ze_rc = zeCommandQueueExecuteCommandLists(device->streams[1].cq, 1, &device->streams[1].cl, device->streams[1].fences[0]);
                LEVEL_ZERO_CHECK_ERROR( "zeCommandQueueExecuteCommandLists ", ze_rc, { continue; } );

                clock_gettime(CLOCK_REALTIME, &timings[did].enter_wait_gpu2cpu);
                do {
                    ze_rc = zeFenceQueryStatus(device->streams[1].fences[0]);
                    if( ZE_RESULT_SUCCESS == ze_rc ) {
                        fprintf(stderr, "STATUS: GEMM ended on device %d of driver %d\n", deviceId, driverId);
                    } else  if( ZE_RESULT_NOT_READY != ze_rc ) {
                        LEVEL_ZERO_CHECK_ERROR( "(progress_stream) zeFenceQueryStatus ", ze_rc, { continue; } );
                    } else {
                        usleep(10);
                    }
                } while(1);
            } else {
                fprintf(stderr, "Skipping device %d which failed at allocating data\n", did);
            }
            did++;
        }
    }

    return EXIT_SUCCESS;
}
