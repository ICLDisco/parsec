/*
 * Copyright (c) 2013-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/** @addtogroup parsec_device
 *  @{
 *
 * @file
 *
 * Computing devices framework component interface.
 *
 * Provides the API to add computational devices to the PaRSEC
 * runtime, including means to move the data between them and the
 * main memory.
 *
 * @section Device Initialization / Finalization
 *
 * All components are activated upon PaRSEC startup and are expected to either
 * complete the initialization with error (to be removed from any further uses
 * by the runtime) or return a correctly initialized module, able to execute the
 * runtime requests. In most cases, a module per hardware device should be the
 * preferred approach, but it is possible to return any combination of modules,
 * even hiding several hardware computing units behind a single module.
 *
 * During the component finalization, which is expected to be called after all
 * modules have been correctly turned off and properly removed, the component
 * complete the last stages of the shutdown, and releases all internal resources
 * binding the process to the hardware.
 *
 */
#ifndef PARSEC_DEVICE_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_H_HAS_BEEN_INCLUDED

#include "parsec/class/list_item.h"
#if defined(PARSEC_PROF_TRACE)
#include "parsec/profiling.h"
#endif  /* defined(PARSEC_PROF_TRACE) */
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/mca/mca.h"
#include "parsec/class/info.h"

 #include <string.h>

BEGIN_C_DECLS

typedef struct parsec_device_module_s parsec_device_module_t;
typedef struct parsec_device_module_s parsec_device_base_module_1_0_0_t;
typedef struct parsec_device_module_s parsec_device_base_module_t;

struct parsec_device_base_component_2_0_0 {
    mca_base_component_2_0_0_t        base_version;
    mca_base_component_data_2_0_0_t   base_data;
    parsec_device_base_module_t     **modules;
};

typedef struct parsec_device_base_component_2_0_0 parsec_device_base_component_2_0_0_t;
typedef struct parsec_device_base_component_2_0_0 parsec_device_base_component_t;

#define PARSEC_DEV_NONE       ((uint8_t)    0x00)
#define PARSEC_DEV_CPU        ((uint8_t)(1 << 0))
#define PARSEC_DEV_RECURSIVE  ((uint8_t)(1 << 1))
#define PARSEC_DEV_CUDA       ((uint8_t)(1 << 2))
#define PARSEC_DEV_HIP        ((uint8_t)(1 << 3))
#define PARSEC_DEV_LEVEL_ZERO ((uint8_t)(1 << 4))
#define PARSEC_DEV_TEMPLATE   ((uint8_t)(1 << 7))
#define PARSEC_DEV_ANY_TYPE   ((uint8_t)    0x3f)
#define PARSEC_DEV_ALL        ((uint8_t)    0x3f)
#define PARSEC_DEV_MAX_NB_TYPE                (7)

#define PARSEC_DEV_GPU_MASK   (PARSEC_DEV_CUDA|PARSEC_DEV_HIP|PARSEC_DEV_LEVEL_ZERO)
#define PARSEC_DEV_IS_GPU(t)  (0 != ((t) & PARSEC_DEV_GPU_MASK))

#define PARSEC_DEV_DATA_ADVICE_PREFETCH              ((int) 0x01)
#define PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE      ((int) 0x02)
#define PARSEC_DEV_DATA_ADVICE_WARMUP                ((int) 0x03)

typedef int   (*parsec_device_attach_f)(parsec_device_module_t*, parsec_context_t*);
typedef int   (*parsec_device_detach_f)(parsec_device_module_t*, parsec_context_t*);
typedef int   (*parsec_device_taskpool_register_f)(parsec_device_module_t*, parsec_taskpool_t*);
typedef int   (*parsec_device_taskpool_unregister_f)(parsec_device_module_t*, parsec_taskpool_t*);
typedef int   (*parsec_device_memory_register_f)(parsec_device_module_t*, parsec_data_collection_t*, void*, size_t);
typedef int   (*parsec_device_memory_unregister_f)(parsec_device_module_t*, parsec_data_collection_t*, void*);
typedef int   (*parsec_device_memory_release_f)(parsec_device_module_t*);

/**
 * @brief Provide hints for data management wrt a given device
 *
 * @details
 *    @param[INOUT] dev: a parsec device
 *    @param[INOUT] data: a parsec data
 *    @param[IN]    cmd: an advice command
 *
 *  All advices can be ignored by any device. They are used to optimize
 *   usage and performance.
 *
 *  cmd can be 
 *
 *    PARSEC_DEV_DATA_ADVICE_PREFETCH: a data copy of data should be
 *        sent to dev. If the device has an old copy of data, a fresh
 *        one should be downloaded.
 *
 *    PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE: mark that this device is
 *        the preferred device to operate Read-Write tasks on this data.
 *
 *    PARSEC_DEV_DATA_ADVICE_WARMUP: inform the device that the data
 *        is in the active set: if the device manages a LRU or another
 *        structure to compute the active set of data, this informs the
 *        device that the data should be considered as part of the active
 *        set, as if a task had just accessed this data.
 */
typedef int   (*parsec_device_data_advise_f)(parsec_device_module_t*, parsec_data_t*, int);
typedef void* (*parsec_device_find_function_f)(parsec_device_module_t*, char*);

/**
 * Reorders the list of pending tasks on the current device based on the
 *   current heuristic implemented by the device
 */
typedef int  (*parsec_device_sort_pending_list_function_f)(parsec_device_module_t*);

/**
 * Schedules some kernel represented by @p task on the device @p module,
 * from the execution stream @p es.
 * 
 * @note: @p task is a void *parameter, because its actual type depends
 *   on the device type.
 */
typedef parsec_hook_return_t (*parsec_device_kernel_scheduler_function_t)( parsec_device_module_t *module, parsec_execution_stream_t *es, void *task);

/**
 * @brief Callback to complete initialization of a device after all
 *   other devices have done their initialization/attachment
 *   Typically used to compute the interconnect matrix between devices
 *
 * @param [INOUT]module: the module to complete the initialization
 * @return PARSEC_SUCCESS or an error code
 */
typedef int (*parsec_device_all_devices_attached_f)(parsec_device_module_t *module);

struct parsec_device_module_s {
    parsec_object_t                        super;
    const parsec_device_base_component_t  *component;
    /* Device Management Functions */
    parsec_device_attach_f                 attach;
    parsec_device_detach_f                 detach;
    parsec_device_taskpool_register_f      taskpool_register;
    parsec_device_taskpool_unregister_f    taskpool_unregister;
    parsec_device_memory_register_f        memory_register;
    parsec_device_memory_unregister_f      memory_unregister;
    parsec_device_memory_release_f         memory_release;
    parsec_device_data_advise_f            data_advise;
    parsec_device_find_function_f          find_function;
    parsec_device_sort_pending_list_function_f sort_pending_list;
    parsec_device_kernel_scheduler_function_t  kernel_scheduler;
    parsec_device_all_devices_attached_f   all_devices_attached;

    parsec_info_object_array_t             infos; /**< Per-device info objects are stored here */
    struct parsec_context_s* context;  /**< The PaRSEC context this device belongs too */
    char* name;  /**< Simple identified for the device */
    uint64_t *data_in_from_device; /**< One counter per device: how many bytes have been copied from this device */
    uint64_t  data_out_to_host;    /**< When a device writes back to the host, it counts it here, to avoid the need for an atomic operation on the data_in_from_device of the host device */
    uint64_t  required_data_in;
    uint64_t  required_data_out;
    uint64_t  executed_tasks;
    uint64_t  nb_data_faults;
    uint64_t  nb_evictions;
    volatile int64_t  nb_discarded; /**< Track number of discarded data copies on this device */
    /* We provide the compute capacity of the device in GFlop/s so that conversion to #nanosec in load estimates is straightforward */
    /* These compute capacities can be useful for users when providing their own
     * time_estimate functions: the user can divide the number of flops for the
     * operation with the device execution rate to produce the time estimate. */
    int64_t   gflops_fp16;  /**< Number of half precision operations per nanosecond (or gflops/s) */
    int64_t   gflops_fp32;  /**< Number of single precision operations per nanosecond (or gflops/s) */
    int64_t   gflops_fp64;  /**< Number of double precision operations per nanosecond (or gflops/s) */
    int64_t   gflops_tf32;  /**< Number of tensor operations per nanosecond (or gflops/s) */
    int64_t   time_estimate_default; /**< An estimate of the time to execute on that device a task that would take 1ns using the aggregate power of all devices. This is the default time_estimate if none is user-set. */
    int64_t   device_load;     /**< Number of nanoseconds of work submitted to the device, and not completed now. This variable is adjusted by the runtime using the time_estimate loads from the tasks. */
    uint8_t gflops_guess; /**< True if the device is not 'known' which entails that the 'gflops' rates have been populated with fallback (arbitrary) values. */
    uint8_t data_in_array_size; /**< Current size of the data_in_from_device array. Used for safety checking */
    uint8_t device_index;
    uint8_t type;
#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_stream_t *profiling;
#endif  /* defined(PROFILING) */
};

PARSEC_OBJ_CLASS_DECLARATION(parsec_device_module_t);

extern uint32_t parsec_nb_devices;
extern int parsec_device_output;

/**
 * @brief Find the best device to execute the kernel based on the compute
 * capability of the device.
 *
 * Returns:
 * PARSEC_SUCCESS - kernel must be executed by the device set in
 *                  this_task->selected_device (for convenience
 *                  this_task->selected_chore is also set)
 *                  this_task->load is set based on the selected device and
 *                  this_task->task_class->time_estimate function to compute
 *                  the load.
 * PARSEC_ERROR   - no device could be selected
 */
PARSEC_DECLSPEC extern int parsec_select_best_device( parsec_task_t* this_task);

/**
 * Initialize the internal structures for managing external devices such as
 * accelerators and GPU. Memory nodes can as well be managed using the same
 * mechnism.
 */
extern int parsec_mca_device_init(void);

/**
 * The runtime will shutdown, all internal structures have to be destroyed.
 */
extern int parsec_mca_device_fini(void);

/**
 * Parse the list of potential devices and see which one would succesfully load
 * and initialize in the current environment.
 */
extern int parsec_mca_device_attach(parsec_context_t*);

/**
 * This call mark the end of the configuration step, no devices can be registered
 * after this point. This gives a reference point regarding the number of available
 * (supported) devices.
 */
extern int parsec_mca_device_registration_complete(parsec_context_t*);
/**
 * Return a positive value if the devices configurations has been completed 
 * by a prior call to parsec_mca_device_registration_complete().
 */
extern int parsec_mca_device_registration_completed(parsec_context_t*);

/**
 * Reset the load of all the devices to force a reconsideration of the load balance
 */
PARSEC_DECLSPEC void parsec_devices_reset_load(parsec_context_t *context);

/**
 * Add a new device to the context. The device provides a list of supported
 * capabilities, and will later be checked to see it's compatibility with the
 * needs of each taskpool.
 *
 * Returns a positive value to signal the device was successfully registered under
 * that index, or a negative value to signal an error during the process.
 */
PARSEC_DECLSPEC int parsec_mca_device_add(parsec_context_t*, parsec_device_module_t*);

/**
 * Retrieve a pointer to the registered device using the provided index.
 */
PARSEC_DECLSPEC parsec_device_module_t* parsec_mca_device_get(uint32_t devindex);

/**
 * True if the device pointed by the index is a valid GPU device
 */
PARSEC_DECLSPEC int parsec_mca_device_is_gpu(uint32_t devindex);

/**
 * Remove the device from the list of enabled devices. All data residing on the
 * device will be copied back on the main memory (or the memory declared as the
 * originator of the data), and all tasks owned by the device will be discarded
 * and moved back into the main scheduling mechanism.
 */
PARSEC_DECLSPEC int parsec_mca_device_remove(parsec_device_module_t* device);

/**
 * Reset the current devices statistics.
 */
PARSEC_DECLSPEC void parsec_mca_device_reset_statistics(parsec_context_t* parsec_context);

/**
 * Dump and reset the current devices statistics.
 */
PARSEC_DECLSPEC void parsec_mca_device_dump_and_reset_statistics(parsec_context_t* parsec_context);

/**
 * Save the current devices statistics in a (newly allocated in *pstats is
 * NULL) array of stats. If *pstat is not NULL, the same array is reused and
 * its values are overwritten.
 */
PARSEC_DECLSPEC void parsec_devices_save_statistics(uint64_t **pstats);

/**
 * Free a statistic array allocated by a prior call to
 * parsec_devices_save_statistics()
 */
PARSEC_DECLSPEC void parsec_devices_free_statistics(uint64_t **pstats);

/**
 * Print the current devices statistics; if start_stats is not NULL, the
 * procedure will substract start_stats before printing.
 */
PARSEC_DECLSPEC void parsec_devices_print_statistics(parsec_context_t *parsec_context, uint64_t *start_stats);

/**
 * Returns the number of devices currently registered with the runtime. This
 * number can change until parsec_mca_device_registration_complete() is
 * called, fact that mark the end of the configuration stage.
 */
static inline int parsec_mca_device_enabled(void)
{
    return parsec_nb_devices;
}

/**
 * Restrict the device type that can be used to execute the taskpool
 */
PARSEC_DECLSPEC void parsec_mca_device_taskpool_restrict( parsec_taskpool_t *tp,
                                                          uint8_t            device_type );

/**
 * Release all additional memory allocated on device.
 *
 * Device 0 (CPU) does not release the memory allocated on it,
 * only devices with local memory (e.g. GPUs) release temporary
 * buffers. This is used to start with an empty cache.
 */
PARSEC_DECLSPEC int parsec_devices_release_memory(void);

/**
 * Provides hints to a device about data
 *
 * Possible advices are PARSEC_DEV_DATA_ADVICE_*
 *   PREFETCH: a copy corresponding to the data should be prefetch
 *             on the device
 *   PREFERRED_DEVICE: this device is the preferred device to own
 *                     the data (may be used when selecting given
 *                     devices)
 *   WARMUP: if the device uses a cache policy, this tells that
 *           the data should be considered as recently used by
 *           the call.
 *
 * The advice may be ignored by the device. Each device sets their
 * policies wrt prefetching and caching through MCA parameters.
 */
PARSEC_DECLSPEC int parsec_advise_data_on_device(parsec_data_t *data, int device, int advice);

/**
 * Find a function is a set of shared libraries specified in paths. If a path
 * points to a directory, the libname is added to pinpoint to the expected shared
 * library. If no functions has been found on the paths the scope of the current
 * application is searched for the function_name. Upon success the pointer to the
 * function is returned, otherwise NULL.
 */
PARSEC_DECLSPEC void*
parsec_device_find_function(const char* function_name,
                            const char* libname,
                            const char* paths[]);

/**
 * Macro for use in components that are of type sched
 */
#define PARSEC_DEVICE_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
        "device", 2, 0, 0

/** @} */

END_C_DECLS

#endif  /* PARSEC_DEVICE_H_HAS_BEEN_INCLUDED */
