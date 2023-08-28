/*
 * Copyright (c) 2021-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_GPU_H
#define PARSEC_DEVICE_GPU_H

#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/mca/device/device.h"

#include "parsec/class/list_item.h"
#include "parsec/class/list.h"
#include "parsec/class/fifo.h"

BEGIN_C_DECLS

#define PARSEC_GPU_USE_PRIORITIES     1
#define PARSEC_GPU_MAX_STREAMS        6
#define PARSEC_MAX_EVENTS_PER_STREAM  4
#define PARSEC_GPU_MAX_WORKSPACE      2

struct parsec_gpu_task_s;
typedef struct parsec_gpu_task_s parsec_gpu_task_t;

struct parsec_gpu_exec_stream_s;
typedef struct parsec_gpu_exec_stream_s parsec_gpu_exec_stream_t;

struct parsec_device_gpu_module_s;
typedef struct parsec_device_gpu_module_s parsec_device_gpu_module_t;

struct parsec_gpu_workspace_s;
typedef struct parsec_gpu_workspace_s parsec_gpu_workspace_t;

/**
 * Callback from the engine upon GPU event completion for each stage of a task.
 * The same prototype is used for calling the user provided submission function.
 */
typedef int (*parsec_complete_stage_function_t)(parsec_device_gpu_module_t  *gpu_device,
                                                parsec_gpu_task_t           **gpu_task,
                                                parsec_gpu_exec_stream_t     *gpu_stream);

/**
 *
 */
typedef int (*parsec_advance_task_function_t)(parsec_device_gpu_module_t  *gpu_device,
                                              parsec_gpu_task_t           *gpu_task,
                                              parsec_gpu_exec_stream_t    *gpu_stream);

/* Function type to transfer data to the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
typedef int (parsec_stage_in_function_t)(parsec_gpu_task_t        *gtask,
                                         uint32_t                  flow_mask,
                                         parsec_gpu_exec_stream_t *gpu_stream);


/* Function type to transfer data from the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
typedef int (parsec_stage_out_function_t)(parsec_gpu_task_t        *gtask,
                                          uint32_t                  flow_mask,
                                          parsec_gpu_exec_stream_t *gpu_stream);

struct parsec_gpu_task_s {
    parsec_list_item_t                list_item;
    uint16_t                          task_type;
    uint16_t                          pushout;
    int32_t                           last_status;
    parsec_advance_task_function_t    submit;
    parsec_complete_stage_function_t  complete_stage;
    parsec_stage_in_function_t       *stage_in;
    parsec_stage_out_function_t      *stage_out;
#if defined(PARSEC_PROF_TRACE)
    int                               prof_key_end;
    uint64_t                          prof_event_id;
    uint32_t                          prof_tp_id;
#endif
    union {
        struct {
            parsec_task_t            *ec;
            uint64_t                  last_data_check_epoch;
            uint64_t                  load;  /* computational load imposed on the device */
            /* These should be set by the DSL */
            const parsec_flow_t      *flow[MAX_PARAM_COUNT];  /* There is no consistent way to access the flows from the task_class,
                                                               * so the DSL need to provide these flows here.
                                                               */
            uint32_t                  flow_nb_elts[MAX_PARAM_COUNT]; /* for each flow, size of the data to be allocated
                                                                      * on the GPU.
                                                                      */
            parsec_data_collection_t *flow_dc[MAX_PARAM_COUNT];     /* for each flow, data collection from which the data
                                                                     * to be transferred logically belongs to.
                                                                     * This gives the user the chance to indicate on the JDF
                                                                     * a data collection to inspect during GPU transfer.
                                                                     * User may want info from the DC (e.g. mtype),
                                                                     * & otherwise remote copies don't have any info.
                                                                     */
            /* These are private and should not be used outside the device driver */
            parsec_data_copy_t       *sources[MAX_PARAM_COUNT];  /* If the driver decides to acquire the data from a different
                                                                  * source, it will temporary store the best candidate here.
                                                                  */
        };
        struct {
            parsec_data_copy_t        *copy;
        };
    };
};

typedef enum parsec_device_gpu_transfer_direction_e {
    parsec_device_gpu_transfer_direction_h2d,
    parsec_device_gpu_transfer_direction_d2h,
    parsec_device_gpu_transfer_direction_d2d
} parsec_device_gpu_transfer_direction_t;

/**
 * @brief Set the device for the calling thread.
 * 
 * @details typically maps to cudaSetDevice or equivalent
 * 
 * @return PARSEC_SUCCESS or a PARSEC error
 */
typedef int (*parsec_device_gpu_set_device_fn_t)(struct parsec_device_gpu_module_s *gpu);

/**
 * @brief Schedules the asynchronous copy of @p bytes bytes from @p source onto @p dest
 *    on the GPU stream of @p gpu_stream. @p direction must reflect the memory space of 
 *    @p source and @p dest.
 * 
 * @details typically maps to cudaMemcpyAsync or equivalent
 * 
 * @return PARSEC_SUCCESS or a PARSEC error
 */
typedef int (*parsec_device_gpu_memcpy_async_fn_t)(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream,
                                                   void *dest, void *source, size_t bytes, parsec_device_gpu_transfer_direction_t direction);

/**
 * @brief Record an event on the GPU @p gpu_stream of GPU @p gpu, with index @p idx.
 * 
 * @details typically maps to cudaRecordEvent or equivalent. The GPU device must have allocated
 *    @p gpu_stream->super.max_events previously (@p 0 <= event_idx < gpu_stream->super.max_events). 
 * 
 * @return PARSEC_SUCCESS or a PARSEC error
 */
typedef int (*parsec_device_gpu_event_record_fn_t)(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx);

/**
 * @brief Record an event on the GPU @p gpu_stream of GPU @p gpu, with index @p idx.
 * 
 * @details typically maps to cudaRecordEvent or equivalent. The GPU device must have allocated
 *    @p gpu_stream->super.max_events previously (@p 0 <= event_idx < gpu_stream->super.max_events). 
 * 
 * @return 0 if the event recorded at @p event_idx in @p gpu_stream is not ready yet
 *         1 if the event recorded at @p event_idx in @p gpu_stream is ready/completed
 *         a negative value which is a PARSEC error otherwise
 */
typedef int (*parsec_device_gpu_event_query_fn_t)(struct parsec_device_gpu_module_s *gpu, struct parsec_gpu_exec_stream_s *gpu_stream, int32_t event_idx);

/**
 * @brief Computes how much memory is available on the GPU. Returns two values:
 *   @p free_mem is the amount of memory available for this process
 *   @p total_mem is the amount of memory on the device (including memory allocated by other processes)
 * 
 * @details typically maps to cudaMemGetInfo or equivalent. 
 * 
 * @return PARSEC_SUCCESS if succesfull, a PARSEC error otherwise (in which case the parameters are undefined)
 */
typedef int (*parsec_device_gpu_memory_info_fn_t)(struct parsec_device_gpu_module_s *gpu, size_t *free_mem, size_t *total_mem);

/**
 * @brief Allocates @p bytes bytes on GPU @p gpu, and returns the address of the allocated memory in @p addr.
 * 
 * @details typically maps to cudaMalloc or equivalent. 
 * 
 * @return PARSEC_SUCCESS if succesfull, a PARSEC error otherwise (in which case @p addr is undefined)
 */
typedef int (*parsec_device_gpu_memory_allocate_fn_t)(struct parsec_device_gpu_module_s *gpu, size_t bytes, void **addr);

/**
 * @brief Frees memory @p addr allocated by @fn parsec_device_gpu_memory_allocate_fn_t on the same GPU @p gpu.
 * 
 * @details typically maps to cudaFree or equivalent. 
 * 
 * @return PARSEC_SUCCESS if succesfull, a PARSEC error otherwise
 */
typedef int (*parsec_device_gpu_memory_free_fn_t)(struct parsec_device_gpu_module_s *gpu, void *addr);

/**
 * @brief Find a function incarnation for the given function name
 * 
 * @param gpu_device the target GPU
 * @param fname the function name to look for
 * @return address of the symbol that implements this function
 */
typedef void* (*parsec_device_gpu_find_incarnation_fn_t)(parsec_device_gpu_module_t* gpu_device, const char* fname);

struct parsec_device_gpu_module_s {
    parsec_device_module_t     super;

    /* This set of base functions is used by the GPU devices to implement their Device Management Functions */
    parsec_device_gpu_set_device_fn_t       gpu_set_device;
    parsec_device_gpu_memcpy_async_fn_t     gpu_memcpy_async;
    parsec_device_gpu_event_query_fn_t      gpu_event_query;
    parsec_device_gpu_event_record_fn_t     gpu_event_record;
    parsec_device_gpu_memory_info_fn_t      gpu_memory_info;
    parsec_device_gpu_memory_allocate_fn_t  gpu_memory_allocate;
    parsec_device_gpu_memory_free_fn_t      gpu_memory_free;
    parsec_device_gpu_find_incarnation_fn_t gpu_find_incarnation;

    uint8_t                    max_exec_streams;
    uint8_t                    num_exec_streams;
    int16_t                    peer_access_mask;  /**< A bit set to 1 represent the capability of
                                                   *   the device to access directly the memory of
                                                   *   the index of the set bit device.
                                                   */
    volatile int32_t           mutex;
    uint64_t                   data_avail_epoch;  /**< Identifies the epoch of the data status on the device. It
                                                   *   is increased every time a new data is made available, so
                                                   *   that we know which tasks can be evaluated for submission.
                                                   */
    parsec_list_t              gpu_mem_lru;   /* Read-only blocks, and fresh blocks */
    parsec_list_t              gpu_mem_owned_lru;  /* Dirty blocks */
    parsec_fifo_t              pending;
    struct zone_malloc_s      *memory;
    parsec_list_item_t        *sort_starting_p;
    parsec_gpu_exec_stream_t **exec_stream;
    size_t                     mem_block_size;
    int64_t                    mem_nb_blocks;
};

struct parsec_gpu_exec_stream_s {
    struct parsec_gpu_task_s        **tasks;
    char                             *name;
    int32_t                           max_events;  /* number of potential events, and tasks */
    int32_t                           executed;    /* number of executed tasks */
    int32_t                           start;  /* circular buffer management start and end positions */
    int32_t                           end;
    parsec_list_t                    *fifo_pending;
    parsec_gpu_workspace_t           *workspace;
    parsec_info_object_array_t        infos; /**< Per-stream info objects are stored here */

#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_stream_t        *profiling;
    int                               prof_event_track_enable;
#endif  /* defined(PROFILING) */
};

typedef struct parsec_gpu_workspace_s {
    void* workspace[PARSEC_GPU_MAX_WORKSPACE];
    int stack_head;
    int total_workspace;
} parsec_gpu_workspace_t;

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/
PARSEC_DECLSPEC extern int parsec_gpu_output_stream;
PARSEC_DECLSPEC extern int parsec_gpu_verbosity;
PARSEC_DECLSPEC extern int32_t parsec_gpu_d2h_max_flows;

/**
 * Debugging functions.
 */
void dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream);
void dump_GPU_state(parsec_device_gpu_module_t* gpu_device);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef parsec_data_copy_t parsec_gpu_data_copy_t;

#include "parsec/data_distribution.h"

/* GPU workspace  ONLY works when PARSEC_ALLOC_GPU_PER_TILE is OFF */
int parsec_gpu_push_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream);
void* parsec_gpu_pop_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream, size_t size);
int parsec_gpu_free_workspace(parsec_device_gpu_module_t * gpu_device);

/* sort pending task list by number of spaces needed */
int parsec_gpu_sort_pending_list(parsec_device_gpu_module_t *gpu_device);
parsec_gpu_task_t* parsec_gpu_create_w2r_task(parsec_device_gpu_module_t *gpu_device, parsec_execution_stream_t *es);
int parsec_gpu_complete_w2r_task(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *w2r_task, parsec_execution_stream_t *es);

void parsec_gpu_enable_debug(void);

#if defined(PARSEC_DEBUG_VERBOSE)
char *parsec_gpu_describe_gpu_task( char *tmp, size_t len, parsec_gpu_task_t *gpu_task );
#endif

#define PARSEC_GPU_TASK_TYPE_KERNEL       0x0000
#define PARSEC_GPU_TASK_TYPE_D2HTRANSFER  0x1000
#define PARSEC_GPU_TASK_TYPE_PREFETCH     0x2000
#define PARSEC_GPU_TASK_TYPE_WARMUP       0x4000
#define PARSEC_GPU_TASK_TYPE_D2D_COMPLETE 0x8000

#if defined(PARSEC_PROF_TRACE)
#define PARSEC_PROFILE_GPU_TRACK_DATA_IN  0x0001
#define PARSEC_PROFILE_GPU_TRACK_DATA_OUT 0x0002
#define PARSEC_PROFILE_GPU_TRACK_OWN      0x0004
#define PARSEC_PROFILE_GPU_TRACK_EXEC     0x0008
#define PARSEC_PROFILE_GPU_TRACK_MEM_USE  0x0010
#define PARSEC_PROFILE_GPU_TRACK_PREFETCH 0x0020

extern int parsec_gpu_trackable_events;
extern int parsec_gpu_movein_key_start;
extern int parsec_gpu_movein_key_end;
extern int parsec_gpu_moveout_key_start;
extern int parsec_gpu_moveout_key_end;
extern int parsec_gpu_own_GPU_key_start;
extern int parsec_gpu_own_GPU_key_end;
extern int parsec_gpu_allocate_memory_key;
extern int parsec_gpu_free_memory_key;
extern int parsec_gpu_use_memory_key_start;
extern int parsec_gpu_use_memory_key_end;
extern int parsec_gpu_prefetch_key_start;
extern int parsec_gpu_prefetch_key_end;
extern int parsec_device_gpu_one_profiling_stream_per_gpu_stream;

void parsec_gpu_init_profiling(void);

typedef struct {
    uint64_t size;
    uint64_t data_key;
    uint64_t dc_id;
} parsec_device_gpu_memory_prof_info_t;
#define PARSEC_DEVICE_GPU_MEMORY_PROF_INFO_CONVERTER "size{int64_t};data_key{uint64_t};dc_id{uint64_t}"

#endif  /* defined(PROFILING) */

extern int parsec_gpu_sort_pending;

void dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream);
void dump_GPU_state(parsec_device_gpu_module_t* gpu_device);

int parsec_device_gpu_memory_reserve( parsec_device_gpu_module_t* gpu_device,
                                      int           memory_percentage,
                                      int           number_blocks,
                                      size_t        eltsize );
int parsec_gpu_attach( parsec_device_module_t* device, parsec_context_t* context );
int parsec_gpu_detach( parsec_device_module_t* device, parsec_context_t* context );
int parsec_gpu_taskpool_register(parsec_device_module_t* device, parsec_taskpool_t* tp);
int parsec_gpu_taskpool_unregister(parsec_device_module_t* device, parsec_taskpool_t* tp);
int parsec_gpu_data_advise(parsec_device_module_t *dev, parsec_data_t *data, int advice);
int parsec_gpu_flush_lru( parsec_device_module_t *device );
int parsec_gpu_memory_release( parsec_device_gpu_module_t* gpu_device );

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on GPU events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_gpu_kernel_scheduler( parsec_execution_stream_t *es,
                             parsec_gpu_task_t    *gpu_task,
                             int which_gpu );

/* Default stage_in function to transfer data to the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_in(parsec_gpu_task_t        *gtask,
                            uint32_t                  flow_mask,
                            parsec_gpu_exec_stream_t *gpu_stream);

/* Default stage_out function to transfer data from the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_out(parsec_gpu_task_t        *gtask,
                             uint32_t                  flow_mask,
                             parsec_gpu_exec_stream_t *gpu_stream);

END_C_DECLS

#endif //PARSEC_DEVICE_GPU_H
