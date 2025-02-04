/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2025      NVIDIA Corporation.  All rights reserved.
 */

#ifndef PARSEC_RUNTIME_H_HAS_BEEN_INCLUDED
#define PARSEC_RUNTIME_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/list.h"

BEGIN_C_DECLS

/**
 * @defgroup parsec_public_runtime Runtime System
 * @ingroup parsec_public
 *   PaRSEC Core routines belonging to the PaRSEC Runtime System.
 *
 *   This is the public API of the PaRSEC runtime system. Functions of
 *   this module are used to manipulate PaRSEC highest level objects.
 *
 *  @{
 */


/**
 * Arena-datatype management.
 */
typedef struct parsec_arena_datatype_s parsec_arena_datatype_t;

/**
 * @brief Define a weak symbol called by PaRSEC in case of fatal error.
 * Can be overwritten by the user level to catch and handle errors.
 * However, this function must not return.
 */
extern void (*parsec_weaksym_exit)(int status);

/**
 * @brief Defines a DAG of tasks
 */
typedef struct parsec_taskpool_s            parsec_taskpool_t;

/**
 * @brief Defines a Task
 */
typedef struct parsec_task_s parsec_task_t;

/**
 * @brief Represents a computing element (usually a system thread)
*/
typedef struct parsec_execution_stream_s    parsec_execution_stream_t;

/**
 * @brief Holds all the resources used by PaRSEC for this process (threads, memory, ...)
 */
typedef struct parsec_context_s           parsec_context_t;

/**
 * @brief Defines shape, allocator and deallocator of temporary data transferred on the network
 */
typedef struct parsec_arena_s             parsec_arena_t;

/**
 * @brief Opaque structure representing a Task Class
 */
typedef struct parsec_task_class_s      parsec_task_class_t;


/**
 * @brief Prototype of a external fini function
 */
typedef void (*parsec_external_fini_cb_t)(void*);

/**
 * @brief External fini function & data
 */
typedef struct parsec_external_fini_s {
    parsec_external_fini_cb_t  cb;   /**< external fini callback */
    void                      *data; /**< external fini callback args */
}parsec_external_fini_t;

/**
 * @brief Prototype of the allocator function
 */
typedef void* (*parsec_data_allocate_t)(size_t matrix_size);

/**
 * @brief Prototype of the deallocator function
 */
typedef void (*parsec_data_free_t)(void *data);

/**
 * @brief Global allocator function that PaRSEC uses (defaults to libc malloc)
 */
extern parsec_data_allocate_t parsec_data_allocate;

/**
 * @brief Global deallocator function that PaRSEC uses (defaults to libc free)
 */
extern parsec_data_free_t     parsec_data_free;

/**
 * @brief A Remote dependency
 */
typedef struct parsec_remote_deps_s     parsec_remote_deps_t;

/**
 * @brief Data and data copy description.
 */
typedef struct parsec_data_s parsec_data_t;
typedef struct parsec_data_copy_s parsec_data_copy_t;

/**
 * @brief Data collection.
 */
typedef struct parsec_data_collection_s parsec_data_collection_t;
typedef parsec_data_collection_t parsec_dc_t;

/**
 * @brief A description of the content of each data movement/copy
 */
typedef struct parsec_dep_data_description_s  parsec_dep_data_description_t;
typedef struct parsec_dep_type_description_s  parsec_dep_type_description_t;

/**
 * @brief A description of the reshape promise
 */
typedef struct parsec_reshape_promise_description_s parsec_reshape_promise_description_t;

/**
 * @brief A description of the thread private memory pool.
 */
typedef struct parsec_thread_mempool_s parsec_thread_mempool_t;

/**
 *
 */
typedef enum parsec_hook_return_e {
    PARSEC_HOOK_RETURN_DONE    =  0,  /* This execution succeeded */
    PARSEC_HOOK_RETURN_AGAIN   = -1,  /* Reschedule later */
    PARSEC_HOOK_RETURN_NEXT    = -2,  /* Try next variant [if any] */
    PARSEC_HOOK_RETURN_DISABLE = -3,  /* Disable the device, something went wrong */
    PARSEC_HOOK_RETURN_ASYNC   = -4,  /* The task is outside our reach, the completion will
                                       * be triggered asynchronously. */
    PARSEC_HOOK_RETURN_ERROR   = -5,  /* Some other major error happened */
} parsec_hook_return_t;

/* In order for the reshaping to work, retry codes should be negative
 * completion ones positive. */
#define PARSEC_HOOK_RETURN_RESHAPE_DONE 1
#define PARSEC_HOOK_RETURN_DONE_NO_RESHAPE 0


/**
 * @brief Create a new PaRSEC execution context
 *
 * @details
 * Create a new execution context, using the number of resources passed
 * with the arguments. Every execution happend in the context of such an
 * execution context. Several contextes can cohexist on disjoint resources
 * in same time.
 *
 * @param[in]    nb_cores the number of cores to use
 * @param[inout] pargc a pointer to the number of arguments passed in pargv
 * @param[inout] pargv an argv-like NULL terminated array of arguments to pass to
 *        the PaRSEC engine.
 * @return the newly created PaRSEC context
 */
parsec_context_t* parsec_init( int nb_cores, int* pargc, char** pargv[]);

/**
 * @brief Obtain the version number of the PaRSEC runtime
 *
 * @details
 * Obtain the version number of the PaRSEC runtime
 *
 * @param[out]   version_major a pointer to the major version number (i.e., 3, in version 3.0.1911)
 * @param[out]   version_minor a pointer to the minor version number (i.e., 0 in version 3.0.1911)
 * @param[out]   version_release a pointer to the patch version number (i.e., 1911 in version 3.0.1911)
 *               Unreleased (e.g., git master) versions will have patch=0
 *
 * @return PARSEC_SUCCESS on success
 */
int parsec_version( int* version_major, int* version_minor, int* version_release);

/**
 * @brief Obtain the version string describing important options used when
 * compiling the PaRSEC runtime
 *
 * @details
 * Obtain the version of the version string describing important options used when compiling
 * the PaRSEC runtime
 *
 * @param[in]    len the length of the output array (in char)
 * @param[out]   version_string a pointer to the array in which the description is output. When 
 *               the version_string is longer than `len`, the output is truncated.
 *
 * @return PARSEC_SUCCESS on success
 * @return PARSEC_ERR_VALUE_OUT_OF_BOUNDS when version_string is truncated
 */
int parsec_version_ex( size_t len, char* version_string);

/**
 * @brief Change the communicator to use with the context. This function is
 * collective across all processes in this context.
 *
 * @details
 * Reset the comm engine associated with the PaRSEC context, and use
 * the communication context opaque_comm_ctx in the future (typically an MPI
 * communicator). The context can only be changed while the PaRSEC runtime
 * is down, more specifically while the communication thread is not active.
 *
 * parsec_context_wait becomes collective across nodes spanning
 * on this communication context.
 *
 * @param[inout] context the PaRSEC context
 * @param[in] opaque_comm_ctx the new communicator object to use
 * @return PARSEC_SUCCESS on success
 */
int parsec_remote_dep_set_ctx( parsec_context_t* context, intptr_t opaque_comm_ctx );


/**
 * @brief Abort PaRSEC context
 *
 * @details
 * Aborts the PaRSEC context. The execution stops at resources on which the
 * context spans. The call does not return.
 *
 * @param[in] pcontext a pointer to the PaRSEC context to abort
 * @param[in] status an integer value transmitted to the OS specific abort
 * method (an exit code)
 * @return this call does not return.
 */
void parsec_abort( parsec_context_t* pcontext, int status);


/**
 * @brief Finalize a PaRSEC context
 *
 * @details
 * Complete all pending operations on the execution context, and release
 * all associated resources. Threads and acclerators attached to this
 * context will be released.
 *
 * @param[inout] pcontext a pointer to the PaRSEC context to finalize
 * @return PARSEC_SUCCESS on success
 */
int parsec_fini( parsec_context_t** pcontext );

/**
 * Setup external finilize routine to be callback during parsec_fini
 */
void parsec_context_at_fini(parsec_external_fini_cb_t cb, void *data);

/**
 * @brief Enqueue a PaRSEC taskpool into the PaRSEC context
 *
 * @details
 * Attach an execution taskpool on a context, in other words on the set of
 * resources associated to this particular context. A matching between
 * the capabilitis of the context and the support from the taskpool will be
 * done during this step, which will basically define if accelerators can
 * be used for the execution.
 *
 * @param[inout] context The parsec context where the tasks generated by the parsec_taskpool_t
 *                are to be executed.
 * @param[inout] tp The parsec taskpool with pending tasks.
 *
 * @return PARSEC_SUCCESS If the enqueue operation succeeded.
 */
int parsec_context_add_taskpool( parsec_context_t* context, parsec_taskpool_t* tp);

/**
 * @brief Detaches a PaRSEC taskpool from the PaRSEC context
 *
 * @details
 * Detaches an execution taskpool from the context it was attached to. The taskpool must be terminated
 * (i.e. waited upon), and no pending tasks or internal runtime actions can be
 * pending on the taskpool.
 *
 * @param[inout] tp The parsec taskpool to be detached.
 *
 * @return PARSEC_SUCCESS If the dequeue operation succeeded.
 */
int parsec_context_remove_taskpool( parsec_taskpool_t* tp );

/**
 * Query PaRSEC context capabilities.
 */

typedef enum parsec_context_query_cmd_e {
    PARSEC_CONTEXT_QUERY_NODES,
    PARSEC_CONTEXT_QUERY_RANK,
    PARSEC_CONTEXT_QUERY_DEVICES,
    PARSEC_CONTEXT_QUERY_DEVICES_FULL_PEER_ACCESS,
    PARSEC_CONTEXT_QUERY_CORES,
    PARSEC_CONTEXT_QUERY_ACTIVE_TASKPOOLS
} parsec_context_query_cmd_t;

/**
 * @brief Query PaRSEC context's properties.
 *
 * @details
 * Query properties of the runtime, such as number of devices of a certain type
 * or number of cores available to the context.
 *
 * @param[in] context the PaRSEC context
 * @param[in] device_type the type of device the query is about
 * @return PARSEC_ERR_NOT_SUPPORTED if the command is not supported, PARSEC_ERR_NOT_FOUND
 *         if the correct answer cannot yet be returned (such as when the PaRSEC context
 *         has not yet properly been initialized), or the answer to the query (always
 *         a positive number).
 */
int parsec_context_query(parsec_context_t* context, parsec_context_query_cmd_t cmd, ... );

/**
 * @brief Start taskpool that were enqueued into the PaRSEC context
 *
 * @details
 * Start the runtime by allowing all the other threads to start executing.
 * This call should be paired with one of the completion calls, test or wait.
 *
 * @param[inout] context the PaRSEC context
 * @return 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and nothing has
 * been activated.
 */
int parsec_context_start(parsec_context_t* context);

/**
 * @brief Check the status of an ongoing execution, started with parsec_start
 *
 * @details
 * Check the status of an ongoing execution, started with parsec_start
 *
 * @param[inout] context The parsec context where the execution is taking place.
 *
 * @return 0 If the execution is still ongoing.
 * @return 1 If the execution is completed, and the parsec_context has no
 *           more pending tasks. All subsequent calls on the same context
 *           will automatically succeed.
 */
int parsec_context_test( parsec_context_t* context );

/**
 * @brief Complete the execution of all PaRSEC taskpools enqueued into this
 *        PaRSEC context
 *
 * @details
 * Progress the execution context until no further operations are available.
 * Upon return from this function, all resources (threads and accelerators)
 * associated with the corresponding context are put in a mode where they are
 * not active. New taskpools enqueued during the progress stage are automatically
 * taken into account, and the caller of this function will not return to the
 * user until all pending taskpools are completed and all other threads are in a
 * sleeping mode.
 *
 * @param[inout] context The parsec context where the execution is taking place.
 *
 * @return PARSEC_SUCCESS   The context has completed and all work associated
 *                          with the context is done.
 * @return less than PARSEC_SUCCESS If something went wrong.
 */
int parsec_context_wait(parsec_context_t* context);

/**
 * @brief Complete the execution of a given PaRSEC taskpool
 *
 * @details
 * Progress the execution context until the given taskpool reaches termination.
 * Upon return from this function, all resources (threads and accelerators)
 * associated with the corresponding context are left in a mode where they are
 * active iff more taskpools are still active. The taskpool must be ready and
 * registered with a started context.
 *
 * @param[inout] tp the taskpool to complete.
 *
 * @return * A negative number to signal an error. Any other value, aka. a positive
 *           number (including 0), to signal successful completion of all work
 *           associated with the taskpool.
 */
int parsec_taskpool_wait(parsec_taskpool_t* tp);

/**
 * @brief Allow the main thread to temporarily join the computation
 *    by executing one (or less) task
 *
 * @details
 * Try to progress the execution context until the given taskpool reaches termination.
 * Upon return from this function, all resources (threads and accelerators)
 * associated with the corresponding context are left in a mode where they are
 * active. The taskpool must be ready and registered with a started context.
 *
 * @param[inout] tp the taskpool to complete.
 *
 * @return * A negative number to signal an error. 0 if the taskpool is completed.
 *    a strictly positive value if some task was executed.
 */
int parsec_taskpool_test(parsec_taskpool_t* tp);

/**
 * @brief taskpool-callback type definition
 *
 * @details
 * The completion callback of a parsec_taskpool. Once the taskpool has been
 * completed, i.e. all the local tasks associated with the taskpool have
 * been executed, and before the taskpool is marked as done, this callback
 * will be triggered. Inside the callback the taskpool should not be
 * modified.
 */
typedef int (*parsec_event_cb_t)(parsec_taskpool_t* parsec_tp, void*);

/**
 * @brief Hook to update runtime task for each taskpool type
 *
 * @details
 * Each PaRSEC taskpool has a counter nb_pending_action and to update
 * that counter we need a function type that will act as a hook. Each
 * taskpool type can attach it's own callback to get desired way to
 * update the nb_pending_action.
 */
typedef int (*parsec_update_ref_t)(parsec_taskpool_t *parsec_tp, int32_t);

/**
 * @brief Overwrite the completion callback and callback data for the
 *        taskpool. This action replaces the old callback, it must be
 *        explicitly saved if necessary.
 *
 * @details
 * Sets the complete callback of a PaRSEC taskpool
 *
 * @param[inout] parsec_taskpool the taskpool on which the callback should
 *               be attached
 * @param[in] complete_cb the function to call when parsec_taskpool is completed
 * @param[inout] complete_data the user-defined data to passe to complete_cb
 *               when it is called
 * @return PARSEC_SUCCESS on success, an error otherwise
 */

int
parsec_taskpool_set_complete_callback(parsec_taskpool_t* parsec_tp,
                                      parsec_event_cb_t complete_cb,
                                      void* complete_data);

/**
 * @brief Get the current completion callback associated with a PaRSEC taskpool
 *
 * @details
 * Returns the current completion callback associated with a parsec_taskpool.
 * Typically used to chain callbacks together, when inserting a new completion
 * callback.
 *
 * @param[in] parsec_taskpool the PaRSEC taskpool on which a callback is set
 * @param[out] complete_cb a function pointer to the corresponding callback
 * @param[out] complete_data a pointer to the data called with the callback
 * @return PARSEC_SUCCESS
 */
int
parsec_taskpool_get_complete_callback(const parsec_taskpool_t* parsec_tp,
                                      parsec_event_cb_t* complete_cb,
                                      void** complete_data);


/**
 * @brief Overwrite the enqueue callback and callback data for the
 *        taskpool. This action replaces the old callback, it must be
 *        explicitly saved if necessary.
 *
 * @details
 * Sets the enqueuing callback of a PaRSEC taskpool
 *
 * @param[inout] parsec_taskpool the taskpool on which the callback should
 *               be attached
 * @param[in] enqueue_cb the function to call when parsec_taskpool is enqueued
 * @param[inout] enqueue_data the user-defined data to passe to enqueue_cb
 *               when it is called
 * @return PARSEC_SUCCESS on success, an error otherwise
 */
int
parsec_taskpool_set_enqueue_callback(parsec_taskpool_t* parsec_tp,
                                     parsec_event_cb_t enqueue_cb,
                                     void* enqueue_data);

/**
 * @brief Get the current enqueuing callback associated with a PaRSEC taskpool
 *
 * @details
 * Returns the current completion callback associated with a parsec_taskpool.
 * Typically used to chain callbacks together, when inserting a new enqueuing
 * callback.
 *
 * @param[in] parsec_taskpool the PaRSEC taskpool on which a callback is set
 * @param[out] enqueue_cb a function pointer to the corresponding callback
 * @param[out] enqueue_data a pointer to the data called with the callback
 * @return PARSEC_SUCCESS
 */
int
parsec_taskpool_get_enqueue_callback(const parsec_taskpool_t* parsec_tp,
                                     parsec_event_cb_t* enqueue_cb,
                                     void** enqueue_data);

/**
 * @brief Retrieve the local taskpool attached to a unique taskpool id
 *
 * @details
 * Converts a taskpool id into the corresponding taskpool.
 *
 * @param[in] taskpool_id the taskpool id to lookup
 * @return NULL if no taskpool is associated with this taskpool_id,
 *         the PaRSEC taskpool pointer otherwise
 */
parsec_taskpool_t* parsec_taskpool_lookup(uint32_t taskpool_id);

/**
 * @brief Reserve a unique ID of a taskpool.
 *
 * @details
 * Reverse an unique ID for the taskpool. Beware that on a distributed environment the
 * connected taskpools must have the same ID.
 *
 * @param[in] taskpool the PaRSEC taskpool for which an ID should be reserved
 * @return the taskpool ID of taskpool (allocates one if taskpool has no taskpool_id)
 */
int parsec_taskpool_reserve_id(parsec_taskpool_t* tp);

/**
 * @brief Register the taskpool with the engine.
 *
 * @details
 * Register the taskpool with the engine. The taskpool must have a unique taskpool, especially
 * in a distributed environment.
 *
 * @param[in] taskpool the taskpoo lto register
 * @return PARSEC_SUCCESS on success, an error otherwise
 */
int parsec_taskpool_register(parsec_taskpool_t* tp);

/**
 * @brief Unregister the taskpool from the engine.
 *
 * @details
 * Unregister the taskpool from the engine. This make the taskpool ID available for
 * future taskpools. Beware that in a distributed environment the connected
 * taskpools must have the same ID.
 *
 * @param[in] taskpool the taskpool to unregister
 * @return PARSEC_SUCCESS on success, an error otherwise
 */
void parsec_taskpool_unregister(parsec_taskpool_t* tp);

/**
 * @brief Globally synchronize taskpool IDs.
 *
 * @details
 *  Globally synchronize taskpool IDs so that next register generates the same
 *  id at all ranks on a given communicator. This is a collective over the communication object
 *  associated with PaRSEC, and can be used to resolve discrepancies introduced by
 *  taskpools not registered over all ranks.
*/
void parsec_taskpool_sync_ids_context( intptr_t comm );

/**
 * @brief Returns the execution stream that corresponds to the calling thread
 *
 * @details
 *  Threads created by parsec during parsec_init register their execution
 *  streams in a Thread Local Storage variable. These threads can lookup
 *  the execution stream using this function. The behavior is undefined
 *  if calling before parsec_init, after parsec_fini, or at any time if
 *  calling from a thread that was not created by parsec or did not call
 *  parsec_init
 */
parsec_execution_stream_t *parsec_my_execution_stream(void);

/**
 * @cond FALSE
 * Sequentially compose two taskpools, triggering the start of next upon
 * completion of start. If start is already a composed taskpool, then next will be
 * appended to the already existing list. These taskpools will execute one after
 * another as if there were sequential.  The resulting compound parsec_taskpool is
 * returned.
 */
parsec_taskpool_t* parsec_compose(parsec_taskpool_t* start, parsec_taskpool_t* next);
/** @endcond */

/**
 * @brief Free the resource allocated in the parsec taskpool.
 *
 * @details
 * Free the resource allocated in the parsec taskpool. The taskpool should be unregistered first.
 * @param[inout] taskpool the taskpool to free
 */
void parsec_taskpool_free(parsec_taskpool_t *tp);

/**
 * @private
 * @brief The final step of a taskpool activation.
 *
 * @details
 * The final step of a taskpool activation. At this point we assume that all the local
 * initializations have been successfully completed for all components, and that the
 * taskpool is ready to be registered with the system, and any potential pending tasks
 * ready to go. If distributed is non 0, then the runtime assumes that the taskpool has
 * a distributed scope and should be registered with the communication engine.
 *
 * The local_task allows for concurrent management of the startup_queue, and provide a way
 * to prevent a task from being added to the scheduler. As the different tasks classes are
 * initialized concurrently, we need a way to prevent the beginning of the tasks generation until
 * all the tasks classes associated with a DAG are completed. Thus, until the synchronization
 * is complete, the task generators are put on hold in the startup_queue. Once the taskpool is
 * ready to advance, and this is the same moment as when the taskpool is ready to be enabled,
 * we reactivate all pending tasks, starting the tasks generation step for all type classes.
 *
 * @param[inout] taskpool the taskpool to enable
 * @param[in] startup_queue a list of tasks that should be fed to the
 *            ready tasks of eu
 * @param[in] local_task the task that is calling parsec_taskpool_enable, and
 *            that might be included in the startup_queue
 * @param[inout] eu the execution unit on which the tasks should be enqueued
 * @param[in] distributed 0 if that taskpool is local, non zero if it exists on all ranks.
 */
int parsec_taskpool_enable(parsec_taskpool_t* tp,
                           parsec_task_t** startup_queue,
                           parsec_task_t* local_task,
                           parsec_execution_stream_t* es,
                           int distributed);

/**
 * @brief Print PaRSEC usage message.
 *
 * @details
 * Print PaRSEC Modular Component Architecture help message.
 */
void parsec_usage(void);

/**
 * @brief Change the priority of an entire taskpool
 *
 * @details
 * Allow to change the default priority of a taskpool. It returns the
 * old priority (the default priority of a taskpool is 0). This function
 * can be used during the lifetime of a taskpool, however, only tasks
 * generated after this call will be impacted.
 *
 * @param[inout] taskpool the taskpool to bump in priority
 * @param[in] new_priority the new priority to set to that taskpool
 * @return The priority of the taskpool before being assigned to new_priority
 */
int32_t parsec_taskpool_set_priority( parsec_taskpool_t* taskpool, int32_t new_priority );

/**
 * @brief Human-readable print function for tasks
 *
 * @details
 * Prints up to size bytes into str, to provide a human-readable description
 * of task.
 * @param[out] str the string that should be initialized
 * @param[in] size the maximum bytes in str
 * @param[in] task the task to represent with str
 * @return str
 */
char* parsec_task_snprintf( char* str, size_t size,
                            const parsec_task_t* task);

/**
 * @brief Opaque structure representing a Task Class
 */
struct parsec_task_class_s;

/**
 * @brief Opaque structure representing the parameters of a task
 */
struct parsec_assignment_s;

/**
 * @brief Human-readable print function for task parameters
 *
 * @details
 * Prints up to size bytes into str, to provide a human-readable description
 * of the task parameters locals, assuming they belong to the task class
 * function.
 *
 * @param[out] str a buffer of size bytes in which to set the assignment representation
 * @param[in] size the number of bytes of str
 * @param[in] function the task class to which locals belong
 * @param[in] locals the set of parameters of the task
 * @return str
 */
char* parsec_snprintf_assignments(char* str, size_t size,
                                  const struct parsec_task_class_s* function,
                                  const struct parsec_assignment_s* locals);

/**  @} */

END_C_DECLS

/* Temporary support for deprecated features. */
#include "parsec/deprecated.h"

#endif  /* PARSEC_RUNTIME_H_HAS_BEEN_INCLUDED */
