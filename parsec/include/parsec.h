/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_H_HAS_BEEN_INCLUDED
#define PARSEC_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/debug.h"

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

/** @brief Define the PaRSEC major version number */
#define PARSEC_VERSION    2
/** @brief Define the PaRSEC minor version number */
#define PARSEC_SUBVERSION 0

/** $brief To check if any parsec function returned error.
  *        Should be used by users to check correctness.
  */
#define PARSEC_CHECK_ERROR(rc, MSG) \
        if( rc < 0 ) {            \
            parsec_warning( "**** Error occurred in file: %s"   \
                            ":%d : "                            \
                            "%s", __FILE__, __LINE__, MSG );    \
            exit(-1);                                           \
        }                                                       \

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
 * @brief Change the communicator to use with the context
 *
 * @details
 * Reset the remote_dep comm engine associated with the PaRSEC context, and use
 * the communication context opaque_comm_ctx in the future (typically an MPI
 * communicator).
 *
 * parsec_context_wait becomes collective accross nodes spanning
 * on this communication context.
 *
 * @param[inout] context the PaRSEC context
 * @param[in] opaque_comm_ctx the new communicator object to use
 * @return PARSEC_SUCCESS on success
 */
int parsec_remote_dep_set_ctx( parsec_context_t* context, void* opaque_comm_ctx );


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
 * @param[inout] dag The parsec object with pending tasks.
 *
 * @return 0 If the enqueue operation succeeded.
 */
int parsec_enqueue( parsec_context_t* context , parsec_taskpool_t* tp);

/**
 * @brief Start taskpool that were enqueued into the PaRSEC context
 *
 * @details
 * Start the runtime by allowing all the other threads to start executing.
 * This call should be paired with one of the completion calls, test or wait.
 *
 * @param[inout] context the PaRSEC context
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads hav
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
 * @return 0 If the execution is completed.
 * @return * Any other error raised by the tasks themselves.
 */
int parsec_context_wait(parsec_context_t* context);

/**
 * @brief taskpool-callback type definition
 *
 * @details
 * The completion callback of a parsec_taskpool. Once the taskpoolhas been
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
 * @brief Setter for the completion callback and data
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

int parsec_set_complete_callback(parsec_taskpool_t* parsec_tp,
                                parsec_event_cb_t complete_cb, void* complete_data);

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
 * @return PARSEC_SUCCESS on success
 */
int parsec_get_complete_callback(const parsec_taskpool_t* parsec_tp,
                                parsec_event_cb_t* complete_cb, void** complete_data);


/**
 * @brief Setter for the enqueuing callback and data
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
int parsec_set_enqueue_callback(parsec_taskpool_t* parsec_tp,
                               parsec_event_cb_t enqueue_cb, void* enqueue_data);

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
 * @return PARSEC_SUCCESS on success
 */
int parsec_get_enqueue_callback(const parsec_taskpool_t* parsec_tp,
                               parsec_event_cb_t* enqueue_cb, void** enqueue_data);

/**
 * @brief Retrieve the local object attached to a unique object id
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
 * connected objects must have the same ID.
 *
 * @param[in] taskpool the PaRSEC taskpool for which an ID should be reserved
 * @return the taskpool ID of taskpool (allocates one if taskpool has no taskpool_id)
 */
int parsec_taskpool_reserve_id(parsec_taskpool_t* tp);

/**
 * @brief Register the object with the engine.
 *
 * @details
 * Register the object with the engine. The taskpool must have a unique taskpool, especially
 * in a distributed environment.
 *
 * @param[in] taskpool the taskpoo lto register
 * @return PARSEC_SUCCESS on success, an error otherwise
 */
int parsec_taskpool_register(parsec_taskpool_t* tp);

/**
 * @brief Unregister the object with the engine.
 *
 * @details
 * Unregister the object with the engine. This make the taskpool ID available for
 * future taskpools. Beware that in a distributed environment the connected objects
 * must have the same ID.
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
 *  id at all ranks. This is a collective over the communication object
 *  associated with PaRSEC, and can be used to resolve discrepancies introduced by
 *  taskpools not registered over all ranks.
*/
void parsec_taskpool_sync_ids(void);

/**
 * @cond FALSE
 * Sequentially compose two taskpools, triggering the start of next upon
 * completion of start. If start is already a composed object, then next will be
 * appended to the already existing list. These taskpoolss will execute one after
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
int32_t parsec_set_priority( parsec_taskpool_t* taskpool, int32_t new_priority );

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
char* parsec_snprintf_execution_context( char* str, size_t size,
                                        const parsec_task_t* task);

/**
 * @brief Opaque structure representing a Task Class
 */
struct parsec_task_class_s;

/**
 * @brief Opaque structure representing the parameters of a task
 */
struct assignment_s;

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
                                  const struct assignment_s* locals);

/**  @} */

END_C_DECLS

#endif  /* PARSEC_H_HAS_BEEN_INCLUDED */
