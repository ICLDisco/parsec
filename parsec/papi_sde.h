#ifndef PAPI_SDE_H_INCLUDED
#define PAPI_SDE_H_INCLUDED

#include "parsec/parsec_config.h"

#if defined(PARSEC_PAPI_SDE)

#include "parsec/papi_sde_interface.h"

PARSEC_DECLSPEC extern papi_handle_t parsec_papi_sde_handle;

typedef enum parsec_papi_sde_hl_counters_e {
    PARSEC_PAPI_SDE_MEM_ALLOC,               /**< How much memory is currently allocated by arenas */
    PARSEC_PAPI_SDE_MEM_USED,                /**< Out of MEM_ALLOC, how much memory is currently 
                                              *   used by 'active' data allocated in arenas */
    PARSEC_PAPI_SDE_TASKS_ENABLED,           /**< How many tasks have become ready at this time */
    PARSEC_PAPI_SDE_TASKS_RETIRED,           /**< How many tasks are done at this time */
    PARSEC_PAPI_SDE_SCHEDULER_PENDING_TASKS, /**< How many tasks are pending at a given time */
    PARSEC_PAPI_SDE_NB_HL_COUNTERS           /**< This must remain last */
} parsec_papi_sde_hl_counters_t;

#define PARSEC_PAPI_SDE_FIRST_BASIC_COUNTER  PARSEC_PAPI_SDE_MEM_ALLOC
#define PARSEC_PAPI_SDE_LAST_BASIC_COUNTER   PARSEC_PAPI_SDE_TASKS_RETIRED
#define PARSEC_PAPI_SDE_NB_BASIC_COUNTERS    ( (int)PARSEC_PAPI_SDE_LAST_BASIC_COUNTER - (int)PARSEC_PAPI_SDE_FIRST_BASIC_COUNTER + 1)

/**
 * Initialize PAPI-SDE for PaRSEC
 *
 * @details
 *   create the papi handle, and provide the description for the
 *   base counters. Register and enable the base counters
 */
void PARSEC_PAPI_SDE_INIT(void);

/**
 * Finalize PAPI-SDE for PaRSEC
 *
 * @details
 *   release the papi handle, unregister the base counters and
 *   free any remaining thread-specific counters.
 */
void PARSEC_PAPI_SDE_FINI(void);

/**
 * Per-thread initialization function
 *
 * @details
 *   Create the TLS storage for each counter, chain the
 *   TLS block to the list of active threads
 */
void PARSEC_PAPI_SDE_THREAD_INIT(void);

/**
 * Per-thread cleanup function
 *
 * @details
 *   removes the TLS block from the list of active threads
 *   and free the allocated resources for this thread.
 */
void PARSEC_PAPI_SDE_THREAD_FINI(void);

/**
 * Set the value of the base counter for the calling thread
 *
 *  @param[IN] cnt the counter to change
 *  @param[IN] value the new value 
 */
void PARSEC_PAPI_SDE_COUNTER_SET(parsec_papi_sde_hl_counters_t cnt, long long int value);

/**
 * Change the value of the base counter for the calling thread
 *
 *  @param[IN] cnt the counter to change
 *  @param[IN] value the amount to add to the current value
 */
void PARSEC_PAPI_SDE_COUNTER_ADD(parsec_papi_sde_hl_counters_t cnt, long long int value);

/**
 * Helper function to unregister a PAPI-SDE counter
 *
 *  @details
 *    this function is just a wrapper to papi_sde_unregister_counter that
 *    merges counter name construction with the unregister call.
 *
 *    @param[IN] format: a printf-like format, defining the counter name
 *       to unregister.
 */
void PARSEC_PAPI_SDE_UNREGISTER_COUNTER(const char *format, ...);

#define PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN 256

#else

#define PARSEC_PAPI_SDE_INIT()                                  do{}while(0)
#define PARSEC_PAPI_SDE_FINI()                                  do{}while(0)
#define PARSEC_PAPI_SDE_THREAD_INIT()                           do{}while(0)
#define PARSEC_PAPI_SDE_THREAD_FINI()                           do{}while(0)
#define PARSEC_PAPI_SDE_COUNTER_SET(cnt, value)                 do{}while(0)
#define PARSEC_PAPI_SDE_COUNTER_ADD(cnt, value)                 do{}while(0)
#define PARSEC_PAPI_SDE_UNREGISTER_COUNTER(...)                 do{}while(0)

#endif /* defined(PARSEC_PAPI_SDE) */

#endif /* PAPI_SDE_H_INCLUDED */
