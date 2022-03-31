#ifndef PAPI_SDE_H_INCLUDED
#define PAPI_SDE_H_INCLUDED

#include "parsec/parsec_config.h"

#if defined(PARSEC_PAPI_SDE)

#include "parsec/sde_lib.h"

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
void parsec_papi_sde_init(void);

/**
 * Finalize PAPI-SDE for PaRSEC
 *
 * @details
 *   release the papi handle, unregister the base counters and
 *   free any remaining thread-specific counters.
 */
void parsec_papi_sde_fini(void);

/**
 * Per-thread initialization function
 *
 * @details
 *   Create the TLS storage for each counter, chain the
 *   TLS block to the list of active threads
 */
void parsec_papi_sde_thread_init(void);

/**
 * Per-thread cleanup function
 *
 * @details
 *   removes the TLS block from the list of active threads
 *   and free the allocated resources for this thread.
 */
void parsec_papi_sde_thread_fini(void);

/**
 * Set the value of the base counter for the calling thread
 *
 *  @param[IN] cnt the counter to change
 *  @param[IN] value the new value 
 */
void parsec_papi_sde_counter_set(parsec_papi_sde_hl_counters_t cnt, long long int value);

/**
 * Change the value of the base counter for the calling thread
 *
 *  @param[IN] cnt the counter to change
 *  @param[IN] value the amount to add to the current value
 */
void parsec_papi_sde_counter_add(parsec_papi_sde_hl_counters_t cnt, long long int value);

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
void parsec_papi_sde_unregister_counter(const char *format, ...);

/**
 * @brief Register new PAPI SDE counter via parsec-provided function call
 * 
 */
void parsec_papi_sde_register_fp_counter(const char *event_name, int flags, int type, papi_sde_fptr_t fn, void *data);

/**
 * @brief Register new PAPI SDE counter with direct memory access
 * 
 */
void parsec_papi_sde_register_counter(const char *event_name, int flags, int type, long long int *ptr);

/**
 * @brief Establish hierarchy between PAPI SDE counters, and define aggregator counters
 * 
 */
void parsec_papi_sde_add_counter_to_group(const char *event_name, const char *group_name, int operand);

/**
 * @brief Provide a description for a given PAPI SDE counter
 * 
 */
void parsec_papi_sde_describe_counter(const char *event_name, const char *description);

#define PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN 256

#define PARSEC_PAPI_SDE_INIT()                                     parsec_papi_sde_init()
#define PARSEC_PAPI_SDE_FINI()                                     parsec_papi_sde_fini()
#define PARSEC_PAPI_SDE_THREAD_INIT()                              parsec_papi_sde_thread_init()
#define PARSEC_PAPI_SDE_THREAD_FINI()                              parsec_papi_sde_thread_fini()
#define PARSEC_PAPI_SDE_COUNTER_SET(_cnt, _value)                  parsec_papi_sde_counter_set((_cnt), (_value))
#define PARSEC_PAPI_SDE_COUNTER_ADD(_cnt, _value)                  parsec_papi_sde_counter_add((_cnt), (_value))
#define PARSEC_PAPI_SDE_UNREGISTER_COUNTER(args...)                parsec_papi_sde_unregister_counter(args)
#define PARSEC_PAPI_SDE_REGISTER_FP_COUNTER(_en, _fl, _t, _fn, _d) parsec_papi_sde_register_fp_counter((_en), (_fl), (_t), (_fn), (_d))
#define PARSEC_PAPI_SDE_REGISTER_COUNTER(_en, _fl, _t, _ptr)       parsec_papi_sde_register_counter((_en), (_fl), (_t), (_ptr))
#define PARSEC_PAPI_SDE_ADD_COUNTER_TO_GROUP(_en, _gn, _op)        parsec_papi_sde_add_counter_to_group((_en), (_gn), (_op))
#define PARSEC_PAPI_SDE_DESCRIBE_COUNTER(_en, _ds)                 parsec_papi_sde_describe_counter((_en), (_ds))


#else

#define PARSEC_PAPI_SDE_INIT()                                     do{}while(0)
#define PARSEC_PAPI_SDE_FINI()                                     do{}while(0)
#define PARSEC_PAPI_SDE_THREAD_INIT()                              do{}while(0)
#define PARSEC_PAPI_SDE_THREAD_FINI()                              do{}while(0)
#define PARSEC_PAPI_SDE_COUNTER_SET(cnt, value)                    do{}while(0)
#define PARSEC_PAPI_SDE_COUNTER_ADD(cnt, value)                    do{}while(0)
#define PARSEC_PAPI_SDE_UNREGISTER_COUNTER(...)                    do{}while(0)
#define PARSEC_PAPI_SDE_REGISTER_FP_COUNTER(_en, _fl, _t, _fn, _d) do{}while(0)
#define PARSEC_PAPI_SDE_REGISTER_COUNTER(_en, _fl, _t, _ptr)       do{}while(0)
#define PARSEC_PAPI_SDE_ADD_COUNTER_TO_GROUP(_en, _gn, _op)        do{}while(0)
#define PARSEC_PAPI_SDE_DESCRIBE_COUNTER(_en, _ds)                 do{}while(0)

#endif /* defined(PARSEC_PAPI_SDE) */

#endif /* PAPI_SDE_H_INCLUDED */
