#ifndef PAPI_SDE_H_INCLUDED
#define PAPI_SDE_H_INCLUDED

typedef enum parsec_papi_sde_hl_counters_e {
    PARSEC_PAPI_SDE_MEM_ALLOC,               /**< How much memory is currently allocated by arenas */
    PARSEC_PAPI_SDE_MEM_USED,                /**< Out of MEM_ALLOC, how much memory is currently 
                                              *   used by 'active' data allocated in arenas */
    PARSEC_PAPI_SDE_SCHEDULER_PENDING_TASKS, /**< How many tasks are pending at a given time */
    PARSEC_PAPI_SDE_TASKS_ENABLED,           /**< How many tasks have become ready at this time */
    PARSEC_PAPI_SDE_TASKS_RETIRED            /**< How many tasks are done at this time */
} parsec_papi_sde_hl_counters_t;

#define PARSEC_PAPI_SDE_FIRST_EVENT  PARSEC_PAPI_SDE_MEM_ALLOC
#define PARSEC_PAPI_SDE_LAST_EVENT   PARSEC_PAPI_SDE_TASKS_RETIRED
#define PARSEC_PAPI_SDE_NB_HL_EVENTS  ( (int)PARSEC_PAPI_SDE_LAST_EVENT - (int)PARSEC_PAPI_SDE_FIRST_EVENT + 1 )

void parsec_papi_sde_init(void);
void parsec_papi_sde_enable_basic_events(int nb_threads);
void parsec_papi_sde_thread_init(void);
void parsec_papi_sde_counter_set(parsec_papi_sde_hl_counters_t cnt, long long int value);
void parsec_papi_sde_counter_add(parsec_papi_sde_hl_counters_t cnt, long long int value);

#endif /* PAPI_SDE_H_INCLUDED */
