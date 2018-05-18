/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DATAREPO_H_HAS_BEEN_INCLUDED
#define PARSEC_DATAREPO_H_HAS_BEEN_INCLUDED

#include "parsec/runtime.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_hash_table.h"

/** @defgroup parsec_internal_datarepo Data Repositories
 *  @ingroup parsec_internal
 *    Data Repositories store data objects into hash tables
 *    for tasks to retrieve them when they become schedulable.
 *  @addtogroup parsec_internal_datarepo
 *  @{
 */

typedef struct data_repo_entry_s data_repo_entry_t;
typedef struct data_repo_head_s  data_repo_head_t;

/**
 * Hash table:
 *  Because threads are allowed to use elements deposited in the hash table
 *  while we are still discovering how many times these elements will be used,
 *  it is necessary to
 *     - use a positive reference counting method only.
 *       Think for example if 10 threads decrement the reference counter, while
 *       the thread that pushes the element is still counting. The reference counter
 *       goes negative. Bad.
 *     - use a retaining facility.
 *       Think for example that the thread that pushes the element computed for now that
 *       the limit is going to be 3. While it's still exploring the dependencies, other
 *       threads use this element 3 times. The element is going to be removed while the
 *       pushing thread is still exploring, and SEGFAULT will occur.
 *
 *  An alternative solution consisted in having a function that will compute how many
 *  times the element will be used at creation time, and keep this for the whole life
 *  of the entry without changing it. But this requires to write a specialized function
 *  dumped by the precompiler, that will do a loop on the predicates. This was ruled out.
 *
 *  The element can still be inserted in the table, counted for some data (not all),
 *  used by some tasks (not all), removed from the table, then re-inserted when a
 *  new data arrives that the previous tasks did not depend upon.
 *
 *  Here is how it is used:
 *    the table is created with data_repo_create_nothreadsafe
 *    entries can be looked up with data_repo_lookup_entry (no side effect on the counters)
 *    entries are created using data_repo_lookup_entry_and_create. This turns the retained flag on.
 *    The same thread that called data_repo_lookup_entry_and_create must eventually call
 *    data_repo_entry_addto_usage_limit to set the usage limit and remove the retained flag.
 *    Between the two calls, any thread can call data_repo_lookup_entry and
 *    data_repo_entry_used_once if the entry has been "used". When data_repo_entry_addto_usage_limit
 *    has been called the same number of times as data_repo_lookup_entry_and_create and data_repo_entry_used_once
 *    has been called N times where N is the sum of the usagelmt parameters of data_repo_lookup_entry_and_create,
 *    the entry is garbage collected from the hash table. Notice that the values pointed by the entry
 *    are not collected.
 */

/**
 * data_repo_entries as mempool manageable elements:
 *  a mempool manageable element must be a parsec_list_item_t,
 *  and it must have a pointer to it's own mempool_thread_t.
 * Thus, we use the parsec_list_item_t to point to the next fields,
 * althgough this is not done atomically at the datarepo level (not
 * needed).
 */

#define PARSEC_DEFAULT_DATAREPO_HASH_LENGTH 4096

struct data_repo_entry_s {
    parsec_list_item_t         data_repo_next_item;
    parsec_thread_mempool_t   *data_repo_mempool_owner;
    void*                      generator;
    parsec_hash_table_item_t   ht_item;
    volatile int32_t           usagecnt;
    volatile int32_t           usagelmt;
    volatile int32_t           retained;
#if defined(PARSEC_SIM)
    int                        sim_exec_date;
#endif
    struct parsec_data_copy_s *data[1];
};

typedef struct data_repo_s {
    parsec_hash_table_t table;
    unsigned int       nbdata;
} data_repo_t;

BEGIN_C_DECLS

data_repo_t*
data_repo_create_nothreadsafe(unsigned int hashsize_hint, parsec_key_fn_t key_functions, void *key_hash_data, unsigned int nbdata);

data_repo_entry_t*
data_repo_lookup_entry(data_repo_t *repo, parsec_key_t key);

/* If using lookup_and_create, don't forget to call add_to_usage_limit on the same entry when
 * you're done counting the number of references, otherwise the entry is non erasable.
 * See comment near the structure definition.
 */
#if defined(PARSEC_DEBUG_NOISIER)

# define data_repo_lookup_entry_and_create(eu, repo, key)               \
    __data_repo_lookup_entry_and_create(eu, repo, key, #repo, __FILE__, __LINE__)
data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_stream_t *eu, data_repo_t *repo, parsec_key_t key,
                                    const char *tablename, const char *file, int line);

# define data_repo_entry_used_once(eu, repo, key)                       \
    __data_repo_entry_used_once(eu, repo, key, #repo, __FILE__, __LINE__)
void
__data_repo_entry_used_once(parsec_execution_stream_t *eu, data_repo_t *repo, parsec_key_t key,
                            const char *tablename, const char *file, int line);

# define data_repo_entry_addto_usage_limit(repo, key, usagelmt)         \
    __data_repo_entry_addto_usage_limit(repo, key, usagelmt, #repo, __FILE__, __LINE__)
void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, parsec_key_t key, uint32_t usagelmt,
                                    const char *tablename, const char *file, int line);

#else

# define data_repo_lookup_entry_and_create(eu, repo, key)   \
    __data_repo_lookup_entry_and_create(eu, repo, key)
data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_stream_t *es, data_repo_t *repo, parsec_key_t key);

# define data_repo_entry_used_once(eu, repo, key) __data_repo_entry_used_once(eu, repo, key)
void
__data_repo_entry_used_once(parsec_execution_stream_t *es, data_repo_t *repo, parsec_key_t key);

# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) \
    __data_repo_entry_addto_usage_limit(repo, key, usagelmt)
void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, parsec_key_t key, uint32_t usagelmt);

#endif

void data_repo_destroy_nothreadsafe(data_repo_t *repo);

/** @} */

END_C_DECLS

#endif  /* PARSEC_DATAREPO_H_HAS_BEEN_INCLUDED */
