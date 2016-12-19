/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _datarepo_h_
#define _datarepo_h_

#include "parsec_config.h"

typedef struct data_repo_entry_s data_repo_entry_t;
typedef struct data_repo_head_s  data_repo_head_t;

#include <stdlib.h>
#include "parsec/sys/atomic.h"
#include "parsec/debug.h"
#include "parsec/execution_unit.h"
#include "parsec/arena.h"

#define MAX_DATAREPO_HASH 4096

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
 * needed)
 *
 * The following #define are here to help port the code.
 */

#define data_repo_next_entry     data_repo_next_item.list_next

struct data_repo_entry_s {
    parsec_list_item_t         data_repo_next_item;
    parsec_thread_mempool_t   *data_repo_mempool_owner;
    void*                     generator;
    uint64_t                  key;
    volatile uint32_t         usagecnt;
    volatile uint32_t         usagelmt;
    volatile uint32_t         retained;
#if defined(PARSEC_SIM)
    int                       sim_exec_date;
#endif
    struct parsec_data_copy_s *data[1];
};

struct data_repo_head_s {
    volatile uint32_t  lock;
    uint32_t           size;
    data_repo_entry_t *first_entry;
};

struct data_repo_s {
    unsigned int      nbentries;
    unsigned int      nbdata;
    data_repo_head_t  heads[1];
};

static inline data_repo_t*
data_repo_create_nothreadsafe(unsigned int hashsize_hint, unsigned int nbdata)
{
    unsigned int hashsize = hashsize_hint * 1.5;
    data_repo_t *res;

    if( hashsize == 0 ) hashsize = 1;
    if( hashsize > MAX_DATAREPO_HASH ) hashsize = MAX_DATAREPO_HASH;

    res = (data_repo_t*)calloc(1, sizeof(data_repo_t) + sizeof(data_repo_head_t) * hashsize);
    res->nbentries = hashsize;
    res->nbdata = nbdata;
    return res;
}

static inline data_repo_entry_t*
data_repo_lookup_entry(data_repo_t *repo, uint64_t key)
{
    data_repo_entry_t *e;
    int h = key % repo->nbentries;

    parsec_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = (data_repo_entry_t *)e->data_repo_next_entry)
        if( e->key == key ) break;
    parsec_atomic_unlock(&repo->heads[h].lock);

    return e;
}

/* If using lookup_and_create, don't forget to call add_to_usage_limit on the same entry when
 * you're done counting the number of references, otherwise the entry is non erasable.
 * See comment near the structure definition.
 */
#if defined(PARSEC_DEBUG_NOISIER)
# define data_repo_lookup_entry_and_create(eu, repo, key) \
    __data_repo_lookup_entry_and_create(eu, repo, key, #repo, __FILE__, __LINE__)
static inline data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key,
                                    const char *tablename, const char *file, int line)
#else
# define data_repo_lookup_entry_and_create(eu, repo, key)       \
    __data_repo_lookup_entry_and_create(eu, repo, key)
static inline data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key)
#endif
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e;

    parsec_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = (data_repo_entry_t *)e->data_repo_next_entry)
        if( e->key == key ) {
            e->retained++; /* Until we update the usage limit */
            parsec_atomic_unlock(&repo->heads[h].lock);
            return e;
        }
    parsec_atomic_unlock(&repo->heads[h].lock);

    e = (data_repo_entry_t*)parsec_thread_mempool_allocate( eu->datarepo_mempools[repo->nbdata] );
#if defined(PARSEC_DEBUG_PARANOID)
    { uint32_t i; for(i = 0; i < repo->nbdata; e->data[i] = NULL, i++);}
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
    e->data_repo_mempool_owner = eu->datarepo_mempools[repo->nbdata];
    e->key = key;
#if defined(PARSEC_SIM)
    e->sim_exec_date = 0;
#endif
    e->usagelmt = 0;
    e->usagecnt = 0;
    e->retained = 1; /* Until we update the usage limit */

    parsec_atomic_lock(&repo->heads[h].lock);
    e->data_repo_next_entry = (volatile parsec_list_item_t *)repo->heads[h].first_entry;
    repo->heads[h].first_entry = e;
    repo->heads[h].size++;
    parsec_atomic_unlock(&repo->heads[h].lock);
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of hash table %s has been allocated with an usage count of %u/%u and is retained %d at %s:%d",
            e, e->key, tablename, e->usagecnt, e->usagelmt, e->retained, file, line);

    return e;
}

#if defined(PARSEC_DEBUG_NOISIER)
# define data_repo_entry_used_once(eu, repo, key) \
    __data_repo_entry_used_once(eu, repo, key, #repo, __FILE__, __LINE__)
static inline void
__data_repo_entry_used_once(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key,
                            const char *tablename, const char *file, int line)
#else
# define data_repo_entry_used_once(eu, repo, key) __data_repo_entry_used_once(eu, repo, key)
static inline void
__data_repo_entry_used_once(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key)
#endif
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e, *p;
    uint32_t r = 0xffffffff;

    parsec_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry)
        if( e->key == key ) {
            r = parsec_atomic_inc_32b(&e->usagecnt);
            break;
        }

#if defined(PARSEC_DEBUG_NOISIER)
    if( NULL == e ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %" PRIu64 " of hash table %s could not be found at %s:%d", key, tablename, file, line);
    }
#endif
    assert( NULL != e );

    if( (e->usagelmt == r) && (0 == e->retained) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d",
                e, e->key, tablename, r, r, file, line);
        if( NULL != p ) {
            p->data_repo_next_entry = e->data_repo_next_entry;
        } else {
            repo->heads[h].first_entry = (data_repo_entry_t*)e->data_repo_next_entry;
        }
        repo->heads[h].size--;
        parsec_atomic_unlock(&repo->heads[h].lock);

        parsec_thread_mempool_free(e->data_repo_mempool_owner, e );
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of HT %s has %u/%u usage count and %s retained: not freeing it at %s:%d",
                     e, e->key, tablename, r, e->usagelmt, e->retained ? "is" : "is not", file, line);
        parsec_atomic_unlock(&repo->heads[h].lock);
    }
    (void)eu;
}

#if defined(PARSEC_DEBUG_NOISIER)
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) \
    __data_repo_entry_addto_usage_limit(repo, key, usagelmt, #repo, __FILE__, __LINE__)
static inline void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, uint64_t key, uint32_t usagelmt,
                                    const char *tablename, const char *file, int line)
#else
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) \
    __data_repo_entry_addto_usage_limit(repo, key, usagelmt)
static inline void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, uint64_t key, uint32_t usagelmt)
#endif
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e, *p;
    uint32_t ov, nv;

    parsec_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry)
        if( e->key == key ) {
            assert(e->retained > 0);
            do {
                ov = e->usagelmt;
                nv = ov + usagelmt;
            } while( !parsec_atomic_cas_32b( &e->usagelmt, ov, nv) );
            e->retained--;
            break;
        }

    assert( NULL != e );

    if( (e->usagelmt == e->usagecnt) && (0 == e->retained) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, file, line);
        if( NULL != p ) {
            p->data_repo_next_entry = e->data_repo_next_entry;
        } else {
            repo->heads[h].first_entry = (data_repo_entry_t*)e->data_repo_next_entry;
        }
        repo->heads[h].size--;
        parsec_atomic_unlock(&repo->heads[h].lock);
        parsec_thread_mempool_free(e->data_repo_mempool_owner, e );
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of hash table %s has a usage count of %u/%u and is %s retained at %s:%d",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more", file, line);
        parsec_atomic_unlock(&repo->heads[h].lock);
    }
}

static inline void data_repo_destroy_nothreadsafe(data_repo_t *repo)
{
#if defined(PARSEC_DEBUG_NOISIER)
    data_repo_entry_t *e;
    int h;

    for( h = 0; h < (int)repo->nbentries; h++ ) {
        for(e = repo->heads[h].first_entry;
            e != NULL;
            e = (data_repo_entry_t*)e->data_repo_next_entry) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%" PRIu64 " of hash table %p has a usage count of %u/%u and is %s retained while the repo is destroyed",
                   e, e->key, repo, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more");
        }
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
    free(repo);
}

#endif /* _datarepo_h_ */
