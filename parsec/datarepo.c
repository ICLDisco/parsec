/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/datarepo.h"
#include "parsec/debug.h"

#define data_repo_next_entry     data_repo_next_item.list_next

data_repo_t*
data_repo_create_nothreadsafe(unsigned int hashsize_hint, unsigned int nbdata)
{
    unsigned int hashsize = hashsize_hint * 1.5;
    data_repo_t *res;

    if( hashsize == 0 ) hashsize = 1;

    res = (data_repo_t*)calloc(1, sizeof(data_repo_t) + sizeof(data_repo_head_t) * hashsize);
    res->nbentries = hashsize;
    res->nbdata = nbdata;
    return res;
}

data_repo_entry_t*
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

data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key
#if defined(PARSEC_DEBUG_NOISIER)
                                    , const char *tablename, const char *file, int line
#endif
                                    )
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e;

    parsec_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = (data_repo_entry_t *)e->data_repo_next_entry) {
        if( e->key == key ) {
            e->retained++; /* Until we update the usage limit */
            parsec_atomic_unlock(&repo->heads[h].lock);
            return e;
        }
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

void
__data_repo_entry_used_once(parsec_execution_unit_t *eu, data_repo_t *repo, uint64_t key
#if defined(PARSEC_DEBUG_NOISIER)
                            , const char *tablename, const char *file, int line
#endif
                            )
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e, *p;
    uint32_t r = 0xffffffff;

    parsec_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry) {
        if( e->key == key ) {
            r = parsec_atomic_inc_32b(&e->usagecnt);
            break;
        }
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

void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, uint64_t key, uint32_t usagelmt
#if defined(PARSEC_DEBUG_NOISIER)
                                    , const char *tablename, const char *file, int line
#endif
                                    )
{
    const int h = key % repo->nbentries;
    data_repo_entry_t *e, *p;
    uint32_t ov, nv;

    parsec_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry) {
        if( e->key == key ) {
            assert(e->retained > 0);
            do {
                ov = e->usagelmt;
                nv = ov + usagelmt;
            } while( !parsec_atomic_cas_32b( &e->usagelmt, ov, nv) );
            e->retained--;
            break;
        }
    }
    assert( NULL != e );

    if( (e->usagelmt == e->usagecnt) && (0 == e->retained) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "entry %p/%" PRIu64 " of hash table %s has a usage count of %u/%u and is"
                             " not retained: freeing it at %s:%d",
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
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "entry %p/%" PRIu64 " of hash table %s has a usage count of %u/%u and is %s retained at %s:%d",
                             e, e->key, tablename, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more", file, line);
        parsec_atomic_unlock(&repo->heads[h].lock);
    }
}

void data_repo_destroy_nothreadsafe(data_repo_t *repo)
{
#if defined(PARSEC_DEBUG_NOISIER)
    data_repo_entry_t *e;
    int h;

    for( h = 0; h < (int)repo->nbentries; h++ ) {
        for(e = repo->heads[h].first_entry;
            e != NULL;
            e = (data_repo_entry_t*)e->data_repo_next_entry) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                                 "entry %p/%" PRIu64 " of hash table %p has a usage count of %u/%u and is"
                                 " %s retained while the repo is destroyed",
                                 e, e->key, repo, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more");
        }
    }
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
    free(repo);
}
