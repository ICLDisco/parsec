/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/datarepo.h"
#include "parsec/utils/debug.h"

data_repo_t*
data_repo_create_nothreadsafe(unsigned int hashsize_hint, parsec_key_fn_t key_functions, void *key_hash_data, unsigned int nbdata)
{
    data_repo_t *res;
    unsigned int base = 1;
    /* We sanitize the hashsize_hint to get a power of 2, and we force the
     * first hash table to be at most 64k entries*/
    for(base = 1; base < 16 && (1U<<base) < hashsize_hint; base++) /*nothing*/;

    res = (data_repo_t*)calloc(1, sizeof(data_repo_t));
    parsec_hash_table_init(&res->table, offsetof(data_repo_entry_t, ht_item),
                           base,
                           key_functions, key_hash_data);
    
    res->nbdata = nbdata;
    return res;
}

data_repo_entry_t*
data_repo_lookup_entry(data_repo_t *repo, parsec_key_t key)
{
    return (data_repo_entry_t *) parsec_hash_table_find(&repo->table, key);
}

data_repo_entry_t*
__data_repo_lookup_entry_and_create(parsec_execution_stream_t *es, data_repo_t *repo, parsec_key_t key
#if defined(PARSEC_DEBUG_NOISIER)
                                    , const char *tablename, const char *file, int line
#endif
                                    )
{
    data_repo_entry_t *e;
    unsigned int i;
#if defined(PARSEC_DEBUG_NOISIER)
    char estr[64];
#endif
    parsec_hash_table_lock_bucket(&repo->table, key);
    e = (data_repo_entry_t*)parsec_hash_table_nolock_find(&repo->table, key);
    if( NULL != e ) {
        e->retained++; /* Until we update the usage limit */
        parsec_hash_table_unlock_bucket(&repo->table, key);
        return e;
    }
    parsec_hash_table_unlock_bucket(&repo->table, key);

    e = (data_repo_entry_t*)parsec_thread_mempool_allocate( es->datarepo_mempools[repo->nbdata] );
    for(i = 0; i < repo->nbdata; e->data[i] = NULL, i++);
    e->generator = NULL;
    e->data_repo_mempool_owner = es->datarepo_mempools[repo->nbdata];
    e->ht_item.key = key;
#if defined(PARSEC_SIM)
    e->sim_exec_date = 0;
#endif
    e->usagelmt = 0;
    e->usagecnt = 0;
    e->retained = 1; /* Until we update the usage limit */

    parsec_hash_table_lock_bucket(&repo->table, key);
    /* We are the only thread that can create this repo entry
     * because the key is unique in this table for the output task, and only
     * one thread executed the task 
     */
    assert(NULL == parsec_hash_table_nolock_find(&repo->table, key));
    parsec_hash_table_nolock_insert(&repo->table, &e->ht_item);
    parsec_hash_table_unlock_bucket(&repo->table, key);
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%s of hash table %s has been allocated with an usage count of %u/%u and is retained %d at %s:%d",
                         e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), tablename, e->usagecnt, e->usagelmt, e->retained, file, line);

    return e;
}

void
__data_repo_entry_used_once(parsec_execution_stream_t *es, data_repo_t *repo, parsec_key_t key
#if defined(PARSEC_DEBUG_NOISIER)
                            , const char *tablename, const char *file, int line
#endif
                            )
{
    data_repo_entry_t *e;
    int32_t r = -1;
#if defined(PARSEC_DEBUG_NOISIER)
    char estr[64];
#endif

    parsec_hash_table_lock_bucket(&repo->table, key);
    e = (data_repo_entry_t*)parsec_hash_table_nolock_find(&repo->table, key);
#if defined(PARSEC_DEBUG_NOISIER)
    if( NULL == e ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %s of hash table %s could not be found at %s:%d",
                             repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), tablename, file, line);
    }
#endif
    assert( NULL != e );
    r = parsec_atomic_fetch_inc_int32(&e->usagecnt) + 1;

    if( (e->usagelmt == r) && (0 == e->retained) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%s of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d",
                             e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), tablename, r, r, file, line);
        parsec_hash_table_nolock_remove(&repo->table, key);
        parsec_hash_table_unlock_bucket(&repo->table, key);
        
        parsec_thread_mempool_free(e->data_repo_mempool_owner, e );
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "entry %p/%s of HT %s has %u/%u usage count and %s retained: not freeing it at %s:%d",
                             e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), tablename, r, e->usagelmt, e->retained ? "is" : "is not", file, line);
        parsec_hash_table_unlock_bucket(&repo->table, key);
    }
    (void)es;
}

void
__data_repo_entry_addto_usage_limit(data_repo_t *repo, parsec_key_t key, uint32_t usagelmt
#if defined(PARSEC_DEBUG_NOISIER)
                                    , const char *tablename, const char *file, int line
#endif
                                    )
{
    data_repo_entry_t *e;
    uint32_t ov, nv;
#if defined(PARSEC_DEBUG_NOISIER)
    char estr[64];
#endif

    parsec_hash_table_lock_bucket(&repo->table, key);
    e = parsec_hash_table_nolock_find(&repo->table, key);
    assert( NULL != e );
    assert(e->retained > 0);
    do {
        ov = e->usagelmt;
        nv = ov + usagelmt;
    } while( !parsec_atomic_cas_int32( &e->usagelmt, ov, nv) );
    e->retained--;

    if( (e->usagelmt == e->usagecnt) && (0 == e->retained) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "entry %p/%s of hash table %s has a usage count of %u/%u and is"
                             " not retained: freeing it at %s:%d",
                             e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data),tablename, e->usagecnt, e->usagelmt, file, line);
        parsec_hash_table_nolock_remove(&repo->table, key);
        parsec_hash_table_unlock_bucket(&repo->table, key);
        parsec_thread_mempool_free(e->data_repo_mempool_owner, e );
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "entry %p/%s of hash table %s has a usage count of %u/%u and is %s retained at %s:%d",
                             e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), tablename, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more", file, line);
        parsec_hash_table_unlock_bucket(&repo->table, key);
    }
}

#if defined(PARSEC_DEBUG_NOISIER)
static void print_data_repo_entry(void *item, void *cb_data)
{
    char estr[64];
    data_repo_t *repo = (data_repo_t*)cb_data;
    data_repo_entry_t *e = (data_repo_entry_t*)item;
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                         "entry %p/%s of hash table %p has a usage count of %u/%u and is"
                         " %s retained while the repo is destroyed",
                         e, repo->table.key_functions.key_print(estr, 64, e->ht_item.key, repo->table.hash_data), repo, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more");
}
#endif

void data_repo_destroy_nothreadsafe(data_repo_t *repo)
{
#if defined(PARSEC_DEBUG_NOISIER)
    parsec_hash_table_for_all(&repo->table, print_data_repo_entry, repo);
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
    parsec_hash_table_fini(&repo->table);
    free(repo);
}
