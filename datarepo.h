#ifndef _datarepo_h_
#define _datarepo_h_

#include <stdlib.h>

#include "atomic.h"
#include "stats.h"
#include "debug.h"

#define DEBUG_HEAVY(p)

static inline void data_repo_atomic_lock( volatile uint32_t* atomic_lock )
{
    while( !dplasma_atomic_cas( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void data_repo_atomic_unlock( volatile uint32_t* atomic_lock )
{
    *atomic_lock = 0;
}

typedef struct gc_data {
    volatile uint32_t refcount;
    uint32_t cache_friendliness;
    void    *data;
} gc_data_t;

#define GC_POINTER(d) ((gc_data_t*)( (uintptr_t)(d) & ~( (uintptr_t)1) ))
#define GC_ENABLED(d) ( ((uintptr_t)(d) & 1) == 1 )
#define GC_DATA(d) (void*)( GC_ENABLED(d)?(GC_POINTER(d)->data):(d) )


#ifdef DPLASMA_DEBUG_HEAVY
#define gc_data_new(d, e) __gc_data_new(d, e, __FILE__, __LINE__)
static inline gc_data_t *__gc_data_new(void *data, uint32_t gc_enabled, const char *file, int line)
#else
#define gc_data_new(d, e) __gc_data_new(d, e)
static inline gc_data_t *__gc_data_new(void *data, uint32_t gc_enabled)
#endif
{
    gc_data_t *d;

    if( gc_enabled != 0 ) {
        d = (gc_data_t*)malloc(sizeof(gc_data_t));
        d->refcount = 0;
        d->data = data;
        assert( ((uintptr_t)d & (uintptr_t)1) == 0 /* Pointers cannot be odd */ );
#if defined(DPLASMA_STATS)
        d->cache_friendliness = gc_enabled;
        DPLASMA_STAT_INCREASE(mem_communications, gc_enabled + sizeof(gc_data_t) + 2 * STAT_MALLOC_OVERHEAD);
#endif
        d = (gc_data_t*)( (uintptr_t)d | (uintptr_t)1 );

        DEBUG_HEAVY(("Allocating the garbage collectable data %p pointing on data %p, at %s:%d\n",
                     d, GC_DATA(d), file, line));
        return d;
    } else {
        return (gc_data_t*)data;
    }
}

#ifdef DPLASMA_DEBUG_HEAVY
#define gc_data_ref(d) __gc_data_ref(d, __FILE__, __LINE__)
#else
#define gc_data_ref(d) __gc_data_ref(d)
#endif

#ifdef DPLASMA_DEBUG_HEAVY
static inline void __gc_data_ref(gc_data_t *d, const char *file, int line)
#else
static inline void __gc_data_ref(gc_data_t *d)
#endif
{
    if( GC_ENABLED(d) ) {
        DEBUG_HEAVY(("%p is referenced by %s:%d\n", d, file, line));
        dplasma_atomic_inc_32b( &GC_POINTER(d)->refcount);
    }
}

#ifdef DPLASMA_DEBUG_HEAVY
#define gc_data_unref(d) __gc_data_unref(d, __FILE__, __LINE__)
#else
#define gc_data_unref(d) __gc_data_unref(d)
#endif

#if defined(USE_MPI)
extern dplasma_atomic_lifo_t* internal_alloc_lifo;
#endif  /* defined(USE_MPI) */

#ifdef DPLASMA_DEBUG_HEAVY
static inline gc_data_t* __gc_data_unref(gc_data_t *d, const char *file, int line)
#else
static inline gc_data_t* __gc_data_unref(gc_data_t *d)
#endif
{
    int nref;
    if( GC_ENABLED(d) ) {
        nref = dplasma_atomic_dec_32b( &GC_POINTER(d)->refcount );
        DEBUG_HEAVY(("%p is unreferenced by %s:%d\n", d, file, line));
        if( 0 == nref ) {
            DEBUG_HEAVY(("Liberating the garbage collectable datar %p pointing on data %p,\n",
                         d, GC_DATA(d)));
            /*printf( "%s:%d Releasing TILE at %p\n", __FILE__, __LINE__, GC_DATA(d));*/
#if defined(USE_MPI)
            {
                dplasma_list_item_t* item = GC_DATA(d);
                DPLASMA_LIST_ITEM_SINGLETON(item);
                dplasma_atomic_lifo_push(internal_alloc_lifo, item);
            }
#else
            free(GC_DATA(d));
#endif  /* defined(USE_MPI) */
#if defined(DPLASMA_DEBUG_HEAVY)
            GC_POINTER(d)->data = NULL;
            GC_POINTER(d)->refcount = 0;
#endif
            DPLASMA_STAT_DECREASE(mem_communications, sizeof(gc_data_t) + 2*STAT_MALLOC_OVERHEAD + d->cache_friendliness);
            free(GC_POINTER(d));
            return NULL;
        }
    }
    return d;
}

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

typedef struct data_repo_entry {
    volatile uint32_t usagecnt;
    volatile uint32_t usagelmt;
    volatile uint32_t retained;
    long int key;
    struct data_repo_entry *next_entry;
    gc_data_t *data[1];
} data_repo_entry_t;

typedef struct data_repo_head {
    volatile uint32_t  lock;
    uint32_t           size;
    data_repo_entry_t *first_entry;
} data_repo_head_t;

typedef struct data_repo {
    unsigned int      nbentries;
    unsigned int      nbdata;
    data_repo_head_t  heads[1];
} data_repo_t;

static inline data_repo_t *data_repo_create_nothreadsafe(unsigned int hashsize, unsigned int nbdata)
{
    data_repo_t *res = (data_repo_t*)calloc(1, sizeof(data_repo_t) + sizeof(data_repo_head_t) * (hashsize-1));
    res->nbentries = hashsize;
    res->nbdata = nbdata;
    DPLASMA_STAT_INCREASE(mem_hashtable, sizeof(data_repo_t) + sizeof(data_repo_head_t) * (hashsize-1) + STAT_MALLOC_OVERHEAD);
    return res;
}

static inline data_repo_entry_t *data_repo_lookup_entry(data_repo_t *repo, long int key)
{
    data_repo_entry_t *e;
    int h = key % repo->nbentries;
    
    data_repo_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = e->next_entry)
        if( e->key == key ) {
            data_repo_atomic_unlock(&repo->heads[h].lock);
            return e;
        }
    data_repo_atomic_unlock(&repo->heads[h].lock);

    return NULL;
}

/* If using lookup_and_create, don't forget to call add_to_usage_limit on the same entry when
 * you're done counting the number of references, otherwise the entry is non erasable.
 * See comment near the structure definition.
 */
static inline data_repo_entry_t *data_repo_lookup_entry_and_create(data_repo_t *repo, long int key)
{
    data_repo_entry_t *e;
    int h = key % repo->nbentries;
    
    data_repo_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = e->next_entry)
        if( e->key == key ) {
            e->retained++; /* Until we update the usage limit */
            data_repo_atomic_unlock(&repo->heads[h].lock);
            return e;
        }

    e = (data_repo_entry_t*)calloc(1, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*));
    e->next_entry = repo->heads[h].first_entry;
    repo->heads[h].first_entry = e;
    e->key = key;
    e->usagelmt = 0;
    e->usagecnt = 0;
    e->retained = 1; /* Until we update the usage limit */
    repo->heads[h].size++;
    DPLASMA_STAT_INCREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
    DPLASMA_STATMAX_UPDATE(counter_hashtable_collisions_size, repo->heads[h].size);
    data_repo_atomic_unlock(&repo->heads[h].lock);
    return e;
}

#if defined(DPLASMA_DEBUG_HEAVY)
# define data_repo_entry_used_once(repo, key) __data_repo_entry_used_once(repo, key, #repo, __FILE__, __LINE__)
static inline void __data_repo_entry_used_once(data_repo_t *repo, long int key, const char *tablename, const char *file, int line)
#else
# define data_repo_entry_used_once(repo, key) __data_repo_entry_used_once(repo, key)
static inline void __data_repo_entry_used_once(data_repo_t *repo, long int key)
#endif
{
    data_repo_entry_t *e, *p;
    int h = key % repo->nbentries;
    uint32_t r = 0xffffffff;

    data_repo_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = e->next_entry)
        if( e->key == key ) {
            r = dplasma_atomic_inc_32b(&e->usagecnt);
            break;
        }

#ifdef DPLASMA_DEBUG_HEAVY
    if( NULL == e ) {
        DEBUG_HEAVY(("entry %ld of hash table %s could not be found at %s:%d\n", key, tablename, file, line));
    }
#endif
    assert( NULL != e );

    if( (e->usagelmt == r) && (0 == e->retained) ) {
        DEBUG_HEAVY(("entry %p/%ld of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d\n",
                     e, e->key, tablename, r, r, file, line));
        if( NULL != p ) {
            p->next_entry = e->next_entry;
        } else {
            repo->heads[h].first_entry = e->next_entry;
        }
        repo->heads[h].size--;
        data_repo_atomic_unlock(&repo->heads[h].lock);
        free(e);
        DPLASMA_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
    } else {
        DEBUG_HEAVY(("entry %p/%ld of hash table %s has %u/%u usage count and %s retained: not freeing it, even if it's used at %s:%d\n",
                     e, e->key, tablename, r, e->usagelmt, e->retained ? "is" : "is not", file, line));
        data_repo_atomic_unlock(&repo->heads[h].lock);
    }
}

#if defined(DPLASMA_DEBUG_HEAVY)
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) __data_repo_entry_addto_usage_limit(repo, key, usagelmt, #repo, __FILE__, __LINE__)
static inline void __data_repo_entry_addto_usage_limit(data_repo_t *repo, long int key, uint32_t usagelmt, const char *tablename, const char *file, int line)
#else
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) __data_repo_entry_addto_usage_limit(repo, key, usagelmt)
static inline void __data_repo_entry_addto_usage_limit(data_repo_t *repo, long int key, uint32_t usagelmt)
#endif
{
    data_repo_entry_t *e, *p;
    uint32_t ov, nv;
    int h = key % repo->nbentries;

    data_repo_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = e->next_entry)
        if( e->key == key ) {
            assert(e->retained > 0);
            do {
                ov = e->usagelmt;
                nv = ov + usagelmt;
            } while( !dplasma_atomic_cas_32b( &e->usagelmt, ov, nv) );
            e->retained--;
            break;
        }

    assert( NULL != e );

    if( (e->usagelmt == e->usagecnt) && (0 == e->retained) ) {
        DEBUG_HEAVY(("entry %p/%ld of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d\n",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, file, line));
        if( NULL != p ) {
            p->next_entry = e->next_entry;
        } else {
            repo->heads[h].first_entry = e->next_entry;
        }
        repo->heads[h].size--;
        data_repo_atomic_unlock(&repo->heads[h].lock);
        free(e);
        DPLASMA_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
    } else {
        DEBUG_HEAVY(("entry %p/%ld of hash table %s has a usage count of %u/%u and is %s retained at %s:%d\n",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more", file, line));
        data_repo_atomic_unlock(&repo->heads[h].lock);
    }
}

static inline void data_repo_destroy_nothreadsafe(data_repo_t *repo)
{
    data_repo_entry_t *e, *n;
    int i;
    for(i = 0; i < repo->nbentries; i++) {
        for(e = repo->heads[i].first_entry;
            e != NULL;
            e = n) {
            n = e->next_entry;
            free(e);
            DPLASMA_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
        }
    }
    DPLASMA_STAT_DECREASE(mem_hashtable,  sizeof(data_repo_t) + sizeof(data_repo_head_t) * (repo->nbentries-1) + STAT_MALLOC_OVERHEAD);
    free(repo);
}

#endif /* _datarepo_h_ */
