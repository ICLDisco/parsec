#ifndef _datarepo_h_
#define _datarepo_h_

#include <stdlib.h>

#include "atomic.h"
#include "debug.h"
#include "stats.h"

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
#define GC_ENABLED(d) ( (uintptr_t)(d) & 1 == 1 )
#define GC_DATA(d) (void*)( GC_ENABLED(d)?(GC_POINTER(d)->data):(d) )

#ifdef DPLASMA_DEBUG
#define gc_data_new(d, e) __gc_data_new(d, e, __FILE__, __LINE__)
#else
#define gc_data_new(d, e) __gc_data_new(d, e)
#endif

#ifdef DPLASMA_DEBUG
static inline gc_data_t *__gc_data_new(void *data, uint32_t gc_enabled, const char *file, int line)
#else
static inline gc_data_t *__gc_data_new(void *data, uint32_t gc_enabled)
#endif
{
    gc_data_t *d;

    if( gc_enabled ) {
        d = (gc_data_t*)malloc(sizeof(gc_data_t));
        d->refcount = 0;
        d->data = data;
        assert( ((uintptr_t)d & (uintptr_t)1) == 0 /* Pointers cannot be odd */ );
        d = (gc_data_t*)( (uintptr_t)d | (uintptr_t)1 );

        DEBUG(("Allocating the garbage collectable data %p pointing on data %p, at %s:%d\n",
               d, d->data, file, line));
        return d;
    } else {
        return (gc_data_t*)data;
    }
}

#ifdef DPLASMA_DEBUG
#define gc_data_ref(d) __gc_data_ref(d, __FILE__, __LINE__)
#else
#define gc_data_ref(d) __gc_data_ref(d)
#endif

#ifdef DPLASMA_DEBUG
static inline void __gc_data_ref(gc_data_t *d, const char *file, int line)
#else
static inline void __gc_data_ref(gc_data_t *d)
#endif
{
    if( GC_ENABLED(d) ) {
        DEBUG(("%p is referenced by %s:%d\n", d, file, line));
        dplasma_atomic_inc_32b( &GC_POINTER(d)->refcount);
    }
}

#ifdef DPLASMA_DEBUG
#define gc_data_unref(d) __gc_data_unref(d, __FILE__, __LINE__)
#else
#define gc_data_unref(d) __gc_data_unref(d)
#endif

#ifdef DPLASMA_DEBUG
static inline gc_data_t* __gc_data_unref(gc_data_t *d, const char *file, int line)
#else
static inline gc_data_t* __gc_data_unref(gc_data_t *d)
#endif
{
    int nref;
    if( GC_ENABLED(d) ) {
        nref = dplasma_atomic_dec_32b( &GC_POINTER(d)->refcount );
        DEBUG(("%p is unreferenced by %s:%d\n", d, file, line));
        if( 0 == nref ) {
            DEBUG(("Liberating the garbage collectable datar %p pointing on data %p,\n",
                   d, GC_DATA(d)));
            free(GC_DATA(d));
#if defined(DPLASMA_DEBUG)
            GC_POINTER(d)->data = NULL;
            GC_POINTER(d)->refcount = 0;
#endif
            free(GC_POINTER(d));
            return NULL;
        }
    }
    return d;
}

typedef struct data_repo_entry {
    volatile uint32_t usagecnt;
    volatile uint32_t usagelmt;
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

static inline data_repo_entry_t *data_repo_lookup_entry(data_repo_t *repo, long int key, int create)
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

    if( create != 0 ) {
        e = (data_repo_entry_t*)calloc(1, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*));
        e->next_entry = repo->heads[h].first_entry;
        repo->heads[h].first_entry = e;
        e->key = key;
        repo->heads[h].size++;
        DPLASMA_STAT_INCREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
        DPLASMA_STATMAX_UPDATE(counter_hashtable_collisions_size, repo->heads[h].size);
    }
    data_repo_atomic_unlock(&repo->heads[h].lock);
    return e;
}

static inline void data_repo_entry_used_once(data_repo_t *repo, long int key)
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

    if( /*(NULL != e) &&*/ (e->usagelmt == r) ) {
        if( NULL != p ) {
            p->next_entry = e->next_entry;
        } else {
            repo->heads[h].first_entry = e->next_entry;
        }
        data_repo_atomic_unlock(&repo->heads[h].lock);
        repo->heads[h].size--;
        free(e);
        DPLASMA_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
    } else {
        data_repo_atomic_unlock(&repo->heads[h].lock);
    }
}

static inline void data_repo_entry_set_usage_limit(data_repo_t *repo, long int key, uint32_t usagelmt)
{
    data_repo_entry_t *e, *p;
    int h = key % repo->nbentries;

    data_repo_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = e->next_entry)
        if( e->key == key ) {
            e->usagelmt = usagelmt;
            break;
        }

    if( /*(NULL != e) &&*/ (usagelmt == e->usagecnt) ) {
        if( NULL != p ) {
            p->next_entry = e->next_entry;
        } else {
            repo->heads[h].first_entry = e->next_entry;
        }
        data_repo_atomic_unlock(&repo->heads[h].lock);
        free(e);
        DPLASMA_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(gc_data_t*) + STAT_MALLOC_OVERHEAD);
    } else {
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
