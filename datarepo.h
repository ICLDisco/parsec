#ifndef _datarepo_h_
#define _datarepo_h_

#include "atomic.h"

static inline void data_repo_atomic_lock( volatile uint32_t* atomic_lock )
{
    while( !dplasma_atomic_cas( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void data_repo_atomic_unlock( volatile uint32_t* atomic_lock )
{
    *atomic_lock = 0;
}

typedef struct data_repo_entry {
    long int key;
    struct data_repo_entry *next_entry;
    unsigned int refcount;
    void *data[1];
} data_repo_entry_t;

typedef struct data_repo_head {
    volatile uint32_t lock;
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
            e->refcount++;
            break;
        }
    if( (NULL == e) && (create != 0) ) {
        e = (data_repo_entry_t*)calloc(1, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(void*));
        e->next_entry = repo->heads[h].first_entry;
        repo->heads[h].first_entry = e;
        e->refcount++;
        e->key = key;
    }
    data_repo_atomic_unlock(&repo->heads[h].lock);
    return e;
}

static inline void data_repo_unref_entry(data_repo_t *repo, long int key)
{
    data_repo_entry_t *e, *p;
    int h = key % repo->nbentries;

    data_repo_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = e->next_entry)
        if( e->key == key ) {
            e->refcount--;
            break;
        }
    if( (NULL != e) && (0 == e->refcount) ) {
        if( NULL != p ) {
            p->next_entry = e->next_entry;
        } else {
            repo->heads[h].first_entry = e->next_entry;
        }
        free(e);
    } 
    data_repo_atomic_unlock(&repo->heads[h].lock);
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
        }
    }
    free(repo);
}

#endif /* _datarepo_h_ */
