#include <assert.h>
#include "parsec/class/hash_table.h"
#include "parsec/class/list.h"
#include <stdio.h>

#define BASEADDROF(item, ht)  (void*)(  ( (char*)(item) ) - ( (ht)->elt_hashitem_offset ) )

void *hash_table_item_lookup(hash_table_t *ht, hash_table_item_t *item)
{
    return BASEADDROF(item, ht);
}

/* To create object of class hash_table that inherits parsec_object_t class */
OBJ_CLASS_INSTANCE(hash_table_t, parsec_object_t, NULL, NULL);

void hash_table_init(hash_table_t *ht, int64_t offset, size_t size_of_table, hash_table_fn_t *hash, void *data)
{
    parsec_atomic_lock_t unlocked = { PARSEC_ATOMIC_UNLOCKED };
    
    ht->size      = size_of_table;
    ht->hash      = hash;
    ht->hash_data = data;
    ht->elt_hashitem_offset = offset;
    
    ht->buckets = malloc(size_of_table * sizeof(hash_table_bucket_t));

    int i;
    for( i=0; i<size_of_table; i++) {
        ht->buckets[i].lock = unlocked;
        ht->buckets[i].first_item = NULL;
    }
}

void hash_table_lock_bucket(hash_table_t *ht, uint64_t key )
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    assert( hash < ht->size );
    parsec_atomic_lock(&ht->buckets[hash].lock);
}

void hash_table_unlock_bucket(hash_table_t *ht, uint64_t key )
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    assert( hash < ht->size );
    parsec_atomic_unlock(&ht->buckets[hash].lock);
}

void hash_table_fini(hash_table_t *ht)
{
    int i;
    for(i=0; i < ht->size; i++) {
        assert(NULL == ht->buckets[i].first_item);
    }

    free(ht->buckets);
    OBJ_RELEASE(ht);
}

static void hash_table_nolock_insert_with_hash(hash_table_t *ht, hash_table_item_t *item, uint32_t hash)
{
    item->next_item = ht->buckets[hash].first_item;
    ht->buckets[hash].first_item = item;
}

static void *hash_table_nolock_find_with_hash(hash_table_t *ht, uint64_t key, uint32_t hash)
{
    hash_table_item_t *current_item;
    for(current_item = ht->buckets[hash].first_item;
        NULL != current_item;
        current_item = current_item->next_item) {
        if( current_item->key == key ) {
            return BASEADDROF(current_item, ht);
        }
    }
    return NULL;
}

static void *hash_table_nolock_remove_with_hash(hash_table_t *ht, uint64_t key, uint32_t hash)
{
    hash_table_item_t *current_item, *prev_item;
    prev_item = NULL;
    for(current_item = ht->buckets[hash].first_item;
        NULL != current_item;
        current_item = prev_item->next_item) {
        if( current_item->key == key ) {
            if( NULL == prev_item ) {
                ht->buckets[hash].first_item = current_item->next_item;
            } else {
                prev_item->next_item = current_item->next_item;
            }
            return BASEADDROF(current_item, ht);
        }
        prev_item = current_item;
    }
    return NULL;
}

void hash_table_nolock_insert(hash_table_t *ht, hash_table_item_t *item)
{
    uint32_t hash = ht->hash(item->key, ht->hash_data);
    assert(hash < ht->size);
    hash_table_nolock_insert_with_hash(ht, item, hash);
}

void *hash_table_nolock_find(hash_table_t *ht, uint64_t key)
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    assert(hash < ht->size);
    return hash_table_nolock_find_with_hash(ht, key, hash);
}

void *hash_table_nolock_remove(hash_table_t *ht, uint64_t key)
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    assert(hash < ht->size);
    return hash_table_nolock_remove_with_hash(ht, key, hash);
}

void hash_table_insert(hash_table_t *ht, hash_table_item_t *item)
{
    uint32_t hash = ht->hash(item->key, ht->hash_data);
    assert( hash < ht->size );
    parsec_atomic_lock(&ht->buckets[hash].lock);
    hash_table_nolock_insert_with_hash(ht, item, hash);
    parsec_atomic_unlock(&ht->buckets[hash].lock);
}

void *hash_table_find(hash_table_t *ht, uint64_t key)
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    void *ret;
    assert( hash < ht->size );
    parsec_atomic_lock(&ht->buckets[hash].lock);
    ret = hash_table_nolock_find_with_hash(ht, key, hash);
    parsec_atomic_unlock(&ht->buckets[hash].lock);
    return ret;
}

void *hash_table_remove(hash_table_t *ht, uint64_t key)
{
    uint32_t hash = ht->hash(key, ht->hash_data);
    void *ret;
    assert( hash < ht->size );
    parsec_atomic_lock(&ht->buckets[hash].lock);
    ret = hash_table_nolock_remove_with_hash(ht, key, hash);
    parsec_atomic_unlock(&ht->buckets[hash].lock);
    return ret;
}

void hash_table_stat(hash_table_t *ht)
{
    double mean = 0.0, M2=0.0, delta, delta2;
    int n = 0, min = -1, max = -1;
    int nb;
    uint32_t i;
    hash_table_item_t *current_item;

    for(i = 0; i < ht->size; i++) {
        nb = 0;
        for(current_item = ht->buckets[i].first_item;
            current_item != NULL;
            current_item = current_item->next_item) {
            nb++;
        }

        n++;
        delta = (double)nb - mean;
        mean += delta/n;
        delta2 = (double)nb - mean;
        M2 += delta*delta2;

        if( min == -1 || nb < min )
            min = nb;
        if( max == -1 || nb > max )
            max = nb;
    }
    printf("table %p: %d lists, of length %d to %d average length: %g and variance %g\n",
           ht, n, min, max, mean, M2/(n-1));
}
