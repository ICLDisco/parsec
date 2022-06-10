/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <assert.h>
#include "parsec/parsec_config.h"
#include "parsec/class/parsec_hash_table.h"
#include "parsec/class/list.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"
#include <stdio.h>

#undef HELPFIRST

/**
 * @brief Bucket for hash tables. There is no need to have this structure public, it
 *        should only be used in this file.
 */
struct parsec_hash_table_bucket_s {
    parsec_atomic_lock_t      lock;             /**< Buckets are lockable for multithread access
                                                 *   We also use this lock to atomically update the
                                                 *   list of elements when needed. */
    int32_t                   cur_len;          /**< Number of elements currently in this bucket */
    parsec_hash_table_item_t *first_item;       /**< Otherwise they are simply chained lists */
};

#define BASEADDROF(item, ht)  (void*)(  ( (char*)(item) ) - ( (ht)->elt_hashitem_offset ) )
#define ITEMADDROF(ptr, ht)   (parsec_hash_table_item_t*)( ((char*)(ptr)) + ( (ht)->elt_hashitem_offset ) )

static int      parsec_hash_table_mca_param_mch_index = -1;
static int32_t  parsec_hash_table_max_collisions_hint = 16; /* We resize if there are that many collisions */
static int      parsec_hash_table_mca_param_mnb_index = -1;
static int32_t  parsec_hash_table_max_table_nb_bits   = 24; /* We will never create a sub-table with more than 1<<parsec_hash_table_max_table_nb_bits buckets
                                                             * NB: if the user calls parsec_hash_table_init with nb_bits > parsec_hash_table_max_table_nb_bits,
                                                             *     we *will* create the first-level table with 1<<nb_bits buckets, despite this value. */

void *parsec_hash_table_item_lookup(parsec_hash_table_t *ht, parsec_hash_table_item_t *item)
{
    return BASEADDROF(item, ht);
}

/* To create object of class parsec_hash_table that inherits parsec_object_t class */
PARSEC_OBJ_CLASS_INSTANCE(parsec_hash_table_t, parsec_object_t, NULL, NULL);

/* If the keys are equal in value, then the item is the right one.
 * This will work for all keys that fit directly in the 64 bits of the
 * parsec_key_t.
 * For keys that are more complex than 64 bits, we first test for
 * the 64 bits hash of the keys, and if these are equal, we take the
 * time to call the actual key_equal function */
#define OPTIMIZED_EQUAL_TEST(_ITEM, _KEY, _HASH64, _HT)                 \
    ( (_ITEM)->key == (_KEY) ||                                         \
      ((_ITEM)->hash64 == (_HASH64) &&                                  \
       (_HT)->key_functions.key_equal((_ITEM)->key,                     \
                                      (_KEY), (_HT)->hash_data)) )


int parsec_hash_tables_init(void)
{
    int v = parsec_hash_table_max_collisions_hint;

    parsec_hash_table_mca_param_mch_index =
        parsec_mca_param_reg_int_name("parsec", "hash_table_max_collisions_hint",
                                      "Sets a hint for the dynamic hash tables implementation: "
                                      "Hash-tables will be resized if a bucket reaches this number of collisions.\n",
                                      false, false, v, &v);
    parsec_hash_table_max_collisions_hint = v;
    if( PARSEC_ERROR == parsec_hash_table_mca_param_mch_index ) {
        return PARSEC_ERROR;
    }

    v = parsec_hash_table_max_table_nb_bits;
    parsec_hash_table_mca_param_mnb_index =
        parsec_mca_param_reg_int_name("parsec", "hash_table_max_table_nb_bits",
                                      "Sets a limit on the maximum number of buckets in a resized table: "
                                      "if a hash table needs to be resized, it will never grow bigger than "
                                      "1<<parsec_hash_table_max_table_nb_bits buckets, even if this creates "
                                      "more than parsec_hash_table_max_collisions_hint collisions.\n",
                                      false, false, v, &v);
    parsec_hash_table_max_table_nb_bits = v;
    if( PARSEC_ERROR == parsec_hash_table_mca_param_mnb_index ) {
        return PARSEC_ERROR;
    }

    return PARSEC_SUCCESS;
}

void parsec_hash_table_init(parsec_hash_table_t *ht, int64_t offset, int nb_bits, parsec_key_fn_t key_functions, void *data)
{
    parsec_atomic_rwlock_t unlock = { PARSEC_RWLOCK_UNLOCKED };
    parsec_hash_table_head_t *head;
    size_t i;
    int v;

    if( parsec_hash_table_mca_param_mch_index != PARSEC_ERROR ) {
        if( parsec_mca_param_lookup_int(parsec_hash_table_mca_param_mch_index, &v) != PARSEC_ERROR ) {
            ht->max_collisions_hint = v;
        }
    }

    if( parsec_hash_table_mca_param_mnb_index != PARSEC_ERROR ) {
        if( parsec_mca_param_lookup_int(parsec_hash_table_mca_param_mnb_index, &v) != PARSEC_ERROR ) {
            ht->max_table_nb_bits = v;
        }
    }

    assert( nb_bits >= 1 && nb_bits <= 16);

    ht->key_functions = key_functions;
    ht->hash_data = data;
    ht->elt_hashitem_offset = offset;
    ht->warning_issued = 0;

    head = malloc(sizeof(parsec_hash_table_head_t));
    head->buckets      = malloc( (1ULL<<nb_bits) * sizeof(parsec_hash_table_bucket_t));
    head->nb_bits      = nb_bits;
    head->used_buckets = 0;
    head->next         = NULL;
    head->next_to_free = NULL;
    ht->rw_hash        = head;
    ht->rw_lock        = unlock;

    for( i = 0; i < (1ULL<<nb_bits); i++) {
        parsec_atomic_lock_init(&head->buckets[i].lock);
        head->buckets[i].cur_len = 0;
        head->buckets[i].first_item = NULL;
    }
}

static uint64_t parsec_hash_table_universal_rehash(parsec_key_t key, int nb_bits) {
    uint64_t k = (uint64_t)(uintptr_t)key;

    /* The goal is to use all bits to create the hash value.
     * Ideally, if keys a and b have the same hash on s bits,
     * they should have different hashes on s+1 bits, so simple
     * modulo is avoided to take into account the case of keys
     * being different on the high bits as well as keys being
     * different on the low bits.
     */

    switch( nb_bits ) {
    /* We unrolled all cases fully to have minimal number of instructions.
     * For very small values (1 or 2 bits), just take the low bits of k
     * For small values (3 - 12 bits), fold first on 32 bits or 16 bits, then fold and shift the words of nb_bits
     * For big values (13-32 bits), fold and shift the words of nb_bits
     * For values that are too large, fold on 32 bits and take all the bits we can */
    case 0:
        assert(nb_bits > 0);
        return ~0ULL;
    case 1:
        return k & 0x1; /* It does not make sense for small values of nb_bits to mix the bits of k */
    case 2:
        return k & 0x3; /* It does not make sense for small values of nb_bits to mix the bits of k */
    case 3:
        k ^= (k >> 32);
        k ^= (k >> 16);
        return ((k >> 13) ^ (k >> 10) ^ (k >> 7) ^ (k >> 4) ^ (k >>1)  ) & 0x7;
    case 4:
        k ^= (k >> 32);
        k ^= (k >> 16);
        return ((k >> 12) ^ (k >> 8) ^ (k >> 4) ^ (k)) & 0xF;
    case 5:
        k ^= (k >> 32);
        k ^= (k >> 16);
        return ((k >> 11) ^ (k >> 6) ^ (k >> 1)) & 0x1F;
    case 6:
        k ^= (k >> 32);
        return ((k >> 26) ^ (k >> 20) ^ (k >> 14) ^ (k >> 2) ^ k) & 0x3F;
    case 7:
        k ^= (k >> 32);
        k ^= (k >> 24);
        return ((k >> 25) ^ (k >> 18) ^ (k >> 11) ^ (k >> 4) ^ k) & 0x7F;
    case 8:
        k ^= (k >> 32);
        return ((k >> 24) ^ (k >> 16) ^ (k >> 8) ^ k) & 0xFF;
    case 9:
        k ^= (k >> 32);
        return ((k >> 23) ^ (k >> 14) ^ (k >> 5) ^ k) & 0x1FF;
    case 10:
        k ^= (k >> 32);
        return ((k >> 22) ^ (k >> 12) ^ (k >> 2) ^ k) & 0x3FF;
    case 11:
        k ^= (k >> 32);
        return ((k >> 21) ^ (k >> 10) ^ k) & 0x7FF;
    case 12:
        k ^= (k >> 32);
        return ((k >> 20) ^ (k >> 8) ^ k) & 0xFFF;
    case 13:
        return ((k >> 51) ^ (k >> 38) ^ (k >> 25) ^ (k >>12) ^ k) & 0x1FFF;
    case 14:
        return ((k >> 50) ^ (k >> 36) ^ (k >> 22) ^ (k >> 8) ^ k) & 0x3FFF;
    case 15:
        return ((k >> 49) ^ (k >> 34) ^ (k >> 19) ^ (k >> 4) ^ k) & 0x7FFF;
    case 16:
        return ((k >> 48) ^ (k >> 32) ^ (k >> 16) ^ k) & 0xFFFF;
    case 17:
        return ((k >> 47) ^ (k >> 30) ^ (k >> 13) ^ k) & 0x1FFFF;
    case 18:
        return ((k >> 46) ^ (k >> 28) ^ (k >> 10) ^ k) & 0x3FFFF;
    case 19:
        return ((k >> 45) ^ (k >> 26) ^ (k >> 7) ^ k) & 0x7FFFF;
    case 20:
        return ((k >> 44) ^ (k >> 24) ^ (k >> 4) ^ k) & 0xFFFFF;
    case 21:
        return ((k >> 43) ^ (k >> 22) ^ (k >> 1) ^ k) & 0x1FFFFF;
    case 22:
        return ((k >> 42) ^ (k >> 20) ^ k) & 0x3FFFFF;
    case 23:
        return ((k >> 41) ^ (k >> 18) ^ k) & 0x7FFFFF;
    case 24:
        return ((k >> 40) ^ (k >> 16) ^ k) & 0xFFFFFF;
    case 25:
        return ((k >> 39) ^ (k >> 14) ^ k) & 0x1FFFFF;
    case 26:
        return ((k >> 38) ^ (k >> 12) ^ k) & 0x3FFFFF;
    case 27:
        return ((k >> 37) ^ (k >> 10) ^ k) & 0x7FFFFF;
    case 28:
        return ((k >> 36) ^ (k >> 8) ^ k) & 0xFFFFFF;
    case 29:
        return ((k >> 35) ^ (k >> 6) ^ k) & 0x1FFFFFF;
    case 30:
        return ((k >> 34) ^ (k >> 4) ^ k) & 0x3FFFFFF;
    case 31:
        return ((k >> 33) ^ (k >> 2) ^ k) & 0x7FFFFFF;
    case 32:
        return ((k >> 32) ^ k) & 0xFFFFFFFF;
    default:
        /* It is unlikely that we reach this level, so we can pay the cost of
         * argument checking */
        if( (nb_bits <= 0) || (nb_bits > 64) ) {
            assert(nb_bits > 0 && nb_bits <= 64);
            return ~0ULL;
        }
        return ((k >> 32) ^ k) & (~0ULL >> (64-nb_bits));
    }
}

void parsec_hash_table_lock_bucket(parsec_hash_table_t *ht, parsec_key_t key )
{
    uint64_t hash;

    parsec_atomic_rwlock_rdlock(&ht->rw_lock);
    hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), ht->rw_hash->nb_bits);
    assert( hash < (1ULL<<ht->rw_hash->nb_bits) );
    parsec_atomic_lock(&ht->rw_hash->buckets[hash].lock);
}

static void parsec_hash_table_resize(parsec_hash_table_t *ht)
{
    parsec_atomic_lock_t unlocked = PARSEC_ATOMIC_UNLOCKED;
    parsec_hash_table_head_t *head;
    parsec_hash_table_head_t *old_head = ht->rw_hash;
    int nb_bits = old_head->nb_bits + 1;
    assert(nb_bits < 32);

    /* count the number of used buckets */
    int32_t used_buckets = 0;
    for (size_t i = 0; i < (1ULL << old_head->nb_bits); ++i) {
        if (NULL != old_head->buckets[i].first_item) {
            ++used_buckets;
        }
    }
    old_head->used_buckets = used_buckets;

    head = malloc(sizeof(parsec_hash_table_head_t));
    head->buckets      = malloc((1ULL<<nb_bits) * sizeof(parsec_hash_table_bucket_t));
    head->nb_bits      = nb_bits;
    head->used_buckets = 0;
    head->next         = old_head;
    head->next_to_free = old_head;
    ht->rw_hash        = head;

    for( size_t i = 0; i < (1ULL<<nb_bits); i++) {
        head->buckets[i].lock = unlocked;
        head->buckets[i].cur_len = 0;
        head->buckets[i].first_item = NULL;
    }
}

void parsec_hash_table_unlock_bucket_impl(parsec_hash_table_t *ht, parsec_key_t key, const char *file, int line)
{
    int resize = 0;
    parsec_hash_table_head_t *cur_head;
    uint64_t hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), ht->rw_hash->nb_bits);

    assert( hash < (1ULL<<ht->rw_hash->nb_bits) );
    if( ht->rw_hash->buckets[hash].cur_len > ht->max_collisions_hint ) {
        if( (int)ht->rw_hash->nb_bits + 1 < ht->max_table_nb_bits )
            resize = 1;
        else {
            if( !ht->warning_issued ) {
                parsec_warning("%s:%d -- Hash table has %d collisions in bucket %lu, but it already spans over %lu buckets. Performance might get very bad if more elements continue to stack in this bucket. Consider allowing larger resize with the MCA parameter parsec_hash_table_max_table_nb_bits",
                               file, line, ht->rw_hash->buckets[hash].cur_len, hash, (1UL<<ht->rw_hash->nb_bits));
                ht->warning_issued = 1;
            }
        }
    }
    cur_head = ht->rw_hash;
    parsec_atomic_unlock(&ht->rw_hash->buckets[hash].lock);
    parsec_atomic_rwlock_rdunlock(&ht->rw_lock);

    if( resize ) {
        parsec_atomic_rwlock_wrlock(&ht->rw_lock);
        if( cur_head == ht->rw_hash ) {
            /* Barring ABA problems, nobody resized the hash table;
             * Good enough hint that it's our role to do so */
            parsec_hash_table_resize(ht);
        }
        /* Otherwise, let's asssume somebody resized already */
        parsec_atomic_rwlock_wrunlock(&ht->rw_lock);
    }
}

void parsec_hash_table_fini(parsec_hash_table_t *ht)
{
    parsec_hash_table_head_t *head, *next;
    head = ht->rw_hash;
    while( NULL != head ) {
        for(size_t i = 0; i < (1ULL<<head->nb_bits); i++) {
            assert(NULL == head->buckets[i].first_item);
        }
        next = head->next_to_free;
        free(head->buckets);
        free(head);
        head = next;
    }
}

void parsec_hash_table_nolock_insert(parsec_hash_table_t *ht, parsec_hash_table_item_t *item)
{
    uint64_t hash;
    parsec_key_t key = item->key;
    hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), ht->rw_hash->nb_bits);
    item->next_item = ht->rw_hash->buckets[hash].first_item;
    item->hash64 = ht->key_functions.key_hash(key, ht->hash_data);
    ht->rw_hash->buckets[hash].first_item = item;
    ht->rw_hash->buckets[hash].cur_len++;
#if defined(PARSEC_DEBUG_NOISIER)
    {
        char estr[64];
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Added item %p/%s into hash table %p in bucket %d",
                             item, ht->key_functions.key_print(estr, 64, item->key, ht->hash_data), ht, hash);
    }
#endif
}

static void *parsec_hash_table_nolock_remove_from_old_tables(parsec_hash_table_t *ht, parsec_key_t key)
{
    parsec_hash_table_head_t *head, *prev_head;
    parsec_hash_table_item_t *current_item, *prev_item;
    int32_t res;
    uint64_t hash;
    uint64_t hash64 = ht->key_functions.key_hash(key, ht->hash_data);
#if defined(HELPFIRST)
    uint64_t hash_main_bucket = parsec_hash_table_universal_rehash(hash64, ht->rw_hash->nb_bits);
#endif
    prev_head = ht->rw_hash;
    for(head = ht->rw_hash->next; NULL != head; head = head->next) {
        hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), head->nb_bits);
        prev_item = NULL;
        parsec_atomic_lock(&head->buckets[hash].lock );
        current_item = head->buckets[hash].first_item;
        while( NULL != current_item ) {
            if( OPTIMIZED_EQUAL_TEST(current_item, key, hash64, ht) ) {
#if defined(PARSEC_DEBUG_NOISIER)
                char estr[64];
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Removed item %p/%s from (old) hash table %p/%p in bucket %d",
                                     BASEADDROF(current_item, ht), ht->key_functions.key_print(estr, 64, key, ht->hash_data), ht, head, hash);
#endif
                if( NULL == prev_item ) {
                    head->buckets[hash].first_item = current_item->next_item;
                } else {
                    prev_item->next_item = current_item->next_item;
                }
                res = --(head->buckets[hash].cur_len);
                if( 0 == res ) {
                    res = parsec_atomic_fetch_dec_int32(&head->used_buckets);
                    if( 1 == res ) {
                        parsec_atomic_cas_ptr(&prev_head->next, head, head->next);
                    }
                }
                parsec_atomic_unlock(&head->buckets[hash].lock );
                return BASEADDROF(current_item, ht);
            }
#if defined(HELPFIRST)
            if( ht->key_functions.key_hash(current_item->key, ht->rw_hash->nb_bits, ht->hash_data) == hash_main_bucket ) {
                /* It's not the target item, but it's an item that goes in the
                 * same bucket as the target item, so we already have the lock
                 * on that bucket in the main table: insert it there costs not
                 * much and would help getting rid of old tables */
                 if( NULL == prev_item ) {
                     head->buckets[hash].first_item = current_item->next_item;
                 } else {
                     prev_item->next_item = current_item->next_item;
                 }
                 res = --(head->buckets[hash].cur_len);
                 if( 0 == res ) {
                     res = parsec_atomic_fetch_dec_int32(&head->used_buckets);
                     if( 1 == res ) {
                         parsec_atomic_cas_ptr(&prev_head->next, head, head->next);
                     }
                 }
                 parsec_hash_table_nolock_insert(ht, current_item);
                 if( NULL == prev_item )
                     current_item = head->buckets[hash].first_item;
                 else
                     current_item = prev_item->next_item;
            } else
#endif
            {
                prev_item = current_item;
                current_item = prev_item->next_item;
            }
        }
        parsec_atomic_unlock(&head->buckets[hash].lock );
        prev_head = head;
    }
    return NULL;
}

#if !defined(HELPFIRST)
static void *parsec_hash_table_nolock_find_in_old_tables(parsec_hash_table_t *ht, parsec_key_t key)
{
    parsec_hash_table_head_t *head, *prev_head = ht->rw_hash;
    parsec_hash_table_item_t *current_item, *prev_item = NULL;
    int res = 0;
    uint64_t hash, hash64 = ht->key_functions.key_hash(key, ht->hash_data);
    for(head = ht->rw_hash->next; NULL != head; head = head->next) {
        prev_item = NULL;
        hash = parsec_hash_table_universal_rehash(hash64, head->nb_bits);
        // We need the lock on the old tables, as some other thread might
        // be removing elements in this bucket, through remove_from_old_tables
        // and that thread relies on the lowlevel table locks
        parsec_atomic_lock( &head->buckets[hash].lock );
        for(current_item = head->buckets[hash].first_item;
            NULL != current_item;
            current_item = current_item->next_item) {
            if( OPTIMIZED_EQUAL_TEST(current_item, key, hash64, ht) ) {
#if defined(PARSEC_DEBUG_NOISIER)
                char estr[64];
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Found item %p/%s into (old) hash table %p/%p in bucket %d",
                                     BASEADDROF(current_item, ht), ht->key_functions.key_print(estr, 64, key, ht->hash_data), ht, head, hash);
#endif
                // We already have the lock on the toplevel table bucket,
                // and we have the lock on the lowlevel table bucket... So
                // use the opportunity to move the element in the toplevel
                if(NULL == prev_item) {
                    head->buckets[hash].first_item = current_item->next_item;
                } else {
                    prev_item->next_item = current_item->next_item;
                }
                current_item->next_item = NULL;
                res = --(head->buckets[hash].cur_len);
                if( 0 == res ) {
                    res = parsec_atomic_fetch_dec_int32(&head->used_buckets);
                    if( 1 == res ) {
                        parsec_atomic_cas_ptr(&prev_head->next, head, head->next);
                    }
                }
                parsec_hash_table_nolock_insert(ht, current_item);
                parsec_atomic_unlock( &head->buckets[hash].lock );
                return BASEADDROF(current_item, ht);
            }
            prev_item = current_item;
        }
        parsec_atomic_unlock( &head->buckets[hash].lock );
        prev_head = head;
    }
    return NULL;
}
#endif

void *parsec_hash_table_nolock_find(parsec_hash_table_t *ht, parsec_key_t key)
{
    parsec_hash_table_item_t *current_item;
    uint64_t hash;
    void *item;
    uint64_t hash64 = ht->key_functions.key_hash(key, ht->hash_data);
    hash = parsec_hash_table_universal_rehash(hash64, ht->rw_hash->nb_bits);
    for(current_item = ht->rw_hash->buckets[hash].first_item;
        NULL != current_item;
        current_item = current_item->next_item) {
        if( OPTIMIZED_EQUAL_TEST(current_item, key, hash64, ht) ) {
#if defined(PARSEC_DEBUG_NOISIER)
            char estr[64];
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Found item %p/%s into hash table %p in bucket %d",
                                 BASEADDROF(current_item, ht), ht->key_functions.key_print(estr, 64, key, ht->hash_data), ht, hash);
#endif
            return BASEADDROF(current_item, ht);
        }
    }
#if defined(HELPFIRST)
    item = parsec_hash_table_nolock_remove_from_old_tables(ht, key);
    if( NULL != item ) {
        current_item = ITEMADDROF(item, ht);
        parsec_hash_table_nolock_insert(ht, current_item);
    }
#else
    item = parsec_hash_table_nolock_find_in_old_tables(ht, key);
#endif
    return item;
}

void *parsec_hash_table_nolock_remove(parsec_hash_table_t *ht, parsec_key_t key)
{
    parsec_hash_table_item_t *current_item, *prev_item;
    uint64_t hash64 = ht->key_functions.key_hash(key, ht->hash_data);
    uint64_t hash = parsec_hash_table_universal_rehash(hash64, ht->rw_hash->nb_bits);
    prev_item = NULL;
    for(current_item = ht->rw_hash->buckets[hash].first_item;
        NULL != current_item;
        current_item = prev_item->next_item) {
        if( OPTIMIZED_EQUAL_TEST(current_item, key, hash64, ht) ) {
            if( NULL == prev_item ) {
                ht->rw_hash->buckets[hash].first_item = current_item->next_item;
            } else {
                prev_item->next_item = current_item->next_item;
            }
            --(ht->rw_hash->buckets[hash].cur_len);
#if defined(PARSEC_DEBUG_NOISIER)
            char estr[64];
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Removed item %p/%s from hash table %p in bucket %d",
                                 BASEADDROF(current_item, ht), ht->key_functions.key_print(estr, 64, key, ht->hash_data), ht, hash);
#endif
            return BASEADDROF(current_item, ht);
        }
        prev_item = current_item;
    }
    return parsec_hash_table_nolock_remove_from_old_tables(ht, key);
}

void parsec_hash_table_insert_impl(parsec_hash_table_t *ht, parsec_hash_table_item_t *item, const char *file, int line)
{
    uint64_t hash;
    parsec_hash_table_head_t *cur_head;
    int resize = 0;
    parsec_atomic_rwlock_rdlock(&ht->rw_lock);
    cur_head = ht->rw_hash;
    hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(item->key, ht->hash_data), ht->rw_hash->nb_bits);
    assert( hash < (1ULL<<ht->rw_hash->nb_bits) );
    parsec_atomic_lock(&ht->rw_hash->buckets[hash].lock);
    parsec_hash_table_nolock_insert(ht, item);
    if( ht->rw_hash->buckets[hash].cur_len > ht->max_collisions_hint ) {
        if( (int)ht->rw_hash->nb_bits + 1 < ht->max_table_nb_bits )
            resize = 1;
        else {
            if( !ht->warning_issued ) {
                parsec_warning("%s:%d -- Hash table has %d collisions in bucket %lu, but it already spans over %lu buckets. Performance might get very bad if more elements continue to stack in this bucket. Consider allowing larger resize with the MCA parameter parsec_hash_table_max_table_nb_bits",
                               file, line, ht->rw_hash->buckets[hash].cur_len, hash, (1UL<<ht->rw_hash->nb_bits));
                ht->warning_issued = 1;
            }
        }
    }
    parsec_atomic_unlock(&ht->rw_hash->buckets[hash].lock);
    parsec_atomic_rwlock_rdunlock(&ht->rw_lock);

    if( resize ) {
        parsec_atomic_rwlock_wrlock(&ht->rw_lock);
        if( cur_head == ht->rw_hash ) {
            /* Barring ABA problems, nobody resized the hash table;
             * Good enough hint that it's our role to do so */
            parsec_hash_table_resize(ht);
        }
        /* Otherwise, let's asssume somebody resized already */
        parsec_atomic_rwlock_wrunlock(&ht->rw_lock);
    }
}

void *parsec_hash_table_find(parsec_hash_table_t *ht, parsec_key_t key)
{
    uint64_t hash;
    void *ret;
    parsec_atomic_rwlock_rdlock(&ht->rw_lock);
    hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), ht->rw_hash->nb_bits);
    assert( hash < (1ULL<<ht->rw_hash->nb_bits) );
    parsec_atomic_lock(&ht->rw_hash->buckets[hash].lock);
    ret = parsec_hash_table_nolock_find(ht, key);
    parsec_atomic_unlock(&ht->rw_hash->buckets[hash].lock);
    parsec_atomic_rwlock_rdunlock(&ht->rw_lock);
    return ret;
}

void *parsec_hash_table_remove(parsec_hash_table_t *ht, parsec_key_t key)
{
    uint64_t hash;
    void *ret;
    parsec_atomic_rwlock_rdlock(&ht->rw_lock);
    hash = parsec_hash_table_universal_rehash(ht->key_functions.key_hash(key, ht->hash_data), ht->rw_hash->nb_bits);
    assert( hash < (1ULL<<ht->rw_hash->nb_bits) );
    parsec_atomic_lock(&ht->rw_hash->buckets[hash].lock);
    ret = parsec_hash_table_nolock_remove(ht, key);
    parsec_atomic_unlock(&ht->rw_hash->buckets[hash].lock);
    parsec_atomic_rwlock_rdunlock(&ht->rw_lock);
    return ret;
}

void parsec_hash_table_stat(parsec_hash_table_t *ht)
{
    parsec_hash_table_head_t *head;
    double mean, M2, delta, delta2;
    int n, min, max;
    int nb;
    uint32_t i, j;
    parsec_hash_table_item_t *current_item;

    for(head = ht->rw_hash, j=0; NULL != head; head = head->next, j++) {
        n = 0;
        min = -1;
        max = -1;
        mean = 0.0;
        M2 = 0.0;
        for(i = 0; i < (1ULL<<head->nb_bits); i++) {
            nb = 0;
            for(current_item = head->buckets[i].first_item;
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
        printf("table %p level %d: %d lists, of length %d to %d average length: %g and variance %g\n",
               ht, j, n, min, max, mean, M2/(n-1));
    }
}

void parsec_hash_table_for_all(parsec_hash_table_t* ht, parsec_hash_elem_fct_t fct, void* cb_data)
{
    parsec_hash_table_head_t *head;
    parsec_hash_table_item_t *current_item;
    void* user_item;

    for( head = ht->rw_hash; NULL != head; head = head->next ) {
        for( size_t i = 0; i < (1ULL<<head->nb_bits); i++ ) {
            current_item = head->buckets[i].first_item;
            /* Iterating the list to check if we have the element */
            while( NULL != current_item ) {
                user_item = parsec_hash_table_item_lookup(ht, current_item);
                current_item = current_item->next_item;
                fct( user_item, cb_data );
            }
        }
    }
}

char *parsec_hash_table_generic_64bits_key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data)
{
    (void)user_data;
    snprintf(buffer, buffer_size, "%016"PRIu64, (uint64_t)(uintptr_t)k);
    return buffer;
}

uint64_t parsec_hash_table_generic_64bits_key_hash(parsec_key_t key, void *user_data)
{
    (void)user_data;
    return (uint64_t)key;
}

parsec_key_fn_t parsec_hash_table_generic_key_fn = {
        .key_equal = parsec_hash_table_generic_64bits_key_equal,
        .key_hash  = parsec_hash_table_generic_64bits_key_hash,
        .key_print = parsec_hash_table_generic_64bits_key_print
};
