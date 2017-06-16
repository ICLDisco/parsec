/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _hash_table_h
#define _hash_table_h

#include "parsec/parsec_config.h"
#include "parsec/class/list_item.h"

/**
 * @defgroup parsec_internal_classes_hashtable Hash Tables
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Hash Tables for PaRSEC
 *
 *  @details
 *    Items are defined with @ref hash_table_item_t.
 *    To make a structure hash-able, declare a field with this type
 *    in the structure, and when creating the Hash Table with @ref hash_table_create,
 *    pass the offset of that field in that structure.
 * 
 *    Then, pass the address of that field in this structure to @ref hash_table_insert,
 *    to insert elements, and @ref hash_table_find / @ref hash_table_remove will return
 *    the pointer to the structure.
 *
 *    Keys are limited to uint64_t at this time.
 */

BEGIN_C_DECLS

typedef struct hash_table_s        hash_table_t;       /**< A Hash Table */
typedef struct hash_table_item_s   hash_table_item_t;  /**< Items stored in a Hash Table */
typedef struct hash_table_bucket_s hash_table_bucket_t;/**< Buckets of a Hash Table */

/**
 * @brief Function pointer of hash function for hash_table 
*/
typedef uint32_t (hash_table_fn_t)(uintptr_t key, void *data);

/**
 * @brief One type of hash table for task, tiles and functions 
 */
struct hash_table_s {
    parsec_object_t       super;                /**< A Hash Table is a PaRSEC object */
    size_t                size;                 /**< It holds size buckets           */
    int64_t               elt_hashitem_offset;  /**< Elements belonging to this hash table have a hash_table_item_t at this offset */
    hash_table_fn_t      *hash;                 /**< Elements are hashed with this function */
    void                 *hash_data;            /**< This is the second parameter of the hashing function */
    hash_table_bucket_t  *buckets;              /**< These are the buckets (that are lists of items) of this table */
};
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(hash_table_t);

/**
 * @brief Hashtable Item
 */
struct hash_table_item_s {
    hash_table_item_t       *next_item;        /**< A hash table item is a chained list */
    uint64_t                 key;              /**< Items are identified with a 64 bits key */
};

/**
 * @brief Create a hash table
 *
 * @details
 *  @arg[inout] ht     the hash table to initialize
 *  @arg[in]    offset the number of bytes between an element pointer and its hash_table_item field
 *  @arg[in]    size   the number of buckets
 *  @arg[in]    fn     the function to hash
 *  @arg[in]    data   the opaque pointer to pass to the hash function
 */
void hash_table_init(hash_table_t *ht, int64_t offset, size_t size_of_table, hash_table_fn_t *hash, void *data);

/**
 * @brief locks the bucket corresponding to this key
 *
 * @details Waits until the bucket corresponding to the key can be locked
 *  and locks it preventing other threads to update this bucket.
 *  @arg[inout] ht  the hash_table
 *  @arg[in]    key the key for which to lock the bucket
 */
void hash_table_lock_bucket(hash_table_t *ht, uint64_t key );

/**
 * @brief unlocks the bucket corresponding to this key
 *
 * @details allow other threads to update this bucket.
 *  @arg[inout] ht  the hash_table
 *  @arg[in]    key the key for which to unlock the bucket
 */
void hash_table_unlock_bucket(hash_table_t *ht, uint64_t key );

/**
 * @brief Destroy a hash table
 *
 * @details
 *   Releases the resources allocated by the hash table.
 *   In debug mode, will assert if the hash table is not empty
 * @arg[inout] ht the hash table to release
 */
void hash_table_fini(hash_table_t *ht);

/**
 * @brief Insert element in a hash table without
 *  locking the bucket
 *
 * @details
 *  This is not thread safe.
 *  @arg[inout] ht   the hash table
 *  @arg[inout] item the item to insert. Its key must be initialized.
 */
void hash_table_nolock_insert(hash_table_t *ht, hash_table_item_t *item);

/**
 * @brief Find elements in the hash table
 *
 * @details
 *  This does lock the bucket while searching for the item.
 *  @arg[in] ht the hash table
 *  @arg[in] key the key of the element to find
 *  @return NULL if the element is not in the table, the element otherwise.
 */
void *hash_table_nolock_find(hash_table_t *ht, uint64_t key);

/**
 * @brief Remove element from the hash table without locking the
 *  bucket.
 *
 * @details
 *  function is not thread-safe.
 *  @arg[inout] ht the hash table
 *  @arg[inout] key the key of the item to remove
 *  @return NULL if the element was not in the table, the element
 *    that was removed from the table otherwise.
 */
void *hash_table_nolock_remove(hash_table_t *ht, uint64_t key);

/**
 * @brief Insert element in the hash table
 *
 * @details
 *  Inserts an element in the hash, assuming it is not already in the hash
 *  table. This function is thread-safe but assumes that the element does
 *  not belong to the table.
 *  @arg[inout] ht the hash table
 *  @arg[inout] item the pointer to the structure with a hash_table_item_t
 *              structure at the right offset (see hash_table_init)
 */
void hash_table_insert(hash_table_t *ht, hash_table_item_t *item);

/**
 * @brief Find element in the hash table wihout locking it
 *
 * @details
 *  This does not lock the bucket, and is not thread safe.
 *  @arg[in] ht the hash table
 *  @arg[in] key the key of the element to find
 *  @return NULL if the element is not in the table, the element otherwise.
 */
void *hash_table_find(hash_table_t *ht, uint64_t key);

/**
 * @brief Remove element from the hash table.
 *
 * @details
 *  @arg[inout] ht the hash table
 *  @arg[inout] key the key of the item to remove
 *  @return NULL if the element was not in the table, the element
 *    that was removed from the table otherwise.
 *
 * @remark this function is thread-safe.
 */
void * hash_table_remove(hash_table_t *ht, uint64_t key);

/**
 * @brief Converts a hash_table_item_t *pointer into its
 *  corresponding data pointer (void*).
 *
 * @details
 *  This is used to iterate over all the elements of a hash
 *  table: first_item, and all item->next will point to
 *  hash_table_item_t *, but the actual user' pointer is
 *  some bytes before this.
 *
 *     @arg[in] ht   the hash table in which elements belong
 *     @arg[in] item a pointer to a hash_table_item_t* belonging
 *                   to ht
 *     @return the pointer to the user data
 */
void *hash_table_item_lookup(hash_table_t *ht, hash_table_item_t *item);

/**
 * @brief Type of a function that can be called on each element of a hash table
 *
 * @details
 * @arg[inout] item the element of the hash table
 * @arg[inout] cb_data an opaque pointer corresponding to the user parameter
 *             (see @ref hash_table_for_all)
 */
typedef void (*hash_elem_fct_t)(void *item, void*cb_data);

/**
 * @brief Call the function passed as argument for all items in the
 *         hash table.
 *
 * @details This function is safe for items removal. In order to allow
 *         items removal, this function does not protect the hash
 *         table, and it is therefore not thread safe.
 *
 *  @arg[in] ht    the hash table
 *  @arg[in] fct   function to apply to all items in the hash table
 *  @arg[in] cb_data data to pass for each element as the first parameter of the fct.
 */
void hash_table_for_all(hash_table_t* ht, hash_elem_fct_t fct, void* cb_data);

END_C_DECLS

/** @} */

#endif

