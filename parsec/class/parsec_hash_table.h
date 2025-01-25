/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _parsec_hash_table_h
#define _parsec_hash_table_h

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/list_item.h"
#include "parsec/class/parsec_rwlock.h"

/**
 * @defgroup parsec_internal_classes_hashtable Hash Tables
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Hash Tables for PaRSEC
 *
 *  @details
 *    Items are defined with @ref parsec_hash_table_item_t.
 *    To make a structure hash-able, declare a field with this type
 *    in the structure, and when creating the Hash Table with @ref parsec_hash_table_init,
 *    pass the offset of that field in that structure.
 *
 *    Then, pass the address of that field in this structure to @ref parsec_hash_table_insert,
 *    to insert elements, and @ref parsec_hash_table_find / @ref parsec_hash_table_remove will return
 *    the pointer to the structure.
 *
 *    Keys are uintptr integers, but users may pass a pointer and provide a user-defined
 *    comparison function to use arbitrary length keys.
 */

BEGIN_C_DECLS

/**
 * @brief Keys are arbitrary opaque objects that are accessed through a set
 *        of user-defined functions.
 */
typedef uintptr_t parsec_key_t;
typedef struct parsec_key_fn_s {
    /** Returns a == b
     *  user_data is the pointer passed to hash_table_create */
    int          (*key_equal)(parsec_key_t a, parsec_key_t b, void *user_data);
    /** Writes up to buffer_size-1 bytes into buffer, creating a human-readable version of k
     *  user_data is the pointer passed to hash_table_create*/
    char *       (*key_print)(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data);
    /** Returns a hash of k, between 0 and 1ULL<<64
     *  If the key is already unique and comparable directly without passing it to key_equal,
     *  this function can return key.
     *  user_data is the pointer passed to hash_table_create */
    uint64_t     (*key_hash)(parsec_key_t k, void *user_data);
} parsec_key_fn_t;

/**
 * @brief Hashtable key handle
 *
 * @details
 *   When locking a bucket, a handle is provided that can be used for finding,
 *   inserting, and removing an element corresponding to the key provided
 *   when the bucket was locked. The handle is valid until the bucket is unlocked.
 */
struct parsec_key_handle_s {
    uint64_t                  hash64;           /**< Is a 64-bits hash of the key */
    uint64_t                  hash;             /**< Is a 64-bits hash of the key,
                                                     trimmed to the current size of the table */
    parsec_key_t              key;              /**< Items are identified with this key */
};

typedef struct parsec_key_handle_s parsec_key_handle_t;

typedef struct parsec_hash_table_s        parsec_hash_table_t;       /**< A Hash Table */
typedef struct parsec_hash_table_item_s   parsec_hash_table_item_t;  /**< Items stored in a Hash Table */
typedef struct parsec_hash_table_bucket_s parsec_hash_table_bucket_t;/**< Buckets of a Hash Table */


/**
 * @brief
 */
typedef struct parsec_hash_table_head_s {
    struct parsec_hash_table_head_s *next;                 /**< Table of smaller size that is not empty */
    struct parsec_hash_table_head_s *next_to_free;         /**< Table of smaller size, chained in allocation order */
    uint32_t                         nb_bits;              /**< This hash table has 1<<nb_bits buckets */
    int32_t                          used_buckets;         /**< Number of buckets still in use in this hash table */
    parsec_hash_table_bucket_t      *buckets;              /**< These are the buckets (that are lists of items) of this table */
} parsec_hash_table_head_t;

/**
 * @brief One type of hash table for task, tiles and functions
 */
struct parsec_hash_table_s {
    parsec_object_t           super;                /**< A Hash Table is a PaRSEC object */
    parsec_atomic_rwlock_t    rw_lock;              /**< 'readers' are threads that manipulate rw_hash (add, delete, find)
                                                     *   but do not resize it; 'writers' are threads that resize
                                                     *   rw_hash */
    int64_t                   elt_hashitem_offset;  /**< Elements belonging to this hash table have a parsec_hash_table_item_t
                                                     *   at this offset */
    parsec_key_fn_t           key_functions;        /**< How to acccess and modify the keys */
    void                     *hash_data;            /**< This is the last parameter of the hashing function */
    int                       max_collisions_hint;  /**< How many collisions before this hash table is resized */
    int                       max_table_nb_bits;    /**< Maximum size of the hash table: above this, if max_collisions_hint
                                                     *   is reached, a warning is issued (once), and elements just get stacked
                                                     *   in the same buckets. */
    int                       warning_issued;       /**< Number of times the warning mentionned above has been issued */
    parsec_hash_table_head_t *rw_hash;              /**< Added elements go in this hash table */
};
PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_hash_table_t);

/**
 * @brief Hashtable Item
 *
 * @details
 *   When inserting an element in the hash table, the key must be set by
 *   the caller; The other fields have meaning only for the hash table,
 *   and only as long as the element remains in the hash table.
 */
struct parsec_hash_table_item_s {
    parsec_hash_table_item_t *next_item;        /**< A hash table item is a chained list */
    uint64_t                  hash64;           /**< Is a 64-bits hash of the key */
    parsec_key_t              key;              /**< Items are identified with this key */
};

/**
 * @brief Initialize the hash table library
 *
 * @details
 *   Declare the MCA parameters used by the library to control the resize options
 *   @return PARSEC_SUCCESS on success, an error in other cases
 */
int parsec_hash_tables_init(void);

/**
 * @brief Create a hash table
 *
 * @details
 *  @arg[inout] ht      the hash table to initialize
 *  @arg[in]    offset  the number of bytes between an element pointer and its parsec_hash_table_item field
 *  @arg[in]    nb_bits log_2 of the number of buckets (how many bits to use on the hash)
 *                      This field is a hint and must be between 1 and 16.
 *  @arg[in]    key_fn  the functions to access and modify these keys
 *  @arg[in]    data    the opaque pointer to pass to the hash function
 */
void parsec_hash_table_init(parsec_hash_table_t *ht, int64_t offset, int nb_bits, parsec_key_fn_t key_functions, void *data);

/**
 * @brief locks the bucket corresponding to this key
 *
 * @details Waits until the bucket corresponding to the key can be locked
 *  and locks it preventing other threads to update this bucket.
 *  The table cannot be resized as long as any bucket is locked.
 *  @arg[inout] ht  the parsec_hash_table
 *  @arg[in]    key the key for which to lock the bucket
 */
void parsec_hash_table_lock_bucket(parsec_hash_table_t *ht, parsec_key_t key );


/**
 * @brief locks the bucket corresponding to this key and sets the bucket handle
 *
 * @details Waits until the bucket corresponding to the key can be locked
 *  and locks it preventing other threads to update this bucket.
 *  The table cannot be resized as long as any bucket is locked.
 *  @arg[inout] ht  the parsec_hash_table
 *  @arg[in]    key the key for which to lock the bucket
 *  @arg[out]   handle the handle identifying the locked bucket
 */
void parsec_hash_table_lock_bucket_handle(parsec_hash_table_t *ht, parsec_key_t key,
                                          parsec_key_handle_t *handle);

/**
 * @brief unlocks the bucket corresponding to this key.
 *
 * @details allow other threads to update this bucket.
 *  This operation might block and resize the table, if non-locking
 *  insertions during the critical section added too many elements.
 *  Use it through the parsec_hash_table_unlock_bucket macro
 *  @arg[inout] ht  the parsec_hash_table
 *  @arg[in]    key the key for which to unlock the bucket
 *  @arg[in]    file the file where the caller of the function is
 *  @arg[in]    line the line at which this function is called
 */
void parsec_hash_table_unlock_bucket_impl(parsec_hash_table_t *ht, parsec_key_t key, const char *file, int line);
#define parsec_hash_table_unlock_bucket(ht, key) parsec_hash_table_unlock_bucket_impl(ht, key, __FILE__, __LINE__)

/**
 * @brief unlocks the bucket identified by the handle.
 *
 * @details allow other threads to update this bucket.
 *  This operation might block and resize the table, if non-locking
 *  insertions during the critical section added too many elements.
 *  Use it through the parsec_hash_table_unlock_bucket_handle macro.
 *  The handle becomes invalid after this call.
 *  @arg[inout] ht  the parsec_hash_table
 *  @arg[in]    handle the handle to the bucket to unlock
 *  @arg[in]    file the file where the caller of the function is
 *  @arg[in]    line the line at which this function is called
 */
void parsec_hash_table_unlock_bucket_handle_impl(parsec_hash_table_t *ht,
                                                 const parsec_key_handle_t *handle,
                                                 const char *file, int line);
#define parsec_hash_table_unlock_bucket_handle(ht, handle) \
  parsec_hash_table_unlock_bucket_handle_impl(ht, handle, __FILE__, __LINE__)

/**
 * @brief Destroy a hash table
 *
 * @details
 *   Releases the resources allocated by the hash table.
 *   In debug mode, will assert if the hash table is not empty
 * @arg[inout] ht the hash table to release
 */
void parsec_hash_table_fini(parsec_hash_table_t *ht);

/**
 * @brief Insert element in a hash table without
 *  locking the bucket.
 *
 *
 * @details
 *  This is not thread safe, and might make this bucket to exceed
 *  the hint provided by the parsec_hash_table_max_collisions MCA parameter
 *
 *  @arg[inout] ht   the hash table
 *  @arg[inout] item the item to insert. Its key must be initialized.
 */
void parsec_hash_table_nolock_insert(parsec_hash_table_t *ht, parsec_hash_table_item_t *item);

/**
 * @brief Insert element identified by handle in a hash table without
 *  locking the bucket.
 *
 *
 * @details
 *  This is not thread safe, and might make this bucket to exceed
 *  the hint provided by the parsec_hash_table_max_collisions MCA parameter.
 *  The item is inserted into the bucket identified by handle, the item's key is
 *  ignored.
 *
 *  @arg[inout] ht   the hash table
 *  @arg[in]    handle the handle identifying the bucket to insert into
 *  @arg[inout] item the item to insert. Its key must be initialized.
 */
void parsec_hash_table_nolock_insert_handle(parsec_hash_table_t *ht,
                                            const parsec_key_handle_t *handle,
                                            parsec_hash_table_item_t *item);

/**
 * @brief Find elements in the hash table
 *
 * @details
 *  This does lock the bucket while searching for the item.
 *  @arg[in] ht the hash table
 *  @arg[in] key the key of the element to find
 *  @return NULL if the element is not in the table, the element otherwise.
 */
void *parsec_hash_table_nolock_find(parsec_hash_table_t *ht, parsec_key_t key);

/**
 * @brief Find elements in the hash table using the handle
 *
 * @details
 *  This does lock the bucket while searching for the item.
 *  @arg[in] ht the hash table
 *  @arg[in] handle the handle for the bucket to find
 *  @return NULL if the element is not in the table, the element otherwise.
 */
void *parsec_hash_table_nolock_find_handle(parsec_hash_table_t *ht,
                                           const parsec_key_handle_t *handle);

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
void *parsec_hash_table_nolock_remove(parsec_hash_table_t *ht, parsec_key_t key);

/**
 * @brief Remove element identified by handle from the hash table without locking the
 *  bucket.
 *
 * @details
 *  function is not thread-safe.
 *  @arg[inout] ht the hash table
 *  @arg[inout] handle the handle of the item to remove
 *  @return NULL if the element was not in the table, the element
 *    that was removed from the table otherwise.
 */
void *parsec_hash_table_nolock_remove_handle(parsec_hash_table_t *ht,
                                             const parsec_key_handle_t *handle);

/**
 * @brief Insert element in the hash table
 *
 * @details
 *  Inserts an element in the hash, assuming it is not already in the hash
 *  table. This function is thread-safe but assumes that the element does
 *  not belong to the table. This might lock the table to redimension it,
 *  if a bucket holds more than parsec_hash_table_max_collisions MCA parameter
 *  Use this through the parsec_hash_table_insert macro.
 *  @arg[inout] ht the hash table
 *  @arg[inout] item the pointer to the structure with a parsec_hash_table_item_t
 *              structure at the right offset (see parsec_hash_table_init)
 *  @arg[in]    file the name of the file where the caller of this function is
 *  @arg[in]    line the line at which the caller is
 */
void parsec_hash_table_insert_impl(parsec_hash_table_t *ht, parsec_hash_table_item_t *item, const char *file, int line);
#define parsec_hash_table_insert(ht, item) parsec_hash_table_insert_impl(ht, item, __FILE__, __LINE__)

/**
 * @brief Find element in the hash table wihout locking it
 *
 * @details
 *  This does not lock the bucket, and is not thread safe.
 *  @arg[in] ht the hash table
 *  @arg[in] key the key of the element to find
 *  @return NULL if the element is not in the table, the element otherwise.
 */
void *parsec_hash_table_find(parsec_hash_table_t *ht, parsec_key_t key);

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
void * parsec_hash_table_remove(parsec_hash_table_t *ht, parsec_key_t key);

/**
 * @brief Converts a parsec_hash_table_item_t *pointer into its
 *  corresponding data pointer (void*).
 *
 * @details
 *  This is used to iterate over all the elements of a hash
 *  table: first_item, and all item->next will point to
 *  parsec_hash_table_item_t *, but the actual user' pointer is
 *  some bytes before this.
 *
 *     @arg[in] ht   the hash table in which elements belong
 *     @arg[in] item a pointer to a parsec_hash_table_item_t* belonging
 *                   to ht
 *     @return the pointer to the user data
 */
void *parsec_hash_table_item_lookup(parsec_hash_table_t *ht, parsec_hash_table_item_t *item);

/**
 * @brief Type of a function that can be called on each element of a hash table
 *
 * @details
 * @arg[inout] item the element of the hash table
 * @arg[inout] cb_data an opaque pointer corresponding to the user parameter
 *             (see @ref parsec_hash_table_for_all)
 */
typedef void (*parsec_hash_elem_fct_t)(void *item, void*cb_data);

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
void parsec_hash_table_for_all(parsec_hash_table_t* ht, parsec_hash_elem_fct_t fct, void* cb_data);

/**
 * @brief a generic key_equal function that can be used for 64 bits keys
 *
 * @details Use this equal function is you simply code no-collisions
 *          identifiers within the 64 bits of the parsec_key_t.
 *          That function simply return a == b, ignoring user_data
 *
 *  @arg[in] a one key
 *  @arg[in] b the other key
 *  @arg[in] use_data ignored parameter
 *  @return true iff a == b
 *
 *  This function is simply NULL in the current implementation, because we test the
 *  equality of the keys in the hash table implementation, and if the keys are equal on
 *  64 bits, in the case they code the key, they are equal (tautology), and if they
 *  point to a key, they point to the same, so they are also equal.
 */
#define parsec_hash_table_generic_64bits_key_equal    NULL

/**
 * @brief a generic key_print function that can be used for 64 bits keys
 *
 * @details simply prints the hexadecimal value of the key in buffer,
 *          assuming that the key is on 64 bits
 *
 *   @arg[inout] buffer a buffer of buffer_size characters
 *   @arg[in] buffer_size the size of buffer
 *   @arg[in] k the key to print
 *   @arg[in] user_data ignored parameter
 *   @return buffer for convenience use in printf statements
 */
char *parsec_hash_table_generic_64bits_key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data);

/**
 * @brief a generic hash function for keys that fit in 64 bits
 *
 * @details computes a hash of k on 64 bits, assuming that k is a
 *          key on 64 bits. This function returns k.
 *
 *    @arg[in] k the key to hash
 *    @arg[in] user_data ignored parameter
 *    @return a 64-bits hash of k on nb_bits
 */
uint64_t parsec_hash_table_generic_64bits_key_hash(parsec_key_t k, void *user_data);

/**
 * @brief displays statistics about the hash table
 *
 * @details displays statistics about the hash table on stdout
 *
 *    @arg[in] ht the hash table
 */
void parsec_hash_table_stat(parsec_hash_table_t *ht);

extern parsec_key_fn_t parsec_hash_table_generic_key_fn;

END_C_DECLS

/** @} */

#endif

