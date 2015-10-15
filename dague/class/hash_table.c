#include <assert.h>
#include "dague/class/hash_table.h"
#include "dague/class/list.h"

/* To create object of class hash_table that inherits dague_object_t
 * class
 */
OBJ_CLASS_INSTANCE(hash_table, dague_object_t, NULL, NULL);

/* To create object of class dague_generic_bucket_t that
 * inherits from dague_list_item_t class.
 */
OBJ_CLASS_INSTANCE(dague_generic_bucket_t, dague_list_item_t, NULL, NULL);

/* Function to create generic hash table
 * Arguments:   - Size of the hash table, or the total number of bucket the
                  table will hold (int)
                - Size of each bucket the table will hold (int)
 * Returns:     - void
*/
void
hash_table_init(hash_table *obj, int size_of_table,
                hash_fn    *hash)
{
    obj->size     = size_of_table;
    obj->hash     = hash;
    obj->bucket_list = calloc(size_of_table, sizeof(dague_list_t *));

    int i;
    for( i=0; i<size_of_table; i++) {
        obj->bucket_list[i] = OBJ_NEW(dague_list_t);
    }
}

/* Function to destroy generic hash table
 * Arguments:   - hash table (hash_table *)
 * Returns:     - void
*/
void
hash_table_fini(hash_table *obj, int size_of_table)
{
    int i;
    for( i=0; i<size_of_table; i++) {
        free(obj->bucket_list[i]);
    }

    free(obj->bucket_list);
    OBJ_RELEASE(obj);
}

/* Bucket element's reference accounting:
 * Everytime we insert a bucket in the hashtable,
 * we increment the object's ref. count. Everytime
 * we find an element in the hash table, we
 * increment the ref. count.
 */

/* Function to insert element in the hash table
 * Arguments:
 * Returns:
 */
void
hash_table_insert
( hash_table *hash_table, dague_generic_bucket_t *bucket,
  uint64_t key, void *value, uint32_t hash )
{
    dague_list_item_t *current_bucket = (dague_list_item_t *)bucket;

    bucket->key     = key;
    bucket->value   = value;

    dague_list_t *bucket_list  = hash_table->bucket_list[hash];
    dague_list_push_back ( bucket_list, current_bucket );
    OBJ_RETAIN(current_bucket);
}

/* Function to find element in the hash table
 * Arguments:
 * Returns:
 */
void *
hash_table_find
( hash_table *hash_table,
  uint64_t key, uint32_t hash )
{
    dague_generic_bucket_t *current_bucket;
    dague_list_t *bucket_list = hash_table->bucket_list[hash];

    dague_list_lock ( bucket_list );

    current_bucket = (dague_generic_bucket_t *) DAGUE_LIST_ITERATOR_FIRST(bucket_list);

    /* Iterating the list to check if we have the element */
    while( current_bucket != (dague_generic_bucket_t *) DAGUE_LIST_ITERATOR_END(bucket_list) ) {
        if( current_bucket->key == key ) {
            OBJ_RETAIN(current_bucket);
            dague_list_unlock ( bucket_list );
            return (void *)current_bucket;
        }
        dague_list_item_t *item = &(current_bucket->super);
        current_bucket = (dague_generic_bucket_t *)DAGUE_LIST_ITERATOR_NEXT(item);
    }

    dague_list_unlock ( bucket_list );
    return (void *)NULL;
}

/* Function to remove element from the hash table
 * Arguments:
 * Returns:
 */
void *
hash_table_remove
( hash_table *hash_table,
  uint64_t key, uint32_t hash )
{
    dague_list_t *bucket_list = hash_table->bucket_list[hash];
    dague_list_item_t *current_bucket = hash_table_find ( hash_table, key, hash );
    /* Making sure if we are trying to remove something it is there in the first place */
    assert(current_bucket != NULL);

    if( current_bucket != NULL ) {
        dague_list_lock ( bucket_list );
        /* The following release is to account for the increment in hash_table_find() */
        OBJ_RELEASE(current_bucket);
        if( current_bucket->super.obj_reference_count == 2 ) {
            dague_list_nolock_remove ( bucket_list, current_bucket );
            /* To account for the increment in hash_table_insert() */
            OBJ_RELEASE(current_bucket);
        }
        dague_list_unlock ( bucket_list );
    }
    return (void *)current_bucket;
}
