#include <assert.h>
#include "dague/class/hash_table.h"
#include "dague/class/list.h"

/* To create object of class hash_table that inherits dague_object_t
 * class
 */
OBJ_CLASS_INSTANCE(hash_table, dague_object_t, NULL, NULL);

/* To create object of class dague_hashtabel_item_t that
 * inherits from dague_list_item_t class.
 */
OBJ_CLASS_INSTANCE(dague_hashtable_item_t, dague_list_item_t, NULL, NULL);

/* To create object of class dague_generic_bucket_t that
 * inherits from dague_hashtable_item_t class.
 */
OBJ_CLASS_INSTANCE(dague_generic_bucket_t, dague_hashtable_item_t, NULL, NULL);

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
    obj->item_list = calloc(size_of_table, sizeof(dague_list_t *));

    int i;
    for( i=0; i<size_of_table; i++) {
        obj->item_list[i] = OBJ_NEW(dague_list_t);
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
        free(obj->item_list[i]);
    }

    free(obj->item_list);
    OBJ_RELEASE(obj);
}

/* item element's reference accounting:
 * Everytime we insert a item in the hashtable,
 * we increment the object's ref. count. Everytime
 * we find an element in the hash table, we
 * increment the ref. count.
 */

/* Function to insert element in the hash table (thread safe)
 * Arguments:
 * Returns:
 */
void
hash_table_insert
( hash_table *hash_table,
  dague_hashtable_item_t *item,
  uint32_t hash )
{
    dague_list_item_t *current_item = (dague_list_item_t *)item;

    dague_list_t *item_list  = hash_table->item_list[hash];

    dague_list_lock ( item_list );
    OBJ_RETAIN(current_item);
    dague_list_nolock_push_back ( item_list, current_item );
    dague_list_unlock ( item_list );
}

/* Function to find element in the hash table (not thread safe)
 * Arguments:
 * Returns:
 */
void *
hash_table_find
( hash_table *hash_table,
  uint64_t key, uint32_t hash )
{
    dague_hashtable_item_t *current_item;
    dague_list_t *item_list = hash_table->item_list[hash];

    current_item = (dague_hashtable_item_t *) DAGUE_LIST_ITERATOR_FIRST(item_list);

    /* Iterating the list to check if we have the element */
    while( current_item != (dague_hashtable_item_t *) DAGUE_LIST_ITERATOR_END(item_list) ) {
        if( current_item->key == key ) {
            return (void *)current_item;
        }
        dague_list_item_t *item = &(current_item->list_item);
        current_item = (dague_hashtable_item_t *)DAGUE_LIST_ITERATOR_NEXT(item);
    }

    return (void *)NULL;
}

/* Function to remove element from the hash table (thread safe)
 * Arguments:
 * Returns:
 */
void *
hash_table_remove
( hash_table *hash_table,
  uint64_t key, uint32_t hash )
{
    dague_list_t *item_list = hash_table->item_list[hash];
    dague_list_item_t *current_item = hash_table_find ( hash_table, key, hash );

    if( current_item != NULL ) {
        dague_list_lock ( item_list );
        OBJ_RELEASE(current_item);
        if( current_item->super.obj_reference_count == 1 ) {
#if defined(DAGUE_DEBUG_ENABLE)
            assert(current_item->refcount == 1);
#endif
            dague_list_nolock_remove ( item_list, current_item );
#if defined(DAGUE_DEBUG_ENABLE)
            assert(current_item->refcount == 0);
#endif
        }
        dague_list_unlock ( item_list );
    }

    return current_item;
}
