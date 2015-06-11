#include "dague/class/hash_table.h"

/* To create object of class hash_table that inherits dague_object_t
 * class 
 */
OBJ_CLASS_INSTANCE(hash_table, dague_object_t, NULL, NULL);

/* Function to create generic hash table
 * Arguments:   - Size of the hash table, or the total number of bucket the
                  table will hold (int) 
                - Size of each bucket the table will hold (int)
 * Returns:     - The hash table (hash_table *)
*/
void
hash_table_init(hash_table *obj, int size_of_table, 
                int size_of_each_bucket, hash_fn *hash)
{
    obj->buckets  = calloc(size_of_table, size_of_each_bucket);
    obj->size     = size_of_table;
    obj->hash     = hash;
}

/* Function to destroy generic hash table
 * Arguments:   - hash table (hash_table *)
 * Returns:     - void
*/
void
hash_table_fini(hash_table *hash_table)
{
    free(hash_table->buckets);
    OBJ_RELEASE(hash_table);
}
