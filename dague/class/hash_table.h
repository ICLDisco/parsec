#include "dague/dague_internal.h"

typedef struct generic_hash_table hash_table;

/* Function pointer of hash function for hash_table */
typedef uint32_t (hash_fn)(uintptr_t key, int size);


/* One type of hash table for task, tiles and functions */
struct generic_hash_table {
    dague_object_t  super; 
    int             size;
    hash_fn         *hash;
    void            **buckets;
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(hash_table);

/* Function to create generic hash table
 * Arguments:   - Size of the hash table, or the total number of bucket the
                  table will hold (int) 
                - Size of each bucket the table will hold (int)
 * Returns:     - The hash table (hash_table *)
*/
void
hash_table_init(hash_table *obj, int size_of_table, 
                int size_of_each_bucket, hash_fn *hash);

/* Function to destroy generic hash table
 * Arguments:   - hash table (hash_table *)
 * Returns:     - void
*/
void
hash_table_fini(hash_table *hash_table);
