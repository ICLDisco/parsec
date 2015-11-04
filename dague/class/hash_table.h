#include "dague_config.h"
#include "dague/class/dague_object.h"
#include "dague/class/list_item.h"
#include "dague/mempool.h"

typedef struct generic_hash_table hash_table;
typedef struct dague_generic_bucket_s dague_generic_bucket_t;

/* Function pointer of hash function for hash_table */
typedef uint32_t (hash_fn)(uintptr_t key, int size);


/* One type of hash table for task, tiles and functions */
struct generic_hash_table {
    dague_object_t    super;
    uint32_t          size;
    hash_fn          *hash;
    void            **bucket_list;
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(hash_table);

/* Generic Bucket for hash tables in PaRSEC.
 */
struct dague_generic_bucket_s {
    dague_list_item_t   super;
    dague_thread_mempool_t *mempool_owner;
    uint64_t            key;
    void               *value;
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_generic_bucket_t);


/* Function to create generic hash table
 * Arguments:   - Size of the hash table, or the total number of bucket the
                  table will hold (int)
                - Size of each bucket the table will hold (int)
 * Returns:     - The hash table (hash_table *)
*/
void
hash_table_init(hash_table *obj, int size_of_table,
                hash_fn    *hash);

/* Function to destroy generic hash table
 * Arguments:   - hash table (hash_table *)
 * Returns:     - void
*/
void
hash_table_fini(hash_table *hash_table, int size_of_table);

/* Function to insert element in the hash table
 * Arguments:
 * Returns:
 */
void
hash_table_insert( hash_table *hash_table, dague_generic_bucket_t *bucket,
                   uintptr_t key, void *value, uint32_t hash );


/* Function to find element in the hash table
 * Arguments:
 * Returns:
 */
void *
hash_table_find( hash_table *hash_table,
                uintptr_t key, uint32_t hash );

/* Function to find element in the hash table
 * Arguments:
 * Returns:
 */
void *
hash_table_remove( hash_table *hash_table,
                uintptr_t key, uint32_t hash );

