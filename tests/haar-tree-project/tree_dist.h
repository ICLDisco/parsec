#ifndef _tree_dist_h
#define _tree_dist_h

#include "dague_config.h"

#include <stdarg.h>
#include <assert.h>
#include <pthread.h>

#include "dague/data_distribution.h"
#include "dague/data_internal.h"

typedef struct tree_dist_s tree_dist_t;
typedef struct node_s      node_t;
typedef struct tree_dist_node_s tree_dist_node_t;
typedef struct tree_hash_bucket_entry_s tree_hash_bucket_entry_t;

struct node_s {
    double d;
    double s;
};

int tree_dist_number_of_potential_nodes(tree_dist_t *treeA);
int tree_dist_number_of_nodes(tree_dist_t *treeA);
int tree_dist_instanciate_node(tree_dist_t *tree, int pnid);
int tree_dist_lookup_node(tree_dist_t *tree, int l, int n);
int tree_dist_level_of_node(tree_dist_t *tree, int nid);
int tree_dist_position_of_node(tree_dist_t *tree, int nid);
int tree_dist_parent_of_node(tree_dist_t *tree, int nid);
int tree_dist_has_left_child(tree_dist_t *tree, int nid);
int tree_dist_has_right_child(tree_dist_t *tree, int nid);
int tree_dist_left_child_of_node(tree_dist_t *tree, int nid);
int tree_dist_right_child_of_node(tree_dist_t *tree, int nid);
static inline int tree_dist_is_leaf(tree_dist_t *tree, int nid) {
    return tree_dist_left_child_of_node(tree, nid) == -1 &&
        tree_dist_right_child_of_node(tree, nid) == -1;
}
void tree_copy_node(tree_dist_t *dst_tree, int dst_nid, node_t *src);
int tree_dist_depth(tree_dist_t *tree);

void tree_dist_insert_node(tree_dist_t *tree, node_t *node, int l, int n);
void tree_dist_insert_data(tree_dist_t *tree, dague_data_t *data, int l, int n);
tree_dist_t *tree_dist_create_empty(int myrank, int nodes);
int tree_dist_depth(tree_dist_t *tree);

int tree_dist_to_dotfile(tree_dist_t *tree, char *filename);

typedef void (tree_walker_node_fn_t)(tree_dist_t *tree, int nid, int l, int n, double s, double d, void *param);
typedef void (tree_walker_child_fn_t)(tree_dist_t *tree, int nid, int pl, int pn, int cl, int cn, void *param);

void walk_tree(tree_walker_node_fn_t *node_fn,
               tree_walker_child_fn_t *child_fn,
               void *fn_param, tree_dist_t *tree);

struct tree_dist_node_s {
    /** nodes are linked by nid in the nodes array
     *  and by (l, n) in the hash table. next_in_hash is used to
     *  handle collisions on the hash buckets. */
    tree_dist_node_t *next_in_hash;
    int nid;
    int l, n;

    dague_data_t *data;
    int rank, vpid;
    int nid_parent, nid_left, nid_right;
};

struct tree_hash_bucket_entry_s {
    tree_dist_node_t *first;
    pthread_rwlock_t rw_lock;
};

/** This structure will hold the actual data */
typedef struct tree_buffer_s {
    struct tree_buffer_s *prev;
    size_t buffer_use;
    size_t buffer_size;
    void *buffer;
} tree_buffer_t;

struct tree_dist_s {
    dague_ddesc_t super;

    /** Actual memory in which the node_t elements reside */
    pthread_mutex_t     buffer_lock;
    tree_buffer_t      *buffers;
    
    /** Array structure: holds nodes one after the other, nids going from 0 to nb_nodes-1 */
    pthread_mutex_t     resize_lock;
    size_t              allocated_nodes;        /**< Size of nodes */
    size_t              nb_nodes;               /**< Number of non-NULL elementsi n nodes */
    tree_dist_node_t  **nodes;

    /** Hash table structure. (l, n) is used as key. */
    size_t                     hash_size;
    tree_hash_bucket_entry_t  *hash_buckets;

    /** High-level information about the tree, updated by a call to tree_refinalize() */
    int dirty;                                  /**< true iff the information below is not up to date */
    int depth;                                  /**< max depth of tree */
    size_t            nb_potential_nodes;
};

#endif
