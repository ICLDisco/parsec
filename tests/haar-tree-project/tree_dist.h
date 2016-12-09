#ifndef _tree_dist_h
#define _tree_dist_h

#include "parsec_config.h"

#include <stdarg.h>
#include <assert.h>
#include <pthread.h>

#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"

typedef struct tree_dist_s tree_dist_t;
typedef struct node_s      node_t;
typedef struct tree_dist_node_s tree_dist_node_t;
typedef struct tree_hash_bucket_entry_s tree_hash_bucket_entry_t;

struct node_s {
    double d;
    double s;
};

void tree_copy_node(tree_dist_t *dst_tree, int dst_nid, node_t *src);
void tree_dist_insert_node(tree_dist_t *tree, node_t *node, int l, int n);
void tree_dist_insert_data(tree_dist_t *tree, parsec_data_t *data, int l, int n);
tree_dist_t *tree_dist_create_empty(int myrank, int nodes);
int tree_dist_lookup_node(tree_dist_t *tree, int n, int l);

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
    int n, l;

    parsec_data_t *data;
    int rank, vpid;
};

/** This structure will hold the actual data */
typedef struct tree_buffer_s {
    struct tree_buffer_s *prev;
    size_t buffer_use;
    size_t buffer_size;
    void *buffer;
} tree_buffer_t;

struct tree_dist_s {
    parsec_ddesc_t super;

    /** Actual memory in which the node_t elements reside */
    pthread_mutex_t     buffer_lock;
    tree_buffer_t      *buffers;

    /** Hash structure: holds nodes one after the other */
    pthread_rwlock_t    resize_lock;
    size_t              allocated_nodes;        /**< Size of of nodes */
    tree_dist_node_t  **nodes;
};

#endif
