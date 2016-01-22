#include "tree_dist.h"
#include <dague/devices/device.h>
#include <dague/data.h>
#include <dague/vpmap.h>

#define DEBUG 0

#if DEBUG
#define debug(toto...) fprintf(stderr, toto)
static void debug_check_tree(tree_dist_t *tree)
{
    int i, p, l, r;
    for(i = 0; i < tree->nb_nodes; i++) {
        p = tree->nodes[i]->nid_parent;
        l = tree->nodes[i]->nid_left;
        r = tree->nodes[i]->nid_right;
        assert( (l==-1) || (tree->nodes[l]->nid_parent == i) );
        assert( (r==-1) || (tree->nodes[r]->nid_parent == i) );
        assert( (p==-1) || (tree->nodes[p]->nid_left == i) || (tree->nodes[p]->nid_right == i) );
#if DEBUG_VERBOSE
        debug("[%d](%d, %d) parent: [%d](%d, %d), left child: [%d](%d, %d), right child: [%d](%d, %d)\n",
              i, tree->nodes[i]->l, tree->nodes[i]->n,
              p, p == -1 ? -1 : tree->nodes[p]->l, p == -1 ? -1 : tree->nodes[p]->n,
              l, l == -1 ? -1 : tree->nodes[l]->l, l == -1 ? -1 : tree->nodes[l]->n,
              r, r == -1 ? -1 : tree->nodes[r]->l, r == -1 ? -1 : tree->nodes[r]->n);
#endif
    }
}
#else
#define debug(toto...)         do {} while(0)
#define debug_check_tree(tree) do {} while(0)
#endif

/***********************************************************************************************
 * Internals
 ***********************************************************************************************/
static int tree_dist_hash_key(tree_dist_t *tree, int l, int n)
{
    return (l ^ n) % tree->hash_size;
}

static int tree_lookup_nid(tree_dist_t *tree, int l, int n)
{
    int hash_key = tree_dist_hash_key(tree, l, n);
    tree_dist_node_t *node;
    int nid;

    pthread_rwlock_rdlock(&tree->hash_buckets[hash_key].rw_lock);
    node = tree->hash_buckets[hash_key].first;
    while( NULL != node ) {
        if( node->l == l && node->n == n ) {
            nid = node->nid;
            pthread_rwlock_unlock(&tree->hash_buckets[hash_key].rw_lock);
            return nid;
        }
        node = node->next_in_hash;
    }
    pthread_rwlock_unlock(&tree->hash_buckets[hash_key].rw_lock);
    return -1;
}

int tree_dist_depth(tree_dist_t *tree)
{
    return tree->depth;
}

static int tree_lookup_or_allocate_nid(tree_dist_t *tree, int l, int n)
{
    tree_dist_node_t *node;
    int nid, cd;
    int hash_key = tree_dist_hash_key(tree, l, n);
    size_t new_size;

    nid =  tree_lookup_nid(tree, l, n);
    if(-1 != nid )
        return nid;

    cd = tree->depth;
    while( l > cd ) {
        if(__sync_bool_compare_and_swap(&tree->depth, cd, l))
            break;
        cd = l;
    }

    nid = __sync_fetch_and_add(&tree->nb_nodes, 1);
    while( tree->nb_nodes >= tree->allocated_nodes ) {
        pthread_mutex_lock(&tree->resize_lock);
        if( tree->nb_nodes >= tree->allocated_nodes ) {
            new_size = (size_t)(nid + 64) > 2*tree->allocated_nodes ? (size_t)(nid + 64) : 2*tree->allocated_nodes;
            tree->nodes = realloc(tree->nodes, new_size * sizeof(tree_dist_node_t*));
            tree->allocated_nodes = new_size;
        }
        pthread_mutex_unlock(&tree->resize_lock);
    }
    node = (tree_dist_node_t*)malloc(sizeof(tree_dist_node_t));
    node->next_in_hash = NULL;
    node->nid = nid;
    node->l = l;
    node->n = n;
    node->data = NULL;
    if(l > 0 ) {
        node->nid_parent = tree_lookup_nid(tree, l-1, n/2);
        if( node->nid_parent != -1 ) {
            if( (n%2) == 0 ) {
                assert( tree->nodes[node->nid_parent] != NULL );
                assert( tree->nodes[node->nid_parent]->nid_left == -1 );
                tree->nodes[node->nid_parent]->nid_left = nid;
            } else {
                assert( tree->nodes[node->nid_parent] != NULL );
                assert( tree->nodes[node->nid_parent]->nid_right == -1 );
                tree->nodes[node->nid_parent]->nid_right = nid;
            }
        }
    } else {
        node->nid_parent = -1;
    }
    node->nid_left = tree_lookup_nid(tree, l+1, n*2);
    if( node->nid_left != -1 ) {
        tree->nodes[node->nid_left]->nid_parent = nid;
    }
    node->nid_right = tree_lookup_nid(tree, l+1, n*2+1);
    if( node->nid_right != -1 ) {
        tree->nodes[node->nid_right]->nid_parent = nid;
    }
    tree->nodes[nid] = node;

    debug("**Created [%d](%d, %d) in tree %p\n", nid, l, n, tree);
    debug_check_tree(tree);

    pthread_rwlock_wrlock(&tree->hash_buckets[hash_key].rw_lock);
    node->next_in_hash = tree->hash_buckets[hash_key].first;
    tree->hash_buckets[hash_key].first = node;
    pthread_rwlock_unlock(&tree->hash_buckets[hash_key].rw_lock);
    __sync_add_and_fetch(&tree->dirty, 1);
    return nid;
}

static void tree_compute_values(tree_dist_t *tree)
{
    int nid;
    int nb_leaves;
    int dirty;

    do {
        dirty = tree->dirty;
        nb_leaves = 0;
        for(nid = 0; nid < (int)tree->nb_nodes; nid++) {
            if( nid >= (int)tree->allocated_nodes )
                break;
            if( tree->nodes[nid] == NULL )
                continue;
            if( tree->nodes[nid]->nid_left == -1 &&
                tree->nodes[nid]->nid_right == -1 )
                nb_leaves++;
        }
        tree->nb_potential_nodes = 2*nb_leaves - 1; /**< This assumes that the tree is complete by part */
    } while(! __sync_bool_compare_and_swap(&tree->dirty, dirty, 0) );
}

/***********************************************************************************************
 * parsec data distribution interface
 ***********************************************************************************************/

static dague_data_key_t tree_dist_data_key(dague_ddesc_t *desc, ...)
{
    va_list ap;
    int l, n;
    int nid;
    va_start(ap, desc);
    l = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_nid((tree_dist_t*)desc, l, n);
    return nid;
}

static uint32_t tree_dist_rank_of_key(dague_ddesc_t *desc, dague_data_key_t k)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    assert(k < tree->nb_nodes);
    assert(NULL != tree->nodes[k]);
    return tree->nodes[k]->n % tree->super.nodes;
}

static uint32_t tree_dist_rank_of(dague_ddesc_t *desc, ...)
{
    va_list ap;
    int l, n;
    int nid;
    va_start(ap, desc);
    l = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_nid((tree_dist_t*)desc, l, n);
    return tree_dist_rank_of_key(desc, nid);
}

static dague_data_t* tree_dist_data_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    void *pos;
    tree_buffer_t *buffer;
    assert(key < tree->nb_nodes);
    assert(tree->nodes[key] != NULL);
    if( tree->nodes[key]->data == NULL ) {
        assert(tree->super.myrank == tree_dist_rank_of_key(desc, key));
        {
            pthread_mutex_lock(&tree->buffer_lock);
            if( tree->buffers == NULL || (tree->buffers->buffer_use + sizeof(node_t) > tree->buffers->buffer_size) ) {
                buffer = (tree_buffer_t *)malloc(sizeof(tree_buffer_t));
                buffer->prev = tree->buffers;
                buffer->buffer_size = (tree->buffers == NULL) ? (sizeof(node_t)) : (2 * tree->buffers->buffer_size);
                buffer->buffer_use  = 0;
                buffer->buffer = (void*)calloc(buffer->buffer_size, 1);
                tree->buffers = buffer;
            }
            pos = tree->buffers->buffer + tree->buffers->buffer_use;
            tree->buffers->buffer_use += sizeof(node_t);
            pthread_mutex_unlock(&tree->buffer_lock);
        }
        dague_data_create(&tree->nodes[key]->data, desc, key, pos, sizeof(node_t));
    }
    return tree->nodes[key]->data;
}

static dague_data_t* tree_dist_data_of(dague_ddesc_t *desc, ...)
{
    va_list ap;
    int l, n;
    int nid;
    va_start(ap, desc);
    l = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_nid((tree_dist_t*)desc, l, n);
    return tree_dist_data_of_key(desc, nid);
}

static int32_t tree_dist_vpid_of_key(dague_ddesc_t *desc, dague_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    assert(key < tree->nb_nodes);
    assert(NULL != tree->nodes[key]);
    return tree->nodes[key]->n % vpmap_get_nb_vp();
}

static int32_t tree_dist_vpid_of(dague_ddesc_t *desc, ...)
{
    va_list ap;
    int l, n;
    int nid;
    va_start(ap, desc);
    l = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_nid((tree_dist_t*)desc, l, n);
    return tree_dist_vpid_of_key(desc, nid);
}

static int tree_dist_register_memory(dague_ddesc_t* desc, struct dague_device_s* device)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    return device->device_memory_register(device, desc,
                                          tree->buffers->buffer,
                                          tree->buffers->buffer_use);
}

static int tree_dist_unregister_memory(dague_ddesc_t* desc, struct dague_device_s* device)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    return device->device_memory_unregister(device, desc, tree->buffers->buffer);
}

#ifdef DAGUE_PROF_TRACE
static int tree_dist_key_to_string(dague_ddesc_t *desc, dague_data_key_t key, char * buffer, uint32_t buffer_size)
{
    if( buffer_size > 0 )
        buffer[0] = '\0';
    return DAGUE_SUCCESS;
}
#endif

/***********************************************************************************************
 * Utilitiy functions to move on the tree
 ***********************************************************************************************/

void tree_dist_insert_node(tree_dist_t *tree, node_t *node, int l, int n)
{
    int nid;

    nid = tree_lookup_or_allocate_nid(tree, l, n);
    assert(tree->nodes[nid] != NULL);
    tree_copy_node(tree, nid, node);
}

void tree_dist_insert_data(tree_dist_t *tree, dague_data_t *data, int l, int n)
{
    int nid;
    nid = tree_lookup_or_allocate_nid(tree, l, n);
    printf("Node at %d, %d is %d\n", l, n, nid);
    assert(tree->nodes[nid] != NULL);
    assert(tree->nodes[nid]->data == NULL);
    tree->nodes[nid]->data = data;
    OBJ_RETAIN(data);
}

int tree_dist_number_of_potential_nodes(tree_dist_t *tree)
{
    if(tree->dirty) {
        tree_compute_values(tree);
    }
    debug("**Found %d potential nodes in that tree\n", (int)tree->nb_potential_nodes);
    return tree->nb_potential_nodes;
}

int tree_dist_number_of_nodes(tree_dist_t *tree)
{
    return tree->nb_nodes;
}

int tree_dist_instanciate_node(tree_dist_t *tree, int pnid)
{
    int parent;
    assert( pnid < (int)tree->nb_nodes );
    assert( tree->nodes[pnid] != NULL );
    debug("**Looking for potential node id [%d](%d, %d)\n", pnid, tree->nodes[pnid]->l, tree->nodes[pnid]->n);
    if( tree->nodes[pnid]->l == 0 ) {
        debug("**Node [%d](%d, %d) is the root: we don't need to instanciate above\n", pnid, tree->nodes[pnid]->l, tree->nodes[pnid]->n);
        return pnid;
    }
    if( tree->nodes[pnid]->nid_parent == -1 ) {
        parent = tree_lookup_or_allocate_nid(tree, tree->nodes[pnid]->l-1, tree->nodes[pnid]->n/2);
        assert(tree->nodes[pnid]->nid_parent == parent);
        debug("**  Instanciated node [%d](%d, %d)\n",
               parent, tree->nodes[parent]->l, tree->nodes[parent]->n);
    } else {
        parent = tree->nodes[pnid]->nid_parent;
        debug("**  Node [%d](%d, %d) already existed\n",
               parent, tree->nodes[parent]->l, tree->nodes[parent]->n);
    }
    debug("**  Node [%d](%d, %d) %s a left child, and %s a right child\n",
           pnid, tree->nodes[pnid]->l, tree->nodes[pnid]->n,
           tree->nodes[pnid]->nid_left == -1 ? "doesn't have":"have",
           tree->nodes[pnid]->nid_right == -1 ? "doesn't have":"have");
    return pnid;
}

int tree_dist_level_of_node(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return tree->nodes[nid]->l;
}

int tree_dist_position_of_node(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return tree->nodes[nid]->n;
}

int tree_dist_parent_of_node(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return tree->nodes[nid]->nid_parent;
}

int tree_dist_has_left_child(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return (-1 != tree->nodes[nid]->nid_left);
}

int tree_dist_has_right_child(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return (-1 != tree->nodes[nid]->nid_right);
}

int tree_dist_left_child_of_node(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return tree->nodes[nid]->nid_left;
}

int tree_dist_right_child_of_node(tree_dist_t *tree, int nid)
{
    assert(nid < (int)tree->nb_nodes);
    assert(NULL != tree->nodes[nid]);
    return tree->nodes[nid]->nid_right;
}

void tree_copy_node(tree_dist_t *tree, int nid, node_t *src)
{
    dague_data_copy_t *data_copy;
    node_t *dst;
    assert(nid < (int)tree->nb_nodes);
    assert(tree->nodes[nid] != NULL);
    if( tree->nodes[nid]->data == NULL ) {
        (void)tree_dist_data_of_key(&tree->super, nid);
    }
    data_copy = dague_data_get_copy(tree->nodes[nid]->data, 0);
    dst = (node_t*)dague_data_copy_get_ptr(data_copy);
    memcpy(dst, src, sizeof(node_t));
}

static int node_has_descendents(tree_dist_t *tree, int l, int n)
{
    if(l > tree->depth+1) return 0;
    if( (tree_lookup_nid(tree, l, n) != -1 ) ||
        (tree_lookup_nid(tree, l+1, n*2) != -1) ||
        (tree_lookup_nid(tree, l+1, n*2+1) != -1) ||
        node_has_descendents(tree, l+1, n*2) ||
        node_has_descendents(tree, l+1, n*2+1) )
        return 1;
    return 0;
}

static void walk_tree(FILE *f, tree_dist_t *tree, int l, int n)
{
    int nid = tree_lookup_nid(tree, l, n);
    double s, d;
    dague_data_copy_t *data_copy;
    node_t *node;

    if( nid != -1 ) {
        data_copy = dague_data_get_copy(tree->nodes[nid]->data, 0);
        if( NULL != data_copy ) {
            node = (node_t*)dague_data_copy_get_ptr(data_copy);
            s = node->s;
            d = node->d;
        } else{
            s = 0.0;
            d = 0.0;
        }
        fprintf(f,  "n%d_%d [label=\"[%d:%d,%d](%g, %g)\"];\n", l, n, nid, l, n, s, d);
    } else {
        fprintf(f,  "n%d_%d [label=\"[#:%d,%d](-)\"];\n", l, n, l, n);
    }
    if( node_has_descendents(tree, l+1, n*2) ) {
        fprintf(f, "n%d_%d -> n%d_%d;\n", l, n, l+1, n*2);
        walk_tree(f, tree, l+1, n*2);
    }
    if( node_has_descendents(tree, l+1, n*2+1) ) {
        fprintf(f, "n%d_%d -> n%d_%d;\n", l, n, l+1, n*2+1);
        walk_tree(f, tree, l+1, n*2+1);
    }
}

int tree_dist_to_dotfile(tree_dist_t *tree, char *filename)
{
    FILE *f = fopen(filename, "w");

    if( f == NULL )
        return -1;

    fprintf(f, "digraph G {\n");
    walk_tree(f, tree, 0, 0);
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

int tree_dist_lookup_node(tree_dist_t *tree, int l, int n)
{
    return tree_lookup_nid(tree, l, n);
}

/***********************************************************************************************
 * Tree creation function
 ***********************************************************************************************/

tree_dist_t *tree_dist_create_empty(int myrank, int nodes)
{
    int i;
    tree_dist_t *res;
    res = (tree_dist_t*)malloc(sizeof(tree_dist_t));

    /** Let's take care of the DAGUE data distribution interface first */
    res->super.myrank = myrank;
    res->super.nodes  = nodes;
    res->super.data_key = tree_dist_data_key;
    res->super.rank_of  = tree_dist_rank_of;
    res->super.rank_of_key = tree_dist_rank_of_key;
    res->super.data_of     = tree_dist_data_of;
    res->super.data_of_key = tree_dist_data_of_key;
    res->super.vpid_of     = tree_dist_vpid_of;
    res->super.vpid_of_key = tree_dist_vpid_of_key;
    res->super.register_memory   = tree_dist_register_memory;
    res->super.unregister_memory = tree_dist_unregister_memory;
    res->super.memory_registration_status = MEMORY_STATUS_UNREGISTERED;
    res->super.key_base = NULL;
#ifdef DAGUE_PROF_TRACE
    res->super.key_to_string = tree_dist_key_to_string;
    res->super.key_dim = "";
    res->super.key     = "";
#endif

    /** Then, the tree-specific info */
    pthread_mutex_init(&res->buffer_lock, NULL);
    res->buffers = NULL;

    pthread_mutex_init(&res->resize_lock, NULL);
    res->allocated_nodes = 0;
    res->nb_nodes = 0;
    res->nodes = NULL;

    res->hash_size = 1023;
    res->hash_buckets = (tree_hash_bucket_entry_t*)calloc(res->hash_size, sizeof(tree_hash_bucket_entry_t));
    for(i = 0; i < (int)res->hash_size; i++) {
        res->hash_buckets[i].first = NULL;
        pthread_rwlock_init(&res->hash_buckets[i].rw_lock, NULL);
    }

    res->dirty = 0;
    res->nb_potential_nodes = 0;
    res->depth = 0;

    return res;
}
