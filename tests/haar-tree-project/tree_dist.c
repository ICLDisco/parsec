#include "tree_dist.h"
#include "parsec/mca/device/device.h"
#include "parsec/data.h"
#include "parsec/vpmap.h"
#include <string.h>

/***********************************************************************************************
 * Internals
 ***********************************************************************************************/
static int tree_dist_hash_key(tree_dist_t *tree, int n, int l)
{
    return (l ^ n) % tree->allocated_nodes;
}

static int tree_lookup_node(tree_dist_t *tree, int n, int l, int *id)
{
    int hash_key = tree_dist_hash_key(tree, n, l);
    int i;

    i = hash_key;
    pthread_rwlock_rdlock( &tree->resize_lock );
    do {
        if( NULL == tree->nodes[i] ) {
            /** Empty spot: (n, l) is not in the tree, or it would be here
             *    -- This uses the fact that nodes are inserted but never removed
             *       from the hash table
             */
            pthread_rwlock_unlock( &tree->resize_lock );
            *id = i;
            return 0;
        }
        if( (tree->nodes[i]->n == n) && (tree->nodes[i]->l == l) ) {
            pthread_rwlock_unlock( &tree->resize_lock );
            if( NULL != tree->nodes[i]->data ) {
                *id = i;
                return 1;
            } else {
                *id = -1;
                return 0;
            }
        }
        i = ((i+1) % tree->allocated_nodes);
    } while(i != hash_key);
    pthread_rwlock_unlock( &tree->resize_lock );
    *id = -1;
    return 0;
}

static int tree_lookup_or_allocate_node(tree_dist_t *tree, int n, int l)
{
    tree_dist_node_t *node = NULL;
    int nid, hash_key, i, j;
    size_t old_size;
    tree_dist_node_t **old_nodes;

    if( tree_lookup_node(tree, n, l, &nid) )
        return nid;

  try_again:
    pthread_rwlock_rdlock( &tree->resize_lock );
    i = hash_key = tree_dist_hash_key(tree, n, l);
    do {
        if( NULL == tree->nodes[i] ) {
            /** Empty spot: (n, l) is not in the tree, or it would be here
             *    -- This uses the fact that nodes are inserted but never removed
             *       from the hash table
             *    -- Try to steal that spot
             */
            if( NULL == node ) {
                node = (tree_dist_node_t *)malloc(sizeof(tree_dist_node_t));
                node->n = n;
                node->l = l;
                node->data = NULL;
            }
            if( parsec_atomic_cas_ptr(&tree->nodes[i], NULL, node) ) {
                pthread_rwlock_unlock( &tree->resize_lock );
                return i;
            }
        }
        if( (tree->nodes[i]->n == n) && (tree->nodes[i]->l == l) ) {
            pthread_rwlock_unlock( &tree->resize_lock );
            if( NULL != node )
                free(node);
            return i;
        }
        i = ((i+1) % tree->allocated_nodes);
    } while(i != hash_key);
    pthread_rwlock_unlock( &tree->resize_lock );

    /** The hash is too small... */
    old_size = tree->allocated_nodes;
    pthread_rwlock_wrlock( &tree->resize_lock );
    if( old_size != tree->allocated_nodes ) {
        /** Somebody else reallocated the hash table, I should have room */
        pthread_rwlock_unlock( &tree->resize_lock );
        goto try_again;
    }
    old_nodes = tree->nodes;
    tree->allocated_nodes = (size_t)( 2 * old_size );
    tree->nodes = calloc(tree->allocated_nodes, sizeof(tree_dist_node_t*));
    /** Rehash everything */
    for(i = 0; i < (int)old_size; i++) {
        if( old_nodes[i] ) {
            /** Since I'm the only one messing with the table, *and* the table size
             *  is bigger, I will always find a spot for every element */
            for(j = tree_dist_hash_key(tree, old_nodes[i]->n, old_nodes[i]->l);
                tree->nodes[j] != NULL; j = (j+1)%tree->allocated_nodes)
                /** Nothing to do */ ;
            tree->nodes[j] = old_nodes[i];
        }
    }
    pthread_rwlock_unlock( &tree->resize_lock );
    free(old_nodes);
    goto try_again;
}

/***********************************************************************************************
 * parsec data distribution interface
 ***********************************************************************************************/

static parsec_data_key_t tree_dist_data_key(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    int nid;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_node((tree_dist_t*)desc, n, l);
    return nid;
}

static uint32_t tree_dist_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t k)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    assert(k < tree->allocated_nodes);
    assert(NULL != tree->nodes[k]);
    return tree->nodes[k]->n % tree->super.nodes;
}

static uint32_t tree_dist_rank_of(parsec_data_collection_t *desc, ...)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    va_list ap;
    int n, l;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    (void)l;
    return n % tree->super.nodes;
}

static parsec_data_t* tree_dist_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    void *pos;
    tree_buffer_t *buffer;
    assert(key < tree->allocated_nodes);
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
            pos = (char*)(tree->buffers->buffer) + tree->buffers->buffer_use;
            tree->buffers->buffer_use += sizeof(node_t);
            pthread_mutex_unlock(&tree->buffer_lock);
        }
        parsec_data_create(&tree->nodes[key]->data, desc, key, pos, sizeof(node_t));
    }
    return tree->nodes[key]->data;
}

static parsec_data_t* tree_dist_data_of(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    int nid;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    nid = tree_lookup_or_allocate_node((tree_dist_t*)desc, n, l);
    return tree_dist_data_of_key(desc, nid);
}

static int32_t tree_dist_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    assert(key < tree->allocated_nodes);
    assert(NULL != tree->nodes[key]);
    return tree->nodes[key]->n % vpmap_get_nb_vp();
}

static int32_t tree_dist_vpid_of(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    (void)l;
    return n % vpmap_get_nb_vp();
}

static int
tree_dist_register_memory(parsec_data_collection_t* desc,
                          parsec_device_module_t* device)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    return device->memory_register(device, desc,
                                   tree->buffers->buffer,
                                   tree->buffers->buffer_use);
}

static int
tree_dist_unregister_memory(parsec_data_collection_t* desc,
                            parsec_device_module_t* device)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    return device->memory_unregister(device, desc, tree->buffers->buffer);
}

static int tree_dist_key_to_string(parsec_data_collection_t *desc, parsec_data_key_t key, char * buffer, uint32_t buffer_size)
{
    (void)desc;
    (void)key;
    if( buffer_size > 0 )
        buffer[0] = '\0';
    (void)desc;
    (void)key;
    return PARSEC_SUCCESS;
}

/***********************************************************************************************
 * Utilitiy functions to move on the tree
 ***********************************************************************************************/

void tree_dist_insert_node(tree_dist_t *tree, node_t *node, int n, int l)
{
    int nid;

    nid = tree_lookup_or_allocate_node(tree, n, l);
    assert(tree->nodes[nid] != NULL);
    tree_copy_node(tree, nid, node);
}

void tree_dist_insert_data(tree_dist_t *tree, parsec_data_t *data, int n, int l)
{
    int nid;
    nid = tree_lookup_or_allocate_node(tree, n, l);
    assert(tree->nodes[nid] != NULL);
    assert(tree->nodes[nid]->data == NULL);
    tree->nodes[nid]->data = data;
    OBJ_RETAIN(data);
}

void tree_copy_node(tree_dist_t *tree, int nid, node_t *src)
{
    parsec_data_copy_t *data_copy;
    node_t *dst;
    assert(nid < (int)tree->allocated_nodes);
    assert(tree->nodes[nid] != NULL);
    if( tree->nodes[nid]->data == NULL ) {
        (void)tree_dist_data_of_key(&tree->super, nid);
    }
    data_copy = parsec_data_get_copy(tree->nodes[nid]->data, 0);
    dst = (node_t*)parsec_data_copy_get_ptr(data_copy);
    memcpy(dst, src, sizeof(node_t));
}

static void walker_print_node(tree_dist_t *tree, int nid, int n, int l, double s, double d, void *param)
{
    FILE *f = (FILE*)param;
    if( nid != -1 ) {
        fprintf(f,  "n%d_%d [label=\"[%d:%d,%d](%g, %g)\"];\n", n, l, nid, n, l, s, d);
    } else {
        fprintf(f,  "n%d_%d [label=\"[#:%d,%d](-)\"];\n", n, l, n, l);
    }
    (void)tree;
}

static void walker_print_child(tree_dist_t *tree, int nid, int pn, int pl, int cn, int cl, void *param)
{
    FILE *f = (FILE*)param;
    fprintf(f, "n%d_%d -> n%d_%d;\n", pn, pl, cn, cl);
    (void)tree;
    (void)nid;
}

static int walk_tree_rec(tree_walker_node_fn_t *node_fn,
                          tree_walker_child_fn_t *child_fn,
                          void *fn_param, tree_dist_t *tree, int n, int l)
{
    int nid;
    double s = 0.0, d = 0.0;
    node_t *node;
    parsec_data_copy_t *data_copy;
    if( tree->super.nodes > 1 ) {
        fprintf(stderr, "tree_dist does not implement distributed tree walking yet.\n");
        return 0;
    }
    if( tree_lookup_node(tree, n, l, &nid) ) {
        data_copy = parsec_data_get_copy(tree->nodes[nid]->data, 0);
        if( NULL != data_copy ) {
            node = (node_t*)parsec_data_copy_get_ptr(data_copy);
            s = node->s;
            d = node->d;
        }
        node_fn(tree, nid, n, l, s, d, fn_param);
        if( walk_tree_rec(node_fn, child_fn, fn_param, tree, n+1, l*2) ) {
            child_fn(tree, nid, n, l, n+1, l*2, fn_param);
        }
        if( walk_tree_rec(node_fn, child_fn, fn_param, tree, n+1, l*2+1) ) {
            child_fn(tree, nid, n, l, n+1, l*2+1, fn_param);
        }
        return 1;
    }
    return 0;
}

void walk_tree(tree_walker_node_fn_t *node_fn,
               tree_walker_child_fn_t *child_fn,
               void *fn_param, tree_dist_t *tree)
{
    (void)walk_tree_rec(node_fn, child_fn, fn_param, tree, 0, 0);
}

int tree_dist_to_dotfile(tree_dist_t *tree, char *filename)
{
    FILE *f = fopen(filename, "w");

    if( f == NULL )
        return -1;

    fprintf(f, "digraph G {\n");
    walk_tree(walker_print_node, walker_print_child, f, tree);
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

int tree_dist_lookup_node(tree_dist_t *tree, int n, int l)
{
    int nid;
    if( tree_lookup_node(tree, n, l, &nid) )
        return nid;
    return -1;
}

/***********************************************************************************************
 * Tree creation function
 ***********************************************************************************************/

tree_dist_t *tree_dist_create_empty(int myrank, int nodes)
{
    tree_dist_t *res;
    res = (tree_dist_t*)malloc(sizeof(tree_dist_t));

    /** Let's take care of the PARSEC data distribution interface first */
    parsec_data_collection_init(&res->super, nodes, myrank);
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
    res->super.key_to_string = tree_dist_key_to_string;
    res->super.key_dim = "";
    res->super.key     = "";

    /** Then, the tree-specific info */
    pthread_mutex_init(&res->buffer_lock, NULL);
    res->buffers = NULL;

    pthread_rwlock_init(&res->resize_lock, NULL);
    res->allocated_nodes = 1;
    res->nodes = (tree_dist_node_t **)calloc(1, sizeof(tree_dist_node_t*));

    return res;
}
