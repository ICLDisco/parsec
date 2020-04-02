#include "tree_dist.h"
#include "parsec/mca/device/device.h"
#include "parsec/data.h"
#include "parsec/vpmap.h"
#include <string.h>

/***********************************************************************************************
 * parsec data distribution interface
 ***********************************************************************************************/

static parsec_data_key_t tree_dist_data_key(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    int64_t k;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    k = ((int64_t)n)<<32 | ((int32_t)l);
    return k;
}

static tree_dist_node_t *lookup_or_create_node(tree_dist_t *tree, parsec_data_key_t key)
{
    tree_dist_node_t *node;

    parsec_hash_table_lock_bucket(&tree->nodes, key);
    node = parsec_hash_table_nolock_find(&tree->nodes, key);
    if(NULL == node) {
        node = (tree_dist_node_t*)malloc(sizeof(tree_dist_node_t));
        node->n = (int32_t) ( (key >> 32) );
        node->l = (int32_t) ( (key & 0xffffffff) );
        node->ht_item.key = key;
        node->data = NULL;
        node->rank = node->n % tree->super.nodes;
        node->vpid = node->n / tree->super.nodes % vpmap_get_nb_vp();
        parsec_hash_table_nolock_insert(&tree->nodes, &node->ht_item);
    }
    parsec_hash_table_unlock_bucket(&tree->nodes, key);
    return node;
}

static uint32_t tree_dist_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t k)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    tree_dist_node_t *node = lookup_or_create_node(tree, k);
    assert(NULL != node);
    return node->n % tree->super.nodes;
}

static uint32_t tree_dist_rank_of(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    return tree_dist_rank_of_key(desc, tree_dist_data_key(desc, n, l));
}

static parsec_data_t* tree_dist_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    tree_dist_node_t *node = lookup_or_create_node(tree, key);
    tree_buffer_t *buffer;
    char *pos;
    if( node->data == NULL ) {
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
        parsec_data_create(&node->data, desc, key, pos, sizeof(node_t));
    }
    return node->data;
}

static parsec_data_t* tree_dist_data_of(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    return tree_dist_data_of_key(desc, tree_dist_data_key(desc, n, l));
}

static int32_t tree_dist_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    tree_dist_t *tree = (tree_dist_t*)desc;
    tree_dist_node_t *node = lookup_or_create_node(tree, key);
    assert(NULL != node);
    return node->vpid;
}

static int32_t tree_dist_vpid_of(parsec_data_collection_t *desc, ...)
{
    va_list ap;
    int n, l;
    va_start(ap, desc);
    n = va_arg(ap, int);
    l = va_arg(ap, int);
    va_end(ap);
    return tree_dist_vpid_of_key(desc,  tree_dist_data_key(desc, n, l));
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
    int n = (int32_t) ( (key >> 32) );
    int l = (int32_t) ( (key & 0xffffffff) );

    (void)desc;
    if( buffer_size > 0 ) {
        snprintf(buffer, buffer_size, "%d, %d", n, l);
    }
    return PARSEC_SUCCESS;
}

/***********************************************************************************************
 * Utilitiy functions to move on the tree
 ***********************************************************************************************/

void tree_dist_insert_node(tree_dist_t *tree, node_t *node, int n, int l)
{
    parsec_data_key_t key =  tree_dist_data_key(&tree->super, n, l);
    tree_dist_node_t *tnode = lookup_or_create_node(tree, key);

    tree_copy_node(tnode, node);
}

void tree_dist_insert_data(tree_dist_t *tree, parsec_data_t *data, int n, int l)
{
    parsec_data_key_t key =  tree_dist_data_key(&tree->super, n, l);
    tree_dist_node_t *node = lookup_or_create_node(tree, key);

    node->data = data;
    PARSEC_OBJ_RETAIN(data);
}

void tree_copy_node(tree_dist_node_t *tnode, node_t *src)
{
    parsec_data_copy_t *data_copy;
    node_t *dst;
    data_copy = parsec_data_get_copy(tnode->data, 0);
    dst = (node_t*)parsec_data_copy_get_ptr(data_copy);
    memcpy(dst, src, sizeof(node_t));
}

static void walker_print_node(tree_dist_t *tree, tree_dist_node_t *node, int n, int l, double s, double d, void *param)
{
    FILE *f = (FILE*)param;
    if( node != NULL ) {
        fprintf(f,  "n%d_%d [label=\"[%p:%d,%d](%g, %g)\"];\n", n, l, node, n, l, s, d);
    } else {
        fprintf(f,  "n%d_%d [label=\"[#:%d,%d](-)\"];\n", n, l, n, l);
    }
    (void)tree;
}

static void walker_print_child(tree_dist_t *tree, tree_dist_node_t *node, int pn, int pl, int cn, int cl, void *param)
{
    FILE *f = (FILE*)param;
    fprintf(f, "n%d_%d -> n%d_%d;\n", pn, pl, cn, cl);
    (void)tree;
    (void)node;
}

static int walk_tree_rec(tree_walker_node_fn_t *node_fn,
                          tree_walker_child_fn_t *child_fn,
                          void *fn_param, tree_dist_t *tree, int n, int l)
{
    tree_dist_node_t *tnode;
    parsec_data_key_t key;
    double s = 0.0, d = 0.0;
    node_t *node;
    parsec_data_copy_t *data_copy;
    if( tree->super.nodes > 1 ) {
        fprintf(stderr, "tree_dist does not implement distributed tree walking yet.\n");
        return 0;
    }
    key = tree_dist_data_key(&tree->super, n, l);
    if( (tnode = parsec_hash_table_find(&tree->nodes, key)) != NULL ) {
        data_copy = parsec_data_get_copy(tnode->data, 0);
        if( NULL != data_copy ) {
            node = (node_t*)parsec_data_copy_get_ptr(data_copy);
            s = node->s;
            d = node->d;
        }
        node_fn(tree, tnode, n, l, s, d, fn_param);
        if( walk_tree_rec(node_fn, child_fn, fn_param, tree, n+1, l*2) ) {
            child_fn(tree, tnode, n, l, n+1, l*2, fn_param);
        }
        if( walk_tree_rec(node_fn, child_fn, fn_param, tree, n+1, l*2+1) ) {
            child_fn(tree, tnode, n, l, n+1, l*2+1, fn_param);
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

int tree_dist_has_node(tree_dist_t *tree, int n, int l)
{
    tree_dist_node_t *tnode;
    parsec_data_key_t key;
    key = tree_dist_data_key(&tree->super, n, l);
    tnode = parsec_hash_table_find(&tree->nodes, key);
    if(NULL == tnode)
        return 0;
    return NULL != tnode->data;
}

/***********************************************************************************************
 * Tree creation function
 ***********************************************************************************************/

static parsec_key_fn_t tree_node_hash_fn_struct = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

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
    res->super.memory_registration_status = PARSEC_MEMORY_STATUS_UNREGISTERED;
    res->super.key_base = NULL;
    res->super.key_to_string = tree_dist_key_to_string;
    res->super.key_dim = NULL;
    res->super.key     = "";

    /** Then, the tree-specific info */
    pthread_mutex_init(&res->buffer_lock, NULL);
    res->buffers = NULL;

    parsec_hash_table_init(&res->nodes, offsetof(tree_dist_node_t, ht_item), 8,
                           tree_node_hash_fn_struct, NULL);

    return res;
}

void tree_dist_node_free(void *item, void*cb_data)
{
    tree_dist_node_t *tnode = (tree_dist_node_t*)item;
    tree_dist_t *tree = (tree_dist_t*)cb_data;
    parsec_hash_table_nolock_remove(&tree->nodes, tnode->ht_item.key);
    free(tnode);
}

void tree_dist_free(tree_dist_t *tree)
{
    parsec_hash_table_for_all(&tree->nodes, tree_dist_node_free, tree);
    if(NULL != tree->buffers) free(tree->buffers);
    parsec_data_collection_destroy(&tree->super);
    free(tree);
}
