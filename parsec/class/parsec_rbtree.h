#ifndef PARSEC_RBTREE_H
#define PARSEC_RBTREE_H

#include "parsec/class/list_item.h"


typedef enum parsec_rbtree_color_e { PARSEC_RBTREE_RED, PARSEC_RBTREE_BLACK } parsec_rbtree_color_e;

typedef struct parsec_rbtree_node_t {
    parsec_list_item_t super; // use prev/next for left/right
    parsec_rbtree_color_e color;
    struct parsec_rbtree_node_t *parent;
} parsec_rbtree_node_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_rbtree_node_t);

typedef struct parsec_rbtree_t {
    parsec_rbtree_node_t *root;
    parsec_rbtree_node_t *nil;
    size_t comp_offset;
} parsec_rbtree_t;

typedef void (parsec_rbtree_visitor_cb)(parsec_rbtree_node_t*, void*);

void parsec_rbtree_init(parsec_rbtree_t *tree, size_t compare_offset);

void parsec_rbtree_insert(parsec_rbtree_t *tree, parsec_rbtree_node_t *node);

parsec_rbtree_node_t* parsec_rbtree_minimum(parsec_rbtree_t *tree, parsec_rbtree_node_t *x);

void parsec_rbtree_remove(parsec_rbtree_t *tree, parsec_rbtree_node_t *z);

parsec_rbtree_node_t* parsec_rbtree_find(parsec_rbtree_t *tree, int data);

parsec_rbtree_node_t* parsec_rbtree_find_or_larger(parsec_rbtree_t *tree, int data);

int parsec_rbtree_update_node(parsec_rbtree_t *tree, parsec_rbtree_node_t *node, int newdata);

void parsec_rbtree_foreach(parsec_rbtree_t *tree, parsec_rbtree_visitor_cb *fn, void *cbdata);

#endif // PARSEC_RBTREE_H